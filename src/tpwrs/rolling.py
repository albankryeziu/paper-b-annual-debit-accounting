"""Chronological look-ahead controller and implemented-block debit ledger.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from __future__ import annotations


from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import numpy as np

from tpwrs.storage import ContractError, ReplayFailure
from tpwrs.contracts import AllowanceSpec, ExperimentCase
from tpwrs.contracts import PhysicalCase
from tpwrs.model import solve_policy
from tpwrs.contracts import PolicySolution
from tpwrs.persistence import load_policy_solution, save_policy_solution
from tpwrs.storage import sha256_array
from tpwrs.execution import validate_policy_solution


@dataclass(frozen=True)
class RollingResult:
    case_id: str
    x_mw: np.ndarray
    u: np.ndarray | None
    final_balance_hours: np.ndarray
    epoch_solutions: tuple[PolicySolution, ...]
    covered_intervals: tuple[tuple[int, int], ...]
    complete_coverage: bool


def _hours_to_intervals(grid, hours: int) -> int:
    delta = grid.uniform_delta
    if delta is None:
        raise ContractError("Primary rolling controller requires a uniform grid.")
    value = hours / delta
    if abs(value - round(value)) > 1e-9:
        raise ContractError("Rolling hours are not an integer number of intervals.")
    return int(round(value))


def solve_rolling(
    case: PhysicalCase,
    experiment: ExperimentCase,
    *,
    q: np.ndarray | None = None,
    use_stage3: bool = False,
    checkpoint_root: str | Path | None = None,
) -> RollingResult:
    if q is not None:
        q = np.asarray(q, float)
        if q.shape != case.requests_mw.shape or not np.isfinite(q).all() or np.any(q <= 0):
            raise ContractError("Rolling q must be finite, positive and aligned.")
        if (
            experiment.rate_matrix_hash is not None
            and sha256_array(q) != experiment.rate_matrix_hash
        ):
            raise ContractError("Rolling q matrix identity mismatch.")
    if experiment.mode != "rolling":
        raise ContractError("solve_rolling requires a rolling ExperimentCase.")
    look = _hours_to_intervals(case.time_grid, int(experiment.lookahead_hours))
    impl = _hours_to_intervals(case.time_grid, int(experiment.implementation_hours))
    balance = experiment.allowance.values.copy()
    T, N = case.n_time, case.n_participants
    x_full = np.zeros((T, N))
    u_full = np.zeros((T, N)) if experiment.rule == "M1" else None
    epochs = []
    covered = []
    previous = None
    from tpwrs.storage import atomic_write_json
    from tpwrs.execution import utc_now

    for start in range(0, T, impl):
        stop = min(T, start + look)
        implement_stop = min(T, start + impl)
        subcase = case.slice_time(start, stop)
        suballow = AllowanceSpec.explicit(
            case.participant_ids,
            balance,
            source="explicit",
            nominal_rho=experiment.allowance.nominal_rho,
        )
        subexp = ExperimentCase(
            case_id=f"{experiment.case_id}__epoch_{start:06d}",
            physical_case_id=subcase.physical_case_id,
            physical_case_hash=subcase.physical_case_hash,
            time_grid_id=subcase.time_grid.grid_id,
            rule=experiment.rule,
            allowance=suballow,
            mode="annual",
            card_id=experiment.card_id,
            card_hash=experiment.card_hash,
            feature_set_id=experiment.feature_set_id,
            information_level=experiment.information_level,
            solver_profile=experiment.solver_profile,
            epsilon_s_abs=experiment.epsilon_s_abs,
            epsilon_s_rel=experiment.epsilon_s_rel,
            epsilon_e_abs=experiment.epsilon_e_abs,
            epsilon_e_rel=experiment.epsilon_e_rel,
            source_root_id=experiment.source_root_id,
            result_root_id=f"{experiment.result_root_id}/epochs/epoch_{start:06d}",
            implementation_hash=experiment.implementation_hash,
        )
        q_sub = None if q is None else np.asarray(q[start:stop], float)
        if experiment.rule == "M3" and q_sub is not None:
            subexp = replace(subexp, rate_matrix_hash=sha256_array(q_sub))
        checkpoint = (
            None if checkpoint_root is None else Path(checkpoint_root) / f"epoch_{start:06d}"
        )
        warm_x = warm_u = None
        if previous is not None:
            overlap = max(0, min(previous.x_mw.shape[0] - impl, subcase.n_time))
            warm_x = np.full(subcase.requests_mw.shape, np.nan)
            if overlap:
                warm_x[:overlap] = np.minimum(
                    previous.x_mw[impl : impl + overlap], subcase.xbar_mw[:overlap]
                )
            if experiment.rule == "M1" and previous.u is not None:
                warm_u = np.full(subcase.requests_mw.shape, np.nan)
                if overlap:
                    warm_u[:overlap] = previous.u[impl : impl + overlap]

        def solve_epoch():
            import os

            old_job = os.environ.get("TPWRS_CURRENT_JOB_ROOT")
            if checkpoint is not None:
                os.environ["TPWRS_CURRENT_JOB_ROOT"] = str(checkpoint)
            try:
                return solve_policy(
                    subcase,
                    subexp,
                    q=q_sub,
                    use_stage3=use_stage3,
                    checkpoint_dir=checkpoint,
                    initial_x=warm_x,
                    initial_u=warm_u,
                )
            finally:
                if old_job is None:
                    os.environ.pop("TPWRS_CURRENT_JOB_ROOT", None)
                else:
                    os.environ["TPWRS_CURRENT_JOB_ROOT"] = old_job

        if checkpoint is not None:
            atomic_write_json(
                Path(checkpoint_root) / "ROLLING_HEARTBEAT.json",
                {
                    "status": "RUNNING",
                    "created_utc": utc_now(),
                    "start_interval": start,
                    "lookahead_stop": stop,
                    "implemented_stop": implement_stop,
                    "total_intervals": T,
                    "minimum_balance_h": float(balance.min()),
                },
            )
            print(
                f"ROLLING {experiment.case_id}: interval {start}/{T}, balance min={balance.min():.6g} h",
                flush=True,
            )
            if (checkpoint / "solution.json").is_file():
                sol = load_policy_solution(
                    checkpoint,
                    expected_experiment_hash=subexp.hash,
                    expected_physical_case_hash=subcase.physical_case_hash,
                )
                validate_policy_solution(subcase, subexp, sol, require_stage3=use_stage3)
            else:
                sol = solve_epoch()
                validate_policy_solution(subcase, subexp, sol, require_stage3=use_stage3)
                save_policy_solution(sol, checkpoint)
        else:
            sol = solve_epoch()
        previous = sol
        k = implement_stop - start
        x_full[start:implement_stop] = sol.x_mw[:k]
        if u_full is not None and sol.u is not None:
            u_full[start:implement_stop] = sol.u[:k]
        dt = case.time_grid.delta[start:implement_stop, None]
        if experiment.rule == "M1":
            debit = np.sum(dt * u_full[start:implement_stop], axis=0)
        elif experiment.rule == "M2":
            debit = np.sum(dt * x_full[start:implement_stop] / case.capacities_mw[None, :], axis=0)
        elif experiment.rule == "M3":
            if q is None:
                raise ContractError("Rolling M3 requires q.")
            debit = np.sum(
                dt
                * np.asarray(q[start:implement_stop])
                * x_full[start:implement_stop]
                / case.capacities_mw[None, :],
                axis=0,
            )
        else:
            raise ContractError("Rolling primary controller supports M1/M2/M3.")
        balance -= debit
        if np.min(balance) < -10 * experiment.solver_profile.feasibility_tol:
            raise ReplayFailure("Rolling entitlement balance became negative.")
        balance = np.maximum(balance, 0.0)
        epochs.append(sol)
        covered.append((start, implement_stop))
    complete = (
        (
            covered[0][0] == 0
            and covered[-1][1] == T
            and all(covered[i][1] == covered[i + 1][0] for i in range(len(covered) - 1))
        )
        if covered
        else False
    )
    return RollingResult(
        experiment.case_id, x_full, u_full, balance, tuple(epochs), tuple(covered), complete
    )
