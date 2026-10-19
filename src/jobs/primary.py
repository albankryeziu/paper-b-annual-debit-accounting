"""Final annual M1-LP/M2 and rolling M1/M2 campaign at rho=0.15.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from __future__ import annotations


import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


# =============================================================================
# Repository setup
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from tpwrs.execution import CampaignLock, utc_now, validate_policy_solution
from tpwrs.config import scientific_fingerprint
from tpwrs.contracts import nominal_experiment
from jobs.certificates import corrected_m1_certificate
from tpwrs.contracts import ExperimentCase, SolverProfile
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.persistence import load_policy_solution, save_policy_solution
from tpwrs.storage import atomic_write_json
from tpwrs.persistence import load_physical_case
from tpwrs.reporting import episode_metrics, reference_state_metrics, residual_index
from tpwrs.model import solve_policy
from tpwrs.rolling import RollingResult, solve_rolling


# =============================================================================
# Frozen final M1/M2 campaign
# =============================================================================

RHO = 0.15
YEARS = (2029, 2030, 2031)

ANNUAL_METHODS = ("M1_LP", "M2")
ROLLING_METHODS = ("M1", "M2")

CERT_TOL = 2e-6
BALANCE_TOL = 1e-7
ROLLING_BUDGET_TOL = 2e-6
FINAL_BALANCE_TOL = 1e-7


# =============================================================================
# Solver profile
# =============================================================================


def _profile() -> SolverProfile:
    from tpwrs.config import qualified_solver_profile

    return qualified_solver_profile()


# =============================================================================
# Case and experiment construction
# =============================================================================


def _year_case(case: object, year: int) -> object:
    rows = np.flatnonzero(case.time_grid.index.year == year)

    if not len(rows):
        raise RuntimeError(f"Physical case omits required year {year}.")

    return case.slice_time(
        int(rows[0]),
        int(rows[-1]) + 1,
    )


def _experiment(
    case: object,
    year: int,
    method: str,
    mode: str,
    implementation_hash: str,
) -> ExperimentCase:

    suffix = f"ROLL_{method}" if mode == "rolling" else method
    rel = f"year_{year}/rho_0.15/{mode}/{method}"

    return nominal_experiment(
        case,
        case_id=f"Y{year}_RHO15_{suffix}",
        rule=method,
        rho=RHO,
        mode=mode,
        solver_profile=_profile(),
        result_root_id=rel,
        implementation_hash=implementation_hash,
    )


# =============================================================================
# Annual metrics
# =============================================================================


def _policy_metrics(
    case: object,
    solution: object,
) -> dict[str, object]:

    x = np.asarray(solution.x_mw, dtype=float)
    dt = case.time_grid.delta[:, None]

    S, S_norm, max_overload, insecure_intervals, violated_pairs = residual_index(case, x)

    physical_fde = np.sum(
        dt * x / case.capacities_mw[None, :],
        axis=0,
    )

    touch = np.asarray(
        solution.u > 0.5 if solution.u is not None else x > 1e-7,
        dtype=bool,
    )

    touched_duration = np.sum(
        dt * touch,
        axis=0,
    )

    if solution.rule == "M1_LP":
        debit = np.sum(
            dt
            * np.divide(
                x,
                case.xbar_mw,
                out=np.zeros_like(x),
                where=case.xbar_mw > 0,
            ),
            axis=0,
        )
    else:
        debit = physical_fde

    return {
        "schema": "TPWRS_M1_M2_CASE_METRICS_V1",
        "case_id": solution.case_id,
        "rule": solution.rule,
        "solution_hash": solution.solution_hash,
        "physical_case_id": case.physical_case_id,
        "physical_case_hash": case.physical_case_hash,
        "allowance_hash": solution.allowance_hash,
        "S_mw_h_row": S,
        "E_restricted_mwh": float(np.sum(dt * x)),
        "rating_normalized_residual_severity_h": S_norm,
        "maximum_overload_mw": max_overload,
        "insecure_intervals": insecure_intervals,
        "violated_row_state_pairs": violated_pairs,
        "affected_participants": int(np.sum(np.any(touch, axis=0))),
        "touched_duration_sum_h": float(touched_duration.sum()),
        "physical_fde_sum_h": float(physical_fde.sum()),
        "physical_fde_hours": physical_fde.tolist(),
        "allowance_usage_hours": debit.tolist(),
        "stage1_status": solution.stage1.status,
        "stage1_certified_bound": solution.stage1.best_bound,
        "stage2_status": solution.stage2.status,
        # Primary annual cases do not use Stage III.
        "stage3_used": solution.stage3 is not None,
        "selector_status": (None if solution.stage3 is None else solution.stage3.status),
        "replay_passed": bool(solution.replay_summary.get("passed")),
        **reference_state_metrics(case, x),
        **episode_metrics(x, case.time_grid.delta),
    }


# =============================================================================
# Rolling metrics
# =============================================================================


def _rolling_metrics(
    case: object,
    experiment: ExperimentCase,
    result: RollingResult,
) -> dict[str, object]:

    x = np.asarray(result.x_mw, dtype=float)
    dt = case.time_grid.delta[:, None]

    S, S_norm, max_overload, insecure_intervals, violated_pairs = residual_index(case, x)

    physical_fde = np.sum(
        dt * x / case.capacities_mw[None, :],
        axis=0,
    )

    touch = np.asarray(
        result.u > 0.5 if result.u is not None else x > 1e-7,
        dtype=bool,
    )

    touched_duration = np.sum(
        dt * touch,
        axis=0,
    )

    if experiment.rule == "M1":
        debit = touched_duration
    else:
        debit = physical_fde

    budget_violation = float(
        np.max(
            np.maximum(
                debit - experiment.allowance.values,
                0.0,
            )
        )
    )

    post_state = case.post_state_mw(x)

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Rolling solutions use only Stage I and Stage II.
    #
    # Stage III is deterministic tie-breaking and is not part of (S,E).
    # Therefore rolling acceptance requires:
    #
    #   - Stage I OPTIMAL
    #   - Stage II OPTIMAL
    #   - independent epoch replay PASS
    #
    # but DOES NOT require Stage III.
    # -------------------------------------------------------------------------

    epochs_pass = all(
        epoch.stage1.status == "OPTIMAL"
        and epoch.stage2.status == "OPTIMAL"
        and bool(epoch.replay_summary.get("passed"))
        for epoch in result.epoch_solutions
    )

    complete_coverage = bool(result.complete_coverage)

    final_balance_ok = float(np.min(result.final_balance_hours)) >= -FINAL_BALANCE_TOL

    budget_ok = budget_violation <= ROLLING_BUDGET_TOL

    network_balance_error = float(np.max(np.abs(post_state.sum(axis=1))))

    network_balance_ok = network_balance_error <= BALANCE_TOL

    passed = bool(
        complete_coverage and final_balance_ok and budget_ok and network_balance_ok and epochs_pass
    )

    return {
        "schema": "TPWRS_M1_M2_ROLLING_METRICS_V1",
        "status": "PASS" if passed else "FAIL",
        "case_id": experiment.case_id,
        "rule": experiment.rule,
        "experiment_hash": experiment.hash,
        "implementation_hash": experiment.implementation_hash,
        "physical_case_id": case.physical_case_id,
        "physical_case_hash": case.physical_case_hash,
        "allowance_hash": experiment.allowance.hash,
        "allowance_hours": experiment.allowance.values.tolist(),
        "allowance_usage_hours": debit.tolist(),
        "maximum_allowance_violation_h": budget_violation,
        "S_mw_h_row": S,
        "E_restricted_mwh": float(np.sum(dt * x)),
        "rating_normalized_residual_severity_h": S_norm,
        "maximum_overload_mw": max_overload,
        "insecure_intervals": insecure_intervals,
        "violated_row_state_pairs": violated_pairs,
        "affected_participants": int(np.sum(np.any(touch, axis=0))),
        "touched_duration_sum_h": float(touched_duration.sum()),
        "physical_fde_sum_h": float(physical_fde.sum()),
        "physical_fde_hours": physical_fde.tolist(),
        "final_balance_hours": (result.final_balance_hours.tolist()),
        "minimum_final_balance_h": float(np.min(result.final_balance_hours)),
        "complete_coverage": complete_coverage,
        "covered_intervals": result.covered_intervals,
        "epoch_solution_hashes": [epoch.solution_hash for epoch in result.epoch_solutions],
        "all_epoch_stage1_stage2_optimal": epochs_pass,
        "all_epoch_replays_pass": epochs_pass,
        # Explicitly document that rolling Stage III is disabled.
        "stage3_used": False,
        "epoch_stage3_required": False,
        "policy_state_balance_max_abs_mw": (network_balance_error),
        "replay_passed": passed,
        **reference_state_metrics(case, x),
        **episode_metrics(x, case.time_grid.delta),
    }


# =============================================================================
# Immutable experiment identity
# =============================================================================


def _freeze(
    path: Path,
    experiment: ExperimentCase,
) -> None:

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = path / "EXPERIMENT_CASE.json"

    payload = {
        "schema": "TPWRS_IMMUTABLE_EXPERIMENT_CASE_V1",
        "status": "FROZEN",
        "experiment_hash": experiment.hash,
        "experiment": asdict(experiment),
    }

    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))

        if existing.get("experiment_hash") != experiment.hash:
            raise RuntimeError(f"Stale experiment identity occupies {path}; use a new output root.")

    else:
        atomic_write_json(
            target,
            payload,
        )


# =============================================================================
# Annual cache
# =============================================================================


def _annual_cache(
    case: object,
    experiment: ExperimentCase,
    path: Path,
    full: bool,
) -> tuple[bool, object | None, str]:

    required = (
        path / "solution.json",
        path / "solution_arrays.npz",
        path / "metrics.json",
        path / "case.log.json",
        path / "EXPERIMENT_CASE.json",
    )

    if not all(item.is_file() for item in required):
        return (
            False,
            None,
            "incomplete",
        )

    try:
        solution = load_policy_solution(
            path,
            expected_experiment_hash=experiment.hash,
            expected_physical_case_hash=case.physical_case_hash,
        )

        if full:
            validate_policy_solution(
                case,
                experiment,
                solution,
                require_stage3=False,
            )

        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))

        log = json.loads((path / "case.log.json").read_text(encoding="utf-8"))

        if metrics.get("solution_hash") != solution.solution_hash:
            raise RuntimeError("metrics solution hash mismatch")

        if not metrics.get("replay_passed"):
            raise RuntimeError("annual replay did not pass")

        if log.get("status") != "PASS":
            raise RuntimeError("annual job log is not PASS")

        if log.get("solution_hash") != solution.solution_hash:
            raise RuntimeError("annual job log solution hash mismatch")

        return (
            True,
            solution,
            "valid accepted cache",
        )

    except Exception as exc:
        return (
            False,
            None,
            f"{type(exc).__name__}: {exc}",
        )


# =============================================================================
# Annual solve
# =============================================================================


def _run_annual(
    case: object,
    experiment: ExperimentCase,
    path: Path,
) -> tuple[object, str]:

    _freeze(
        path,
        experiment,
    )

    valid, solution, detail = _annual_cache(
        case,
        experiment,
        path,
        True,
    )

    if valid:
        return solution, "REUSE"
    if detail != "incomplete":
        raise RuntimeError(
            f"Existing annual cache is invalid and will not be overwritten: {path}: {detail}"
        )

    started = time.perf_counter()
    started_utc = utc_now()

    atomic_write_json(
        path / "case.log.json",
        {
            "schema": "TPWRS_HEAVY_JOB_LOG_V1",
            "status": "RUNNING",
            "started_utc": started_utc,
            "experiment_hash": experiment.hash,
            "cache_rejection": detail,
        },
    )

    try:
        # Primary annual solutions use only Stage I and Stage II; Stage III is not part of the reported scientific value.
        solution = solve_policy(
            case,
            experiment,
            use_stage3=False,
            checkpoint_dir=path,
        )

        acceptance = validate_policy_solution(
            case,
            experiment,
            solution,
            require_stage3=False,
        )

        save_policy_solution(
            solution,
            path,
        )

        metrics = {
            **_policy_metrics(
                case,
                solution,
            ),
            "implementation_hash": experiment.implementation_hash,
            "allowance_hours": (experiment.allowance.values.tolist()),
        }

        atomic_write_json(
            path / "metrics.json",
            metrics,
        )

        atomic_write_json(
            path / "case.log.json",
            {
                "schema": "TPWRS_HEAVY_JOB_LOG_V1",
                "status": "PASS",
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "runtime_s": (time.perf_counter() - started),
                "experiment_hash": experiment.hash,
                "solution_hash": solution.solution_hash,
                "acceptance": acceptance,
            },
        )

        return solution, "RUN"

    except BaseException as exc:
        atomic_write_json(
            path / "case.log.json",
            {
                "schema": "TPWRS_HEAVY_JOB_LOG_V1",
                "status": "FAILED",
                "started_utc": started_utc,
                "failed_utc": utc_now(),
                "failure": (f"{type(exc).__name__}: {exc}"),
            },
        )

        raise


# =============================================================================
# Rolling serialization
# =============================================================================


def _save_rolling(
    path: Path,
    result: RollingResult,
    metrics: dict[str, object],
) -> None:

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, name = tempfile.mkstemp(
        prefix=".rolling.",
        suffix=".npz",
        dir=path,
    )

    os.close(fd)

    tmp = Path(name)
    arrays = path / "rolling_arrays.npz"

    try:
        payload = {
            "x_mw": result.x_mw,
            "final_balance_hours": (result.final_balance_hours),
        }

        if result.u is not None:
            payload["u"] = result.u

        np.savez_compressed(
            tmp,
            **payload,
        )

        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())

        os.replace(
            tmp,
            arrays,
        )

    finally:
        if tmp.exists():
            tmp.unlink()

    atomic_write_json(
        path / "rolling_result.json",
        {
            **metrics,
            "arrays_file": arrays.name,
            "arrays_sha256": sha256_file(arrays),
        },
    )


# =============================================================================
# Rolling cache
# =============================================================================


def _rolling_cache(
    case: object,
    experiment: ExperimentCase,
    path: Path,
) -> tuple[bool, str]:

    result_path = path / "rolling_result.json"

    arrays_path = path / "rolling_arrays.npz"

    log_path = path / "case.log.json"

    experiment_path = path / "EXPERIMENT_CASE.json"

    required = (
        result_path,
        arrays_path,
        log_path,
        experiment_path,
    )

    if not all(item.is_file() for item in required):
        return (
            False,
            "incomplete",
        )

    try:
        metrics = json.loads(result_path.read_text(encoding="utf-8"))

        job = json.loads(log_path.read_text(encoding="utf-8"))

        if metrics.get("experiment_hash") != experiment.hash:
            raise RuntimeError("rolling experiment hash mismatch")

        if metrics.get("physical_case_hash") != case.physical_case_hash:
            raise RuntimeError("rolling physical-case hash mismatch")

        if metrics.get("implementation_hash") != experiment.implementation_hash:
            raise RuntimeError("rolling implementation hash mismatch")

        if not metrics.get("replay_passed"):
            raise RuntimeError("rolling replay did not pass")

        if metrics.get("stage3_used") is not False:
            raise RuntimeError(
                "rolling cache was not generated under final Stage-I/II-only protocol"
            )

        if job.get("status") != "PASS":
            raise RuntimeError("rolling job log is not PASS")

        if sha256_file(arrays_path) != metrics.get("arrays_sha256"):
            raise RuntimeError("rolling arrays hash mismatch")

        hashes = metrics.get(
            "epoch_solution_hashes",
            [],
        )

        expected_epochs = int(np.ceil(case.n_time / 24.0))

        if len(hashes) != expected_epochs:
            raise RuntimeError("epoch registry incomplete")

        for start, digest in zip(
            range(
                0,
                case.n_time,
                24,
            ),
            hashes,
        ):
            epoch = load_policy_solution(path / "epochs" / f"epoch_{start:06d}")

            if epoch.solution_hash != digest:
                raise RuntimeError(f"epoch {start} solution hash mismatch")

            if epoch.stage1.status != "OPTIMAL":
                raise RuntimeError(f"epoch {start} Stage I is not OPTIMAL")

            if epoch.stage2.status != "OPTIMAL":
                raise RuntimeError(f"epoch {start} Stage II is not OPTIMAL")

            if epoch.stage3 is not None:
                raise RuntimeError(f"epoch {start} unexpectedly contains Stage III")

            if not epoch.replay_summary.get("passed"):
                raise RuntimeError(f"epoch {start} replay failed")

        return (
            True,
            "valid accepted cache",
        )

    except Exception as exc:
        return (
            False,
            f"{type(exc).__name__}: {exc}",
        )


# =============================================================================
# Rolling solve
# =============================================================================


def _run_rolling(
    case: object,
    experiment: ExperimentCase,
    path: Path,
) -> str:

    _freeze(
        path,
        experiment,
    )

    valid, detail = _rolling_cache(
        case,
        experiment,
        path,
    )

    if valid:
        return "REUSE"
    if detail != "incomplete":
        raise RuntimeError(
            f"Existing rolling cache is invalid and will not be overwritten: {path}: {detail}"
        )

    started = time.perf_counter()
    started_utc = utc_now()

    atomic_write_json(
        path / "case.log.json",
        {
            "schema": "TPWRS_HEAVY_JOB_LOG_V1",
            "status": "RUNNING",
            "started_utc": started_utc,
            "experiment_hash": experiment.hash,
            "cache_rejection": detail,
            "stage3_used": False,
        },
    )

    try:
        # ---------------------------------------------------------------------
        # FINAL ROLLING PROTOCOL:
        #
        # Stages I and II only.
        #
        # Stage III is omitted because it is deterministic tie-breaking and is
        # not part of the reported lexicographic value (S,E). Omitting it saves
        # one optimization per rolling epoch without changing the scientific
        # Stage-I/Stage-II objectives.
        # ---------------------------------------------------------------------

        result = solve_rolling(
            case,
            experiment,
            use_stage3=False,
            checkpoint_root=(path / "epochs"),
        )

        metrics = _rolling_metrics(
            case,
            experiment,
            result,
        )

        if not metrics["replay_passed"]:
            raise RuntimeError("full chronological replay failed")

        _save_rolling(
            path,
            result,
            metrics,
        )

        atomic_write_json(
            path / "case.log.json",
            {
                "schema": "TPWRS_HEAVY_JOB_LOG_V1",
                "status": "PASS",
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "runtime_s": (time.perf_counter() - started),
                "experiment_hash": experiment.hash,
                "epoch_count": len(result.epoch_solutions),
                "stage3_used": False,
                "full_replay_passed": True,
            },
        )

        return "RUN"

    except BaseException as exc:
        atomic_write_json(
            path / "case.log.json",
            {
                "schema": "TPWRS_HEAVY_JOB_LOG_V1",
                "status": "FAILED",
                "started_utc": started_utc,
                "failed_utc": utc_now(),
                "stage3_used": False,
                "failure": (f"{type(exc).__name__}: {exc}"),
            },
        )

        raise


# =============================================================================
# Dry-run campaign plan
# =============================================================================


def _plan(
    case: object,
    results: Path,
    implementation_hash: str,
) -> dict[str, object]:

    jobs: list[dict[str, object]] = []

    for year in YEARS:
        year_case = _year_case(
            case,
            year,
        )

        for mode, methods in (
            ("annual", ANNUAL_METHODS),
            ("rolling", ROLLING_METHODS),
        ):
            for method in methods:
                experiment = _experiment(
                    year_case,
                    year,
                    method,
                    mode,
                    implementation_hash,
                )

                rel = f"year_{year}/rho_0.15/{mode}/{method}"

                if mode == "annual":
                    valid = _annual_cache(
                        year_case,
                        experiment,
                        results / rel,
                        False,
                    )[0]
                else:
                    valid = _rolling_cache(
                        year_case,
                        experiment,
                        results / rel,
                    )[0]

                jobs.append(
                    {
                        "job_id": experiment.case_id,
                        "phase": mode,
                        "year": year,
                        "rho": RHO,
                        "method": method,
                        "physical_case_hash": (year_case.physical_case_hash),
                        "allowance_hash": (experiment.allowance.hash),
                        "dependencies": [
                            "frozen physical case",
                            "solver environment",
                        ],
                        "output": rel,
                        "cache_state": ("VALID" if valid else "ABSENT_OR_INVALID"),
                        "action": ("REUSE" if valid else "FUTURE_RUN"),
                    }
                )

    annual_count = sum(job["phase"] == "annual" for job in jobs)

    rolling_count = sum(job["phase"] == "rolling" for job in jobs)

    passed = annual_count == 6 and rolling_count == 6 and len(jobs) == 12

    return {
        "schema": "TPWRS_FINAL_M1_M2_DRY_RUN_V2",
        "status": ("PASS" if passed else "FAIL"),
        "created_utc": utc_now(),
        "no_solver_model_built": True,
        "no_scientific_optimization": True,
        "heavy_job_concurrency": 1,
        "annual_case_count": annual_count,
        "rolling_case_count": rolling_count,
        "total_heavy_policy_case_count": len(jobs),
        "m3_case_count": 0,
        "card_or_training_job_count": 0,
        "rolling_stage3_used": False,
        "physical_case_id": case.physical_case_id,
        "physical_case_hash": case.physical_case_hash,
        "implementation_hash": implementation_hash,
        "jobs": jobs,
    }


# =============================================================================
# Restart / lock self-test
# =============================================================================


def _self_test() -> int:

    base = ROOT / "artifacts" / "tmp"
    base.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="m1_m2_resume_",
        dir=base,
    ) as name:
        root = Path(name)
        lock_path = root / ".campaign.lock"

        duplicate_rejected = False

        with CampaignLock(
            lock_path,
            "synthetic",
        ):
            try:
                CampaignLock(
                    lock_path,
                    "synthetic",
                ).acquire()
            except Exception:
                duplicate_rejected = True

        reacquired = False

        with CampaignLock(
            lock_path,
            "synthetic",
        ):
            reacquired = True

        status_path = root / "RUN_STATUS.json"

        atomic_write_json(
            status_path,
            {
                "campaign_hash": "synthetic",
                "status": "RUNNING",
                "jobs": {"job_a": {"status": "PASS"}},
            },
        )

        loaded = json.loads(status_path.read_text(encoding="utf-8"))

        loaded["status"] = "RUNNING"
        loaded["resumed_utc"] = utc_now()
        loaded["invocation_count"] = 2

        atomic_write_json(
            status_path,
            loaded,
        )

        resumed = json.loads(status_path.read_text(encoding="utf-8"))

        resume_ok = (
            resumed.get("campaign_hash") == "synthetic"
            and resumed.get(
                "jobs",
                {},
            )
            .get(
                "job_a",
                {},
            )
            .get("status")
            == "PASS"
            and resumed.get("invocation_count") == 2
        )

    payload = {
        "schema": "TPWRS_M1_M2_RESUME_SELF_TEST_V1",
        "status": ("PASS" if (duplicate_rejected and reacquired and resume_ok) else "FAIL"),
        "duplicate_lock_rejected": (duplicate_rejected),
        "released_lock_reacquired": (reacquired),
        "completed_job_preserved_on_resume": (resume_ok),
        "scientific_optimization_executed": False,
        "created_utc": utc_now(),
    }

    atomic_write_json(
        ROOT / "artifacts" / "M1_M2_RESUME_SELF_TEST.json",
        payload,
    )

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )

    return 0 if payload["status"] == "PASS" else 1


# =============================================================================
# Campaign status helper
# =============================================================================


def _update(
    manifest: dict[str, object],
    path: Path,
    job_id: str,
    status: str,
    **extra: object,
) -> None:

    previous = dict(
        manifest.setdefault(
            "jobs",
            {},
        ).get(
            job_id,
            {},
        )
    )

    previous.update(
        {
            "status": status,
            "updated_utc": utc_now(),
            **extra,
        }
    )

    manifest["jobs"][job_id] = previous

    manifest["completed_cases"] = sorted(
        key for key, value in manifest["jobs"].items() if value.get("status") == "PASS"
    )

    manifest["failed_cases"] = sorted(
        key for key, value in manifest["jobs"].items() if value.get("status") == "FAILED"
    )

    atomic_write_json(
        path,
        manifest,
    )


# =============================================================================
# Main campaign execution
# =============================================================================


def _execute(
    case: object,
    args: argparse.Namespace,
    implementation_hash: str,
    manifest: dict[str, object],
    status_path: Path,
) -> None:

    for year in YEARS:
        year_case = _year_case(
            case,
            year,
        )

        annual: dict[str, object] = {}
        annual_experiments: dict[str, object] = {}
        rolling: dict[str, dict[str, object]] = {}

        # ---------------------------------------------------------------------
        # Annual M1-LP and M2
        # ---------------------------------------------------------------------

        for method in ANNUAL_METHODS:
            experiment = _experiment(
                year_case,
                year,
                method,
                "annual",
                implementation_hash,
            )
            annual_experiments[method] = experiment

            rel = Path(experiment.result_root_id)

            _update(
                manifest,
                status_path,
                experiment.case_id,
                "RUNNING",
                output=str(rel),
                experiment_hash=experiment.hash,
            )

            try:
                solution, action = _run_annual(
                    year_case,
                    experiment,
                    args.results / rel,
                )

                annual[method] = solution

                _update(
                    manifest,
                    status_path,
                    experiment.case_id,
                    "PASS",
                    action=action,
                    solution_hash=(solution.solution_hash),
                )

            except BaseException as exc:
                _update(
                    manifest,
                    status_path,
                    experiment.case_id,
                    "FAILED",
                    failure=(f"{type(exc).__name__}: {exc}"),
                )

                raise

        # ---------------------------------------------------------------------
        # Rolling M1 and M2
        # ---------------------------------------------------------------------

        for method in ROLLING_METHODS:
            experiment = _experiment(
                year_case,
                year,
                method,
                "rolling",
                implementation_hash,
            )

            rel = Path(experiment.result_root_id)

            _update(
                manifest,
                status_path,
                experiment.case_id,
                "RUNNING",
                output=str(rel),
                experiment_hash=experiment.hash,
            )

            try:
                action = _run_rolling(
                    year_case,
                    experiment,
                    args.results / rel,
                )

                rolling[method] = json.loads(
                    (args.results / rel / "rolling_result.json").read_text(encoding="utf-8")
                )

                _update(
                    manifest,
                    status_path,
                    experiment.case_id,
                    "PASS",
                    action=action,
                )

            except BaseException as exc:
                _update(
                    manifest,
                    status_path,
                    experiment.case_id,
                    "FAILED",
                    failure=(f"{type(exc).__name__}: {exc}"),
                )

                raise

        # ---------------------------------------------------------------------
        # Strict annual M1 certificate
        # ---------------------------------------------------------------------

        lower_bound = annual["M1_LP"].stage1.best_bound

        if lower_bound is None:
            raise RuntimeError("M1-LP Stage-I certified lower bound is missing.")

        m2_value = annual["M2"].s_value_mw_h_row

        m1lp_value = annual["M1_LP"].s_value_mw_h_row

        rolling_m1_value = rolling["M1"]["S_mw_h_row"]

        strict_certificate = corrected_m1_certificate(
            year_case,
            annual["M2"].x_mw,
            annual_experiments["M2"].allowance.values,
            lower_bound,
        )

        certificate = {
            "schema": "TPWRS_M1_M2_STRICT_CERTIFICATE_R10_V1",
            "year": year,
            "M2_vs_M1": strict_certificate,
            "scalar_bracket": {
                "S_M2": m2_value,
                "S_M1LP_replay_value": (m1lp_value),
                "M1LP_certified_lower_bound": (lower_bound),
                "S_rolling_M1": (rolling_m1_value),
                "M2_le_M1LP": bool(m2_value <= m1lp_value + CERT_TOL),
                "M1LP_lower_le_rolling_M1": bool(lower_bound <= rolling_m1_value + CERT_TOL),
                # The certificate is an annual scalar statement.
                "pointwise_bound_claim": False,
            },
        }

        certificate_path = args.results / f"year_{year}" / "rho_0.15" / "M1_M2_CERTIFICATE.json"

        atomic_write_json(
            certificate_path,
            certificate,
        )
        atomic_write_json(
            certificate_path.with_name("B03_M1_M2_CERTIFICATE.json"),
            {
                **strict_certificate,
                "year": year,
                "scientific_fingerprint": implementation_hash,
                "source_m2_solution_hash": annual["M2"].solution_hash,
                "source_m1lp_solution_hash": annual["M1_LP"].solution_hash,
            },
        )


# =============================================================================
# Main
# =============================================================================


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Authoritative final rho=0.15 M1/M2 campaign runner (12 sequential heavy cases)."
        )
    )

    parser.add_argument(
        "--case",
        type=Path,
        default=(ROOT / "artifacts" / "real_case" / "physical_case" / "FINAL_DUTCH_CASE"),
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=(ROOT / "results_m1_m2_final"),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Synthetic restart/lock test only
    # -------------------------------------------------------------------------

    if args.self_test:
        return _self_test()

    # -------------------------------------------------------------------------
    # Freeze current implementation identity
    # -------------------------------------------------------------------------

    implementation_hash = scientific_fingerprint()

    case = load_physical_case(args.case)

    # -------------------------------------------------------------------------
    # Dry run: no scientific optimization
    # -------------------------------------------------------------------------

    if args.dry_run:
        plan = _plan(
            case,
            args.results,
            implementation_hash,
        )

        args.results.mkdir(
            parents=True,
            exist_ok=True,
        )

        atomic_write_json(
            args.results / "DRY_RUN_PLAN.json",
            plan,
        )

        atomic_write_json(
            ROOT / "artifacts" / "M1_M2_DRY_RUN_STATUS.json",
            plan,
        )

        print(
            json.dumps(
                plan,
                indent=2,
                sort_keys=True,
            )
        )

        return 0 if plan["status"] == "PASS" else 1

    # -------------------------------------------------------------------------
    # Real campaign
    # -------------------------------------------------------------------------

    args.results.mkdir(
        parents=True,
        exist_ok=True,
    )

    campaign_identity = {
        "physical_case_hash": (case.physical_case_hash),
        "years": YEARS,
        "rho": RHO,
        "annual_methods": (ANNUAL_METHODS),
        "rolling_methods": (ROLLING_METHODS),
        "annual_stage3_used": False,
        "rolling_stage3_used": False,
        "implementation_hash": (implementation_hash),
        "solver_profile": asdict(_profile()),
        "heavy_job_concurrency": 1,
    }

    campaign_hash = sha256_json(campaign_identity)

    status_path = args.results / "RUN_STATUS.json"

    with CampaignLock(
        args.results / ".campaign.lock",
        campaign_hash,
    ):
        # ================================================================
        # NEW CAMPAIGN OR AUTOMATIC RESUME
        # ================================================================

        if status_path.is_file():
            manifest = json.loads(status_path.read_text(encoding="utf-8"))

            if manifest.get("campaign_hash") != campaign_hash:
                raise RuntimeError(
                    "Existing output belongs to a different immutable campaign. "
                    "Do not overwrite it. Use a new results directory."
                )

            manifest.update(
                {
                    "status": "RUNNING",
                    "resumed_utc": utc_now(),
                    "invocation_count": (
                        int(
                            manifest.get(
                                "invocation_count",
                                1,
                            )
                        )
                        + 1
                    ),
                }
            )

            print(
                "\n"
                "============================================================\n"
                "M1/M2 CAMPAIGN — AUTOMATIC RESUME\n"
                "============================================================\n"
                f"Completed cases : "
                f"{len(manifest.get('completed_cases', []))}/12\n"
                f"Results root    : {args.results}\n"
                "============================================================\n",
                flush=True,
            )

        else:
            manifest = {
                "schema": "TPWRS_FINAL_M1_M2_RUN_V2",
                "status": "RUNNING",
                "started_utc": utc_now(),
                "invocation_count": 1,
                "campaign_hash": campaign_hash,
                **campaign_identity,
                "physical_case_id": case.physical_case_id,
                "runner_sha256": sha256_file(Path(__file__)),
                "jobs": {},
                "completed_cases": [],
                "failed_cases": [],
            }

            print(
                "\n"
                "============================================================\n"
                "M1/M2 CAMPAIGN — NEW RUN\n"
                "============================================================\n"
                "Total heavy cases : 12\n"
                f"Results root      : {args.results}\n"
                "============================================================\n",
                flush=True,
            )

        atomic_write_json(
            status_path,
            manifest,
        )

        try:
            _execute(
                case,
                args,
                implementation_hash,
                manifest,
                status_path,
            )

            manifest.update(
                {
                    "status": "COMPLETE",
                    "completed_utc": utc_now(),
                }
            )

            atomic_write_json(
                status_path,
                manifest,
            )

            atomic_write_json(
                args.results / "CAMPAIGN_COMPLETE.json",
                {
                    "schema": "TPWRS_FINAL_M1_M2_COMPLETE_V2",
                    "status": "PASS",
                    "campaign_hash": campaign_hash,
                    "completed_utc": utc_now(),
                    "heavy_policy_case_count": 12,
                    "annual_stage3_used": False,
                    "rolling_stage3_used": False,
                },
            )

            print(
                "\n"
                "============================================================\n"
                "M1_M2_CAMPAIGN=COMPLETE\n"
                "============================================================\n",
                flush=True,
            )

            return 0

        except BaseException as exc:
            manifest.update(
                {
                    "status": "FAILED",
                    "failed_utc": utc_now(),
                    "failure": (f"{type(exc).__name__}: {exc}"),
                }
            )

            atomic_write_json(
                status_path,
                manifest,
            )

            raise
