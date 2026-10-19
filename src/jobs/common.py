"""Shared campaign identity, cached-policy and metric helpers.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from __future__ import annotations


from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from tpwrs.execution import validate_policy_solution
from tpwrs.config import scientific_fingerprint
from tpwrs.provenance import require_compatible_status
from tpwrs.contracts import nominal_experiment
from tpwrs.contracts import ExperimentCase, SolverProfile
from tpwrs.storage import sha256_array, sha256_file, sha256_json
from tpwrs.cards import validate_preheldout_audit
from tpwrs.persistence import load_card
from tpwrs.persistence import load_policy_solution, save_policy_solution
from tpwrs.storage import atomic_write_json
from tpwrs.persistence import load_physical_case
from tpwrs.cards import q_from_card
from tpwrs.cards import matched_decomposition
from tpwrs.reporting import episode_metrics, reference_state_metrics, residual_index
from tpwrs.model import solve_policy
from tpwrs.rolling import RollingResult, solve_rolling


EVALUATION_YEARS = (2029, 2030, 2031)
FROZEN_LEXICOGRAPHIC_ABS_TOL = 1e-6
FROZEN_LEXICOGRAPHIC_REL_TOL = 1e-9


def frozen_matched_decomposition(
    s_m2: float,
    s_m3: float,
    s_matched: float,
) -> dict[str, float]:
    """Apply the locked numerical envelope to the matched-exposure identity."""
    return matched_decomposition(
        s_m2,
        s_m3,
        s_matched,
        tol=FROZEN_LEXICOGRAPHIC_ABS_TOL,
        rel_tol=FROZEN_LEXICOGRAPHIC_REL_TOL,
    )


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def exact_profile(name: str = "final_v3_exact") -> SolverProfile:
    from tpwrs.config import qualified_solver_profile

    return qualified_solver_profile()


def year_case(case: Any, year: int) -> Any:
    rows = np.flatnonzero(case.time_grid.index.year == int(year))
    if not len(rows):
        raise RuntimeError(f"Frozen physical case omits year {year}.")
    return case.slice_time(int(rows[0]), int(rows[-1]) + 1)


def load_case(path: Path | None = None) -> Any:
    return load_physical_case(
        path or (ROOT / "artifacts" / "real_case" / "physical_case" / "FINAL_DUTCH_CASE")
    )


def _rho_tag(rho: float) -> str:
    return f"RHO{int(round(100.0 * float(rho))):02d}"


def load_frozen_card(
    case: Any,
    cards_root: Path,
    rho: float,
) -> tuple[Any, dict[str, Any], Path]:
    build_status_path = cards_root / "CARD_BUILD_STATUS.json"
    if not build_status_path.is_file():
        raise FileNotFoundError(f"Card build status is missing: {build_status_path}")
    build_status, _ = require_compatible_status(
        build_status_path,
        use="FROZEN_CARD_LOADING",
    )
    if build_status.get("schema") != "TPWRS_ALLOWANCE_SPECIFIC_CARD_BUILD_V3":
        raise RuntimeError("Card build status schema is not the current contract.")
    if build_status.get("status") != "PASS":
        raise RuntimeError("Card build status is not PASS.")
    tag = _rho_tag(rho)
    card = load_card(cards_root / f"{tag}_SELECTED_CARD.json")
    freeze_path = cards_root / f"{tag}_CARD_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    metadata = freeze.get("freeze_metadata", {})

    if freeze.get("schema") != "TPWRS_V3_CARD_FREEZE_V2":
        raise RuntimeError(f"rho={rho:.2f}: selected-card freeze schema is not current.")
    if freeze.get("selected_card_hash") != card.hash:
        raise RuntimeError(f"rho={rho:.2f}: selected-card hash mismatch.")
    if freeze.get("heldout_access_audit") != "PASS":
        raise RuntimeError(f"rho={rho:.2f}: held-out access audit is not PASS.")
    if metadata.get("physical_case_hash") != case.physical_case_hash:
        raise RuntimeError(f"rho={rho:.2f}: card physical-case hash mismatch.")
    if metadata.get("allowance_contract_hash") != card.allowance_hash:
        raise RuntimeError(f"rho={rho:.2f}: card allowance-contract hash mismatch.")

    model_path = cards_root / f"rho_{rho:.2f}" / f"MODEL_{card.model_family}.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if sha256_file(model_path) != card.source_hashes.get("model_bundle"):
        raise RuntimeError(f"rho={rho:.2f}: frozen model-bundle hash mismatch.")

    audit_path_value = metadata.get("heldout_access_audit_path")
    audit_hash = metadata.get("heldout_access_audit_sha256")
    if not audit_path_value or not audit_hash:
        raise RuntimeError("Card freeze lacks a bound held-out-access inventory.")
    audit_path = Path(audit_path_value)
    if not audit_path.is_absolute():
        audit_path = cards_root / audit_path
    try:
        audit_payload, validated_hash = validate_preheldout_audit(
            audit_path, expected_validation_year=int(freeze.get("validation_year", 2028))
        )
    except Exception as exc:
        raise RuntimeError(f"rho={rho:.2f}: held-out audit artifact is invalid.") from exc
    if validated_hash != audit_hash:
        raise RuntimeError(f"rho={rho:.2f}: held-out audit artifact hash mismatch.")
    if metadata.get("heldout_access_audit_schema") != audit_payload.get("schema"):
        raise RuntimeError(f"rho={rho:.2f}: held-out audit schema binding mismatch.")

    return card, freeze, model_path


def q_for_year(
    case: Any,
    year: int,
    card: Any,
    cards_root: Path,
    features_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    model_path = (
        cards_root
        / f"rho_{0.15 if card.card_id.startswith('RHO15') else 0.03:.2f}"
        / f"MODEL_{card.model_family}.joblib"
    )
    if sha256_file(model_path) != card.source_hashes.get("model_bundle"):
        raise RuntimeError("Frozen model-bundle hash differs from selected card.")

    feature_gate = json.loads((features_root / "FEATURE_GATE.json").read_text(encoding="utf-8"))
    feature_path = features_root / f"F0_{year}.parquet"
    expected = feature_gate.get("outputs", {}).get(str(year), {})
    if (
        feature_gate.get("status") != "PASS"
        or not feature_path.is_file()
        or sha256_file(feature_path) != expected.get("sha256")
    ):
        raise RuntimeError(f"{year}: F0 feature artifact hash mismatch.")

    frame = pd.read_parquet(feature_path)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True).map(
        lambda value: value.isoformat()
    )
    frame["participant_id"] = frame["participant_id"].astype(str)

    expected_index = pd.MultiIndex.from_product(
        [case.time_grid.timestamps_utc, case.participant_ids],
        names=["timestamp_utc", "participant_id"],
    )
    actual_index = pd.MultiIndex.from_frame(frame[["timestamp_utc", "participant_id"]])
    if not actual_index.equals(expected_index):
        raise RuntimeError(f"{year}: F0 rows are not exact time-major physical-case coordinates.")

    bundle = joblib.load(model_path)
    score = np.asarray(bundle.predict(frame), dtype=float).reshape(
        case.n_time,
        case.n_participants,
    )
    q = q_from_card(card, case.participant_directions, score)
    return np.asarray(q, dtype=float), score


def verify_card_allowance(
    experiment: ExperimentCase,
    card: Any,
    freeze: dict[str, Any],
    year: int,
) -> None:
    metadata = freeze.get("freeze_metadata", {})
    expected_hash = metadata.get("annual_allowance_hashes", {}).get(str(year))
    expected_vector_hash = metadata.get("allowance_vector_hashes", {}).get(str(year))
    if freeze.get("selected_card_hash") != card.hash:
        raise RuntimeError("Selected-card freeze hash mismatch.")
    if expected_hash != experiment.allowance.hash:
        raise RuntimeError(f"{experiment.case_id}: nominal allowance differs from card freeze.")
    if expected_vector_hash != sha256_array(experiment.allowance.values):
        raise RuntimeError(
            f"{experiment.case_id}: nominal allowance vector differs from card freeze."
        )


def make_experiment(
    case: Any,
    *,
    case_id: str,
    rule: str,
    rho: float,
    mode: str,
    result_root_id: str,
    implementation_hash: str,
    card: Any | None = None,
    q: np.ndarray | None = None,
    allowance: Any | None = None,
    information_level: str | None = None,
) -> ExperimentCase:
    profile = exact_profile()
    if allowance is None:
        experiment = nominal_experiment(
            case,
            case_id=case_id,
            rule=rule,
            rho=rho,
            mode=mode,
            card_id=card.card_id if rule == "M3" else None,
            card_hash=card.hash if rule == "M3" else None,
            feature_set_id=card.feature_set_id if rule == "M3" else None,
            solver_profile=profile,
            result_root_id=result_root_id,
            implementation_hash=implementation_hash,
        )
    else:
        experiment = ExperimentCase(
            case_id=case_id,
            physical_case_id=case.physical_case_id,
            physical_case_hash=case.physical_case_hash,
            time_grid_id=case.time_grid.grid_id,
            rule=rule,
            allowance=allowance,
            mode=mode,
            card_id=None if card is None else card.card_id,
            card_hash=None if card is None else card.hash,
            feature_set_id=None if card is None else card.feature_set_id,
            information_level=information_level or "POST_M3_MATCHED_EXPOSURE",
            lookahead_hours=48 if mode == "rolling" else None,
            implementation_hours=24 if mode == "rolling" else None,
            solver_profile=profile,
            source_root_id="FINAL_DUTCH_CASE",
            result_root_id=result_root_id,
            implementation_hash=implementation_hash,
        )
    if information_level is not None:
        experiment = replace(experiment, information_level=information_level)
    if q is not None:
        experiment = replace(experiment, rate_matrix_hash=sha256_array(q))
    return experiment


def freeze_experiment(path: Path, experiment: ExperimentCase) -> None:
    path.mkdir(parents=True, exist_ok=True)
    target = path / "EXPERIMENT_CASE.json"
    payload = {
        "schema": "TPWRS_IMMUTABLE_EXPERIMENT_CASE_V1",
        "status": "FROZEN",
        "experiment_hash": experiment.hash,
        "experiment": asdict(experiment),
    }
    if target.is_file():
        current = json.loads(target.read_text(encoding="utf-8"))
        if current.get("experiment_hash") != experiment.hash:
            raise RuntimeError(f"Stale experiment identity occupies {path}; use a new result root.")
    else:
        atomic_write_json(target, payload)


def solution_metrics(case: Any, solution: Any) -> dict[str, Any]:
    x = np.asarray(solution.x_mw, dtype=float)
    dt = case.time_grid.delta[:, None]
    S, S_norm, max_overload, insecure, pairs = residual_index(case, x)
    physical = np.sum(
        dt * x / case.capacities_mw[None, :],
        axis=0,
    )
    touch = np.asarray(
        solution.u > 0.5 if solution.u is not None else x > 1e-7,
        dtype=bool,
    )
    touched = np.sum(dt * touch, axis=0)
    weighted = (
        None
        if solution.q is None
        else np.sum(
            dt * solution.q * x / case.capacities_mw[None, :],
            axis=0,
        )
    )
    post = case.post_state_mw(x)
    return {
        "schema": "TPWRS_POST_PRIMARY_CASE_METRICS_V1",
        "case_id": solution.case_id,
        "rule": solution.rule,
        "solution_hash": solution.solution_hash,
        "physical_case_id": case.physical_case_id,
        "physical_case_hash": case.physical_case_hash,
        "source_hashes": dict(case.source_hashes),
        "allowance_hash": solution.allowance_hash,
        "S_mw_h_row": S,
        "stage1_optimum_mw_h_row": solution.stage1.objective_value,
        "E_restricted_mwh": float(np.sum(dt * x)),
        "rating_normalized_residual_severity_h": S_norm,
        "maximum_overload_mw": max_overload,
        "insecure_intervals": insecure,
        "violated_row_state_pairs": pairs,
        "affected_participants": int(np.sum(np.any(touch, axis=0))),
        "touched_duration_sum_h": float(touched.sum()),
        "physical_fde_sum_h": float(physical.sum()),
        "physical_fde_hours": physical.tolist(),
        "weighted_m3_debit_hours": (None if weighted is None else weighted.tolist()),
        "closure_rounds": solution.closure_rounds,
        "active_row_count": solution.active_row_count,
        "stage1_status": solution.stage1.status,
        "stage1_certified_bound": solution.stage1.best_bound,
        "stage2_status": solution.stage2.status,
        "stage3_status": (None if solution.stage3 is None else solution.stage3.status),
        "background_balance_max_abs_mw": float(np.max(np.abs(case.p_background_mw.sum(axis=1)))),
        "requested_state_balance_max_abs_mw": float(
            np.max(np.abs(case.requested_state_mw.sum(axis=1)))
        ),
        "policy_state_balance_max_abs_mw": float(np.max(np.abs(post.sum(axis=1)))),
        "full_row_replay_max_violation_mw": float(
            solution.replay_summary.get("max_row_violation_mw", np.nan)
        ),
        "replay_passed": bool(solution.replay_summary.get("passed")),
        **reference_state_metrics(case, x),
        **episode_metrics(x, case.time_grid.delta),
    }


def annual_cache(
    case: Any,
    experiment: ExperimentCase,
    root: Path,
    *,
    full: bool,
) -> tuple[bool, Any | None, str]:
    expected = (
        root / "solution.json",
        root / "solution_arrays.npz",
        root / "metrics.json",
        root / "case.log.json",
        root / "EXPERIMENT_CASE.json",
    )
    if not all(path.is_file() for path in expected):
        return False, None, "expected artifacts are incomplete"
    try:
        solution = load_policy_solution(
            root,
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
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        log = json.loads((root / "case.log.json").read_text(encoding="utf-8"))
        if (
            metrics.get("solution_hash") != solution.solution_hash
            or not metrics.get("replay_passed")
            or log.get("status") != "PASS"
            or log.get("solution_hash") != solution.solution_hash
        ):
            raise RuntimeError("Annual cache identity/acceptance marker failed.")
        return True, solution, "valid accepted cache"
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def run_annual(
    case: Any,
    experiment: ExperimentCase,
    q: np.ndarray | None,
    root: Path,
) -> tuple[Any, str]:
    freeze_experiment(root, experiment)
    valid, solution, rejection = annual_cache(
        case,
        experiment,
        root,
        full=True,
    )
    if valid:
        return solution, "REUSE"
    if rejection != "expected artifacts are incomplete":
        raise RuntimeError(
            f"Existing annual cache is invalid and will not be overwritten: {root}: {rejection}"
        )

    started = time.perf_counter()
    started_utc = utc_now()
    atomic_write_json(
        root / "case.log.json",
        {
            "schema": "TPWRS_EXPENSIVE_JOB_LOG_V2",
            "status": "RUNNING",
            "started_utc": started_utc,
            "experiment_hash": experiment.hash,
            "implementation_hash": experiment.implementation_hash,
            "cache_rejection": rejection,
        },
    )
    try:
        solution = solve_policy(
            case,
            experiment,
            q=q,
            use_stage3=False,
            checkpoint_dir=root,
        )
        acceptance = validate_policy_solution(
            case,
            experiment,
            solution,
            require_stage3=False,
        )
        save_policy_solution(solution, root)
        metrics = solution_metrics(case, solution)
        metrics.update(
            {
                "implementation_hash": experiment.implementation_hash,
                "card_id": experiment.card_id,
                "card_hash": experiment.card_hash,
                "feature_set_id": experiment.feature_set_id,
                "rate_matrix_hash": experiment.rate_matrix_hash,
                "allowance_hours": experiment.allowance.values.tolist(),
            }
        )
        atomic_write_json(root / "metrics.json", metrics)
        atomic_write_json(
            root / "case.log.json",
            {
                "schema": "TPWRS_EXPENSIVE_JOB_LOG_V2",
                "status": "PASS",
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "runtime_s": time.perf_counter() - started,
                "experiment_hash": experiment.hash,
                "implementation_hash": experiment.implementation_hash,
                "solution_hash": solution.solution_hash,
                "stage_statuses": [
                    solution.stage1.status,
                    solution.stage2.status,
                    None if solution.stage3 is None else solution.stage3.status,
                ],
                "acceptance": acceptance,
                "cache_rejection": rejection,
            },
        )
        return solution, "RUN"
    except BaseException as exc:
        atomic_write_json(
            root / "case.log.json",
            {
                "schema": "TPWRS_EXPENSIVE_JOB_LOG_V2",
                "status": "FAILED",
                "started_utc": started_utc,
                "failed_utc": utc_now(),
                "runtime_s": time.perf_counter() - started,
                "experiment_hash": experiment.hash,
                "implementation_hash": experiment.implementation_hash,
                "failure": f"{type(exc).__name__}: {exc}",
                "cache_rejection": rejection,
            },
        )
        raise


def rolling_metrics(
    case: Any,
    experiment: ExperimentCase,
    result: RollingResult,
    q: np.ndarray | None,
) -> dict[str, Any]:
    x = np.asarray(result.x_mw, dtype=float)
    dt = case.time_grid.delta[:, None]
    S, S_norm, max_overload, insecure, pairs = residual_index(case, x)
    physical = np.sum(
        dt * x / case.capacities_mw[None, :],
        axis=0,
    )
    touch = np.asarray(
        result.u > 0.5 if result.u is not None else x > 1e-7,
        dtype=bool,
    )
    touched = np.sum(dt * touch, axis=0)
    weighted = (
        None
        if q is None
        else np.sum(
            dt * q * x / case.capacities_mw[None, :],
            axis=0,
        )
    )
    if experiment.rule == "M1":
        debit = touched
    elif experiment.rule == "M2":
        debit = physical
    elif experiment.rule == "M3":
        if weighted is None:
            raise RuntimeError("Rolling M3 metrics require q.")
        debit = weighted
    else:
        raise RuntimeError(f"Unsupported rolling rule {experiment.rule}.")

    budget_violation = float(
        np.max(np.maximum(np.asarray(debit) - experiment.allowance.values, 0.0))
    )
    post = case.post_state_mw(x)
    epochs_pass = all(
        bool(solution.replay_summary.get("passed"))
        and solution.stage1.status == "OPTIMAL"
        and solution.stage2.status == "OPTIMAL"
        and solution.stage3 is None
        for solution in result.epoch_solutions
    )
    replay_passed = bool(
        result.complete_coverage
        and np.min(result.final_balance_hours) >= -1e-7
        and budget_violation <= 2e-6
        and np.max(np.abs(post.sum(axis=1))) <= 1e-7
        and epochs_pass
    )
    return {
        "schema": "TPWRS_POST_PRIMARY_ROLLING_METRICS_V1",
        "status": "PASS" if replay_passed else "FAIL",
        "case_id": experiment.case_id,
        "rule": experiment.rule,
        "experiment_hash": experiment.hash,
        "implementation_hash": experiment.implementation_hash,
        "physical_case_id": case.physical_case_id,
        "physical_case_hash": case.physical_case_hash,
        "source_hashes": dict(case.source_hashes),
        "card_id": experiment.card_id,
        "card_hash": experiment.card_hash,
        "feature_set_id": experiment.feature_set_id,
        "rate_matrix_hash": experiment.rate_matrix_hash,
        "allowance_hash": experiment.allowance.hash,
        "allowance_hours": experiment.allowance.values.tolist(),
        "allowance_usage_hours": np.asarray(debit).tolist(),
        "maximum_allowance_violation_h": budget_violation,
        "S_mw_h_row": S,
        "E_restricted_mwh": float(np.sum(dt * x)),
        "rating_normalized_residual_severity_h": S_norm,
        "maximum_overload_mw": max_overload,
        "insecure_intervals": insecure,
        "violated_row_state_pairs": pairs,
        "affected_participants": int(np.sum(np.any(touch, axis=0))),
        "touched_duration_sum_h": float(touched.sum()),
        "physical_fde_sum_h": float(physical.sum()),
        "physical_fde_hours": physical.tolist(),
        "weighted_m3_debit_hours": (None if weighted is None else weighted.tolist()),
        "final_balance_hours": result.final_balance_hours.tolist(),
        "minimum_final_balance_h": float(np.min(result.final_balance_hours)),
        "complete_coverage": bool(result.complete_coverage),
        "covered_intervals": result.covered_intervals,
        "epoch_solution_hashes": [solution.solution_hash for solution in result.epoch_solutions],
        "all_epoch_stage1_stage2_optimal": epochs_pass,
        "all_epoch_replays_pass": epochs_pass,
        "stage3_used": False,
        "policy_state_balance_max_abs_mw": float(np.max(np.abs(post.sum(axis=1)))),
        "replay_passed": replay_passed,
        **reference_state_metrics(case, x),
        **episode_metrics(x, case.time_grid.delta),
    }


def _save_rolling(
    path: Path,
    result: RollingResult,
    metrics: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays = path / "rolling_arrays.npz"
    fd, name = tempfile.mkstemp(prefix=".rolling.", suffix=".npz", dir=path)
    os.close(fd)
    temporary = Path(name)
    try:
        payload: dict[str, np.ndarray] = {
            "x_mw": result.x_mw,
            "final_balance_hours": result.final_balance_hours,
        }
        if result.u is not None:
            payload["u"] = result.u
        np.savez_compressed(temporary, **payload)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, arrays)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_write_json(
        path / "rolling_result.json",
        {
            **metrics,
            "arrays_file": arrays.name,
            "arrays_sha256": sha256_file(arrays),
        },
    )


def rolling_cache(
    case: Any,
    experiment: ExperimentCase,
    q: np.ndarray | None,
    root: Path,
    *,
    full: bool,
) -> tuple[bool, dict[str, Any] | None, str]:
    result_path = root / "rolling_result.json"
    arrays_path = root / "rolling_arrays.npz"
    log_path = root / "case.log.json"
    freeze_path = root / "EXPERIMENT_CASE.json"
    if not all(path.is_file() for path in (result_path, arrays_path, log_path, freeze_path)):
        return False, None, "expected artifacts are incomplete"
    try:
        metrics = json.loads(result_path.read_text(encoding="utf-8"))
        log = json.loads(log_path.read_text(encoding="utf-8"))
        if (
            metrics.get("status") != "PASS"
            or metrics.get("experiment_hash") != experiment.hash
            or metrics.get("physical_case_hash") != case.physical_case_hash
            or metrics.get("implementation_hash") != experiment.implementation_hash
            or not metrics.get("replay_passed")
            or metrics.get("stage3_used") is not False
            or log.get("status") != "PASS"
        ):
            raise RuntimeError("Rolling identity/replay/job marker failed.")
        if sha256_file(arrays_path) != metrics.get("arrays_sha256"):
            raise RuntimeError("Rolling array hash mismatch.")

        with np.load(arrays_path, allow_pickle=False) as data:
            x = np.asarray(data["x_mw"], dtype=float)
            balance = np.asarray(data["final_balance_hours"], dtype=float)
            u = np.asarray(data["u"], dtype=float) if "u" in data.files else None
        if x.shape != case.requests_mw.shape:
            raise RuntimeError("Rolling x shape mismatch.")
        if balance.shape != (case.n_participants,):
            raise RuntimeError("Rolling balance shape mismatch.")

        hashes = tuple(metrics.get("epoch_solution_hashes", ()))
        expected_epochs = int(np.ceil(case.n_time / 24.0))
        if len(hashes) != expected_epochs:
            raise RuntimeError("Rolling epoch registry is incomplete.")
        epochs: list[Any] = []
        for start, digest in zip(range(0, case.n_time, 24), hashes):
            epoch = load_policy_solution(root / "epochs" / f"epoch_{start:06d}")
            if epoch.solution_hash != digest:
                raise RuntimeError(f"Epoch {start} solution hash mismatch.")
            if epoch.stage1.status != "OPTIMAL" or epoch.stage2.status != "OPTIMAL":
                raise RuntimeError(f"Epoch {start} is not Stage-I/II optimal.")
            if epoch.stage3 is not None:
                raise RuntimeError(f"Epoch {start} unexpectedly contains Stage III.")
            if not epoch.replay_summary.get("passed"):
                raise RuntimeError(f"Epoch {start} replay failed.")
            epochs.append(epoch)

        if full:
            check = rolling_metrics(
                case,
                experiment,
                RollingResult(
                    experiment.case_id,
                    x,
                    u,
                    balance,
                    tuple(epochs),
                    tuple(tuple(map(int, pair)) for pair in metrics.get("covered_intervals", ())),
                    bool(metrics.get("complete_coverage")),
                ),
                q,
            )
            if (
                not check["replay_passed"]
                or abs(check["S_mw_h_row"] - float(metrics["S_mw_h_row"])) > 2e-6
                or abs(check["E_restricted_mwh"] - float(metrics["E_restricted_mwh"])) > 2e-6
            ):
                raise RuntimeError("Independent full rolling replay differs.")
        return True, metrics, "valid accepted cache"
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def run_rolling_case(
    case: Any,
    experiment: ExperimentCase,
    q: np.ndarray | None,
    root: Path,
) -> tuple[dict[str, Any], str]:
    freeze_experiment(root, experiment)
    valid, metrics, rejection = rolling_cache(
        case,
        experiment,
        q,
        root,
        full=True,
    )
    if valid:
        return metrics, "REUSE"
    if rejection != "expected artifacts are incomplete":
        raise RuntimeError(
            f"Existing rolling cache is invalid and will not be overwritten: {root}: {rejection}"
        )

    started = time.perf_counter()
    started_utc = utc_now()
    atomic_write_json(
        root / "case.log.json",
        {
            "schema": "TPWRS_EXPENSIVE_JOB_LOG_V2",
            "status": "RUNNING",
            "started_utc": started_utc,
            "experiment_hash": experiment.hash,
            "implementation_hash": experiment.implementation_hash,
            "cache_rejection": rejection,
            "stage3_used": False,
        },
    )
    try:
        result = solve_rolling(
            case,
            experiment,
            q=q,
            use_stage3=False,
            checkpoint_root=root / "epochs",
        )
        metrics = rolling_metrics(case, experiment, result, q)
        if not metrics["replay_passed"]:
            raise RuntimeError(f"{experiment.case_id}: rolling replay failed.")
        _save_rolling(root, result, metrics)
        atomic_write_json(
            root / "case.log.json",
            {
                "schema": "TPWRS_EXPENSIVE_JOB_LOG_V2",
                "status": "PASS",
                "started_utc": started_utc,
                "completed_utc": utc_now(),
                "runtime_s": time.perf_counter() - started,
                "experiment_hash": experiment.hash,
                "implementation_hash": experiment.implementation_hash,
                "epoch_count": len(result.epoch_solutions),
                "stage3_used": False,
                "full_replay_passed": True,
            },
        )
        return metrics, "RUN"
    except BaseException as exc:
        atomic_write_json(
            root / "case.log.json",
            {
                "schema": "TPWRS_EXPENSIVE_JOB_LOG_V2",
                "status": "FAILED",
                "started_utc": started_utc,
                "failed_utc": utc_now(),
                "runtime_s": time.perf_counter() - started,
                "experiment_hash": experiment.hash,
                "implementation_hash": experiment.implementation_hash,
                "stage3_used": False,
                "failure": f"{type(exc).__name__}: {exc}",
                "cache_rejection": rejection,
            },
        )
        raise


def load_policy_array(directory: Path, *, rolling: bool) -> np.ndarray:
    if rolling:
        metadata = json.loads((directory / "rolling_result.json").read_text(encoding="utf-8"))
        arrays_path = directory / metadata["arrays_file"]
        if sha256_file(arrays_path) != metadata["arrays_sha256"]:
            raise RuntimeError(f"Rolling policy array hash mismatch: {directory}")
        with np.load(arrays_path, allow_pickle=False) as data:
            return np.asarray(data["x_mw"], dtype=float)
    solution = load_policy_solution(directory)
    return np.asarray(solution.x_mw, dtype=float)


def require_m1_m2_freeze(baseline_root: Path, case: Any) -> dict[str, Any]:
    path = baseline_root / "M1_M2_EVIDENCE_FREEZE.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run jobs/primary_freeze.py first; missing {path}")
    payload, identity_mode = require_compatible_status(
        path,
        use="UPSTREAM_STAGE04_NUMERICAL_EVIDENCE",
        fingerprint_fields=("implementation_hash_recorded_by_run",),
    )
    if payload.get("physical_case_hash") != case.physical_case_hash:
        raise RuntimeError("M1/M2 evidence freeze physical-case hash mismatch.")
    if int(payload.get("case_count", -1)) != 12:
        raise RuntimeError("M1/M2 evidence freeze does not contain twelve cases.")
    if identity_mode == "CURRENT_FINGERPRINT":
        for relative, digest in payload.get("source_hashes", {}).items():
            if not (ROOT / relative).is_file() or sha256_file(ROOT / relative) != digest:
                raise RuntimeError(
                    f"Scientific source differs from frozen primary evidence: {relative}"
                )
    for key, record in payload["cases"].items():
        parts = key.split("/")
        year = parts[0]
        if len(parts) == 2:
            target = baseline_root / f"year_{year}/rho_0.15/M1_M2_CERTIFICATE.json"
            if sha256_file(target) != record["sha256"]:
                raise RuntimeError(f"Primary certificate changed: {target}")
            continue
        mode, rule = parts[1:]
        folder = baseline_root / f"year_{year}/rho_0.15/{mode}/{rule}"
        names = {
            "solution_json_sha256": "solution.json",
            "solution_arrays_sha256": "solution_arrays.npz",
            "metrics_sha256": "metrics.json",
            "log_sha256": "case.log.json",
            "arrays_sha256": "rolling_arrays.npz",
            "metadata_sha256": "rolling_result.json",
        }
        for field, name in names.items():
            if field in record and sha256_file(folder / name) != record[field]:
                raise RuntimeError(f"Primary frozen artifact changed: {folder / name}")
    return payload


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp.parquet",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_fitting_tables(card, cards_root, features_root):
    """Bind sensitivity reconstruction to the selected card's frozen fitting tables."""
    for year in (2025, 2026, 2027):
        paths = {
            f"teacher_table_{year}": Path(cards_root) / "rho_0.03" / f"teacher_rows_{year}.parquet",
            f"feature_table_{year}": Path(features_root) / f"F0_{year}.parquet",
        }
        for key, path in paths.items():
            if not path.is_file() or sha256_file(path) != card.source_hashes.get(key):
                raise RuntimeError(f"Frozen fitting table changed or unbound: {key}")
