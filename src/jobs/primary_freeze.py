"""Validation and evidence freeze for the final duration/depth campaign.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from __future__ import annotations


import os
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpwrs.execution import validate_policy_solution
from tpwrs.config import scientific_fingerprint, scientific_source_hashes
from tpwrs.contracts import AllowanceSpec, ExperimentCase, SolverProfile
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.persistence import load_policy_solution
from tpwrs.storage import atomic_write_json
from tpwrs.persistence import load_physical_case
from tpwrs.reporting import residual_index


YEARS = (2029, 2030, 2031)
ANNUAL = ("M1_LP", "M2")
ROLLING = ("M1", "M2")


def _year_case(case: object, year: int) -> object:
    rows = np.flatnonzero(case.time_grid.index.year == year)
    if not len(rows):
        raise RuntimeError(f"Frozen case omits year {year}.")
    return case.slice_time(int(rows[0]), int(rows[-1]) + 1)


def _experiment_from_file(path: Path) -> ExperimentCase:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "TPWRS_IMMUTABLE_EXPERIMENT_CASE_V1":
        raise RuntimeError(f"Unsupported experiment freeze schema: {path}")
    value = dict(payload["experiment"])
    allowance_value = dict(value["allowance"])
    allowance_value["participant_ids"] = tuple(allowance_value["participant_ids"])
    allowance_value["values_hours"] = tuple(allowance_value["values_hours"])
    value["allowance"] = AllowanceSpec(**allowance_value)
    value["solver_profile"] = SolverProfile(**dict(value["solver_profile"]))
    experiment = ExperimentCase(**value)
    if experiment.hash != payload.get("experiment_hash"):
        raise RuntimeError(f"Experiment freeze hash mismatch: {path}")
    return experiment


def _rolling_audit(
    case: object,
    experiment: ExperimentCase,
    directory: Path,
) -> dict[str, object]:
    metadata_path = directory / "rolling_result.json"
    arrays_path = directory / "rolling_arrays.npz"
    log_path = directory / "case.log.json"
    for path in (metadata_path, arrays_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    log = json.loads(log_path.read_text(encoding="utf-8"))
    if metadata.get("experiment_hash") != experiment.hash:
        raise RuntimeError("Rolling experiment hash mismatch.")
    if metadata.get("physical_case_hash") != case.physical_case_hash:
        raise RuntimeError("Rolling physical-case hash mismatch.")
    if metadata.get("allowance_hash") != experiment.allowance.hash:
        raise RuntimeError("Rolling allowance hash mismatch.")
    if metadata.get("stage3_used") is not False:
        raise RuntimeError("Final M1/M2 rolling evidence must be Stage-I/II only.")
    if metadata.get("replay_passed") is not True or log.get("status") != "PASS":
        raise RuntimeError("Rolling replay/job acceptance marker failed.")
    if sha256_file(arrays_path) != metadata.get("arrays_sha256"):
        raise RuntimeError("Rolling arrays hash mismatch.")

    with np.load(arrays_path, allow_pickle=False) as data:
        x = np.asarray(data["x_mw"], dtype=float)
        balance = np.asarray(data["final_balance_hours"], dtype=float)
        u = np.asarray(data["u"], dtype=float) if "u" in data.files else None

    if x.shape != case.requests_mw.shape:
        raise RuntimeError("Rolling restriction array shape mismatch.")
    if balance.shape != (case.n_participants,):
        raise RuntimeError("Rolling final-balance array shape mismatch.")

    if not np.isfinite(x).all() or not np.isfinite(balance).all():
        raise RuntimeError("Rolling artifact contains nonfinite values.")
    if np.min(x) < -1e-7 or np.max(x - case.xbar_mw) > 1e-7:
        raise RuntimeError("Rolling action bounds failed independent replay.")
    if experiment.rule == "M1":
        if (
            u is None
            or not np.isfinite(u).all()
            or np.max(np.abs(u - np.rint(u))) > 1e-7
            or np.min(u) < -1e-7
            or np.max(u) > 1 + 1e-7
        ):
            raise RuntimeError("Rolling M1 support variables are invalid.")
        if np.max(x - case.xbar_mw * u) > 1e-7 or np.max(case.materiality_floor_mw * u - x) > 1e-7:
            raise RuntimeError("Rolling M1 materiality links failed independent replay.")
    dt = case.time_grid.delta[:, None]
    if experiment.rule == "M1":
        if u is None:
            raise RuntimeError("Rolling M1 cache omits support variables.")
        debit = np.sum(dt * u, axis=0)
    else:
        debit = np.sum(
            dt * x / case.capacities_mw[None, :],
            axis=0,
        )
    maximum_budget_violation = float(np.max(np.maximum(debit - experiment.allowance.values, 0.0)))
    if not np.allclose(balance, experiment.allowance.values - debit, rtol=0, atol=2e-6):
        raise RuntimeError("Rolling terminal ledger does not telescope to annual debit.")
    if maximum_budget_violation > 2e-6:
        raise RuntimeError("Rolling independent allowance replay failed.")
    if float(np.min(balance)) < -1e-7:
        raise RuntimeError("Rolling final balance is negative.")

    S, S_norm, max_overload, insecure, pairs = residual_index(case, x)
    if abs(float(metadata["S_mw_h_row"]) - S) > 2e-6:
        raise RuntimeError("Rolling S differs from independent replay.")
    E = float(np.sum(dt * x))
    if abs(float(metadata["E_restricted_mwh"]) - E) > 2e-6:
        raise RuntimeError("Rolling E differs from independent replay.")

    hashes = tuple(metadata.get("epoch_solution_hashes", ()))
    expected_epochs = int(np.ceil(case.n_time / 24.0))
    if len(hashes) != expected_epochs:
        raise RuntimeError("Rolling epoch registry is incomplete.")
    for start, expected_hash in zip(range(0, case.n_time, 24), hashes):
        epoch = load_policy_solution(directory / "epochs" / f"epoch_{start:06d}")
        length = min(24, case.n_time - start)
        if not np.allclose(x[start : start + length], epoch.x_mw[:length], rtol=0, atol=1e-7):
            raise RuntimeError(f"Rolling implemented block differs from epoch {start}.")
        if epoch.solution_hash != expected_hash:
            raise RuntimeError(f"Rolling epoch {start} hash mismatch.")
        if epoch.stage1.status != "OPTIMAL" or epoch.stage2.status != "OPTIMAL":
            raise RuntimeError(f"Rolling epoch {start} is not Stage-I/II optimal.")
        if epoch.stage3 is not None:
            raise RuntimeError(f"Rolling epoch {start} unexpectedly has Stage III.")
        if not epoch.replay_summary.get("passed"):
            raise RuntimeError(f"Rolling epoch {start} replay failed.")

    return {
        "experiment_hash": experiment.hash,
        "allowance_hash": experiment.allowance.hash,
        "arrays_sha256": sha256_file(arrays_path),
        "metadata_sha256": sha256_file(metadata_path),
        "log_sha256": sha256_file(log_path),
        "S_mw_h_row": S,
        "E_restricted_mwh": E,
        "rating_normalized_residual_severity_h": S_norm,
        "maximum_overload_mw": max_overload,
        "insecure_intervals": insecure,
        "violated_row_state_pairs": pairs,
        "epoch_count": len(hashes),
        "minimum_final_balance_h": float(np.min(balance)),
        "maximum_budget_violation_h": maximum_budget_violation,
        "replay_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the completed M1/M2 evidence for reuse by post-primary scripts; no source "
            "patch is required."
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
        default=ROOT / "results_m1_m2_final",
    )
    args = parser.parse_args()

    status_path = args.results / "RUN_STATUS.json"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    run_status = json.loads(status_path.read_text(encoding="utf-8"))
    if run_status.get("status") != "COMPLETE":
        raise RuntimeError("M1/M2 run is not COMPLETE. Freeze only after all twelve cases finish.")
    if len(run_status.get("completed_cases", ())) != 12:
        raise RuntimeError("M1/M2 run does not contain exactly twelve completed cases.")
    if run_status.get("failed_cases"):
        raise RuntimeError("M1/M2 run manifest contains failed cases.")

    runner_path = ROOT / "jobs" / "primary.py"
    if sha256_file(runner_path) != run_status.get("runner_sha256"):
        raise RuntimeError(
            "The M1/M2 runner changed after the campaign started. "
            "Do not freeze mixed-provenance results."
        )
    current_implementation_hash = scientific_fingerprint()
    if current_implementation_hash != run_status.get("implementation_hash"):
        raise RuntimeError(
            "A scientific module used by the M1/M2 campaign changed after "
            "the campaign started. Freeze the evidence before installing "
            "the post-primary patch."
        )

    case = load_physical_case(args.case)
    if run_status.get("physical_case_hash") != case.physical_case_hash:
        raise RuntimeError("M1/M2 run physical-case hash mismatch.")

    # Bind exactly the same complete source closure used by production identity.
    source_hashes = scientific_source_hashes()

    cases: dict[str, dict[str, object]] = {}
    for year in YEARS:
        year_case = _year_case(case, year)
        for rule in ANNUAL:
            directory = args.results / f"year_{year}/rho_0.15/annual/{rule}"
            experiment = _experiment_from_file(directory / "EXPERIMENT_CASE.json")
            solution = load_policy_solution(
                directory,
                expected_experiment_hash=experiment.hash,
                expected_physical_case_hash=year_case.physical_case_hash,
            )
            acceptance = validate_policy_solution(
                year_case,
                experiment,
                solution,
                require_stage3=False,
            )
            metrics_path = directory / "metrics.json"
            log_path = directory / "case.log.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            log = json.loads(log_path.read_text(encoding="utf-8"))
            if metrics.get("solution_hash") != solution.solution_hash:
                raise RuntimeError(f"{year} {rule}: metrics solution hash mismatch.")
            if log.get("status") != "PASS":
                raise RuntimeError(f"{year} {rule}: job log is not PASS.")
            key = f"{year}/annual/{rule}"
            cases[key] = {
                "experiment_hash": experiment.hash,
                "allowance_hash": experiment.allowance.hash,
                "solution_hash": solution.solution_hash,
                "solution_json_sha256": sha256_file(directory / "solution.json"),
                "solution_arrays_sha256": sha256_file(directory / "solution_arrays.npz"),
                "metrics_sha256": sha256_file(metrics_path),
                "log_sha256": sha256_file(log_path),
                "acceptance": acceptance,
            }

        for rule in ROLLING:
            directory = args.results / f"year_{year}/rho_0.15/rolling/{rule}"
            experiment = _experiment_from_file(directory / "EXPERIMENT_CASE.json")
            cases[f"{year}/rolling/{rule}"] = _rolling_audit(
                year_case,
                experiment,
                directory,
            )

        certificate_path = args.results / f"year_{year}/rho_0.15/M1_M2_CERTIFICATE.json"
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        if not certificate.get("M2_vs_M1", {}).get("passed"):
            # A negative result is scientifically admissible, but it must be
            # explicit in the freeze rather than silently called a certificate.
            certificate_status = "NO_STRICT_SEPARATION"
        else:
            certificate_status = "PASS"
        cases[f"{year}/certificate"] = {
            "status": certificate_status,
            "sha256": sha256_file(certificate_path),
            "payload": certificate,
        }

    payload = {
        "schema": "TPWRS_M1_M2_EVIDENCE_FREEZE_V1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "physical_case_id": case.physical_case_id,
        "physical_case_hash": case.physical_case_hash,
        "run_manifest_sha256": sha256_file(status_path),
        "run_campaign_hash": run_status.get("campaign_hash"),
        "runner_sha256": sha256_file(runner_path),
        "implementation_hash_recorded_by_run": run_status.get("implementation_hash"),
        "complete_source_closure_hash": sha256_json(source_hashes),
        "source_hashes": source_hashes,
        "case_count": 12,
        "cases": cases,
        "claim_boundary": (
            "ANNUAL SCALAR M1 CERTIFICATE ONLY; HOURLY CUMULATIVE CURVES "
            "ARE CHRONOLOGY DIAGNOSTICS, NOT POINTWISE OPTIMALITY BOUNDS"
        ),
    }
    freeze_path = args.results / "M1_M2_EVIDENCE_FREEZE.json"
    atomic_write_json(freeze_path, payload)
    atomic_write_json(
        ROOT / "artifacts" / "M1_M2_EVIDENCE_FREEZE.json",
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
