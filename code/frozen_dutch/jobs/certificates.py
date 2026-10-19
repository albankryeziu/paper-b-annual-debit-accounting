"""Reporting-level conservative witnesses and strict M1 certificates."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
from typing import Any

import numpy as np

from tpwrs.config import ROOT, scientific_fingerprint
from tpwrs.persistence import load_policy_solution
from tpwrs.replay import strict_m1_certificate
from tpwrs.reporting import residual_index
from tpwrs.storage import atomic_write_json, sha256_array, sha256_file


def _debit_guard(allowance: float, n_time: int) -> float:
    """Roundoff guard for a nonnegative annual dot product, in hours."""
    scale = max(1.0, abs(float(allowance)))
    return max(
        16.0 * abs(float(np.spacing(scale))),
        64.0 * np.finfo(float).eps * max(1, int(n_time)) * scale,
    )


def conservative_m2_witness(
    case: Any,
    selected_x_mw: np.ndarray,
    allowance_hours: np.ndarray,
) -> dict[str, Any]:
    """Construct a distinct, conservatively M2-feasible certificate witness.

    Only participant-wise downward scaling is used.  The selected policy is
    never edited or relabelled.  A small arithmetic guard makes the resulting
    nonnegative debit strictly no larger than the stated allowance.
    """
    selected = np.asarray(selected_x_mw, dtype=float)
    allowance = np.asarray(allowance_hours, dtype=float)
    if selected.shape != case.requests_mw.shape:
        raise ValueError("Selected M2 policy shape does not match the physical case.")
    if allowance.shape != (case.n_participants,):
        raise ValueError("M2 allowance axis does not match the participant axis.")
    if not np.isfinite(selected).all() or not np.isfinite(allowance).all():
        raise ValueError("Certificate witness inputs must be finite.")
    if np.min(selected, initial=0.0) < 0.0:
        raise ValueError("The selected M2 policy contains a negative action.")
    if np.max(selected - case.xbar_mw, initial=0.0) > 0.0:
        raise ValueError("The selected M2 policy exceeds its action box.")
    if np.min(allowance, initial=0.0) < 0.0:
        raise ValueError("M2 allowance cannot be negative.")

    dt = np.asarray(case.time_grid.delta, dtype=float)
    capacities = np.asarray(case.capacities_mw, dtype=float)
    selected_debit = np.asarray(
        [
            math.fsum((dt * selected[:, n] / capacities[n]).tolist())
            for n in range(case.n_participants)
        ],
        dtype=float,
    )
    factors = np.ones(case.n_participants, dtype=float)
    guards = np.asarray(
        [_debit_guard(value, case.n_time) for value in allowance], dtype=float
    )
    for n, (debit, limit, guard) in enumerate(
        zip(selected_debit, allowance, guards)
    ):
        target = max(0.0, float(limit) - float(guard))
        if debit > target:
            if debit <= 0.0:
                factors[n] = 0.0
            else:
                factors[n] = float(np.nextafter(target / debit, 0.0))
    witness = selected * factors[None, :]
    witness_debit = np.asarray(
        [
            math.fsum((dt * witness[:, n] / capacities[n]).tolist())
            for n in range(case.n_participants)
        ],
        dtype=float,
    )
    if np.min(witness, initial=0.0) < 0.0:
        raise RuntimeError("Certificate witness is negative.")
    if np.max(witness - case.xbar_mw, initial=0.0) > 0.0:
        raise RuntimeError("Certificate witness exceeds the action box.")
    if np.max(witness_debit - allowance, initial=0.0) > 0.0:
        raise RuntimeError("Certificate witness is not conservatively M2-feasible.")
    return {
        "selected_x_mw": selected,
        "selected_debit_hours": selected_debit,
        "selected_budget_residual_hours": selected_debit - allowance,
        "witness_x_mw": witness,
        "witness_scaling_factors": factors,
        "witness_debit_hours": witness_debit,
        "witness_budget_residual_hours": witness_debit - allowance,
        "debit_roundoff_guard_hours": guards,
        "witness_hash": sha256_array(witness),
    }


def corrected_m1_certificate(
    case: Any,
    selected_x_mw: np.ndarray,
    allowance_hours: np.ndarray,
    m1lp_certified_lower_bound: float,
) -> dict[str, Any]:
    witness = conservative_m2_witness(case, selected_x_mw, allowance_hours)
    selected_s = float(residual_index(case, witness["selected_x_mw"])[0])
    witness_s = float(residual_index(case, witness["witness_x_mw"])[0])
    lower = float(m1lp_certified_lower_bound)
    if not np.isfinite([selected_s, witness_s, lower]).all():
        raise ValueError("Strict certificate inputs must be finite.")
    epsilon = max(1e-6, 1e-9 * max(1.0, abs(witness_s), abs(lower)))
    strict = strict_m1_certificate(witness_s, lower, epsilon)
    result = {
        "schema": "TPWRS_B03_CONSERVATIVE_M2_WITNESS_CERTIFICATE_V1",
        "status": "PASS" if strict.passed else "FAIL",
        "selected_replay_S_mw_h_row": selected_s,
        "selected_maximum_budget_residual_h": float(
            np.max(witness["selected_budget_residual_hours"], initial=0.0)
        ),
        "selected_maximum_budget_violation_h": float(
            np.max(
                np.maximum(witness["selected_budget_residual_hours"], 0.0),
                initial=0.0,
            )
        ),
        "witness_scaling_factors": witness["witness_scaling_factors"].tolist(),
        "witness_debit_hours": witness["witness_debit_hours"].tolist(),
        "witness_maximum_budget_residual_h": float(
            np.max(witness["witness_budget_residual_hours"], initial=0.0)
        ),
        "debit_roundoff_guard_hours": witness[
            "debit_roundoff_guard_hours"
        ].tolist(),
        "witness_S_mw_h_row": witness_s,
        "m1lp_certified_lower_bound": lower,
        "certificate_epsilon": epsilon,
        "guaranteed_margin_mw_h_row": float(strict.guaranteed_margin),
        "witness_hash": witness["witness_hash"],
        "selected_policy_preserved": True,
        "strict_certificate": asdict(strict),
    }
    return result


def write_corrected_certificate(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    atomic_write_json(target, payload)
    return target


def certificate_artifact_record(path: str | Path) -> dict[str, str]:
    target = Path(path)
    return {"path": target.name, "sha256": sha256_file(target)}


def certify_existing_stage04(
    root: Path = ROOT,
    *,
    results_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Re-certify stored 2029--2031 M2 arrays without running optimization."""
    from jobs.common import load_case, require_m1_m2_freeze, year_case

    base = Path(results_root or (root / "results_m1_m2_final"))
    output = Path(output_root or (base / "r10_certificates"))
    case = load_case(root / "artifacts/real_case/physical_case/FINAL_DUTCH_CASE")
    require_m1_m2_freeze(base, case)
    records = []
    for year in (2029, 2030, 2031):
        current = year_case(case, year)
        m2_root = base / f"year_{year}/rho_0.15/annual/M2"
        lp_root = base / f"year_{year}/rho_0.15/annual/M1_LP"
        m2 = load_policy_solution(m2_root)
        lp = load_policy_solution(lp_root)
        m2_experiment = __import__("json").loads(
            (m2_root / "EXPERIMENT_CASE.json").read_text(encoding="utf-8")
        )["experiment"]
        lower = lp.stage1.best_bound
        if lower is None:
            raise RuntimeError(f"{year}: M1-LP certified lower bound is missing.")
        payload = corrected_m1_certificate(
            current,
            m2.x_mw,
            np.asarray(m2_experiment["allowance"]["values_hours"], dtype=float),
            float(lower),
        )
        payload.update(
            year=year,
            scientific_fingerprint=scientific_fingerprint(),
            physical_case_hash=current.physical_case_hash,
            source_m2_solution_hash=m2.solution_hash,
            source_m1lp_solution_hash=lp.solution_hash,
            source_m2_solution_json_sha256=sha256_file(m2_root / "solution.json"),
            source_m2_solution_arrays_sha256=sha256_file(m2_root / "solution_arrays.npz"),
            source_m1lp_solution_json_sha256=sha256_file(lp_root / "solution.json"),
            source_m1lp_solution_arrays_sha256=sha256_file(lp_root / "solution_arrays.npz"),
        )
        target = output / f"year_{year}/B03_M1_M2_CERTIFICATE.json"
        write_corrected_certificate(target, payload)
        records.append(
            {
                "year": year,
                "status": payload["status"],
                "path": target.relative_to(root).as_posix(),
                "sha256": sha256_file(target),
                "selected_replay_S_mw_h_row": payload["selected_replay_S_mw_h_row"],
                "selected_maximum_budget_residual_h": payload[
                    "selected_maximum_budget_residual_h"
                ],
                "witness_S_mw_h_row": payload["witness_S_mw_h_row"],
                "m1lp_certified_lower_bound": payload[
                    "m1lp_certified_lower_bound"
                ],
                "certificate_epsilon": payload["certificate_epsilon"],
                "guaranteed_margin_mw_h_row": payload[
                    "guaranteed_margin_mw_h_row"
                ],
                "witness_hash": payload["witness_hash"],
            }
        )
    status = "PASS" if all(row["status"] == "PASS" for row in records) else "FAIL"
    aggregate = {
        "schema": "TPWRS_B03_STAGE04_RECERTIFICATION_V1",
        "status": status,
        "scientific_fingerprint": scientific_fingerprint(),
        "optimization_rerun": False,
        "selected_policy_arrays_modified": False,
        "records": records,
    }
    atomic_write_json(output / "B03_STATUS.json", aggregate)
    if status != "PASS":
        raise RuntimeError("At least one corrected B03 certificate did not pass.")
    return aggregate
