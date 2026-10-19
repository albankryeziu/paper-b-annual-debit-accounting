"""Exact reuse contract for a bitwise-flat frozen M3 card."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tpwrs.persistence import load_policy_solution
from tpwrs.config import ROOT
from tpwrs.storage import atomic_write_json, sha256_array, sha256_file, sha256_json


def is_exact_flat_card(card: Any) -> bool:
    rates = [float(value) for vector in card.rates.values() for value in vector]
    return bool(rates) and all(value == 1.0 for value in rates)


def require_exact_flat_card(card: Any) -> None:
    if not is_exact_flat_card(card):
        raise RuntimeError("Exact core reuse requires every serialized card rate to equal 1.0 bitwise.")


def matched_stage1_value(solution: Any) -> float:
    """Return the certified M2-MATCHED Stage-I optimum, not Stage-II policy replay."""
    value = float(solution.stage1.objective_value)
    if not np.isfinite(value):
        raise RuntimeError("Certified M2-MATCHED Stage-I objective is non-finite.")
    return value


def load_exact_core_bundle(core_root: Path, year: int) -> dict[str, Any]:
    base = Path(core_root) / f"year_{int(year)}/rho_0.03/annual"
    roots = {name: base / name for name in ("M2", "M3", "M2_MATCHED")}
    solutions = {name: load_policy_solution(path) for name, path in roots.items()}
    metrics = {
        name: json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for name, path in roots.items()
    }
    logs = {
        name: json.loads((path / "case.log.json").read_text(encoding="utf-8"))
        for name, path in roots.items()
    }
    for name in roots:
        if metrics[name].get("solution_hash") != solutions[name].solution_hash:
            raise RuntimeError(f"Core {name} metrics/solution hash mismatch for {year}.")
        if metrics[name].get("replay_passed") is not True:
            raise RuntimeError(f"Core {name} replay is not PASS for {year}.")
        if logs[name].get("status") != "PASS":
            raise RuntimeError(f"Core {name} case log is not PASS for {year}.")
    m3 = solutions["M3"]
    matched = solutions["M2_MATCHED"]
    if m3.q is None or not np.array_equal(np.asarray(m3.q, dtype=float), np.ones_like(m3.q, dtype=float)):
        raise RuntimeError(f"Core M3 q matrix is not bitwise all ones for {year}.")
    m3_experiment = json.loads((roots["M3"] / "EXPERIMENT_CASE.json").read_text(encoding="utf-8"))
    matched_experiment = json.loads((roots["M2_MATCHED"] / "EXPERIMENT_CASE.json").read_text(encoding="utf-8"))
    matched_allowance = matched_experiment.get("experiment", {}).get("allowance", {})
    if matched_allowance.get("source_solution_hash") != m3.solution_hash:
        raise RuntimeError(f"Core matched allowance is not bound to the M3 solution for {year}.")
    if matched.allowance_hash != sha256_json(matched_allowance):
        raise RuntimeError(f"Core matched allowance hash differs from its experiment for {year}.")
    return {
        "roots": roots,
        "solutions": solutions,
        "metrics": metrics,
        "m3_experiment": m3_experiment,
        "matched_experiment": matched_experiment,
        "source_files": {
            f"{name}_{kind}": {
                "path": (root / filename).resolve().relative_to(ROOT.resolve()).as_posix(),
                "sha256": sha256_file(root / filename),
            }
            for name, root in roots.items()
            for kind, filename in (
                ("solution", "solution.json"),
                ("arrays", "solution_arrays.npz"),
                ("metrics", "metrics.json"),
                ("experiment", "EXPERIMENT_CASE.json"),
                ("case_log", "case.log.json"),
            )
        },
    }


def exact_equivalence_certificate(
    *,
    year: int,
    variant_kind: str,
    variant_value: Any,
    final_card: Any,
    selected_parent_card: Any,
    core_bundle: dict[str, Any],
    physical_case_hash: str,
    scientific_fingerprint: str,
) -> dict[str, Any]:
    require_exact_flat_card(selected_parent_card)
    require_exact_flat_card(final_card)
    if (
        final_card.allowance_hash != selected_parent_card.allowance_hash
        or final_card.feature_set_id != selected_parent_card.feature_set_id
        or final_card.model_family != selected_parent_card.model_family
        or dict(final_card.thresholds) != dict(selected_parent_card.thresholds)
    ):
        raise RuntimeError("Sensitivity card changed frozen allowance/model/threshold lineage.")
    if (
        final_card.hash != selected_parent_card.hash
        and final_card.source_hashes.get("selected_parent_card") != selected_parent_card.hash
    ):
        raise RuntimeError("Sensitivity card is not bound to the selected parent card.")
    solutions = core_bundle["solutions"]
    m3 = solutions["M3"]
    matched = solutions["M2_MATCHED"]
    metrics = core_bundle.get("metrics", {})
    if metrics.get("M3", {}).get("solution_hash") != m3.solution_hash:
        raise RuntimeError("Core M3 metrics are not bound to the source solution hash.")
    if metrics.get("M2_MATCHED", {}).get("solution_hash") != matched.solution_hash:
        raise RuntimeError("Core matched metrics are not bound to the source solution hash.")
    matched_allowance = core_bundle["matched_experiment"]["experiment"]["allowance"]
    if matched_allowance.get("source_solution_hash") != m3.solution_hash:
        raise RuntimeError("Matched allowance is not bound to the source M3 solution.")
    if sha256_json(matched_allowance) != matched.allowance_hash:
        raise RuntimeError("Matched allowance hash does not match the certified solution.")
    if m3.q is None or not np.array_equal(np.asarray(m3.q, dtype=float), np.ones_like(m3.q, dtype=float)):
        raise RuntimeError("Source core M3 q matrix is not bitwise all ones.")
    m3_experiment = core_bundle["m3_experiment"]["experiment"]
    matched_experiment = core_bundle["matched_experiment"]["experiment"]
    if m3_experiment.get("physical_case_hash") != physical_case_hash or matched_experiment.get("physical_case_hash") != physical_case_hash:
        raise RuntimeError("Core experiment physical-case identity changed.")
    if m3_experiment.get("card_hash") != selected_parent_card.hash or matched_experiment.get("card_hash") != selected_parent_card.hash:
        raise RuntimeError("Core experiment card identity differs from the selected parent.")
    if m3_experiment.get("rate_matrix_hash") != sha256_array(np.asarray(m3.q, dtype=float)):
        raise RuntimeError("Core experiment q-matrix hash differs from the source solution.")
    payload = {
        "schema": "TPWRS_R10_EXACT_FLAT_OPERATOR_EQUIVALENCE_V1",
        "status": "PASS",
        "year": int(year),
        "variant_kind": str(variant_kind),
        "variant_value": variant_value,
        "proof": {
            "selected_parent_rates_bitwise_one": True,
            "derived_card_rates_bitwise_one": True,
            "core_q_matrix_bitwise_one": True,
            "operator_problem_identical_to_core_m3": True,
            "matched_problem_identical_to_core_m2_matched": True,
            "reason": (
                "q_selected=1 implies q_raw=1; q=1 satisfies ordering, bounds, "
                "and weighted mean exactly with zero projection distance."
            ),
        },
        "physical_case_hash": physical_case_hash,
        "selected_parent_card_hash": selected_parent_card.hash,
        "final_card_hash": final_card.hash,
        "source_m3_solution_hash": m3.solution_hash,
        "source_matched_solution_hash": matched.solution_hash,
        "source_matched_allowance_hash": matched.allowance_hash,
        "source_matched_allowance_parent_solution_hash": core_bundle[
            "matched_experiment"
        ]["experiment"]["allowance"]["source_solution_hash"],
        "source_files": core_bundle["source_files"],
        "scientific_fingerprint": scientific_fingerprint,
    }
    payload["certificate_hash"] = sha256_json(payload)
    return payload


def write_equivalence_certificate(path: Path, payload: dict[str, Any]) -> str:
    atomic_write_json(path, payload)
    digest = sha256_file(path)
    if payload.get("certificate_hash") != sha256_json(
        {key: value for key, value in payload.items() if key != "certificate_hash"}
    ):
        raise RuntimeError("Exact-equivalence certificate self-hash is inconsistent.")
    return digest
