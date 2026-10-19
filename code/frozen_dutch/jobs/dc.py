from __future__ import annotations


# ============================================================================
# scripts.run_independent_dc_selected_states (consolidated; see SOURCE_MAP.json)
# ============================================================================

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from jobs.selection import _policy_catalog
from jobs.common import load_case, year_case
from tpwrs.engineering import mapped_dc_audit
from tpwrs.storage import sha256_file
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.config import scientific_fingerprint

DC_ABSOLUTE_TOLERANCE_MW = 1e-4
DC_RELATIVE_RATING_TOLERANCE = 1e-8
DC_SIGN_MATERIALITY_MW = 1e-6


def independent_dc_acceptance(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen rowwise magnitude, sign, and overload acceptance gate."""
    required = {"frozen_flow_mw", "independent_flow_mw", "rating_mw"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Independent DC acceptance missing columns: {missing}")
    frame = rows.copy()
    frozen = pd.to_numeric(frame["frozen_flow_mw"], errors="coerce").to_numpy(float)
    independent = pd.to_numeric(frame["independent_flow_mw"], errors="coerce").to_numpy(float)
    rating = pd.to_numeric(frame["rating_mw"], errors="coerce").to_numpy(float)
    if not np.isfinite(frozen).all() or not np.isfinite(independent).all() or not np.isfinite(rating).all() or np.any(rating <= 0.0):
        raise ValueError("Independent DC acceptance requires finite flows and positive ratings.")
    tolerance = np.maximum(
        DC_ABSOLUTE_TOLERANCE_MW,
        DC_RELATIVE_RATING_TOLERANCE * rating,
    )
    error = np.abs(independent - frozen)
    frozen_sign = np.where(
        frozen > DC_SIGN_MATERIALITY_MW,
        1,
        np.where(frozen < -DC_SIGN_MATERIALITY_MW, -1, 0),
    )
    independent_sign = np.where(
        independent > DC_SIGN_MATERIALITY_MW,
        1,
        np.where(independent < -DC_SIGN_MATERIALITY_MW, -1, 0),
    )
    sign_agreement = frozen_sign == independent_sign
    frozen_overloaded = np.abs(frozen) > rating + tolerance
    independent_overloaded = np.abs(independent) > rating + tolerance
    false_safe = (~frozen_overloaded) & independent_overloaded
    false_overload = frozen_overloaded & (~independent_overloaded)
    frame["acceptance_tolerance_mw"] = tolerance
    frame["magnitude_within_tolerance"] = error <= tolerance
    frame["frozen_material_sign"] = frozen_sign
    frame["independent_material_sign"] = independent_sign
    frame["material_sign_agreement"] = sign_agreement
    frame["false_safe_beyond_tolerance"] = false_safe
    frame["false_overload_beyond_tolerance"] = false_overload
    sign_share = float(np.mean(sign_agreement)) if len(frame) else 1.0
    summary = {
        "row_count": int(len(frame)),
        "per_row_tolerance_formula": "max(1e-4 MW, 1e-8*rating_mw)",
        "flow_sign_materiality_mw": DC_SIGN_MATERIALITY_MW,
        "required_material_sign_agreement": 1.0,
        "material_sign_agreement_share": sign_share,
        "maximum_abs_error_mw": float(np.max(error, initial=0.0)),
        "maximum_error_to_tolerance_ratio": float(
            np.max(error / tolerance, initial=0.0)
        ),
        "magnitude_failure_count": int(np.sum(error > tolerance)),
        "false_safe_count_beyond_tolerance": int(np.sum(false_safe)),
        "false_overload_count_beyond_tolerance": int(np.sum(false_overload)),
    }
    summary["passed"] = bool(
        summary["magnitude_failure_count"] == 0
        and sign_share == 1.0
        and summary["false_safe_count_beyond_tolerance"] == 0
        and summary["false_overload_count_beyond_tolerance"] == 0
    )
    return frame, summary


def _load_network_source() -> Path:
    from tpwrs.config import source_path
    return source_path("reduced_network", verify=True)


def _load_network(path: Path) -> Any:
    import pandapower as pp

    suffix = path.suffix.lower()
    if suffix == ".json":
        return pp.from_json(str(path))
    if suffix in {".p", ".pickle", ".pkl"}:
        return pp.from_pickle(str(path))
    raise RuntimeError(f"Unsupported pandapower network serialization: {suffix}")


def _prepare_topology(network: Any, state_id: str) -> tuple[Any, list[int]]:
    import pandapower as pp

    net = deepcopy(network)

    # Disable all original active-power injections. The independent DC path
    # receives the exact balanced V3 nodal injection through one synthetic
    # element per bus. Network topology/branch parameters remain unchanged.
    for table_name in (
        "load",
        "sgen",
        "storage",
        "ward",
        "xward",
    ):
        table = getattr(net, table_name, None)
        if table is not None and len(table):
            if "in_service" in table:
                table.loc[:, "in_service"] = False

    if hasattr(net, "gen") and len(net.gen):
        if "in_service" in net.gen:
            net.gen.loc[:, "in_service"] = False
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        net.ext_grid.loc[:, "in_service"] = False

    # One independent slack is sufficient because every supplied V3 injection
    # is balanced. Its solved active power should therefore be numerical zero.
    if hasattr(net, "ext_grid") and len(net.ext_grid):
        slack_bus = int(net.ext_grid.iloc[0].bus)
        net.ext_grid.iloc[0, net.ext_grid.columns.get_loc("in_service")] = True
    elif hasattr(net, "gen") and len(net.gen):
        slack_bus = int(net.gen.iloc[0].bus)
        net.gen.iloc[0, net.gen.columns.get_loc("in_service")] = True
        net.gen.iloc[0, net.gen.columns.get_loc("p_mw")] = 0.0
        if "slack" in net.gen.columns:
            net.gen.iloc[0, net.gen.columns.get_loc("slack")] = True
    else:
        slack_bus = int(net.bus.index[0])
        pp.create_ext_grid(net, slack_bus)

    if state_id != "INTACT":
        if not state_id.startswith("OUT::"):
            raise RuntimeError(f"Unsupported network state ID {state_id}.")
        element_id = state_id[len("OUT::") :]
        kind, raw_index = element_id.split("::", 1)
        index = int(raw_index)
        table = getattr(net, kind, None)
        if table is None or index not in table.index:
            raise RuntimeError(f"Outaged element {element_id} is not in source network.")
        table.at[index, "in_service"] = False

    injection_indices = [
        int(pp.create_sgen(net, int(bus), p_mw=0.0, q_mvar=0.0))
        for bus in net.bus.index
    ]
    return net, injection_indices


def _set_injection(
    net: Any,
    injection_indices: list[int],
    bus_ids: tuple[str, ...],
    injection: np.ndarray,
) -> None:
    values = np.asarray(injection, dtype=float)
    if values.shape != (len(bus_ids),):
        raise RuntimeError("Independent DC injection axis mismatch.")
    if abs(float(values.sum())) > 1e-7:
        raise RuntimeError("Independent DC received an unbalanced injection.")

    source_bus_order = tuple(map(int, net.bus.index))
    if set(source_bus_order) != set(map(int, bus_ids)):
        raise RuntimeError("V3 and pandapower bus ID sets differ.")
    value_by_bus = {
        int(bus): float(value)
        for bus, value in zip(map(int, bus_ids), values)
    }
    for bus, sgen_index in zip(source_bus_order, injection_indices):
        net.sgen.at[sgen_index, "p_mw"] = value_by_bus[bus]


def _extract_rows(net: Any, state: Any) -> np.ndarray:
    values = []
    for row_id in state.row_ids:
        kind, raw_index = str(row_id).split("::", 1)
        index = int(raw_index)
        if kind == "line":
            values.append(float(net.res_line.at[index, "p_from_mw"]))
        elif kind == "trafo":
            values.append(float(net.res_trafo.at[index, "p_hv_mw"]))
        else:
            raise RuntimeError(f"Unsupported monitored element type {kind}.")
    return np.asarray(values, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independent pandapower DC replay of the frozen selected V3 "
            "policy-hour/state registry."
        )
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=ROOT / "artifacts/real_case/physical_case/FINAL_DUTCH_CASE",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "results_m1_m2_final",
    )
    parser.add_argument(
        "--m3-results",
        type=Path,
        default=ROOT / "results_m3_final",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            ROOT
            / "results_engineering_validation"
            / "selected_registry"
            / "SELECTED_POLICY_STATE_CASES.csv"
        ),
    )
    parser.add_argument(
        "--registry-freeze",
        type=Path,
        default=(
            ROOT
            / "results_engineering_validation"
            / "selected_registry"
            / "SELECTED_STATE_REGISTRY_FREEZE.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_engineering_validation/independent_dc",
    )
    args = parser.parse_args()

    import pandapower as pp

    case = load_case(args.case)
    freeze = json.loads(args.registry_freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS":
        raise RuntimeError("Selected-state registry freeze is not PASS.")
    if freeze.get("physical_case_hash") != case.physical_case_hash:
        raise RuntimeError("Selected-state registry physical-case hash mismatch.")
    if sha256_file(args.registry) != freeze["outputs"]["cases"]["sha256"]:
        raise RuntimeError("Selected-state registry case table hash mismatch.")

    existing_status_path = args.output / "STATUS.json"
    if existing_status_path.is_file():
        existing = json.loads(existing_status_path.read_text(encoding="utf-8"))
        outputs = existing.get("outputs", {})
        existing_valid = (
            existing.get("status") == "PASS"
            and existing.get("scientific_fingerprint") == scientific_fingerprint()
            and existing.get("registry_freeze_sha256") == sha256_file(args.registry_freeze)
            and all(
                (args.output / record["path"]).is_file()
                and sha256_file(args.output / record["path"]) == record["sha256"]
                for record in outputs.values()
            )
        )
        if existing_valid:
            print("INDEPENDENT_SELECTED_DC=ALREADY_COMPLETE")
            return 0

    registry = pd.read_csv(args.registry)
    source_path = _load_network_source()
    source_network = _load_network(source_path)
    states = {state.state_id: state for state in case.network_states}
    topology_cache: dict[str, tuple[Any, list[int]]] = {}
    catalogs = {
        year: _policy_catalog(
            year,
            year_case(case, year),
            args.baseline,
            args.m3_results,
        )
        for year in sorted(registry["year"].astype(int).unique())
    }

    frozen_rows: list[dict[str, Any]] = []
    independent_rows: list[dict[str, Any]] = []
    case_summary: list[dict[str, Any]] = []

    for row in registry.itertuples(index=False):
        year = int(row.year)
        time_index = int(row.time_index)
        state_id = str(row.state_id)
        policy_state = str(row.policy_state)
        selected_case = year_case(case, year)
        state = states[state_id]
        state_vector = np.asarray(
            catalogs[year][policy_state]["state"],
            dtype=float,
        )[time_index]
        frozen_flow = state.H @ state_vector + state.h_mw

        if state_id not in topology_cache:
            topology_cache[state_id] = _prepare_topology(source_network, state_id)
        net, injection_indices = topology_cache[state_id]
        _set_injection(
            net,
            injection_indices,
            selected_case.bus_ids,
            state_vector,
        )
        pp.rundcpp(
            net,
            calculate_voltage_angles=True,
            check_connectivity=True,
            init="flat",
        )
        if not bool(getattr(net, "converged", False)):
            raise RuntimeError(f"Independent DC did not converge for {row.case_id}.")
        independent_flow = _extract_rows(net, state)

        if len(independent_flow) != len(frozen_flow):
            raise RuntimeError("Independent/frozen monitored-row count mismatch.")

        for row_id, rating, frozen_value, independent_value in zip(
            state.row_ids,
            state.ratings_mw,
            frozen_flow,
            independent_flow,
        ):
            common = {
                "case_id": str(row.case_id),
                "state_id": state_id,
                "row_id": str(row_id),
                "rating_mw": float(rating),
                "year": year,
                "time_index": time_index,
                "timestamp_utc": row.timestamp_utc,
                "policy_state": policy_state,
                "selection_group": row.selection_group,
            }
            frozen_rows.append(
                {**common, "frozen_flow_mw": float(frozen_value)}
            )
            independent_rows.append(
                {**common, "independent_flow_mw": float(independent_value)}
            )
        case_summary.append(
            {
                "case_id": str(row.case_id),
                "year": year,
                "time_index": time_index,
                "state_id": state_id,
                "policy_state": policy_state,
                "independent_dc_converged": True,
                "slack_injection_mw": (
                    float(net.res_ext_grid.p_mw.sum())
                    if hasattr(net, "res_ext_grid") and len(net.res_ext_grid)
                    else float("nan")
                ),
            }
        )

    frozen_frame = pd.DataFrame(frozen_rows)
    independent_frame = pd.DataFrame(independent_rows)
    merged, summary = mapped_dc_audit(
        frozen_frame,
        independent_frame,
    )
    merged, acceptance = independent_dc_acceptance(merged)
    accepted = bool(acceptance["passed"])

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(args.output / "FROZEN_DC_ROWS.csv", frozen_frame)
    atomic_write_csv(args.output / "INDEPENDENT_DC_ROWS.csv", independent_frame)
    atomic_write_csv(args.output / "DC_COMPARISON_ROWS.csv", merged)
    atomic_write_csv(
        args.output / "DC_CASE_SUMMARY.csv",
        pd.DataFrame(case_summary),
    )
    status = {
        "schema": "TPWRS_SELECTED_STATE_INDEPENDENT_DC_V1",
        "status": "PASS" if accepted else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_fingerprint": scientific_fingerprint(),
        "physical_case_hash": case.physical_case_hash,
        "registry_freeze_sha256": sha256_file(args.registry_freeze),
        "network_source_sha256": sha256_file(source_path),
        "comparison": summary,
        "acceptance": acceptance,
        "claim_boundary": (
            "INDEPENDENT PANDAPOWER DC SOFTWARE PATH ON THE SAME FROZEN "
            "TOPOLOGY/PARAMETERS; NOT AN INDEPENDENT NETWORK DATASET OR AC PROOF."
        ),
        "outputs": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
            }
            for name, path in {
                "frozen": args.output / "FROZEN_DC_ROWS.csv",
                "independent": args.output / "INDEPENDENT_DC_ROWS.csv",
                "comparison": args.output / "DC_COMPARISON_ROWS.csv",
                "cases": args.output / "DC_CASE_SUMMARY.csv",
            }.items()
        },
    }
    atomic_write_json(args.output / "STATUS.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if accepted else 2
