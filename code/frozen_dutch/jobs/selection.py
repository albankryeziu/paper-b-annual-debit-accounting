from __future__ import annotations


# ============================================================================
# scripts.build_selected_state_registry (consolidated; see SOURCE_MAP.json)
# ============================================================================

import argparse
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

from jobs.common import EVALUATION_YEARS, load_case, load_policy_array, require_m1_m2_freeze, year_case
from tpwrs.storage import sha256_array, sha256_file, sha256_json
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.config import scientific_fingerprint


SELECTION_COUNT_PER_GROUP = 2
SELECTION_GROUPS = (
    "REQUEST_STRESS",
    "RHO15_ROLLING_M1_M2_FLOW_DIFFERENCE",
    "RHO03_ANNUAL_M2_M3_FLOW_DIFFERENCE",
)


def _flows(case: Any, state: np.ndarray) -> np.ndarray:
    return state @ case.flat_H.T + case.flat_h[None, :]


def _hourly_violation(case: Any, state: np.ndarray) -> np.ndarray:
    output = np.zeros(case.n_time, dtype=float)
    for start in range(0, case.n_time, 128):
        stop = min(case.n_time, start + 128)
        flow = (
            state[start:stop] @ case.flat_H.T
            + case.flat_h[None, :]
        )
        output[start:stop] = (
            np.maximum(
                np.abs(flow) - case.flat_ratings[None, :],
                0.0,
            ).sum(axis=1)
            * case.time_grid.delta[start:stop]
        )
    return output


def _hourly_flow_difference(
    case: Any,
    left_x: np.ndarray,
    right_x: np.ndarray,
) -> np.ndarray:
    left = case.post_state_mw(left_x)
    right = case.post_state_mw(right_x)
    output = np.zeros(case.n_time, dtype=float)
    for start in range(0, case.n_time, 128):
        stop = min(case.n_time, start + 128)
        left_flow = left[start:stop] @ case.flat_H.T + case.flat_h[None, :]
        right_flow = right[start:stop] @ case.flat_H.T + case.flat_h[None, :]
        output[start:stop] = np.sum(
            np.abs(left_flow - right_flow),
            axis=1,
        )
    return output


def _top_indices(
    values: np.ndarray,
    count: int,
    *,
    excluded: set[int] | None = None,
) -> list[int]:
    excluded = excluded or set()
    order = np.argsort(
        -np.asarray(values, dtype=float),
        kind="mergesort",
    )
    selected: list[int] = []
    for index in order:
        index = int(index)
        if index in excluded:
            continue
        selected.append(index)
        if len(selected) == count:
            break
    return selected


def _state_slices(case: Any) -> dict[str, slice]:
    result: dict[str, slice] = {}
    cursor = 0
    for state in case.network_states:
        result[state.state_id] = slice(cursor, cursor + len(state.row_ids))
        cursor += len(state.row_ids)
    if cursor != case.n_flat_rows:
        raise RuntimeError("Network-state slices do not cover the flat row registry.")
    return result


def _policy_catalog(
    year: int,
    year_case_value: Any,
    baseline: Path,
    m3: Path,
) -> dict[str, dict[str, Any]]:
    entries = {
        "BACKGROUND": {
            "state": year_case_value.p_background_mw,
            "source": "PHYSICAL_CASE::BACKGROUND",
            "source_hash": year_case_value.physical_case_hash,
        },
        "REQUEST": {
            "state": year_case_value.requested_state_mw,
            "source": "PHYSICAL_CASE::REQUESTED_STATE",
            "source_hash": year_case_value.physical_case_hash,
        },
    }

    definitions = {
        "RHO15_ANNUAL_M1_LP": (
            baseline / f"year_{year}/rho_0.15/annual/M1_LP",
            False,
        ),
        "RHO15_ANNUAL_M2": (
            baseline / f"year_{year}/rho_0.15/annual/M2",
            False,
        ),
        "RHO15_ROLLING_M1": (
            baseline / f"year_{year}/rho_0.15/rolling/M1",
            True,
        ),
        "RHO15_ROLLING_M2": (
            baseline / f"year_{year}/rho_0.15/rolling/M2",
            True,
        ),
        "RHO15_ANNUAL_M3": (
            m3 / f"year_{year}/rho_0.15/annual/M3",
            False,
        ),
        "RHO15_ROLLING_M3": (
            m3 / f"year_{year}/rho_0.15/rolling/M3",
            True,
        ),
        "RHO03_ANNUAL_M2": (
            m3 / f"year_{year}/rho_0.03/annual/M2",
            False,
        ),
        "RHO03_ANNUAL_M3": (
            m3 / f"year_{year}/rho_0.03/annual/M3",
            False,
        ),
        "RHO03_ANNUAL_M2_MATCHED": (
            m3 / f"year_{year}/rho_0.03/annual/M2_MATCHED",
            False,
        ),
        "RHO03_ROLLING_M2": (
            m3 / f"year_{year}/rho_0.03/rolling/M2",
            True,
        ),
        "RHO03_ROLLING_M3": (
            m3 / f"year_{year}/rho_0.03/rolling/M3",
            True,
        ),
    }

    for label, (path, rolling) in definitions.items():
        x = load_policy_array(path, rolling=rolling)
        state = year_case_value.post_state_mw(x)
        source_file = (
            path / ("rolling_result.json" if rolling else "solution.json")
        )
        entries[label] = {
            "state": state,
            "source": str(path),
            "source_hash": sha256_file(source_file),
            "x_hash": sha256_array(x),
        }
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the selected V3 policy-hour/state registry before any "
            "independent DC or AC output is opened."
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
        "--output",
        type=Path,
        default=ROOT / "results_engineering_validation/selected_registry",
    )
    args = parser.parse_args()

    case = load_case(args.case)
    require_m1_m2_freeze(args.baseline, case)
    m3_status = json.loads(
        (args.m3_results / "M3_CORE_STATUS.json").read_text(encoding="utf-8")
    )
    if m3_status.get("status") != "COMPLETE":
        raise RuntimeError("Corrected M3 core campaign must complete first.")

    args.output.mkdir(parents=True, exist_ok=True)
    freeze_path = args.output / "SELECTED_STATE_REGISTRY_FREEZE.json"
    row_path = args.output / "SELECTED_POLICY_STATE_CASES.csv"
    hour_path = args.output / "SELECTED_HOURS.csv"

    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if (
            freeze.get("status") == "PASS"
            and freeze.get("physical_case_hash") == case.physical_case_hash
            and freeze.get("scientific_fingerprint") == scientific_fingerprint()
            and row_path.is_file()
            and hour_path.is_file()
            and sha256_file(row_path) == freeze["outputs"]["cases"]["sha256"]
            and sha256_file(hour_path) == freeze["outputs"]["hours"]["sha256"]
        ):
            current_inputs = {}
            for year in EVALUATION_YEARS:
                catalog = _policy_catalog(year, year_case(case, year), args.baseline, args.m3_results)
                current_inputs.update({f"{year}/{label}": str(record["source_hash"])
                                       for label, record in catalog.items()})
            if current_inputs != freeze.get("input_hashes"):
                raise RuntimeError("Selected-state freeze belongs to changed policy inputs. Do not reuse or overwrite it.")
            print("SELECTED_STATE_REGISTRY=ALREADY_FROZEN")
            return 0
        raise RuntimeError(
            "A stale or incomplete selected-state freeze exists. "
            "Move it aside before rebuilding; never overwrite a viewed registry."
        )

    case_rows: list[dict[str, Any]] = []
    hour_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}

    for year in EVALUATION_YEARS:
        current = year_case(case, year)
        catalog = _policy_catalog(
            year,
            current,
            args.baseline,
            args.m3_results,
        )
        for label, record in catalog.items():
            input_hashes[f"{year}/{label}"] = str(record["source_hash"])

        request_score = _hourly_violation(
            current,
            current.requested_state_mw,
        )
        rolling_difference = _hourly_flow_difference(
            current,
            load_policy_array(
                args.baseline / f"year_{year}/rho_0.15/rolling/M1",
                rolling=True,
            ),
            load_policy_array(
                args.baseline / f"year_{year}/rho_0.15/rolling/M2",
                rolling=True,
            ),
        )
        m3_difference = _hourly_flow_difference(
            current,
            load_policy_array(
                args.m3_results / f"year_{year}/rho_0.03/annual/M2",
                rolling=False,
            ),
            load_policy_array(
                args.m3_results / f"year_{year}/rho_0.03/annual/M3",
                rolling=False,
            ),
        )

        scores = {
            "REQUEST_STRESS": request_score,
            "RHO15_ROLLING_M1_M2_FLOW_DIFFERENCE": rolling_difference,
            "RHO03_ANNUAL_M2_M3_FLOW_DIFFERENCE": m3_difference,
        }
        selected_hours: list[tuple[str, int, float]] = []
        used: set[int] = set()
        for group in SELECTION_GROUPS:
            chosen = _top_indices(
                scores[group],
                SELECTION_COUNT_PER_GROUP,
                excluded=used,
            )
            for time_index in chosen:
                used.add(time_index)
                selected_hours.append(
                    (group, time_index, float(scores[group][time_index]))
                )
        # Deterministically fill rare duplicates/degenerate groups from request
        # stress so every year retains six unique selected hours.
        if len(selected_hours) < 6:
            for time_index in _top_indices(
                request_score,
                6 - len(selected_hours),
                excluded=used,
            ):
                used.add(time_index)
                selected_hours.append(
                    ("REQUEST_STRESS_FILL", time_index, float(request_score[time_index]))
                )

        slices = _state_slices(current)
        policy_labels = tuple(catalog)
        for selection_group, time_index, selection_score in selected_hours:
            timestamp = current.time_grid.timestamps_utc[time_index]
            request_row_scores = {}
            spread_row_scores = {}
            for state_id, part in slices.items():
                request_flow = (
                    current.requested_state_mw[time_index]
                    @ current.flat_H[part].T
                    + current.flat_h[part]
                )
                request_row_scores[state_id] = float(
                    np.maximum(
                        np.abs(request_flow) - current.flat_ratings[part],
                        0.0,
                    ).sum()
                )
                policy_flows = []
                for label in policy_labels:
                    state = np.asarray(catalog[label]["state"], dtype=float)
                    policy_flows.append(
                        state[time_index] @ current.flat_H[part].T
                        + current.flat_h[part]
                    )
                stacked = np.vstack(policy_flows)
                spread_row_scores[state_id] = float(
                    np.sum(np.max(stacked, axis=0) - np.min(stacked, axis=0))
                )

            state_ids = [
                "INTACT",
                max(request_row_scores, key=request_row_scores.get),
                max(spread_row_scores, key=spread_row_scores.get),
            ]
            state_ids = list(dict.fromkeys(state_ids))

            hour_rows.append(
                {
                    "year": year,
                    "time_index": time_index,
                    "timestamp_utc": timestamp,
                    "selection_group": selection_group,
                    "selection_score": selection_score,
                    "selected_state_ids": json.dumps(state_ids),
                    "request_stress_score": float(request_score[time_index]),
                    "rho15_rolling_m1_m2_flow_difference": float(
                        rolling_difference[time_index]
                    ),
                    "rho03_annual_m2_m3_flow_difference": float(
                        m3_difference[time_index]
                    ),
                }
            )

            for state_id in state_ids:
                for policy_state, record in catalog.items():
                    case_rows.append(
                        {
                            "case_id": (
                                f"{year}_{time_index:05d}_{state_id}_"
                                f"{policy_state}"
                            ),
                            "year": year,
                            "time_index": time_index,
                            "timestamp_utc": timestamp,
                            "selection_group": selection_group,
                            "selection_score": selection_score,
                            "state_id": state_id,
                            "policy_state": policy_state,
                            "policy_source": record["source"],
                            "policy_source_hash": record["source_hash"],
                            "policy_x_hash": record.get("x_hash"),
                        }
                    )

    hours = pd.DataFrame(hour_rows).sort_values(["year", "time_index"])
    cases = pd.DataFrame(case_rows).sort_values(
        ["year", "time_index", "state_id", "policy_state"]
    )
    atomic_write_csv(hour_path, hours)
    atomic_write_csv(row_path, cases)

    freeze = {
        "schema": "TPWRS_SELECTED_STATE_REGISTRY_FREEZE_V1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_fingerprint": scientific_fingerprint(),
        "physical_case_id": case.physical_case_id,
        "physical_case_hash": case.physical_case_hash,
        "selection_algorithm": {
            "version": 1,
            "groups": SELECTION_GROUPS,
            "hours_per_group": SELECTION_COUNT_PER_GROUP,
            "deduplication": "stable descending score; request-stress fill",
            "state_selection": (
                "INTACT plus worst request-overload state plus state with "
                "largest cross-policy active-flow spread"
            ),
            "selected_policy_states": sorted(cases["policy_state"].unique()),
            "validation_outcomes_accessed": False,
        },
        "input_hashes": input_hashes,
        "outputs": {
            "hours": {
                "path": hour_path.name,
                "rows": len(hours),
                "sha256": sha256_file(hour_path),
            },
            "cases": {
                "path": row_path.name,
                "rows": len(cases),
                "sha256": sha256_file(row_path),
            },
        },
        "claim_boundary": (
            "DETERMINISTIC STRESS-STATE SELECTION BEFORE INDEPENDENT DC/AC "
            "OUTCOMES; NOT A RANDOM SAMPLE OR ANNUAL FEASIBILITY CLAIM."
        ),
    }
    atomic_write_json(freeze_path, freeze)
    print(
        "SELECTED_STATE_REGISTRY=FROZEN "
        f"hours={len(hours)} cases={len(cases)}"
    )
    return 0
