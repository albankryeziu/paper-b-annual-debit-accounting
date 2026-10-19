"""Hash-bound serialization of physical cases and policy solutions.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from __future__ import annotations


import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from tpwrs.storage import ContractError
from tpwrs.storage import sha256_file
from tpwrs.storage import atomic_write_json
from tpwrs.contracts import NetworkState, PhysicalCase
from tpwrs.contracts import TimeGrid


def save_physical_case(case: PhysicalCase, directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "capacities_mw": case.capacities_mw,
        "requests_mw": case.requests_mw,
        "contract_mask": case.contract_mask.astype(np.uint8),
        "eligibility_mask": case.eligibility_mask.astype(np.uint8),
        "xbar_mw": case.xbar_mw,
        "p_background_mw": case.p_background_mw,
        "A": case.A,
    }
    for i, state in enumerate(case.network_states):
        arrays[f"H_{i}"] = state.H
        arrays[f"h_{i}"] = state.h_mw
        arrays[f"rating_{i}"] = state.ratings_mw
    npz = root / "physical_case_arrays.npz"
    # np.savez_compressed is not atomic.  A killed annual build previously
    # could leave a valid-looking but truncated cache.  Write in the target
    # directory, fsync, then replace so restart logic observes all or nothing.
    fd, tmp_name = tempfile.mkstemp(prefix=".physical_case_arrays.", suffix=".npz", dir=str(root))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, npz)
    finally:
        if tmp.exists():
            tmp.unlink()
    meta = {
        "schema": "TPWRS_V3_PHYSICAL_CASE_V1",
        "physical_case_id": case.physical_case_id,
        "physical_case_hash": case.physical_case_hash,
        "time_grid": {
            "timestamps_utc": case.time_grid.timestamps_utc,
            "delta_t_hours": case.time_grid.delta_t_hours,
            "timezone": case.time_grid.timezone,
        },
        "participant_ids": case.participant_ids,
        "bus_ids": case.bus_ids,
        "participant_directions": case.participant_directions,
        "participant_set_id": case.participant_set_id,
        "contingency_set_id": case.contingency_set_id,
        "balancing_key_hash": case.balancing_key_hash,
        "source_hashes": dict(case.source_hashes),
        "network_states": [
            {
                "state_id": s.state_id,
                "row_ids": s.row_ids,
                "state_kind": s.state_kind,
                "outaged_element_id": s.outaged_element_id,
                "source_hash": s.source_hash,
                "index": i,
            }
            for i, s in enumerate(case.network_states)
        ],
        "arrays_file": "physical_case_arrays.npz",
        "arrays_sha256": sha256_file(npz),
    }
    atomic_write_json(root / "physical_case.json", meta)
    return meta


def load_physical_case(directory: str | Path) -> PhysicalCase:
    root = Path(directory)
    meta = json.loads((root / "physical_case.json").read_text(encoding="utf-8"))
    npz = root / meta["arrays_file"]
    if meta.get("schema") != "TPWRS_V3_PHYSICAL_CASE_V1":
        raise ContractError("Unsupported physical-case schema.")
    if sha256_file(npz) != meta["arrays_sha256"]:
        raise ContractError("Physical-case arrays hash mismatch.")
    with np.load(npz, allow_pickle=False) as a:
        grid = TimeGrid(
            tuple(meta["time_grid"]["timestamps_utc"]),
            tuple(meta["time_grid"]["delta_t_hours"]),
            meta["time_grid"].get("timezone", "UTC"),
            False,
        )
        states = []
        for s in meta["network_states"]:
            i = int(s["index"])
            states.append(
                NetworkState(
                    s["state_id"],
                    tuple(s["row_ids"]),
                    a[f"H_{i}"],
                    a[f"h_{i}"],
                    a[f"rating_{i}"],
                    s.get("state_kind", "INTACT"),
                    s.get("outaged_element_id"),
                    s.get("source_hash"),
                )
            )
        case = PhysicalCase(
            grid,
            tuple(meta["participant_ids"]),
            tuple(meta["bus_ids"]),
            a["capacities_mw"],
            a["requests_mw"],
            a["contract_mask"].astype(bool),
            a["eligibility_mask"].astype(bool),
            a["p_background_mw"],
            a["A"],
            tuple(states),
            None
            if meta.get("participant_directions") is None
            else tuple(meta["participant_directions"]),
            meta.get("participant_set_id", "UNSPECIFIED"),
            meta.get("contingency_set_id", "UNSPECIFIED"),
            meta.get("balancing_key_hash", "UNSPECIFIED"),
            meta.get("source_hashes", {}),
            meta["physical_case_id"],
        )
        if "xbar_mw" not in a.files or not np.array_equal(a["xbar_mw"], case.xbar_mw):
            raise ContractError("Stored xbar does not exactly equal chi*min(P,y_req).")
    if case.physical_case_hash != meta["physical_case_hash"]:
        raise ContractError("Reconstructed physical-case hash mismatch.")
    return case


def prepare_fitting_inputs(case: PhysicalCase, directory: str | Path) -> dict:
    """Input-stage partition: fitting never loads the all-year NPZ.

    Future contractual allowance vectors are metadata, not future requests,
    background states, features, or policy outcomes.
    """
    from tpwrs.contracts import AllowanceSpec

    root = Path(directory)
    years = case.time_grid.index.year
    fit = np.flatnonzero(years <= 2028)
    if not len(fit) or set(years[fit]) != {2025, 2026, 2027, 2028}:
        raise ContractError("Fitting partition requires 2025--2028.")
    subset = case.slice_time(int(fit[0]), int(fit[-1]) + 1)
    meta = save_physical_case(subset, root / "FITTING_2025_2028")
    allowances = {}
    for rho in (0.15, 0.03):
        allowances[f"{rho:.2f}"] = {}
        for year in sorted(set(years)):
            ids = np.flatnonzero(years == year)
            annual = case.slice_time(int(ids[0]), int(ids[-1]) + 1)
            value = AllowanceSpec.from_nominal(
                annual.participant_ids, annual.contract_mask, annual.time_grid, rho
            )
            allowances[f"{rho:.2f}"][str(year)] = {
                "hash": value.hash,
                "values_hours": value.values.tolist(),
            }
    record = {
        "schema": "TPWRS_FITTING_PARTITION_V1",
        "physical_case_hash": case.physical_case_hash,
        "physical_case_id": case.physical_case_id,
        "fitting_case_hash": subset.physical_case_hash,
        "fitting_metadata_sha256": sha256_file(root / "FITTING_2025_2028/physical_case.json"),
        "allowances": allowances,
        "allowed_years": [2025, 2026, 2027, 2028],
    }
    from tpwrs.storage import sha256_json

    record["record_hash"] = sha256_json(record)
    atomic_write_json(root / "FITTING_INPUTS.json", record)
    return record


def load_fitting_case(directory: str | Path):
    root = Path(directory)
    record = json.loads((root / "FITTING_INPUTS.json").read_text(encoding="utf-8"))
    from tpwrs.storage import sha256_json

    if record.get("record_hash") != sha256_json(
        {k: v for k, v in record.items() if k != "record_hash"}
    ):
        raise ContractError("Fitting partition record changed.")
    # Only metadata is read from the full case. Never open its all-year arrays.
    parent = json.loads((root / "physical_case.json").read_text(encoding="utf-8"))
    if (
        record.get("schema") != "TPWRS_FITTING_PARTITION_V1"
        or record.get("physical_case_hash") != parent["physical_case_hash"]
    ):
        raise ContractError("Fitting partition parent identity mismatch.")
    if (
        sha256_file(root / "FITTING_2025_2028/physical_case.json")
        != record["fitting_metadata_sha256"]
    ):
        raise ContractError("Fitting partition metadata changed.")
    case = load_physical_case(root / "FITTING_2025_2028")
    if case.physical_case_hash != record["fitting_case_hash"] or set(case.time_grid.index.year) != {
        2025,
        2026,
        2027,
        2028,
    }:
        raise ContractError("Held-out physical input in fitting partition.")
    return case, record


from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from tpwrs.storage import ContractError
from tpwrs.storage import sha256_file
from tpwrs.storage import atomic_write_json
from tpwrs.contracts import PolicySolution, StageCertificate


def save_policy_solution(solution: PolicySolution, directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays = root / "solution_arrays.npz"
    payload = {
        "x_mw": solution.x_mw,
        "has_u": np.asarray([solution.u is not None], np.uint8),
        "has_q": np.asarray([solution.q is not None], np.uint8),
    }
    if solution.u is not None:
        payload["u"] = solution.u
    if solution.q is not None:
        payload["q"] = solution.q
    fd, name = tempfile.mkstemp(prefix=".solution_arrays.", suffix=".npz", dir=root)
    os.close(fd)
    tmp = Path(name)
    try:
        np.savez_compressed(tmp, **payload)
        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, arrays)
    finally:
        if tmp.exists():
            tmp.unlink()
    meta = {
        "schema": "TPWRS_POLICY_SOLUTION_V1",
        "solution_hash": solution.solution_hash,
        "case_id": solution.case_id,
        "experiment_hash": solution.experiment_hash,
        "physical_case_id": solution.physical_case_id,
        "physical_case_hash": solution.physical_case_hash,
        "rule": solution.rule,
        "allowance_hash": solution.allowance_hash,
        "stage1": asdict(solution.stage1),
        "stage2": asdict(solution.stage2),
        "stage3": None if solution.stage3 is None else asdict(solution.stage3),
        "s_value_mw_h_row": solution.s_value_mw_h_row,
        "e_value_mwh": solution.e_value_mwh,
        "m2_budget_duals": dict(solution.m2_budget_duals),
        "closure_rounds": solution.closure_rounds,
        "active_row_count": solution.active_row_count,
        "replay_summary": dict(solution.replay_summary),
        "arrays_file": arrays.name,
        "arrays_sha256": sha256_file(arrays),
    }
    atomic_write_json(root / "solution.json", meta)
    return meta


def load_policy_solution(
    directory: str | Path,
    *,
    expected_experiment_hash: str | None = None,
    expected_physical_case_hash: str | None = None,
) -> PolicySolution:
    root = Path(directory)
    meta = json.loads((root / "solution.json").read_text(encoding="utf-8"))
    if meta.get("schema") != "TPWRS_POLICY_SOLUTION_V1":
        raise ContractError("Unsupported policy-solution schema.")
    arrays = root / meta["arrays_file"]
    if sha256_file(arrays) != meta["arrays_sha256"]:
        raise ContractError("Policy-solution arrays hash mismatch.")
    if expected_experiment_hash is not None and meta["experiment_hash"] != expected_experiment_hash:
        raise ContractError("Cached solution experiment hash mismatch.")
    if (
        expected_physical_case_hash is not None
        and meta["physical_case_hash"] != expected_physical_case_hash
    ):
        raise ContractError("Cached solution physical-case hash mismatch.")

    def stage(value: dict[str, Any] | None) -> StageCertificate | None:
        return None if value is None else StageCertificate(**value)

    with np.load(arrays, allow_pickle=False) as z:
        solution = PolicySolution(
            meta["case_id"],
            meta["experiment_hash"],
            meta["physical_case_id"],
            meta["physical_case_hash"],
            meta["rule"],
            meta["allowance_hash"],
            z["x_mw"],
            z["u"] if bool(z["has_u"][0]) else None,
            z["q"] if bool(z["has_q"][0]) else None,
            stage(meta["stage1"]),
            stage(meta["stage2"]),
            stage(meta.get("stage3")),
            float(meta["s_value_mw_h_row"]),
            float(meta["e_value_mwh"]),
            meta.get("m2_budget_duals", {}),
            int(meta.get("closure_rounds", 0)),
            int(meta.get("active_row_count", 0)),
            meta.get("replay_summary", {}),
        )
    if solution.solution_hash != meta["solution_hash"]:
        raise ContractError("Reconstructed policy-solution hash mismatch.")
    return solution


from dataclasses import asdict
import json
from pathlib import Path

from tpwrs.storage import ContractError
from tpwrs.contracts import CardSpec
from tpwrs.storage import atomic_write_json


def card_to_json_dict(card: CardSpec) -> dict[str, object]:
    value = asdict(card)
    value["thresholds"] = {
        str(k): (
            None
            if not all(__import__("math").isfinite(float(x)) for x in v)
            else list(map(float, v))
        )
        for k, v in card.thresholds.items()
    }
    return value


def save_card(card: CardSpec, path: str | Path) -> Path:
    payload = {
        "schema": "TPWRS_CARD_SPEC_V1",
        "card_hash": card.hash,
        "card": card_to_json_dict(card),
    }
    return atomic_write_json(path, payload)


def load_card(path: str | Path) -> CardSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "TPWRS_CARD_SPEC_V1":
        raise ContractError("Unsupported card schema.")
    value = payload["card"]
    value["thresholds"] = {
        str(k): ((-float("inf"), float("inf")) if v is None else tuple(map(float, v)))
        for k, v in value["thresholds"].items()
    }
    value["rates"] = {str(k): tuple(map(float, v)) for k, v in value["rates"].items()}
    value["q_bounds"] = tuple(map(float, value["q_bounds"]))
    value["fit_years"] = tuple(map(int, value.get("fit_years", (2025, 2026, 2027))))
    card = CardSpec(**value)
    if card.hash != payload["card_hash"]:
        raise ContractError("Reconstructed card hash mismatch.")
    return card
