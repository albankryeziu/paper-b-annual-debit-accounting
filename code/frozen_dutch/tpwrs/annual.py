from __future__ import annotations


# ============================================================================
# tpwrs_v3.annual_lp_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import math
import os
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tpwrs.storage import ContractError, SolverFailure
from tpwrs.contracts import ExperimentCase
from tpwrs.storage import sha256_array, sha256_json, sha256_file
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.contracts import PhysicalCase
from tpwrs.replay import physical_objectives, replay_solution
from tpwrs.contracts import PolicySolution, StageCertificate
from tpwrs.replay import ANNUAL_SUPPORT_FACE_CONTRACT, annual_support_face_numeric_tolerance
from tpwrs.compression import add_active_mvar

ProgressCallback = Callable[[dict[str, object]], None]

# V3 replaces the explicit directed-row/slack master by the exact support-function
# projection of the hourly L1 overload objective.  Physical screening and replay
# are unchanged: only the LP representation/solution architecture changes.
ENGINE_VERSION = "ANNUAL_SUPPORT_OA_V3_20260905"
SCREEN_TOL_MW = 1e-9
USEFUL_ACTION_TOL = 0.0
INITIAL_NEAR_BINDING_MARGIN_MW = 1.0  # diagnostic/legacy row-registry only
GEOMETRY_CHUNK_HOURS = 128
ORACLE_CUT_ABS_TOL = 1e-10


@dataclass(frozen=True)
class AnnualGeometry:
    active_hours: np.ndarray
    participants_by_hour: tuple[np.ndarray, ...]
    offsets: np.ndarray
    participant_coordinate: np.ndarray
    local_hour_coordinate: np.ndarray
    xbar_coordinate: np.ndarray
    dt_coordinate: np.ndarray
    potential_pos: tuple[np.ndarray, ...]
    potential_neg: tuple[np.ndarray, ...]
    potential_pos_base: tuple[np.ndarray, ...]
    potential_neg_base: tuple[np.ndarray, ...]
    # Retained for diagnostics/backward-compatible closed-row hints.  These rows
    # are no longer materialized one-for-one in the solver model.
    seed_pos: tuple[np.ndarray, ...]
    seed_neg: tuple[np.ndarray, ...]
    candidate_action_coordinates: int
    retained_action_coordinates: int
    dominated_action_coordinates_removed: int
    potential_directed_rows: int
    seed_directed_rows: int
    constant_s_mw_h_row: float
    geometry_hash: str

    @property
    def variable_local_hours(self) -> np.ndarray:
        return np.flatnonzero(np.diff(self.offsets) > 0).astype(np.int32, copy=False)


@dataclass(frozen=True)
class SupportCut:
    """One exact hourly support-function selector.

    For one hour, any subset of positive and negative directed hinge rows defines
    a globally valid affine lower support of the sum of positive parts.  The
    oracle selects exactly the rows positive at its current point, making the cut
    tight there.  Persisting selectors rather than floating coefficients makes a
    restart hint independently reconstructible from the immutable physical case.
    """

    local_hour: int
    pos_rows: np.ndarray
    neg_rows: np.ndarray


_GEOMETRY_CACHE: OrderedDict[tuple[str, str], AnnualGeometry] = OrderedDict()
GEOMETRY_CACHE_LIMIT = 2


def _memory_snapshot() -> dict[str, float]:
    from tpwrs.execution import memory_snapshot
    return memory_snapshot()


def _emit(callback: ProgressCallback | None, event: str, **payload: object) -> None:
    if callback is None:
        return
    callback(
        {
            "event": event,
            "time_utc_epoch_s": time.time(),
            "engine": ENGINE_VERSION,
            **payload,
            **_memory_snapshot(),
        }
    )


def _default_progress_callback(root: Path) -> ProgressCallback:
    events = root / "ENGINE_EVENTS.jsonl"

    def callback(record: dict[str, object]) -> None:
        payload = {"created_epoch_s": time.time(), **record}
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        atomic_write_json(root / "HEARTBEAT.json", payload)
        event = str(payload.get("event", ""))
        if event in {
            "geometry_complete",
            "geometry_reused",
            "support_cuts_added",
            "support_cuts_reused",
            "stage_start",
            "stage_complete",
            "solver_heartbeat",
            "solve_complete",
        }:
            print(
                f"[{event}] stage={payload.get('stage','-')} "
                f"round={payload.get('closure_round','-')} "
                f"rows={payload.get('model_rows','-')} "
                f"cols={payload.get('model_columns','-')} "
                f"nnz={payload.get('model_nonzeros','-')} "
                f"cuts={payload.get('support_cuts', payload.get('total_support_cuts','-'))} "
                f"privateGB={payload.get('private_gb','-')}",
                flush=True,
            )

    return callback


def _positive_allowance_hash(experiment: ExperimentCase) -> str:
    return sha256_array((experiment.allowance.values > 0.0).astype(np.uint8))


def _screen_geometry(
    case: PhysicalCase,
    experiment: ExperimentCase,
    *,
    callback: ProgressCallback | None = None,
) -> AnnualGeometry:
    """Exact physical reduction before optimization.

    1. A requested-state secure hour has the lexicographic optimum x_t=0: Stage I
       cannot improve below zero there and every nonzero action weakly worsens
       Stage I while strictly worsening Stage II and consuming an upper-bounded
       entitlement.  Such hours are removed exactly.
    2. The action-box extrema identify every directed row that can possibly have
       positive hinge loss anywhere in the admissible box.
    3. An action coordinate that cannot decrease any such directed hinge argument
       is dominated by zero and is removed exactly.
    """
    cache_key = (case.physical_case_hash, _positive_allowance_hash(experiment))
    cached = _GEOMETRY_CACHE.get(cache_key)
    if cached is not None:
        _GEOMETRY_CACHE.move_to_end(cache_key)
        _emit(callback, "geometry_reused", geometry_hash=cached.geometry_hash)
        return cached

    started = time.perf_counter()
    H = case.flat_H
    h = case.flat_h
    ratings = case.flat_ratings
    G = case.flat_G
    G_pos = np.maximum(G, 0.0)
    G_neg = np.maximum(-G, 0.0)
    xbar = case.xbar_mw
    request_state = case.requested_state_mw
    allowance_positive = experiment.allowance.values > 0.0
    dt = case.time_grid.delta

    active_hours: list[int] = []
    potential_pos: list[np.ndarray] = []
    potential_neg: list[np.ndarray] = []
    potential_pos_base: list[np.ndarray] = []
    potential_neg_base: list[np.ndarray] = []
    seed_pos: list[np.ndarray] = []
    seed_neg: list[np.ndarray] = []
    participants: list[np.ndarray] = []
    constant_s = 0.0
    candidate_count = 0
    retained_count = 0
    potential_count = 0
    seed_count = 0

    _emit(callback, "geometry_start", n_time=case.n_time, n_rows=case.n_flat_rows)

    for start in range(0, case.n_time, GEOMETRY_CHUNK_HOURS):
        stop = min(start + GEOMETRY_CHUNK_HOURS, case.n_time)
        base = request_state[start:stop] @ H.T + h[None, :]
        active_local = np.flatnonzero(np.any(np.abs(base) > ratings[None, :], axis=1))
        if not len(active_local):
            continue

        xb = np.asarray(xbar[start:stop][active_local], dtype=float).copy()
        xb[:, ~allowance_positive] = 0.0
        active_base = base[active_local]
        pos_upper = active_base + xb @ G_pos.T
        neg_upper = -active_base + xb @ G_neg.T

        for j, chunk_local in enumerate(active_local):
            global_hour = start + int(chunk_local)
            base_row = active_base[j]
            pos = np.flatnonzero(pos_upper[j] >= ratings - SCREEN_TOL_MW).astype(np.int32, copy=False)
            neg = np.flatnonzero(neg_upper[j] >= ratings - SCREEN_TOL_MW).astype(np.int32, copy=False)

            pos_seed = np.intersect1d(
                pos,
                np.flatnonzero(base_row >= ratings - INITIAL_NEAR_BINDING_MARGIN_MW),
                assume_unique=True,
            ).astype(np.int32, copy=False)
            neg_seed = np.intersect1d(
                neg,
                np.flatnonzero(-base_row >= ratings - INITIAL_NEAR_BINDING_MARGIN_MW),
                assume_unique=True,
            ).astype(np.int32, copy=False)

            candidates = np.flatnonzero((xbar[global_hour] > 0.0) & allowance_positive).astype(np.int32, copy=False)
            candidate_count += int(len(candidates))
            useful = np.zeros(len(candidates), dtype=bool)
            if len(candidates) and len(pos):
                useful |= np.any(G[np.ix_(pos, candidates)] < -USEFUL_ACTION_TOL, axis=0)
            if len(candidates) and len(neg):
                useful |= np.any(G[np.ix_(neg, candidates)] > USEFUL_ACTION_TOL, axis=0)
            kept = candidates[useful]
            retained_count += int(len(kept))

            if not len(kept):
                constant_s += float(dt[global_hour] * np.maximum(np.abs(base_row) - ratings, 0.0).sum())

            active_hours.append(global_hour)
            potential_pos.append(pos)
            potential_neg.append(neg)
            potential_pos_base.append(np.asarray(base_row[pos], dtype=float).copy())
            potential_neg_base.append(np.asarray(base_row[neg], dtype=float).copy())
            seed_pos.append(pos_seed)
            seed_neg.append(neg_seed)
            participants.append(kept)
            potential_count += int(len(pos) + len(neg))
            seed_count += int(len(pos_seed) + len(neg_seed))

        _emit(
            callback,
            "geometry_chunk",
            hours_scanned=stop,
            active_hours=len(active_hours),
            candidate_action_coordinates=candidate_count,
            retained_action_coordinates=retained_count,
            potential_directed_rows=potential_count,
        )

    active_arr = np.asarray(active_hours, dtype=np.int32)
    offsets = np.zeros(len(active_arr) + 1, dtype=np.int64)
    if len(active_arr):
        offsets[1:] = np.cumsum(np.fromiter((len(x) for x in participants), dtype=np.int64))
    if int(offsets[-1]):
        participant_coordinate = np.concatenate(participants).astype(np.int32, copy=False)
        local_hour_coordinate = np.repeat(np.arange(len(active_arr), dtype=np.int32), np.diff(offsets))
        global_hour_coordinate = active_arr[local_hour_coordinate]
        xbar_coordinate = xbar[global_hour_coordinate, participant_coordinate]
        dt_coordinate = dt[global_hour_coordinate]
    else:
        participant_coordinate = np.empty(0, dtype=np.int32)
        local_hour_coordinate = np.empty(0, dtype=np.int32)
        xbar_coordinate = np.empty(0, dtype=float)
        dt_coordinate = np.empty(0, dtype=float)

    geometry_hash = sha256_json(
        {
            "engine": ENGINE_VERSION,
            "physical_case_hash": case.physical_case_hash,
            "allowance_positive_hash": _positive_allowance_hash(experiment),
            "active_hours": sha256_array(active_arr),
            "participant_coordinate": sha256_array(participant_coordinate),
            "local_hour_coordinate": sha256_array(local_hour_coordinate),
            "potential_pos": sha256_json([sha256_array(x) for x in potential_pos]),
            "potential_neg": sha256_json([sha256_array(x) for x in potential_neg]),
            "constant_s": float(constant_s),
        }
    )
    result = AnnualGeometry(
        active_arr,
        tuple(participants),
        offsets,
        participant_coordinate,
        local_hour_coordinate,
        np.asarray(xbar_coordinate, dtype=float),
        np.asarray(dt_coordinate, dtype=float),
        tuple(potential_pos),
        tuple(potential_neg),
        tuple(potential_pos_base),
        tuple(potential_neg_base),
        tuple(seed_pos),
        tuple(seed_neg),
        candidate_count,
        retained_count,
        candidate_count - retained_count,
        potential_count,
        seed_count,
        float(constant_s),
        geometry_hash,
    )
    _GEOMETRY_CACHE[cache_key] = result
    _GEOMETRY_CACHE.move_to_end(cache_key)
    while len(_GEOMETRY_CACHE) > GEOMETRY_CACHE_LIMIT:
        _GEOMETRY_CACHE.popitem(last=False)
    _emit(
        callback,
        "geometry_complete",
        geometry_hash=geometry_hash,
        runtime_s=time.perf_counter() - started,
        active_hours=len(active_arr),
        variable_hours=int(len(result.variable_local_hours)),
        candidate_action_coordinates=candidate_count,
        retained_action_coordinates=retained_count,
        dominated_action_coordinates_removed=candidate_count - retained_count,
        potential_directed_rows=potential_count,
        legacy_initial_seed_directed_rows=seed_count,
        constant_s_mw_h_row=constant_s,
    )
    return result


def build_geometry_audit(case: PhysicalCase, experiment: ExperimentCase) -> dict[str, object]:
    """Cheap model-size audit for the exact projected support master."""
    geo = _screen_geometry(case, experiment)
    R = int(geo.retained_action_coordinates)
    K = int(len(geo.variable_local_hours))
    N = int(case.n_participants)
    # One initial tight support cut per variable hour.  Each cut has at most the
    # local x coordinates plus one theta coefficient.  Every x has exactly one
    # annual-budget coefficient for the supported debit laws.
    initial_support_nnz_upper = int(R + (R + K))
    return {
        "schema": "TPWRS_ANNUAL_GEOMETRY_V2_SUPPORT_OA",
        "engine": ENGINE_VERSION,
        "geometry_hash": geo.geometry_hash,
        "active_hours": int(len(geo.active_hours)),
        "variable_hours": K,
        "candidate_action_coordinates": geo.candidate_action_coordinates,
        "retained_action_coordinates": R,
        "dominated_action_coordinates_removed": geo.dominated_action_coordinates_removed,
        "potential_directed_rows": geo.potential_directed_rows,
        "legacy_initial_seed_directed_rows": geo.seed_directed_rows,
        "constant_s_mw_h_row": geo.constant_s_mw_h_row,
        "legacy_explicit_seed_rows": int(N + geo.seed_directed_rows),
        "legacy_explicit_seed_columns": int(R + geo.seed_directed_rows),
        "support_initial_rows": int(N + K),
        "support_initial_columns": int(R + K),
        "support_initial_nonzeros_upper_bound": initial_support_nnz_upper,
        "explicit_security_slack_variables": 0,
        "support_epigraph_variables": K,
    }


# ---------------------------------------------------------------------------
# Backward-compatible directed-row registry.  It is a warm hint only; the new
# engine does not materialize these rows one-for-one.
# ---------------------------------------------------------------------------

def _encode_rows(included_pos: Sequence[Iterable[int]], included_neg: Sequence[Iterable[int]], n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray([h * n_rows + int(r) for h, rows in enumerate(included_pos) for r in sorted(set(map(int, rows)))], dtype=np.int64)
    neg = np.asarray([h * n_rows + int(r) for h, rows in enumerate(included_neg) for r in sorted(set(map(int, rows)))], dtype=np.int64)
    return pos, neg


def _decode_rows(keys: Iterable[int], n_rows: int, n_hours: int) -> list[set[int]]:
    out = [set() for _ in range(n_hours)]
    for raw in keys:
        value = int(raw)
        h, r = divmod(value, n_rows)
        if 0 <= h < n_hours and 0 <= r < n_rows:
            out[h].add(r)
    return out


def load_closed_row_seed(directory: str | Path, *, geometry_hash: str, n_rows: int, n_hours: int) -> tuple[list[set[int]], list[set[int]]] | None:
    root = Path(directory)
    meta_path = root / "CLOSED_ROWS_META.json"
    rows_path = root / "CLOSED_ROWS.npz"
    if not meta_path.is_file() or not rows_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("geometry_hash") != geometry_hash or meta.get("arrays_sha256") != sha256_file(rows_path):
            return None
        with np.load(rows_path, allow_pickle=False) as z:
            return _decode_rows(z["pos_keys"], n_rows, n_hours), _decode_rows(z["neg_keys"], n_rows, n_hours)
    except Exception:
        return None


def _save_closed_rows(directory: Path, *, geometry: AnnualGeometry, n_rows: int, included_pos: Sequence[Iterable[int]], included_neg: Sequence[Iterable[int]], stage: str, closure_round: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pos, neg = _encode_rows(included_pos, included_neg, n_rows)
    tmp = directory / ".CLOSED_ROWS.tmp.npz"
    np.savez_compressed(tmp, pos_keys=pos, neg_keys=neg)
    os.replace(tmp, directory / "CLOSED_ROWS.npz")
    atomic_write_json(
        directory / "CLOSED_ROWS_META.json",
        {
            "schema": "TPWRS_CLOSED_DIRECTED_ROWS_V2_SUPPORT_REGISTRY",
            "engine": ENGINE_VERSION,
            "geometry_hash": geometry.geometry_hash,
            "arrays_sha256": sha256_file(directory / "CLOSED_ROWS.npz"),
            "stage": stage,
            "closure_round": int(closure_round),
            "positive_directed_rows": int(len(pos)),
            "negative_directed_rows": int(len(neg)),
            "total_directed_rows": int(len(pos) + len(neg)),
            "note": "registry/warm hint only; rows are aggregated into hourly support cuts",
        },
    )


# ---------------------------------------------------------------------------
# Exact hourly hinge/support representation.
# ---------------------------------------------------------------------------

def _validated_subset(rows: np.ndarray, potential: np.ndarray) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.int32).reshape(-1)
    if len(arr) == 0:
        return arr
    arr = np.unique(arr)
    idx = np.searchsorted(potential, arr)
    if np.any(idx >= len(potential)) or np.any(potential[idx] != arr):
        raise SolverFailure("Support-cut selector contains a row outside the exact action-box universe.")
    return arr


def _base_for_subset(rows: np.ndarray, potential: np.ndarray, base: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.empty(0, dtype=float)
    idx = np.searchsorted(potential, rows)
    if np.any(idx >= len(potential)) or np.any(potential[idx] != rows):
        raise SolverFailure("Support selector/base alignment failed.")
    return base[idx]


def _support_cut_key(cut: SupportCut) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([int(cut.local_hour)], dtype=np.int32).tobytes())
    digest.update(np.asarray(cut.pos_rows, dtype=np.int32).tobytes())
    digest.update(b"|")
    digest.update(np.asarray(cut.neg_rows, dtype=np.int32).tobytes())
    return digest.hexdigest()


def _canonical_support_cut(geometry: AnnualGeometry, local_hour: int, pos_rows: np.ndarray, neg_rows: np.ndarray) -> SupportCut:
    h = int(local_hour)
    if not (0 <= h < len(geometry.active_hours)) or len(geometry.participants_by_hour[h]) == 0:
        raise SolverFailure("Support cut refers to a fixed/nonexistent hour.")
    return SupportCut(
        h,
        _validated_subset(pos_rows, geometry.potential_pos[h]),
        _validated_subset(neg_rows, geometry.potential_neg[h]),
    )


def _support_cut_coefficients(case: PhysicalCase, geometry: AnnualGeometry, cut: SupportCut) -> tuple[np.ndarray, float]:
    h = int(cut.local_hour)
    participants = geometry.participants_by_hour[h]
    dt = float(case.time_grid.delta[int(geometry.active_hours[h])])
    rating = case.flat_ratings
    grad = np.zeros(len(participants), dtype=float)
    intercept = 0.0
    if len(cut.pos_rows):
        base = _base_for_subset(cut.pos_rows, geometry.potential_pos[h], geometry.potential_pos_base[h])
        grad += np.asarray(case.flat_G[np.ix_(cut.pos_rows, participants)].sum(axis=0), dtype=float).reshape(-1)
        intercept += float(np.sum(base - rating[cut.pos_rows]))
    if len(cut.neg_rows):
        base = _base_for_subset(cut.neg_rows, geometry.potential_neg[h], geometry.potential_neg_base[h])
        grad -= np.asarray(case.flat_G[np.ix_(cut.neg_rows, participants)].sum(axis=0), dtype=float).reshape(-1)
        intercept += float(np.sum(-base - rating[cut.neg_rows]))
    return dt * grad, dt * intercept


def _evaluate_hour_support(case: PhysicalCase, geometry: AnnualGeometry, local_hour: int, x_local: np.ndarray) -> tuple[float, SupportCut]:
    """Evaluate the exact screened hourly overload and return a tight support cut."""
    h = int(local_hour)
    participants = geometry.participants_by_hour[h]
    xh = np.asarray(x_local, dtype=float).reshape(-1)
    if xh.shape != (len(participants),):
        raise ValueError("Hourly support oracle x dimension mismatch.")
    rating = case.flat_ratings
    dt = float(case.time_grid.delta[int(geometry.active_hours[h])])
    pos_rows = geometry.potential_pos[h]
    neg_rows = geometry.potential_neg[h]

    if len(pos_rows):
        pos_z = geometry.potential_pos_base[h] + case.flat_G[np.ix_(pos_rows, participants)] @ xh - rating[pos_rows]
        pos_mask = pos_z > 0.0
        pos_selector = pos_rows[pos_mask]
        pos_value = float(np.maximum(pos_z, 0.0).sum())
    else:
        pos_selector = np.empty(0, dtype=np.int32)
        pos_value = 0.0
    if len(neg_rows):
        neg_z = -geometry.potential_neg_base[h] - case.flat_G[np.ix_(neg_rows, participants)] @ xh - rating[neg_rows]
        neg_mask = neg_z > 0.0
        neg_selector = neg_rows[neg_mask]
        neg_value = float(np.maximum(neg_z, 0.0).sum())
    else:
        neg_selector = np.empty(0, dtype=np.int32)
        neg_value = 0.0
    return dt * (pos_value + neg_value), SupportCut(h, pos_selector.astype(np.int32, copy=False), neg_selector.astype(np.int32, copy=False))


def _evaluate_support(case: PhysicalCase, geometry: AnnualGeometry, x_coordinate: np.ndarray) -> tuple[np.ndarray, list[SupportCut]]:
    variable = geometry.variable_local_hours
    values = np.zeros(len(variable), dtype=float)
    cuts: list[SupportCut] = []
    for k, h_raw in enumerate(variable):
        h = int(h_raw)
        start, stop = int(geometry.offsets[h]), int(geometry.offsets[h + 1])
        values[k], cut = _evaluate_hour_support(case, geometry, h, x_coordinate[start:stop])
        cuts.append(cut)
    return values, cuts


def _initial_zero_support_cuts(case: PhysicalCase, geometry: AnnualGeometry) -> list[SupportCut]:
    cuts: list[SupportCut] = []
    rating = case.flat_ratings
    for h_raw in geometry.variable_local_hours:
        h = int(h_raw)
        pos = geometry.potential_pos[h]
        neg = geometry.potential_neg[h]
        # At x=0 no matrix multiplication is needed.  The tight selector is
        # simply the set of baseline directed violations.
        pos_selector = pos[geometry.potential_pos_base[h] - rating[pos] > 0.0]
        neg_selector = neg[-geometry.potential_neg_base[h] - rating[neg] > 0.0]
        cut = SupportCut(h, pos_selector.astype(np.int32, copy=False), neg_selector.astype(np.int32, copy=False))
        # Active variable hours are requested-state overloaded, so this is
        # normally nonempty.  theta>=0 already represents an empty selector.
        if len(cut.pos_rows) or len(cut.neg_rows):
            cuts.append(cut)
    return cuts


def _cut_pool_arrays(cuts: Sequence[SupportCut]) -> dict[str, np.ndarray]:
    hours = np.asarray([c.local_hour for c in cuts], dtype=np.int32)
    pos_offsets = np.zeros(len(cuts) + 1, dtype=np.int64)
    neg_offsets = np.zeros(len(cuts) + 1, dtype=np.int64)
    if cuts:
        pos_offsets[1:] = np.cumsum([len(c.pos_rows) for c in cuts], dtype=np.int64)
        neg_offsets[1:] = np.cumsum([len(c.neg_rows) for c in cuts], dtype=np.int64)
    pos_rows = np.concatenate([c.pos_rows for c in cuts]).astype(np.int32, copy=False) if int(pos_offsets[-1]) else np.empty(0, dtype=np.int32)
    neg_rows = np.concatenate([c.neg_rows for c in cuts]).astype(np.int32, copy=False) if int(neg_offsets[-1]) else np.empty(0, dtype=np.int32)
    return {"hours": hours, "pos_offsets": pos_offsets, "neg_offsets": neg_offsets, "pos_rows": pos_rows, "neg_rows": neg_rows}


def _save_support_cut_pool(directory: Path, geometry: AnnualGeometry, cuts: Sequence[SupportCut], *, stage: str, closure_round: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays = _cut_pool_arrays(cuts)
    tmp = directory / ".SUPPORT_CUTS.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, directory / "SUPPORT_CUTS.npz")
    atomic_write_json(
        directory / "SUPPORT_CUTS_META.json",
        {
            "schema": "TPWRS_SUPPORT_CUT_POOL_V1",
            "engine": ENGINE_VERSION,
            "geometry_hash": geometry.geometry_hash,
            "arrays_sha256": sha256_file(directory / "SUPPORT_CUTS.npz"),
            "stage": stage,
            "closure_round": int(closure_round),
            "cut_count": int(len(cuts)),
            "selector_positive_rows": int(sum(len(c.pos_rows) for c in cuts)),
            "selector_negative_rows": int(sum(len(c.neg_rows) for c in cuts)),
        },
    )


def load_support_cut_seed(directory: str | Path, *, geometry: AnnualGeometry) -> tuple[SupportCut, ...] | None:
    """Load only mathematically valid selector cuts for this exact geometry.

    Persisted coefficients are deliberately not trusted.  The file stores row
    selectors; coefficients are reconstructed from the current immutable case at
    model build time.  A corrupt/foreign hint is discarded rather than accepted.
    """
    root = Path(directory)
    meta_path = root / "SUPPORT_CUTS_META.json"
    array_path = root / "SUPPORT_CUTS.npz"
    if not meta_path.is_file() or not array_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("geometry_hash") != geometry.geometry_hash or meta.get("arrays_sha256") != sha256_file(array_path):
            return None
        with np.load(array_path, allow_pickle=False) as z:
            hours = np.asarray(z["hours"], dtype=np.int32)
            po = np.asarray(z["pos_offsets"], dtype=np.int64)
            no = np.asarray(z["neg_offsets"], dtype=np.int64)
            pr = np.asarray(z["pos_rows"], dtype=np.int32)
            nr = np.asarray(z["neg_rows"], dtype=np.int32)
        if len(po) != len(hours) + 1 or len(no) != len(hours) + 1 or po[0] != 0 or no[0] != 0 or po[-1] != len(pr) or no[-1] != len(nr):
            return None
        cuts: list[SupportCut] = []
        for j, h in enumerate(hours):
            cuts.append(_canonical_support_cut(geometry, int(h), pr[po[j]:po[j + 1]], nr[no[j]:no[j + 1]]))
        return tuple(cuts)
    except Exception:
        return None


def _rows_from_cuts(geometry: AnnualGeometry, cuts: Sequence[SupportCut]) -> tuple[list[set[int]], list[set[int]]]:
    pos = [set() for _ in geometry.active_hours]
    neg = [set() for _ in geometry.active_hours]
    for cut in cuts:
        pos[cut.local_hour].update(map(int, cut.pos_rows))
        neg[cut.local_hour].update(map(int, cut.neg_rows))
    return pos, neg


def _cuts_from_directed_hint(geometry: AnnualGeometry, hint: tuple[list[set[int]], list[set[int]]] | None) -> list[SupportCut]:
    if hint is None:
        return []
    if len(hint) != 2 or len(hint[0]) != len(geometry.active_hours) or len(hint[1]) != len(geometry.active_hours):
        return []
    out: list[SupportCut] = []
    for h_raw in geometry.variable_local_hours:
        h = int(h_raw)
        try:
            cut = _canonical_support_cut(
                geometry,
                h,
                np.asarray(sorted(hint[0][h]), dtype=np.int32),
                np.asarray(sorted(hint[1][h]), dtype=np.int32),
            )
        except Exception:
            continue
        if len(cut.pos_rows) or len(cut.neg_rows):
            out.append(cut)
    return out


def _build_budget_matrix(case: PhysicalCase, experiment: ExperimentCase, geometry: AnnualGeometry, q: np.ndarray | None) -> sp.csr_matrix:
    R = geometry.retained_action_coordinates
    if R == 0:
        return sp.csr_matrix((case.n_participants, 0), dtype=float)
    participant = geometry.participant_coordinate
    local_h = geometry.local_hour_coordinate
    global_h = geometry.active_hours[local_h]
    if experiment.rule == "M1_LP":
        coeff = geometry.dt_coordinate / geometry.xbar_coordinate
    elif experiment.rule in {"M2", "M2_MATCHED"}:
        coeff = geometry.dt_coordinate / case.capacities_mw[participant]
    elif experiment.rule == "M3":
        if q is None:
            raise ContractError("M3 annual engine requires q.")
        coeff = geometry.dt_coordinate * q[global_h, participant] / case.capacities_mw[participant]
    else:
        raise ContractError(f"Unsupported continuous annual rule {experiment.rule}.")
    return sp.csr_matrix((np.asarray(coeff, dtype=float), (participant, np.arange(R, dtype=np.int64))), shape=(case.n_participants, R))


def _reconstruct_active_x(case: PhysicalCase, geometry: AnnualGeometry, x_coordinate: np.ndarray) -> np.ndarray:
    full = np.zeros((case.n_time, case.n_participants), dtype=float)
    if geometry.retained_action_coordinates:
        global_h = geometry.active_hours[geometry.local_hour_coordinate]
        full[global_h, geometry.participant_coordinate] = x_coordinate
    return full


def _stage_certificate(label: str, objective: float, runtime: float, metadata: dict[str, object]) -> StageCertificate:
    lower = float(metadata.get("dual_certificate", {}).get("lower_bound", objective))
    gap = max(0.0, float(objective) - lower)
    rel = gap / max(1.0, abs(float(objective)))
    return StageCertificate(
        label,
        "OPTIMAL",
        float(objective),
        float(runtime),
        lower,
        rel,
        "Gurobi OPTIMAL support master; exact hourly hinge oracle closed and independent full replay passed",
        metadata,
    )


def _support_master_dual_bound(
    *,
    c_x: np.ndarray,
    upper_x: np.ndarray,
    budget_matrix: sp.csr_matrix,
    allowance: np.ndarray,
    budget_pi: np.ndarray,
    support_records: Sequence[tuple[np.ndarray, np.ndarray, sp.csr_matrix, object]],
    constant_s: float,
    stage1_cap_rhs: float | None,
    stage1_cap_pi: float | None,
) -> dict[str, float | str]:
    """Reconstruct a valid numerical dual bound for the projected master.

    Each support row is a'x-theta_h <= -alpha.  In Stage I, its multiplier
    zeta>=0 must satisfy sum_{cuts at h} zeta<=1.  In Stage II the right-hand
    side is eta=-pi_cap.  Projecting solver multipliers into these conditions and
    minimizing the resulting Lagrangian over 0<=x<=upper gives a valid lower
    bound in original x coordinates.  Missing support cuts have zero duals, so
    the bound also extends to the complete epigraph LP.
    """
    c = np.asarray(c_x, dtype=float).reshape(-1)
    upper = np.asarray(upper_x, dtype=float).reshape(-1)
    lam_raw = -np.asarray(budget_pi, dtype=float).reshape(-1)
    lam = np.maximum(lam_raw, 0.0)
    budget_projection = float(np.max(np.abs(lam - lam_raw), initial=0.0))
    eta = 1.0 if stage1_cap_rhs is None else max(0.0, -float(stage1_cap_pi if stage1_cap_pi is not None else 0.0))

    all_hours: list[np.ndarray] = []
    all_raw: list[np.ndarray] = []
    for hours, _intercepts, _ax, con in support_records:
        raw = -np.asarray(con.Pi, dtype=float).reshape(-1)
        if len(raw) != len(hours):
            raise SolverFailure("Support-cut dual dimension mismatch.")
        all_hours.append(np.asarray(hours, dtype=np.int32))
        all_raw.append(np.maximum(raw, 0.0))
    if all_hours:
        hours_flat = np.concatenate(all_hours)
        raw_flat = np.concatenate(all_raw)
        n_hours = int(hours_flat.max(initial=-1)) + 1
        sums = np.bincount(hours_flat, weights=raw_flat, minlength=n_hours)
        scale = np.ones_like(sums)
        mask = sums > eta
        scale[mask] = eta / sums[mask]
        z_flat = raw_flat * scale[hours_flat]
    else:
        hours_flat = np.empty(0, dtype=np.int32)
        raw_flat = np.empty(0, dtype=float)
        z_flat = np.empty(0, dtype=float)

    r = c + np.asarray(budget_matrix.T @ lam).reshape(-1)
    alpha_term = 0.0
    cursor = 0
    magnitude = abs(float(np.dot(np.asarray(allowance, dtype=float), lam)))
    max_support_projection = 0.0
    for hours, intercepts, ax, _con in support_records:
        count = len(hours)
        z = z_flat[cursor:cursor + count]
        raw = raw_flat[cursor:cursor + count]
        cursor += count
        if count:
            r += np.asarray(ax.T @ z).reshape(-1)
            alpha_piece = float(np.dot(np.asarray(intercepts, dtype=float), z))
            alpha_term += alpha_piece
            magnitude += abs(alpha_piece)
            max_support_projection = max(max_support_projection, float(np.max(np.abs(z - raw), initial=0.0)))

    terms = [-float(np.dot(np.asarray(allowance, dtype=float), lam)), alpha_term]
    if stage1_cap_rhs is None:
        terms.append(float(constant_s))
    else:
        terms.append(eta * (float(constant_s) - float(stage1_cap_rhs)))
    box_term = float(np.dot(upper, np.minimum(r, 0.0)))
    terms.append(box_term)
    magnitude += abs(float(constant_s)) + abs(box_term) + float(np.dot(np.abs(r), upper))
    raw_bound = math.fsum(terms)
    guard = 64.0 * np.finfo(float).eps * max(1.0, magnitude)
    return {
        "lower_bound": float(raw_bound - guard),
        "unadjusted_lower_bound": float(raw_bound),
        "floating_point_guard": float(guard),
        "budget_sign_projection_max": budget_projection,
        "support_projection_max": float(max_support_projection),
        "cap_multiplier": float(eta if stage1_cap_rhs is not None else 0.0),
        "bound_kind": "NUMERICAL_SUPPORT_MASTER_DUAL_LOWER_BOUND",
    }


def solve_annual_gurobi(
    case: PhysicalCase,
    experiment: ExperimentCase,
    *,
    q: np.ndarray | None = None,
    progress_callback: ProgressCallback | None = None,
    checkpoint_dir: str | Path | None = None,
    initial_directed_rows: tuple[list[set[int]], list[set[int]]] | None = None,
    maximum_closure_rounds: int = 100,
) -> PolicySolution:
    """Solve annual continuous rules by exact projected hourly support OA.

    The complete scientific LP is unchanged.  Explicit row-wise security slacks
    are projected out using

        sum_j [z_j(x_t)]_+ = max_{u in [0,1]^m} u' z(x_t).

    One theta variable per variable hour represents this value.  The oracle adds
    a tight affine support cut obtained by choosing u_j=1 exactly on the currently
    positive directed hinge rows.  Because the support set has finitely many
    extreme selectors, exact arithmetic terminates finitely.  Numerically, a
    stage is accepted only after the support gap is below the fixed certification
    tolerance and independent complete-row replay agrees.
    """
    if experiment.mode != "annual":
        raise ContractError("Support annual LP engine requires annual mode.")
    if experiment.rule not in {"M1_LP", "M2", "M3", "M2_MATCHED"}:
        raise ContractError(f"Support annual LP engine does not support {experiment.rule}.")
    if experiment.solver_profile.backend not in {"gurobi", "auto"}:
        raise ContractError("Support annual LP engine requires Gurobi.")
    case.verify_experiment_identity(experiment.physical_case_id, experiment.physical_case_hash, experiment.time_grid_id)
    q_arr = None if q is None else np.asarray(q, dtype=float)
    if experiment.rule == "M3":
        if q_arr is None or q_arr.shape != case.requests_mw.shape or np.any(~np.isfinite(q_arr)) or np.any(q_arr <= 0):
            raise ContractError("M3 requires positive q aligned with the case.")
        if experiment.rate_matrix_hash is not None and sha256_array(q_arr) != experiment.rate_matrix_hash:
            raise ContractError("M3 q differs from the immutable experiment hash.")

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:  # pragma: no cover - host dependency
        raise SolverFailure(f"Gurobi import failed: {exc}") from exc

    checkpoint = None if checkpoint_dir is None else Path(checkpoint_dir)
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
        if progress_callback is None:
            progress_callback = _default_progress_callback(checkpoint)

    geometry = _screen_geometry(case, experiment, callback=progress_callback)
    audit = build_geometry_audit(case, experiment)
    if checkpoint is not None:
        atomic_write_json(checkpoint / "GEOMETRY_AUDIT.json", audit)

    if geometry.retained_action_coordinates == 0:
        x_full = np.zeros_like(case.requests_mw, dtype=float)
        S, E = physical_objectives(case, x_full)
        meta = {
            **audit,
            "model_rows": 0,
            "model_columns": 0,
            "model_nonzeros": 0,
            "dual_certificate": {"lower_bound": float(S), "bound_kind": "DEGENERATE_EXACT"},
            **_memory_snapshot(),
        }
        stage1 = _stage_certificate("I", S, 0.0, meta)
        stage2 = _stage_certificate("II", E, 0.0, {**meta, "dual_certificate": {"lower_bound": float(E), "bound_kind": "DEGENERATE_EXACT"}})
        temp = PolicySolution(
            experiment.case_id, experiment.hash, case.physical_case_id, case.physical_case_hash,
            experiment.rule, experiment.allowance.hash, x_full, None, q_arr, stage1, stage2,
            None, S, E, {pid: 0.0 for pid in case.participant_ids} if experiment.rule == "M2" else {},
            0, 0, {},
        )
        replay = replay_solution(case, experiment, temp)
        if not replay.passed:
            raise SolverFailure("Degenerate annual solution failed replay.")
        return PolicySolution(
            temp.case_id, temp.experiment_hash, temp.physical_case_id, temp.physical_case_hash,
            temp.rule, temp.allowance_hash, temp.x_mw, temp.u, temp.q, temp.stage1, temp.stage2,
            temp.stage3, temp.s_value_mw_h_row, temp.e_value_mwh, temp.m2_budget_duals, 0, 0,
            {"passed": True, "engine": ENGINE_VERSION, "geometry_hash": geometry.geometry_hash,
             "max_balance_residual_mw": replay.max_balance_residual_mw,
             "max_budget_violation_h": replay.max_budget_violation_h,
             "max_x_bound_violation_mw": replay.max_x_bound_violation_mw,
             "max_row_violation_mw": replay.max_row_violation_mw},
        )

    profile = experiment.solver_profile
    model = gp.Model(experiment.case_id)
    execution = None
    try:
        from tpwrs.execution import ExecutionBudget
        R = int(geometry.retained_action_coordinates)
        variable_hours = geometry.variable_local_hours
        K = int(len(variable_hours))
        execution = ExecutionBudget(checkpoint, progress_callback)
        # Route the guard from the exact compressed coordinate count before
        # allocating the native master.  This keeps full-year production under
        # the pre-allocation headroom cap while bounded teacher masters avoid a
        # false production classification based on the potential row universe.
        execution.configure(model, model_size=(R + K, 0, 0))
        model.Params.OutputFlag = 1 if progress_callback is not None or profile.output_flag else 0
        model.Params.LogToConsole = model.Params.OutputFlag
        model.Params.Presolve = 2 if profile.presolve else 0
        model.Params.FeasibilityTol = min(max(profile.feasibility_tol, 1e-9), 1e-2)
        model.Params.OptimalityTol = min(max(profile.feasibility_tol, 1e-9), 1e-2)
        model.Params.Threads = int(profile.threads)
        model.Params.Seed = int(profile.seed)
        model.Params.Method = {"auto": -1, "primal_simplex": 0, "dual_simplex": 1, "barrier": 2}[profile.lp_method]
        if profile.time_limit_s is not None:
            model.Params.TimeLimit = min(float(profile.time_limit_s), execution.max_stage_s)
        if checkpoint is not None:
            model.Params.LogFile = str(checkpoint / "solver.log")

        theta_index = np.full(len(geometry.active_hours), -1, dtype=np.int32)
        theta_index[variable_hours] = np.arange(K, dtype=np.int32)

        lb = np.zeros(R + K, dtype=float)
        ub = np.concatenate((geometry.xbar_coordinate, np.full(K, GRB.INFINITY, dtype=float)))
        zvar = add_active_mvar(model,R + K, lb=lb, ub=ub, name="annual_projected")
        x = zvar[:R]
        theta = zvar[R:]

        budget_matrix = _build_budget_matrix(case, experiment, geometry, q_arr)
        budget_full = sp.hstack((budget_matrix, sp.csr_matrix((case.n_participants, K))), format="csr")
        budget_con = model.addMConstr(budget_full, zvar, GRB.LESS_EQUAL, experiment.allowance.values, name="annual_budget")
        e_coeff = np.asarray(geometry.dt_coordinate, dtype=float)

        # Support records: (local-hour ids, intercepts, x-only cut matrix, Gurobi MConstr).
        support_records: list[tuple[np.ndarray, np.ndarray, sp.csr_matrix, object]] = []
        all_cuts: list[SupportCut] = []
        cut_keys: set[str] = set()
        cut_batch = 0

        def add_support_cuts(candidates: Sequence[SupportCut], *, reason: str) -> int:
            nonlocal cut_batch
            accepted: list[SupportCut] = []
            gradients: list[np.ndarray] = []
            intercepts: list[float] = []
            hours: list[int] = []
            for raw in candidates:
                cut = _canonical_support_cut(geometry, raw.local_hour, raw.pos_rows, raw.neg_rows)
                if not (len(cut.pos_rows) or len(cut.neg_rows)):
                    continue  # theta>=0 is exactly the empty-selector support.
                key = _support_cut_key(cut)
                if key in cut_keys:
                    continue
                grad, intercept = _support_cut_coefficients(case, geometry, cut)
                cut_keys.add(key)
                accepted.append(cut)
                gradients.append(grad)
                intercepts.append(float(intercept))
                hours.append(int(cut.local_hour))
            if not accepted:
                return 0

            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
            x_rows: list[int] = []
            x_cols: list[int] = []
            x_data: list[float] = []
            for j, (cut, grad) in enumerate(zip(accepted, gradients)):
                h = cut.local_hour
                start, stop = int(geometry.offsets[h]), int(geometry.offsets[h + 1])
                if len(grad) != stop - start:
                    raise SolverFailure("Support-cut local gradient dimension mismatch.")
                nz = np.flatnonzero(grad != 0.0)
                if len(nz):
                    global_cols = start + nz
                    rows.extend([j] * len(nz)); cols.extend(map(int, global_cols)); data.extend(map(float, grad[nz]))
                    x_rows.extend([j] * len(nz)); x_cols.extend(map(int, global_cols)); x_data.extend(map(float, grad[nz]))
                rows.append(j); cols.append(R + int(theta_index[h])); data.append(-1.0)
            A = sp.csr_matrix((data, (rows, cols)), shape=(len(accepted), R + K))
            A_x = sp.csr_matrix((x_data, (x_rows, x_cols)), shape=(len(accepted), R))
            rhs = -np.asarray(intercepts, dtype=float)
            con = model.addMConstr(A, zvar, GRB.LESS_EQUAL, rhs, name=f"hour_support_{cut_batch}")
            model.update()
            support_records.append((np.asarray(hours, dtype=np.int32), np.asarray(intercepts, dtype=float), A_x, con))
            all_cuts.extend(accepted)
            cut_batch += 1
            _emit(
                progress_callback,
                "support_cuts_added",
                reason=reason,
                newly_added_support_cuts=len(accepted),
                total_support_cuts=len(all_cuts),
                represented_selector_rows=int(sum(len(c.pos_rows) + len(c.neg_rows) for c in accepted)),
                explicit_security_slack_variables=0,
                model_rows=int(model.NumConstrs),
                model_columns=int(model.NumVars),
                model_nonzeros=int(model.NumNZs),
            )
            return len(accepted)

        # Mandatory tight cut at x=0 for each variable hour.  This replaces hundreds
        # of thousands of baseline row slacks by at most one row per hour.
        add_support_cuts(_initial_zero_support_cuts(case, geometry), reason="zero_action_hourly_support")

        if checkpoint is not None:
            prior_cuts = load_support_cut_seed(checkpoint, geometry=geometry)
            if prior_cuts:
                added = add_support_cuts(prior_cuts, reason="same_method_restart_support_pool")
                _emit(progress_callback, "support_cuts_reused", source="same_method_checkpoint", cuts_added=added)
        # Cross-rule legacy hint (e.g. 2028 M1-LP -> M2).  Any selector subset is a
        # valid support cut, so this can improve the start without becoming trusted
        # scientific evidence.
        hinted = _cuts_from_directed_hint(geometry, initial_directed_rows)
        if hinted:
            added = add_support_cuts(hinted, reason="prior_method_directed_selector_hint")
            _emit(progress_callback, "support_cuts_reused", source="prior_method", cuts_added=added)

        if checkpoint is not None:
            _save_support_cut_pool(checkpoint, geometry, all_cuts, stage="INITIAL_MASTER", closure_round=0)

        current_stage: dict[str, object] = {"name": "BUILD", "closure_round": 0}
        peak_private = 0.0
        last_heartbeat = [0.0]

        def heartbeat_cb(gmodel, where):
            nonlocal peak_private
            execution.check_callback(gmodel)
            now = time.monotonic()
            if now - last_heartbeat[0] < 30.0:
                return
            last_heartbeat[0] = now
            mem = _memory_snapshot()
            peak_private = max(peak_private, float(mem.get("private_gb", 0.0)))
            payload: dict[str, object] = {
                "stage": current_stage["name"],
                "closure_round": current_stage["closure_round"],
                "model_rows": current_stage.get("model_rows"),
                "model_columns": current_stage.get("model_columns"),
                "model_nonzeros": current_stage.get("model_nonzeros"),
                "support_cuts": len(all_cuts),
                "active_hours": int(len(geometry.active_hours)),
                "active_x_coordinates": R,
                "support_epigraph_variables": K,
                "explicit_security_slack_variables": 0,
            }
            try:
                if where == GRB.Callback.SIMPLEX:
                    payload.update(
                        simplex_iterations=float(gmodel.cbGet(GRB.Callback.SPX_ITRCNT)),
                        simplex_objective=float(gmodel.cbGet(GRB.Callback.SPX_OBJVAL)),
                        simplex_primal_infeasibility=float(gmodel.cbGet(GRB.Callback.SPX_PRIMINF)),
                        simplex_dual_infeasibility=float(gmodel.cbGet(GRB.Callback.SPX_DUALINF)),
                    )
            except Exception:
                pass
            _emit(progress_callback, "solver_heartbeat", **payload)

        stage1_runtime_total = 0.0
        stage2_runtime_total = 0.0
        closure_log: list[dict[str, object]] = []
        stage1_iterations: list[dict[str, object]] = []
        stage2_iterations: list[dict[str, object]] = []
        support_cert_tol = annual_support_face_numeric_tolerance(profile)
        cut_add_tol = max(ORACLE_CUT_ABS_TOL, 0.1 * float(profile.feasibility_tol))

        def prepare_stage(name: str, closure_round: int) -> None:
            current_stage.update(
                name=name,
                closure_round=closure_round,
                model_rows=int(model.NumConstrs),
                model_columns=int(model.NumVars),
                model_nonzeros=int(model.NumNZs),
            )
            execution.set_stage(name)
            _emit(
                progress_callback,
                "stage_start",
                stage=name,
                closure_round=closure_round,
                support_cuts=len(all_cuts),
                active_hours=int(len(geometry.active_hours)),
                active_x_coordinates=R,
                support_epigraph_variables=K,
                explicit_security_slack_variables=0,
                model_rows=int(model.NumConstrs),
                model_columns=int(model.NumVars),
                model_nonzeros=int(model.NumNZs),
            )

        recovered_stages=set()
        def recover_arithmetic(stage):
            if stage in recovered_stages:
                return False
            recovered_stages.add(stage)
            model.Params.FeasibilityTol=1e-9
            model.Params.OptimalityTol=1e-9
            model.Params.NumericFocus=3
            model.Params.Quad=1
            model.reset()
            closure_log.append({"stage":stage,"action":"SAME_MODEL_NUMERIC_RECOVERY",
                                "scientific_tolerances_unchanged":True})
            return True

        # ---------------- Stage I: exact support closure ----------------
        s_star = math.nan
        stage1_x_coordinate: np.ndarray | None = None
        final_stage1_dual: dict[str, object] = {}
        stage1_round = 0
        while True:
            if stage1_round > maximum_closure_rounds:
                raise SolverFailure("Exact support-function Stage-I closure exceeded maximum_closure_rounds.")
            model.setObjective(float(geometry.constant_s_mw_h_row) + theta.sum(), GRB.MINIMIZE)
            model.update()
            prepare_stage("STAGE_I", stage1_round)
            started = time.perf_counter()
            model.optimize(heartbeat_cb)
            execution.check_status(model)
            runtime = time.perf_counter() - started
            stage1_runtime_total += runtime
            if model.Status != GRB.OPTIMAL:
                raise SolverFailure(f"Stage I requires OPTIMAL support master; Gurobi status={model.Status}.")

            z_value = np.asarray(zvar.X, dtype=float)
            x_value = z_value[:R]
            theta_value = z_value[R:]
            phi, oracle_cuts = _evaluate_support(case, geometry, x_value)
            master_value = float(geometry.constant_s_mw_h_row + theta_value.sum())
            solver_master_value = float(model.ObjVal)
            if abs(master_value - solver_master_value) > max(
                support_cert_tol, experiment.stage1_tolerance(max(abs(master_value), abs(solver_master_value)))
            ):
                raise SolverFailure("Stage-I support-master objective reconstruction disagrees with Gurobi ObjVal.")
            true_value = float(geometry.constant_s_mw_h_row + phi.sum())
            hour_gap = phi - theta_value
            positive_gap = np.maximum(hour_gap, 0.0)
            total_gap = max(0.0, true_value - master_value)
            max_hour_gap = float(np.max(positive_gap, initial=0.0))
            final_stage1_dual = _support_master_dual_bound(
                c_x=np.zeros(R), upper_x=geometry.xbar_coordinate, budget_matrix=budget_matrix,
                allowance=experiment.allowance.values, budget_pi=np.asarray(budget_con.Pi),
                support_records=support_records, constant_s=geometry.constant_s_mw_h_row,
                stage1_cap_rhs=None, stage1_cap_pi=None,
            )
            mem = _memory_snapshot(); peak_private = max(peak_private, float(mem.get("private_gb", 0.0)))
            info = {
                "closure_round": stage1_round,
                "master_lower_objective": master_value,
                "solver_master_objective": solver_master_value,
                "replayed_support_objective": true_value,
                "support_gap": total_gap,
                "max_hour_support_gap": max_hour_gap,
                "runtime_s": runtime,
                "simplex_iterations": float(model.IterCount),
                "support_cuts": len(all_cuts),
                "model_rows": int(model.NumConstrs), "model_columns": int(model.NumVars), "model_nonzeros": int(model.NumNZs),
                "dual_certificate": final_stage1_dual, **mem,
            }
            stage1_iterations.append(info)
            if checkpoint is not None:
                atomic_write_json(
                    checkpoint / "STAGE1_LATEST.json",
                    {"status": "OPTIMAL_SUPPORT_MASTER", "full_support_closure_pending": total_gap > support_cert_tol, **info},
                )
            _emit(progress_callback, "stage_complete", stage="STAGE_I", **info)

            # A cut is needed exactly where the current theta underestimates the true
            # hourly hinge value.  If the global gap is already below the stronger
            # support certification tolerance, the reduced optimum and a feasible
            # full-epigraph point coincide to that tolerance.
            if total_gap <= support_cert_tol and max_hour_gap <= support_cert_tol:
                # Independent complete-row objective replay of the Stage-I point.
                stage1_full_x = _reconstruct_active_x(case, geometry, x_value)
                stage1_physical_s, _ = physical_objectives(case, stage1_full_x)
                if abs(stage1_physical_s - true_value) > max(
                    support_cert_tol,
                    experiment.stage1_tolerance(max(abs(stage1_physical_s), abs(true_value))),
                ):
                    raise SolverFailure(
                        f"Stage-I complete-row replay S={stage1_physical_s} disagrees with "
                        f"screened support S={true_value}."
                    )
                s_star = float(stage1_physical_s)
                stage1_x_coordinate = x_value.copy()
                break
            needed = [oracle_cuts[k] for k in range(K) if positive_gap[k] > cut_add_tol]
            if not needed and total_gap > support_cert_tol:
                if K:
                    needed = [oracle_cuts[int(np.argmax(positive_gap))]]
            added = add_support_cuts(needed, reason="stage1_support_oracle")
            closure_log.append(
                {"closure_round": stage1_round, "stage": "STAGE_I", "S_master": master_value,
                 "S_replayed": true_value, "support_gap": total_gap, "max_hour_support_gap": max_hour_gap,
                 "new_support_cuts": int(added), "total_support_cuts": int(len(all_cuts))}
            )
            if added == 0:
                if recover_arithmetic("STAGE_I"):
                    stage1_round += 1
                    continue
                raise SolverFailure(
                    f"Stage-I support closure stalled with unresolved support gap {total_gap:.3e}."
                )
            stage1_round += 1
            if checkpoint is not None:
                _save_support_cut_pool(checkpoint, geometry, all_cuts, stage="STAGE_I", closure_round=stage1_round)
                atomic_write_csv(checkpoint / "CLOSURE_LOG.csv", pd.DataFrame(closure_log))

        # ---------------- Stage II: exact lexicographic face closure ----------------
        assert stage1_x_coordinate is not None
        cap_rhs = float(s_star + experiment.stage1_tolerance(s_star))
        stage1_cap = model.addConstr(float(geometry.constant_s_mw_h_row) + theta.sum() <= cap_rhs, name="lex_stage1_cap")
        model.setObjective(e_coeff @ x, GRB.MINIMIZE)
        model.update()
        e_star = math.nan
        final_stage2_dual: dict[str, object] = {}
        stage2_round = 0
        final_x_coordinate: np.ndarray | None = None
        final_phi: np.ndarray | None = None
        final_oracle_cuts: list[SupportCut] = []
        while True:
            if stage1_round + stage2_round > maximum_closure_rounds:
                raise SolverFailure("Exact support-function Stage-II closure exceeded maximum_closure_rounds.")
            prepare_stage("STAGE_II", stage1_round + stage2_round)
            started = time.perf_counter()
            model.optimize(heartbeat_cb)
            execution.check_status(model)
            runtime = time.perf_counter() - started
            stage2_runtime_total += runtime
            if model.Status != GRB.OPTIMAL:
                raise SolverFailure(f"Stage II requires OPTIMAL support master; Gurobi status={model.Status}.")
            z_value = np.asarray(zvar.X, dtype=float)
            x_value = z_value[:R]
            theta_value = z_value[R:]
            phi, oracle_cuts = _evaluate_support(case, geometry, x_value)
            true_s = float(geometry.constant_s_mw_h_row + phi.sum())
            e_value = float(np.dot(e_coeff, x_value))
            solver_e_value = float(model.ObjVal)
            if abs(e_value - solver_e_value) > max(
                support_cert_tol, experiment.stage2_tolerance(max(abs(e_value), abs(solver_e_value)))
            ):
                raise SolverFailure("Stage-II restricted-energy objective reconstruction disagrees with Gurobi ObjVal.")
            hour_gap = phi - theta_value
            positive_gap = np.maximum(hour_gap, 0.0)
            cap_violation = max(0.0, true_s - cap_rhs)
            final_stage2_dual = _support_master_dual_bound(
                c_x=e_coeff, upper_x=geometry.xbar_coordinate, budget_matrix=budget_matrix,
                allowance=experiment.allowance.values, budget_pi=np.asarray(budget_con.Pi),
                support_records=support_records, constant_s=geometry.constant_s_mw_h_row,
                stage1_cap_rhs=cap_rhs, stage1_cap_pi=float(stage1_cap.Pi),
            )
            mem = _memory_snapshot(); peak_private = max(peak_private, float(mem.get("private_gb", 0.0)))
            info = {
                "closure_round": stage1_round + stage2_round,
                "objective": e_value,
                "solver_objective": solver_e_value,
                "true_stage1_at_candidate": true_s,
                "stage1_cap": cap_rhs,
                "true_stage1_cap_violation": cap_violation,
                "max_hour_support_gap": float(np.max(np.maximum(hour_gap, 0.0), initial=0.0)),
                "runtime_s": runtime, "simplex_iterations": float(model.IterCount), "support_cuts": len(all_cuts),
                "model_rows": int(model.NumConstrs), "model_columns": int(model.NumVars), "model_nonzeros": int(model.NumNZs),
                "dual_certificate": final_stage2_dual, **mem,
            }
            stage2_iterations.append(info)
            if checkpoint is not None:
                atomic_write_json(
                    checkpoint / "STAGE2_LATEST.json",
                    {"status": "OPTIMAL_SUPPORT_MASTER", "true_stage1_cap_closed": true_s <= cap_rhs + support_cert_tol, **info},
                )
            _emit(progress_callback, "stage_complete", stage="STAGE_II", **info)

            if true_s <= cap_rhs + support_cert_tol:
                physical_s,_=physical_objectives(case,_reconstruct_active_x(case,geometry,x_value))
                if physical_s>cap_rhs+support_cert_tol:
                    if recover_arithmetic("STAGE_II"):
                        stage2_round+=1
                        continue
                    raise SolverFailure("Stage-II complete physical replay violates the unchanged Stage-I face.")
                e_star = e_value
                final_x_coordinate = x_value
                final_phi = phi
                final_oracle_cuts = oracle_cuts
                break
            needed = [oracle_cuts[k] for k in range(K) if positive_gap[k] > cut_add_tol]
            if not needed and cap_violation > support_cert_tol:
                if K:
                    needed = [oracle_cuts[int(np.argmax(positive_gap))]]
            added = add_support_cuts(needed, reason="stage2_true_stage1_cap_oracle")
            closure_log.append(
                {"closure_round": stage1_round + stage2_round, "stage": "STAGE_II", "E": e_value,
                 "true_S": true_s, "stage1_cap": cap_rhs, "cap_violation": cap_violation,
                 "new_support_cuts": int(added), "total_support_cuts": int(len(all_cuts))}
            )
            if added == 0:
                if recover_arithmetic("STAGE_II"):
                    stage2_round += 1
                    continue
                raise SolverFailure(
                    f"Stage-II support closure stalled with true Stage-I cap violation {cap_violation:.3e}."
                )
            stage2_round += 1
            if checkpoint is not None:
                _save_support_cut_pool(checkpoint, geometry, all_cuts, stage="STAGE_II", closure_round=stage1_round + stage2_round)
                atomic_write_csv(checkpoint / "CLOSURE_LOG.csv", pd.DataFrame(closure_log))

        assert final_x_coordinate is not None and final_phi is not None
        x_full = _reconstruct_active_x(case, geometry, final_x_coordinate)
        S, E = physical_objectives(case, x_full)
        screened_S = float(geometry.constant_s_mw_h_row + final_phi.sum())
        screen_replay_tol = max(support_cert_tol, experiment.stage1_tolerance(max(abs(S), abs(screened_S))))
        if abs(S - screened_S) > screen_replay_tol:
            raise SolverFailure(
                f"Complete physical replay S={S} disagrees with exact screened support S={screened_S}; "
                "action-box screening is not certified."
            )
        if S > cap_rhs + support_cert_tol:
            raise SolverFailure(f"Replayed Stage-II S={S} exceeds Stage-I cap {cap_rhs}.")
        if abs(E - e_star) > max(support_cert_tol, experiment.stage2_tolerance(e_star)):
            raise SolverFailure("Stage-II replay E does not match the support-master objective.")

        budget_duals: dict[str, float] = {}
        if experiment.rule == "M2":
            try:
                pi = np.asarray(budget_con.Pi, dtype=float).reshape(-1)
                if pi.shape != (case.n_participants,):
                    raise ValueError("budget dual dimension mismatch")
                mu = -pi
                if np.min(mu, initial=0.0) < -10.0 * profile.feasibility_tol:
                    raise SolverFailure(f"M2 budget dual sign failure: minimum mu={mu.min():.3e}.")
                budget_duals = {pid: max(0.0, float(mu[n])) for n, pid in enumerate(case.participant_ids)}
            except Exception as exc:
                raise SolverFailure(f"M2 budget dual extraction failed: {exc}") from exc

        # Add the final Stage-II oracle selector to the row registry even when the
        # cap was already satisfied and no additional solver cut was necessary.
        registry_cuts = list(all_cuts)
        registry_cuts.extend(final_oracle_cuts)
        registry_pos, registry_neg = _rows_from_cuts(geometry, registry_cuts)
        unique_directed = int(sum(len(registry_pos[h]) + len(registry_neg[h]) for h in range(len(geometry.active_hours))))
        unique_physical_rows = int(sum(len(registry_pos[h] | registry_neg[h]) for h in range(len(geometry.active_hours))))

        model_rows = int(model.NumConstrs)
        model_columns = int(model.NumVars)
        model_nonzeros = int(model.NumNZs)
        total_rounds = int(stage1_round + stage2_round)
        common_meta = {
            **audit,
            "engine": ENGINE_VERSION,
            "geometry_hash": geometry.geometry_hash,
            "support_formulation": "HOURLY_HINGE_SUPPORT_FUNCTION_OUTER_APPROXIMATION",
            "support_cuts": int(len(all_cuts)),
            "support_selector_unique_directed_rows": unique_directed,
            "explicit_security_slack_variables": 0,
            "closure_rounds": total_rounds,
            "model_rows": model_rows,
            "model_columns": model_columns,
            "model_nonzeros": model_nonzeros,
            "peak_private_gb": peak_private if os.name == "nt" else None,
            "face_numeric_contract": ANNUAL_SUPPORT_FACE_CONTRACT,
            "support_certification_tolerance": support_cert_tol,
        }
        stage1 = _stage_certificate("I", s_star, stage1_runtime_total, {**common_meta, "iterations": stage1_iterations, "dual_certificate": final_stage1_dual})
        stage2 = _stage_certificate("II", e_star, stage2_runtime_total, {**common_meta, "iterations": stage2_iterations, "dual_certificate": final_stage2_dual})

        temp = PolicySolution(
            experiment.case_id, experiment.hash, case.physical_case_id, case.physical_case_hash,
            experiment.rule, experiment.allowance.hash, x_full, None, q_arr, stage1, stage2, None,
            S, E, budget_duals, total_rounds, unique_physical_rows, {},
        )
        replay = replay_solution(case, experiment, temp)
        if not replay.passed:
            raise SolverFailure(f"Final projected annual solution failed independent replay: {replay.details}.")

        replay_summary = {
            "passed": True,
            "max_balance_residual_mw": replay.max_balance_residual_mw,
            "max_budget_violation_h": replay.max_budget_violation_h,
            "max_x_bound_violation_mw": replay.max_x_bound_violation_mw,
            "max_row_violation_mw": replay.max_row_violation_mw,
            "engine": ENGINE_VERSION,
            "geometry_hash": geometry.geometry_hash,
            "active_hours": int(len(geometry.active_hours)),
            "variable_hours": K,
            "candidate_action_coordinates": geometry.candidate_action_coordinates,
            "retained_action_coordinates": R,
            "potential_directed_rows": geometry.potential_directed_rows,
            "support_cuts": int(len(all_cuts)),
            "support_selector_unique_directed_rows": unique_directed,
            "final_closed_directed_rows": unique_directed,
            "model_rows": model_rows,
            "model_columns": model_columns,
            "model_nonzeros": model_nonzeros,
            "explicit_security_slack_variables": 0,
            "support_epigraph_variables": K,
            "peak_private_gb": peak_private if os.name == "nt" else None,
        }
        final = PolicySolution(
            temp.case_id, temp.experiment_hash, temp.physical_case_id, temp.physical_case_hash,
            temp.rule, temp.allowance_hash, temp.x_mw, temp.u, temp.q, temp.stage1, temp.stage2,
            temp.stage3, temp.s_value_mw_h_row, temp.e_value_mwh, temp.m2_budget_duals,
            temp.closure_rounds, temp.active_row_count, replay_summary,
        )

        if checkpoint is not None:
            _save_support_cut_pool(checkpoint, geometry, all_cuts, stage="COMPLETE", closure_round=total_rounds)
            _save_closed_rows(
                checkpoint, geometry=geometry, n_rows=case.n_flat_rows,
                included_pos=registry_pos, included_neg=registry_neg,
                stage="COMPLETE_SUPPORT_REGISTRY", closure_round=total_rounds,
            )
            atomic_write_csv(checkpoint / "CLOSURE_LOG.csv", pd.DataFrame(closure_log))
            atomic_write_json(
                checkpoint / "ENGINE_STATUS.json",
                {
                    "schema": "TPWRS_SUPPORT_ANNUAL_ENGINE_STATUS_V2",
                    "status": "PASS",
                    "engine": ENGINE_VERSION,
                    "experiment_hash": experiment.hash,
                    "geometry_hash": geometry.geometry_hash,
                    "S": S,
                    "E": E,
                    "stage1_runtime_s": stage1_runtime_total,
                    "stage2_runtime_s": stage2_runtime_total,
                    "closure_rounds": total_rounds,
                    "support_cuts": int(len(all_cuts)),
                    "explicit_security_slack_variables": 0,
                    "model_rows": model_rows,
                    "model_columns": model_columns,
                    "model_nonzeros": model_nonzeros,
                    "peak_private_gb": peak_private if os.name == "nt" else None,
                    "full_replay_passed": True,
                },
            )

        _emit(
            progress_callback,
            "solve_complete",
            S=S, E=E,
            stage1_runtime_s=stage1_runtime_total,
            stage2_runtime_s=stage2_runtime_total,
            closure_rounds=total_rounds,
            support_cuts=len(all_cuts),
            active_x_coordinates=R,
            support_epigraph_variables=K,
            explicit_security_slack_variables=0,
            model_rows=model_rows,
            model_columns=model_columns,
            model_nonzeros=model_nonzeros,
            peak_private_gb=peak_private,
        )
        return final

    finally:
        if execution is not None:
            execution.close()
        model.dispose()


def load_compatible_closed_row_seed(case: PhysicalCase, experiment: ExperimentCase, directory: str | Path) -> tuple[list[set[int]], list[set[int]]] | None:
    """Load a prior rule's valid selector-row registry only when geometry matches."""
    geometry = _screen_geometry(case, experiment)
    return load_closed_row_seed(directory, geometry_hash=geometry.geometry_hash, n_rows=case.n_flat_rows, n_hours=len(geometry.active_hours))
