from __future__ import annotations


# ============================================================================
# tpwrs_v3.dutch.component_data (consolidated; see SOURCE_MAP.json)
# ============================================================================
from dataclasses import dataclass
import numpy as np
from tpwrs.storage import ContractError

@dataclass(frozen=True)
class ComponentPanel:
    province_ids: tuple[str,...]
    demand_mw: np.ndarray
    pv_mw: np.ndarray
    wind_onshore_mw: np.ndarray
    wind_offshore_mw: np.ndarray
    def __post_init__(self):
        arrays=[np.asarray(self.demand_mw,float),np.asarray(self.pv_mw,float),np.asarray(self.wind_onshore_mw,float),np.asarray(self.wind_offshore_mw,float)]
        if any(a.shape!=arrays[0].shape for a in arrays) or arrays[0].ndim!=2 or arrays[0].shape[1]!=len(self.province_ids):
            raise ContractError("Component trajectories must have identical (time, province) shape.")
        if any(np.any(a<0) or np.any(~np.isfinite(a)) for a in arrays): raise ContractError("Component trajectories must be finite and nonnegative.")


# ============================================================================
# tpwrs_v3.dutch.participant_effects (consolidated; see SOURCE_MAP.json)
# ============================================================================
import numpy as np
from tpwrs.storage import ContractError
from tpwrs.contracts import NetworkState


def participant_flow_effects(network_states: tuple[NetworkState,...], A: np.ndarray) -> tuple[tuple[str,...],np.ndarray]:
    A=np.asarray(A,float)
    if np.max(np.abs(A.sum(axis=0)))>1e-9: raise ContractError("Participant action matrix must be balanced before sensitivity calculation.")
    rows=[]; effects=[]
    for state in network_states:
        G=state.H@A
        for r,row_id in enumerate(state.row_ids): rows.append(f"{state.state_id}::{row_id}"); effects.append(G[r])
    return tuple(rows),np.asarray(effects,float).T


# ============================================================================
# tpwrs_v3.dutch.case_assembler (consolidated; see SOURCE_MAP.json)
# ============================================================================

from typing import Mapping, Sequence
import numpy as np

from tpwrs.contracts import NetworkState, PhysicalCase
from tpwrs.contracts import TimeGrid


def assemble_physical_case(*, time_grid: TimeGrid, participant_ids: Sequence[str], bus_ids: Sequence[str], capacities_mw: np.ndarray, requests_mw: np.ndarray, contract_mask: np.ndarray, eligibility_mask: np.ndarray, p_background_mw: np.ndarray, A: np.ndarray, network_states: Sequence[NetworkState], participant_directions: Sequence[str]|None=None, participant_set_id: str, contingency_set_id: str, balancing_key_hash: str, source_hashes: Mapping[str,str], declared_case_id: str|None=None) -> PhysicalCase:
    return PhysicalCase(time_grid,tuple(map(str,participant_ids)),tuple(map(str,bus_ids)),capacities_mw,requests_mw,contract_mask,eligibility_mask,p_background_mw,A,tuple(network_states),None if participant_directions is None else tuple(map(str,participant_directions)),participant_set_id,contingency_set_id,balancing_key_hash,dict(source_hashes),declared_case_id)


# ============================================================================
# tpwrs_v3.dutch.state_builder (consolidated; see SOURCE_MAP.json)
# ============================================================================
from tpwrs.compression import add_active_mvar, add_mconstr_active, set_mobjective_active, active_hstack, active_vstack, active_np_concatenate, active_np_hstack, active_np_vstack
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from tpwrs.storage import ContractError, SolverFailure
from tpwrs.storage import sha256_array
from tpwrs.contracts import NetworkState

def validate_allocation_matrix(W: np.ndarray, *, name: str, tol: float=1e-10) -> np.ndarray:
    a = np.asarray(W, float)
    if a.ndim != 2 or np.any(a < -tol) or np.any(~np.isfinite(a)):
        raise ContractError(f'{name}: allocation matrix must be finite and nonnegative.')
    sums = a.sum(axis=0)
    if np.max(np.abs(sums - 1.0)) > tol:
        raise ContractError(f'{name}: each source column must conserve its provincial/source total.')
    return a

def _component(W: np.ndarray, values: np.ndarray, T: int, name: str) -> np.ndarray:
    W = validate_allocation_matrix(W, name=name)
    v = np.asarray(values, float)
    if v.ndim != 2 or v.shape[0] != T or v.shape[1] != W.shape[1]:
        raise ContractError(f'{name}: trajectory shape mismatch.')
    return v @ W.T

def build_nondispatchable_state(demand_mw: np.ndarray, pv_mw: np.ndarray, wind_on_mw: np.ndarray, wind_off_mw: np.ndarray, W_D: np.ndarray, W_PV: np.ndarray, W_ON: np.ndarray, W_OFF: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    T = np.asarray(demand_mw).shape[0]
    D = _component(W_D, demand_mw, T, 'W_D')
    PV = _component(W_PV, pv_mw, T, 'W_PV')
    ON = _component(W_ON, wind_on_mw, T, 'W_ON')
    OFF = _component(W_OFF, wind_off_mw, T, 'W_OFF')
    p = -D + PV + ON + OFF
    report = {'demand_conservation_max_mw': float(np.max(np.abs(D.sum(axis=1) - np.asarray(demand_mw).sum(axis=1)))), 'pv_conservation_max_mw': float(np.max(np.abs(PV.sum(axis=1) - np.asarray(pv_mw).sum(axis=1)))), 'onshore_conservation_max_mw': float(np.max(np.abs(ON.sum(axis=1) - np.asarray(wind_on_mw).sum(axis=1)))), 'offshore_conservation_max_mw': float(np.max(np.abs(OFF.sum(axis=1) - np.asarray(wind_off_mw).sum(axis=1))))}
    return (p, report)

def _key(key: np.ndarray, B: int, name: str) -> np.ndarray:
    k = np.asarray(key, float).reshape(-1)
    if k.shape != (B,) or np.any(k < 0) or k.sum() <= 0:
        raise ContractError(f'{name}: invalid balancing key.')
    return k / k.sum()

def balance_with_frozen_keys(p_nd_mw: np.ndarray, upward_key: np.ndarray, downward_key: np.ndarray, *, upward_capacity_mw: np.ndarray | None=None, downward_capacity_mw: np.ndarray | None=None, tol: float=1e-08) -> tuple[np.ndarray, dict[str, float]]:
    p = np.asarray(p_nd_mw, float).copy()
    T, B = p.shape
    up = _key(upward_key, B, 'upward_key')
    down = _key(downward_key, B, 'downward_key')
    upcap = np.full(B, np.inf) if upward_capacity_mw is None else np.asarray(upward_capacity_mw, float)
    dncap = np.full(B, np.inf) if downward_capacity_mw is None else np.asarray(downward_capacity_mw, float)
    for t in range(T):
        residual = float(p[t].sum())
        if residual < -tol:
            need = -residual
            raw = need * up
            if np.any(raw > upcap + tol):
                raise ContractError('Upward balancing capacity is insufficient for frozen key.')
            p[t] += raw
        elif residual > tol:
            need = residual
            raw = need * down
            if np.any(raw > dncap + tol):
                raise ContractError('Downward balancing capacity is insufficient for frozen key.')
            p[t] -= raw
    maxbal = float(np.max(np.abs(p.sum(axis=1))))
    if maxbal > tol:
        raise ContractError(f'Balanced state residual {maxbal:g} MW exceeds tolerance.')
    return (p, {'max_balance_residual_mw': maxbal, 'upward_key_sum': float(up.sum()), 'downward_key_sum': float(down.sum())})

def security_adjust_background(p_initial_mw: np.ndarray, action_matrix: np.ndarray, action_lower_mw: np.ndarray, action_upper_mw: np.ndarray, network_states: tuple[NetworkState, ...], *, epsilon_s: float=1e-08) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build a deterministic bounded security-adjusted public-data background state.

    This is a case-construction operation, not a remedial variable in the parent
    M1/M2/M3 optimization.  Stage 1 minimizes aggregate thermal slack per hour;
    Stage 2 minimizes L1 adjustment at the Stage-1 value.
    """
    p0 = np.asarray(p_initial_mw, float)
    R = np.asarray(action_matrix, float)
    T, B = p0.shape
    if R.shape[0] != B or np.max(np.abs(R.sum(axis=0))) > 1e-09:
        raise ContractError('Background adjustment actions must be balanced.')
    lo = np.asarray(action_lower_mw, float)
    hi = np.asarray(action_upper_mw, float)
    K = R.shape[1]
    if lo.shape != (K,) or hi.shape != (K,) or np.any(lo > hi):
        raise ContractError('Background action bounds invalid.')
    H = active_np_vstack([s.H for s in network_states])
    h = active_np_concatenate([s.h_mw for s in network_states])
    rating = active_np_concatenate([s.ratings_mw for s in network_states])
    GR = H @ R
    actions = np.zeros((T, K))
    adjusted = np.zeros_like(p0)
    total_s = 0.0
    for t in range(T):
        base = H @ p0[t] + h
        M = len(rating)
        n = K + 2 * M
        c = np.r_[np.zeros(K), np.ones(2 * M)]
        A = []
        b = []
        for r in range(M):
            row = np.zeros(n)
            row[:K] = GR[r]
            row[K + r] = -1
            A.append(row)
            b.append(rating[r] - base[r])
            row = np.zeros(n)
            row[:K] = -GR[r]
            row[K + M + r] = -1
            A.append(row)
            b.append(rating[r] + base[r])
        bounds = list(zip(lo, hi)) + [(0, None)] * (2 * M)
        s1 = linprog(c, A_ub=np.asarray(A), b_ub=np.asarray(b), bounds=bounds, method='highs')
        if not s1.success:
            raise SolverFailure(f'Background security adjustment Stage I failed at t={t}: {s1.message}')
        S = float(s1.fun)
        total_s += S
        n2 = K + 2 * K + 2 * M
        c2 = np.r_[np.zeros(K), np.ones(2 * K), np.zeros(2 * M)]
        A2 = []
        b2 = []
        for r in range(M):
            row = np.zeros(n2)
            row[:K] = GR[r]
            row[K + 2 * K + r] = -1
            A2.append(row)
            b2.append(rating[r] - base[r])
            row = np.zeros(n2)
            row[:K] = -GR[r]
            row[K + 2 * K + M + r] = -1
            A2.append(row)
            b2.append(rating[r] + base[r])
        for k in range(K):
            row = np.zeros(n2)
            row[k] = 1
            row[K + k] = -1
            A2.append(row)
            b2.append(0)
            row = np.zeros(n2)
            row[k] = -1
            row[K + K + k] = -1
            A2.append(row)
            b2.append(0)
        row = np.zeros(n2)
        row[K + 2 * K:] = 1
        A2.append(row)
        b2.append(S + epsilon_s)
        bounds2 = list(zip(lo, hi)) + [(0, None)] * (2 * K + 2 * M)
        s2 = linprog(c2, A_ub=np.asarray(A2), b_ub=np.asarray(b2), bounds=bounds2, method='highs')
        if not s2.success:
            raise SolverFailure(f'Background security adjustment Stage II failed at t={t}: {s2.message}')
        actions[t] = s2.x[:K]
        adjusted[t] = p0[t] + R @ actions[t]
    return (adjusted, actions, {'aggregate_background_residual_violation_mw': float(total_s), 'max_balance_residual_mw': float(np.max(np.abs(adjusted.sum(axis=1))))})

def build_component_state_from_provincial_panel(panel: 'object', weights: 'object', bus_ids: tuple[int, ...]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Construct pND from four separately conserved provincial components."""
    import pandas as pd
    if not isinstance(panel, pd.DataFrame) or not isinstance(weights, pd.DataFrame):
        raise ContractError('Component panel and allocation weights must be pandas frames.')
    required = {'timestamp_utc', 'province', 'prov_load_MW', 'prov_solar_MW', 'prov_wind_on_MW', 'prov_wind_off_MW'}
    if not required.issubset(panel.columns):
        raise ContractError(f'Component panel missing {sorted(required - set(panel.columns))}')
    w_required = {'bus', 'area_id', 'load', 'solar', 'onshore_wind', 'offshore_wind'}
    if not w_required.issubset(weights.columns):
        raise ContractError(f'Allocation table missing {sorted(w_required - set(weights.columns))}')
    work = panel.copy()
    work['timestamp_utc'] = pd.to_datetime(work.timestamp_utc, utc=True)
    if work.duplicated(['timestamp_utc', 'province']).any():
        raise ContractError('Duplicate timestamp/province component rows.')
    areas = tuple(sorted(map(str, work.province.unique())))
    expected = set(map(str, weights.area_id.unique()))
    if set(areas) != expected:
        raise ContractError('Province IDs in panel and allocation table differ.')
    timestamps = pd.DatetimeIndex(sorted(work.timestamp_utc.unique()))
    if len(work) != len(timestamps) * len(areas):
        raise ContractError('Component panel is not an exact timestamp/province product.')
    aligned = weights.set_index('bus').reindex(list(bus_ids))
    if aligned.isna().all(axis=1).any():
        raise ContractError('Allocation table omits retained bus IDs.')
    source_columns = {'demand': 'prov_load_MW', 'pv': 'prov_solar_MW', 'wind_on': 'prov_wind_on_MW', 'wind_off': 'prov_wind_off_MW'}
    weight_columns = {'demand': 'load', 'pv': 'solar', 'wind_on': 'onshore_wind', 'wind_off': 'offshore_wind'}
    nodal: dict[str, np.ndarray] = {}
    audits = {}
    for key, source in source_columns.items():
        values = work.pivot(index='timestamp_utc', columns='province', values=source).reindex(index=timestamps, columns=areas)
        if values.isna().any().any() or np.any(values.to_numpy(float) < -1e-09):
            raise ContractError(f'{key}: missing, nonfinite, or negative component values.')
        W = np.column_stack([aligned[weight_columns[key]].where(aligned.area_id.astype(str).eq(area), 0.0).to_numpy(float) for area in areas])
        W = validate_allocation_matrix(W, name=f'W_{key}')
        out = values.to_numpy(float) @ W.T
        nodal[key] = out
        nodal[{'demand': 'W_D', 'pv': 'W_PV', 'wind_on': 'W_ON', 'wind_off': 'W_OFF'}[key]] = W
        audits[key] = {'allocation_column_sum_max_abs_error': float(np.max(np.abs(W.sum(axis=0) - 1.0))), 'nodal_conservation_max_abs_mw': float(np.max(np.abs(out.sum(axis=1) - values.sum(axis=1).to_numpy(float)))), 'weight_matrix_hash': sha256_array(W)}
    pnd = -nodal['demand'] + nodal['pv'] + nodal['wind_on'] + nodal['wind_off']
    audit = {'schema': 'TPWRS_COMPONENT_ALLOCATION_AUDIT_V1', 'status': 'PASS', 'timestamps': len(timestamps), 'provinces': len(areas), 'buses': len(bus_ids), 'components': audits, 'scenario_not_forecast': True, 'timestamp_start': timestamps[0].isoformat(), 'timestamp_end': timestamps[-1].isoformat()}
    nodal['timestamps_utc'] = np.asarray([x.isoformat() for x in timestamps], dtype=str)
    return (pnd, nodal, audit)

def security_adjust_background_squared_reference(p_nd_mw: np.ndarray, device_bus_matrix: np.ndarray, device_lower_mw: np.ndarray, device_upper_mw: np.ndarray, reference_dispatch_mw: np.ndarray, reference_weights: np.ndarray, network_states: tuple[NetworkState, ...], *, closure_tolerance_mw: float=1e-06, stage1_tolerance_mw: float=1e-08, progress: object | None=None) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Exact two-stage background builder with full-row replay closure.

    Feasible hours are solved directly on the zero-slack Stage-I face using
    deterministic convex QP row generation.  Only if that face is infeasible
    is the full all-row L1 slack problem built, followed by the frozen squared
    reference-distance objective on its optimum face.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        raise SolverFailure('Gurobi is required for the squared-reference background builder.') from exc
    pnd = np.asarray(p_nd_mw, float)
    C = np.asarray(device_bus_matrix, float)
    ref = np.asarray(reference_dispatch_mw, float)
    T, B = pnd.shape
    K = C.shape[1]
    if C.shape[0] != B or ref.shape != (T, K):
        raise ContractError('Background device/reference axes do not align.')
    lo = np.asarray(device_lower_mw, float)
    hi = np.asarray(device_upper_mw, float)
    if lo.ndim == 1:
        lo = np.broadcast_to(lo, (T, K))
    if hi.ndim == 1:
        hi = np.broadcast_to(hi, (T, K))
    if lo.shape != (T, K) or hi.shape != (T, K) or np.any(lo > hi):
        raise ContractError('Background device bounds are invalid.')
    w = np.asarray(reference_weights, float).reshape(-1)
    if w.shape != (K,) or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ContractError('Squared-reference weights must be positive and device aligned.')
    if np.any(ref < lo - 1e-07) or np.any(ref > hi + 1e-07):
        raise ContractError('Reference dispatch violates declared device bounds.')
    H = active_np_vstack([s.H for s in network_states])
    h = active_np_concatenate([s.h_mw for s in network_states])
    rating = active_np_concatenate([s.ratings_mw for s in network_states])
    G = H @ C
    csum = C.sum(axis=0)
    result = np.zeros((T, K))
    pbg = np.zeros_like(pnd)
    slack_total = np.zeros(T)
    raw_violation_max = np.zeros(T)
    slack_replay_residual = np.zeros(T)
    closure_rounds = np.zeros(T, dtype=int)
    selected_counts = np.zeros(T, dtype=int)
    statuses = []

    def configure(model: object) -> None:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = 0
        model.Params.NumericFocus = 2

    def replay(base: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flow = base + G @ g
        return (flow, np.maximum(np.abs(flow) - rating, 0.0))
    for t in range(T):
        base = H @ pnd[t] + h
        _, v0 = replay(base, ref[t])
        balance_rhs = -float(pnd[t].sum())
        balance_ref = float(csum @ ref[t] - balance_rhs)
        if abs(balance_ref) <= 1e-07 and float(v0.max()) <= closure_tolerance_mw:
            gstar = ref[t].copy()
            status = 'REFERENCE_SECURE'
            rounds = 0
            selected = set()
        else:
            model = gp.Model(f'background_hard_{t}')
            configure(model)
            g = add_active_mvar(model, K, lb=lo[t], ub=hi[t], name='g')
            model.addConstr(csum @ g == balance_rhs, name='balance')
            model.setObjective(0.5 * gp.quicksum((float(w[k]) * (g[k] - float(ref[t, k])) * (g[k] - float(ref[t, k])) for k in range(K))), GRB.MINIMIZE)
            selected = set(map(int, np.flatnonzero(v0 > closure_tolerance_mw)))
            added: set[int] = set()
            rounds = 0
            hard_feasible = False
            while rounds < 50:
                for r in sorted(selected - added):
                    model.addConstr(G[r] @ g <= float(rating[r] - base[r]), name=f'up_{r}')
                    model.addConstr(-G[r] @ g <= float(rating[r] + base[r]), name=f'dn_{r}')
                    added.add(r)
                model.optimize()
                rounds += 1
                if model.Status == GRB.INFEASIBLE:
                    break
                if model.Status != GRB.OPTIMAL:
                    raise SolverFailure(f'Background hard QP t={t} status={model.Status}.')
                candidate = np.asarray(g.X, float)
                _, violation = replay(base, candidate)
                new = set(map(int, np.flatnonzero(violation > closure_tolerance_mw))) - selected
                if not new:
                    gstar = candidate
                    hard_feasible = True
                    status = 'ZERO_SLACK_QP'
                    break
                selected.update(new)
            if not hard_feasible:
                model = gp.Model(f'background_slack_{t}')
                configure(model)
                g = add_active_mvar(model, K, lb=lo[t], ub=hi[t], name='g')
                sp = add_active_mvar(model, len(rating), lb=0.0, name='sp')
                sm = add_active_mvar(model, len(rating), lb=0.0, name='sm')
                model.addConstr(csum @ g == balance_rhs, name='balance')
                model.addConstr(G @ g - sp <= rating - base, name='upper')
                model.addConstr(-G @ g - sm <= rating + base, name='lower')
                model.setObjective(sp.sum() + sm.sum(), GRB.MINIMIZE)
                model.optimize()
                if model.Status != GRB.OPTIMAL:
                    raise SolverFailure(f'Background Stage I t={t} status={model.Status}.')
                S = float(model.ObjVal)
                model.addConstr(sp.sum() + sm.sum() <= S + stage1_tolerance_mw, name='stage1_face')
                model.setObjective(0.5 * gp.quicksum((float(w[k]) * (g[k] - float(ref[t, k])) * (g[k] - float(ref[t, k])) for k in range(K))), GRB.MINIMIZE)
                model.optimize()
                if model.Status != GRB.OPTIMAL:
                    raise SolverFailure(f'Background Stage II t={t} status={model.Status}.')
                gstar = np.asarray(g.X, float)
                slack_total[t] = S
                status = 'POSITIVE_MINIMUM_SLACK'
                selected = set(range(len(rating)))
        flow, violation = replay(base, gstar)
        if slack_total[t] <= stage1_tolerance_mw and float(violation.max()) > closure_tolerance_mw:
            raise SolverFailure(f'Background row closure failed at t={t}: {float(violation.max())} MW.')
        replayed_stage1 = float(violation.sum())
        replay_error = float(violation.max()) if slack_total[t] <= stage1_tolerance_mw else abs(replayed_stage1 - float(slack_total[t]))
        if replay_error > max(10 * closure_tolerance_mw, 1e-09 * max(1.0, float(slack_total[t]))):
            raise SolverFailure(f'Background Stage-I objective replay failed at t={t}: solved={slack_total[t]}, replayed={replayed_stage1}.')
        state = pnd[t] + C @ gstar
        if abs(state.sum()) > 2e-06:
            raise SolverFailure(f'Background balance replay failed at t={t}: {state.sum()} MW.')
        result[t] = gstar
        pbg[t] = state
        raw_violation_max[t] = float(violation.max())
        slack_replay_residual[t] = replay_error
        closure_rounds[t] = rounds
        selected_counts[t] = len(selected)
        statuses.append(status)
        if progress is not None:
            progress(t + 1, T, status, float(slack_total[t]), float(violation.max()))
    audit = {'schema': 'TPWRS_BACKGROUND_TWO_STAGE_AUDIT_V1', 'status': 'PASS', 'hours': T, 'devices': K, 'flat_network_rows': len(rating), 'aggregate_minimum_stage1_slack_mw': float(slack_total.sum()), 'positive_slack_hours': int(np.sum(slack_total > stage1_tolerance_mw)), 'max_balance_residual_mw': float(np.max(np.abs(pbg.sum(axis=1)))), 'max_raw_thermal_violation_mw': float(raw_violation_max.max(initial=0.0)), 'max_stage1_objective_replay_error_mw': float(slack_replay_residual.max(initial=0.0)), 'max_closure_rounds': int(closure_rounds.max(initial=0)), 'max_selected_hard_rows': int(selected_counts.max(initial=0)), 'status_counts': {key: statuses.count(key) for key in sorted(set(statuses))}, 'stage2_objective': 'ONE_HALF_WEIGHTED_SQUARED_DISTANCE_TO_PUBLIC_REFERENCE', 'row_closure_tolerance_mw': closure_tolerance_mw, 'stage1_face_tolerance_mw': stage1_tolerance_mw}
    return (pbg, result, audit)


# ============================================================================
# tpwrs_v3.dutch.validation_2018 (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.storage import ContractError


def compare_2018_reference(reference_injections: np.ndarray, rebuilt_injections: np.ndarray, reference_loading_pct: np.ndarray|None=None, rebuilt_loading_pct: np.ndarray|None=None) -> dict[str,float]:
    ref=np.asarray(reference_injections,float); new=np.asarray(rebuilt_injections,float)
    if ref.shape!=new.shape: raise ContractError("2018 injection arrays must align.")
    report={"load_generation_total_mae_mw":float(np.mean(np.abs(ref-new))),"balance_reference_max_mw":float(np.max(np.abs(ref.sum(axis=1)))),"balance_rebuilt_max_mw":float(np.max(np.abs(new.sum(axis=1))))}
    if reference_loading_pct is not None and rebuilt_loading_pct is not None:
        r=np.asarray(reference_loading_pct,float); n=np.asarray(rebuilt_loading_pct,float)
        if r.shape!=n.shape: raise ContractError("2018 loading arrays must align.")
        report.update({"branch_loading_mae_pct":float(np.mean(np.abs(r-n))),"reference_p95_loading_pct":float(np.quantile(r,0.95)),"rebuilt_p95_loading_pct":float(np.quantile(n,0.95))})
    return report
