from __future__ import annotations


# ============================================================================
# tpwrs_v3.engineering_audits (consolidated; see SOURCE_MAP.json)
# ============================================================================

from typing import Iterable

import numpy as np
import pandas as pd

from tpwrs.storage import ContractError


def mapped_dc_audit(
    frozen: pd.DataFrame,
    independent: pd.DataFrame,
    *,
    key_columns: tuple[str, ...] = ("case_id", "state_id", "row_id"),
    frozen_flow: str = "frozen_flow_mw",
    independent_flow: str = "independent_flow_mw",
    rating_col: str = "rating_mw",
    near_limit_fraction: float = 0.10,
    top_k: int = 10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare the frozen affine evaluator with a separately executed DC path.

    ``independent`` is the reference for the false-safe/false-overload labels:

    * false safe: frozen evaluator says safe while independent DC says overloaded;
    * false overload: frozen evaluator says overloaded while independent DC says safe.
    """
    if near_limit_fraction < 0:
        raise ContractError("near_limit_fraction must be nonnegative.")
    if top_k < 1:
        raise ContractError("top_k must be positive.")

    required_common = set(key_columns)
    for frame, label, flow_name in (
        (frozen, "frozen", frozen_flow),
        (independent, "independent", independent_flow),
    ):
        missing = sorted((required_common | {flow_name, rating_col}) - set(frame.columns))
        if missing:
            raise ContractError(f"{label} DC audit missing columns {missing}.")
        if frame.duplicated(list(key_columns)).any():
            raise ContractError(f"{label} DC audit contains duplicate case/state/row keys.")

    left = frozen.copy()
    right = independent.copy()
    # Carry the rating from the frozen registry only.  If the independent path
    # also reports a rating, verify it after the merge rather than allowing
    # pandas suffix ambiguity.
    if rating_col in right.columns:
        right = right.rename(columns={rating_col: f"{rating_col}_independent"})

    merged = left.merge(
        right,
        on=list(key_columns),
        validate="one_to_one",
        suffixes=("_f", "_i"),
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ContractError(
            "Independent-DC row/state mapping is not one-to-one and complete."
        )

    f = pd.to_numeric(merged[frozen_flow], errors="coerce").to_numpy(float)
    g = pd.to_numeric(merged[independent_flow], errors="coerce").to_numpy(float)
    rating = pd.to_numeric(merged[rating_col], errors="coerce").to_numpy(float)

    if (
        not np.all(np.isfinite(f))
        or not np.all(np.isfinite(g))
        or not np.all(np.isfinite(rating))
        or np.any(rating <= 0.0)
    ):
        raise ContractError("DC audit flows/ratings must be finite and ratings positive.")

    independent_rating_col = f"{rating_col}_independent"
    if independent_rating_col in merged:
        independent_rating = pd.to_numeric(
            merged[independent_rating_col], errors="coerce"
        ).to_numpy(float)
        tolerance = np.maximum(1e-8, 1e-10 * rating)
        if (
            not np.all(np.isfinite(independent_rating))
            or np.max(np.abs(independent_rating - rating), initial=0.0)
            > float(np.max(tolerance, initial=1e-8))
        ):
            raise ContractError("Frozen and independent DC ratings do not align.")

    error = g - f
    merged["abs_error_mw"] = np.abs(error)
    merged["normalized_error"] = np.abs(error) / rating
    merged["sign_agreement"] = (
        (np.sign(f) == np.sign(g))
        | ((np.abs(f) < 1e-8) & (np.abs(g) < 1e-8))
    )

    frozen_overloaded = np.abs(f) > rating
    independent_overloaded = np.abs(g) > rating

    # Correct physical interpretation.
    merged["false_safe"] = (~frozen_overloaded) & independent_overloaded
    merged["false_overload"] = frozen_overloaded & (~independent_overloaded)

    near_limit = np.abs(np.abs(f) / rating - 1.0) <= near_limit_fraction
    merged["near_limit"] = near_limit

    overlaps: list[float] = []
    group_columns = list(key_columns[:-1])
    for _, part in merged.groupby(group_columns, sort=False, dropna=False):
        k = min(int(top_k), len(part))
        if k == 0:
            continue
        frozen_values = np.abs(part[frozen_flow].to_numpy(float))
        independent_values = np.abs(part[independent_flow].to_numpy(float))
        frozen_rows = set(np.argpartition(frozen_values, -k)[-k:].tolist())
        independent_rows = set(np.argpartition(independent_values, -k)[-k:].tolist())
        overlaps.append(len(frozen_rows & independent_rows) / float(k))

    summary = {
        "row_count": float(len(merged)),
        "mae_mw": float(np.mean(np.abs(error))) if len(error) else 0.0,
        "rmse_mw": float(np.sqrt(np.mean(error * error))) if len(error) else 0.0,
        "max_abs_error_mw": float(np.max(np.abs(error), initial=0.0)),
        "max_normalized_error": float(
            np.max(np.abs(error) / rating, initial=0.0)
        ),
        "sign_agreement_share": (
            float(np.mean(merged["sign_agreement"])) if len(merged) else 1.0
        ),
        "false_safe_count": float(np.sum(merged["false_safe"])),
        "false_overload_count": float(np.sum(merged["false_overload"])),
        "near_limit_row_count": float(np.sum(near_limit)),
        "near_limit_false_safe_rate": (
            float(np.mean(merged.loc[near_limit, "false_safe"]))
            if np.any(near_limit)
            else 0.0
        ),
        "near_limit_false_overload_rate": (
            float(np.mean(merged.loc[near_limit, "false_overload"]))
            if np.any(near_limit)
            else 0.0
        ),
        "topk_overlap_mean": float(np.mean(overlaps)) if overlaps else 1.0,
    }
    return merged, summary


def _numeric_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> pd.Series | None:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return None


def selected_ac_summary(
    case_rows: pd.DataFrame,
    branch_rows: pd.DataFrame,
    voltage_rows: pd.DataFrame,
    loading_rows: pd.DataFrame | None = None,
    *,
    narrow_voltage_limits: tuple[float, float] = (0.95, 1.05),
    restoration_voltage_limits: tuple[float, float] = (0.90, 1.10),
) -> dict[str, float | int | bool | str]:
    """Compile selected-state AC evidence without equating convergence and feasibility.

    Adverse nonlinear outcomes are reported rather than converted into a failed
    software status.  The claim is selected-state consistency/stress only, not
    annual AC-OPF feasibility or comprehensive N-1 security.
    """
    if "converged" not in case_rows.columns:
        raise ContractError("Selected AC case table lacks a converged column.")

    ac_col = next(
        (c for c in ("ac_p_mw", "ac_p_from_mw") if c in branch_rows.columns),
        None,
    )
    dc_col = next(
        (c for c in ("dc_p_mw", "dc_p_from_mw") if c in branch_rows.columns),
        None,
    )
    if ac_col is None or dc_col is None:
        raise ContractError("Selected AC branch table lacks AC/DC active-power columns.")
    if "vm_pu" not in voltage_rows.columns:
        raise ContractError("Selected AC voltage table lacks vm_pu.")

    converged = case_rows["converged"].astype(bool)
    valid = branch_rows[[ac_col, dc_col]].apply(pd.to_numeric, errors="coerce").dropna()
    error = valid[ac_col] - valid[dc_col]

    sign_mask = (valid[ac_col].abs() > 1e-8) | (valid[dc_col].abs() > 1e-8)
    sign_agreement = (
        float(
            (
                np.sign(valid.loc[sign_mask, ac_col])
                == np.sign(valid.loc[sign_mask, dc_col])
            ).mean()
        )
        if sign_mask.any()
        else 1.0
    )
    spearman = (
        float(valid[ac_col].corr(valid[dc_col], method="spearman"))
        if len(valid) >= 2
        else float("nan")
    )

    voltage = pd.to_numeric(voltage_rows["vm_pu"], errors="coerce").dropna()
    if voltage.empty:
        raise ContractError("Selected AC voltage table has no finite voltage values.")
    narrow_low, narrow_high = map(float, narrow_voltage_limits)
    restore_low, restore_high = map(float, restoration_voltage_limits)
    if not (0.0 < restore_low <= narrow_low < narrow_high <= restore_high):
        raise ContractError("Voltage reporting/restoration limits are inconsistent.")

    loading_source = loading_rows if loading_rows is not None else case_rows
    loading = _numeric_column(
        loading_source,
        (
            "loading_percent",
            "loading_pct",
            "maximum_branch_loading_percent",
            "max_loading_percent",
        ),
    )
    maximum_loading = (
        float(loading.max(skipna=True))
        if loading is not None and loading.notna().any()
        else float("nan")
    )

    thermal_count = _numeric_column(
        case_rows,
        ("thermal_violation_count", "thermal_violations"),
    )
    reactive_count = _numeric_column(
        case_rows,
        ("reactive_limit_violation_count", "reactive_limit_violations"),
    )
    voltage_count = _numeric_column(
        case_rows,
        ("voltage_violation_count", "voltage_violations"),
    )

    outside_narrow = (voltage < narrow_low) | (voltage > narrow_high)
    outside_restoration = (voltage < restore_low) | (voltage > restore_high)

    feasibility_columns_available = all(
        series is not None
        for series in (thermal_count, reactive_count)
    )
    ac_feasible_cases = None
    if feasibility_columns_available:
        min_case_voltage = _numeric_column(case_rows, ("minimum_voltage_pu", "min_vm_pu"))
        max_case_voltage = _numeric_column(case_rows, ("maximum_voltage_pu", "max_vm_pu"))
        if min_case_voltage is not None and max_case_voltage is not None:
            feasible = (
                converged
                & min_case_voltage.ge(restore_low)
                & max_case_voltage.le(restore_high)
                & thermal_count.fillna(np.inf).le(0)
                & reactive_count.fillna(np.inf).le(0)
            )
            ac_feasible_cases = int(feasible.sum())
        else:
            feasibility_columns_available = False

    return {
        "attempted_cases": int(len(case_rows)),
        "converged_cases": int(converged.sum()),
        "nonconverged_cases": int((~converged).sum()),
        "convergence_share": float(converged.mean()) if len(case_rows) else 0.0,
        "ac_dc_active_pair_count": int(len(valid)),
        "ac_dc_active_mae_mw": (
            float(error.abs().mean()) if len(error) else float("nan")
        ),
        "ac_dc_active_rmse_mw": (
            float(np.sqrt(np.mean(error * error))) if len(error) else float("nan")
        ),
        "ac_dc_active_sign_agreement_share": sign_agreement,
        "ac_dc_active_spearman": spearman,
        "minimum_voltage_pu": float(voltage.min()),
        "maximum_voltage_pu": float(voltage.max()),
        "voltage_rows_outside_narrow_count": int(outside_narrow.sum()),
        "voltage_rows_outside_narrow_share": float(outside_narrow.mean()),
        "voltage_rows_outside_restoration_count": int(outside_restoration.sum()),
        "voltage_rows_outside_restoration_share": float(outside_restoration.mean()),
        "maximum_element_loading_percent": maximum_loading,
        "thermal_violation_count": (
            int(thermal_count.fillna(0).sum()) if thermal_count is not None else -1
        ),
        "reactive_limit_violation_count": (
            int(reactive_count.fillna(0).sum()) if reactive_count is not None else -1
        ),
        "reported_case_voltage_violation_count": (
            int(voltage_count.fillna(0).sum()) if voltage_count is not None else -1
        ),
        "ac_feasibility_assessed": bool(feasibility_columns_available),
        "ac_feasible_cases": -1 if ac_feasible_cases is None else ac_feasible_cases,
        "convergence_is_not_feasibility": True,
        "claim_boundary": (
            "SELECTED_POLICY_STATE_NONLINEAR_CONSISTENCY_AND_STRESS_ONLY; "
            "NOT_ANNUAL_AC_OPF; NOT_COMPREHENSIVE_N_MINUS_1"
        ),
    }


# ============================================================================
# tpwrs_v3.ac_inputs (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Explicit reactive-input contract. Never infer missing Q as zero."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tpwrs.storage import sha256_file, sha256_array, sha256_json
from tpwrs.storage import atomic_write_json


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def aligned_matrix(path,timestamps,ids,id_column,value_column):
    p=Path(path)
    frame=pd.read_parquet(p) if p.suffix.lower()=='.parquet' else pd.read_csv(p)
    required={'timestamp_utc',id_column,value_column}
    if required-set(frame): raise ValueError(f'Missing fields in {p}: {required-set(frame)}')
    frame['timestamp_utc']=pd.to_datetime(frame['timestamp_utc'],utc=True,errors='raise')
    frame[id_column]=frame[id_column].astype(str)
    if frame.duplicated(['timestamp_utc',id_column]).any():raise ValueError(f'Duplicate AC input keys: {p}')
    target=pd.DatetimeIndex(pd.to_datetime(list(timestamps),utc=True));ids=list(map(str,ids))
    wide=frame.pivot(index='timestamp_utc',columns=id_column,values=value_column)
    if set(wide.columns)!=set(ids) or not wide.index.equals(target):
        raise ValueError(f'AC inputs must have exactly the frozen time/ID axes: {p}; no interpolation is applied.')
    values=wide.loc[target,ids].to_numpy(float)
    if not np.isfinite(values).all():raise ValueError(f'AC inputs contain missing/nonfinite values: {p}')
    return values


def validate_connection_weights(frame,bus_ids,participant_ids):
    if {'bus_id','participant_id','weight'}-set(frame):raise ValueError('Connection weights require bus_id,participant_id,weight.')
    f=frame.copy();f['bus_id']=f.bus_id.astype(str);f['participant_id']=f.participant_id.astype(str)
    if f.duplicated(['bus_id','participant_id']).any():raise ValueError('Duplicate participant Q connection weights.')
    if set(f.bus_id)-set(map(str,bus_ids)) or set(f.participant_id)!=set(participant_ids):raise ValueError('Participant Q map IDs differ from physical case.')
    matrix=f.pivot(index='bus_id',columns='participant_id',values='weight').reindex(index=list(bus_ids),columns=list(participant_ids)).fillna(0).to_numpy(float)
    if not np.isfinite(matrix).all() or np.any(matrix<0) or not np.allclose(matrix.sum(axis=0),1,rtol=0,atol=1e-10):
        raise ValueError('Participant Q connection columns must be nonnegative and sum to one.')
    return matrix


def freeze(config_path,case,output,network_path,network):
    cfg=read_json(config_path);root=Path(config_path).parent;output=Path(output)
    if cfg.get('q_sign_convention')!='positive_net_injection_mvar':raise ValueError('Declare q_sign_convention=positive_net_injection_mvar.')
    if cfg.get('q_fixed_excludes_voltage_controlled_generators_and_shunts') is not True:
        raise ValueError('Q background must explicitly exclude PV/slack-generated Q and separately modeled shunts.')
    if not cfg.get('provenance') or str(cfg['provenance']).strip().lower() in {'todo','required','replace me','unknown'}:raise ValueError('An AC input provenance statement is mandatory.')
    files={}
    for name in ('q_fixed','generator_dispatch','participant_connection_weights'):
        item=cfg.get(name)
        if not isinstance(item,dict) or not item.get('path'):raise ValueError(f'AC input missing: {name}')
        p=Path(item['path']);p=p if p.is_absolute() else root/p
        if not p.is_file():raise FileNotFoundError(p)
        digest=sha256_file(p)
        if item.get('sha256') and digest!=item['sha256']:raise ValueError(f'AC source changed: {name}')
        files[name]={'path':str(p.resolve()),'sha256':digest}
    gen_ids=tuple(str(i) for i in network.gen.index[network.gen.in_service.astype(bool)])
    q=aligned_matrix(files['q_fixed']['path'],case.time_grid.timestamps_utc,case.bus_ids,'bus_id','q_injection_mvar')
    if gen_ids:
        pg=aligned_matrix(files['generator_dispatch']['path'],case.time_grid.timestamps_utc,gen_ids,'generator_id','p_mw')
    else:
        frame=pd.read_csv(files['generator_dispatch']['path'])
        if len(frame):raise ValueError('No source PV generators but nonempty generator dispatch supplied.')
        pg=np.empty((case.n_time,0))
    connection=validate_connection_weights(pd.read_csv(files['participant_connection_weights']['path']),case.bus_ids,case.participant_ids)
    # Preserve actual source Q limits and voltage targets; absence is not a feasible result.
    for label,table in (('gen',network.gen.loc[list(map(int,gen_ids))] if gen_ids else network.gen.iloc[:0]),
                        ('ext_grid',network.ext_grid[network.ext_grid.in_service.astype(bool)])):
        for column in ('vm_pu','min_q_mvar','max_q_mvar'):
            if len(table) and (column not in table or not np.isfinite(table[column].to_numpy(float)).all()):
                raise ValueError(f'Source {label} lacks finite {column}; selected AC input freeze blocked.')
        if len(table) and np.any(table.min_q_mvar>table.max_q_mvar):raise ValueError(f'Invalid Q bounds for {label}.')
    identity={'physical_case_hash':case.physical_case_hash,'time_grid_id':case.time_grid.grid_id,
        'bus_ids':list(case.bus_ids),'participant_ids':list(case.participant_ids),'generator_ids':list(gen_ids),
        'network_sha256':sha256_file(network_path),'source_files':files,'configuration':cfg}
    identity_hash=sha256_json(identity);outmeta=output/'AC_INPUT_FREEZE.json'
    if outmeta.is_file():
        old=read_json(outmeta)
        if old.get('identity_hash')!=identity_hash:raise RuntimeError('AC inputs already frozen under another identity. Do not overwrite viewed engineering inputs.')
        if sha256_file(output/'ac_inputs.npz')!=old['arrays_sha256']:raise ValueError('Frozen AC arrays were altered.')
        return old
    output.mkdir(parents=True,exist_ok=True)
    import os,tempfile
    fd,tmp=tempfile.mkstemp(dir=output,suffix='.npz');os.close(fd)
    try:
        np.savez_compressed(tmp,q_fixed_mvar=q,generator_p_mw=pg,participant_q_connection=connection)
        os.replace(tmp,output/'ac_inputs.npz')
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
    meta={'schema':'TPWRS_EXPLICIT_AC_INPUTS_V1','status':'PASS','identity_hash':identity_hash,
          'arrays_sha256':sha256_file(output/'ac_inputs.npz'),**identity,
          'claim_boundary':'Explicit selected-state reconstruction inputs, not proof of feasibility.'}
    atomic_write_json(outmeta,meta);return meta


def load(directory,case,network_path):
    root=Path(directory);meta=read_json(root/'AC_INPUT_FREEZE.json')
    if meta.get('status')!='PASS' or meta.get('physical_case_hash')!=case.physical_case_hash:
        raise ValueError('AC input freeze does not match physical case.')
    identity={k:meta[k] for k in ('physical_case_hash','time_grid_id','bus_ids','participant_ids','generator_ids','network_sha256','source_files','configuration')}
    if sha256_json(identity)!=meta.get('identity_hash'):raise ValueError('AC freeze identity was altered.')
    if meta['bus_ids']!=list(case.bus_ids) or meta['participant_ids']!=list(case.participant_ids):raise ValueError('AC input axes changed.')
    if meta['network_sha256']!=sha256_file(network_path) or meta['arrays_sha256']!=sha256_file(root/'ac_inputs.npz'):raise ValueError('AC network/array hash mismatch.')
    with np.load(root/'ac_inputs.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    return meta,arrays


# ============================================================================
# tpwrs_v3.ac_reconstruction (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Selected AC reconstruction, separate from the parent affine DC optimizer."""
from copy import deepcopy
import numpy as np


def prepare(network,state_id,bus_ids,p_injection,q_fixed,generator_ids,generator_p):
    import pandapower as pp
    net=deepcopy(network)
    if set(map(str,net.bus.index))!=set(bus_ids):raise ValueError('AC/native bus ID sets differ.')
    # These composite/control elements require explicit mappings; silently
    # dropping their physics is not allowed.
    for name in ('ward','xward','dcline','trafo3w','ssc','svc','tcsc'):
        table=getattr(net,name,None)
        if table is not None and len(table) and ('in_service' not in table or table.in_service.any()):
            raise ValueError(f'Unsupported in-service AC source element: {name}. Supply an audited equivalent explicitly.')
    for name in ('load','sgen','storage'):
        table=getattr(net,name,None)
        if table is not None and len(table):table.loc[:,'in_service']=False
    live_ext=net.ext_grid.index[net.ext_grid.in_service.astype(bool)]
    if len(live_ext)!=1:raise ValueError('Selected AC requires the source to specify one reference grid; no slack redistribution is invented.')
    if state_id!='INTACT':
        if not state_id.startswith('OUT::'):raise ValueError(f'Unsupported contingency: {state_id}')
        kind,index=state_id[5:].split('::',1)
        if kind not in ('line','trafo'):raise ValueError('Only declared line/transformer outage states are supported.')
        net[kind].at[int(index),'in_service']=False
    fixedp=np.asarray(p_injection,float).copy();q=np.asarray(q_fixed,float)
    if fixedp.shape!=(len(bus_ids),) or q.shape!=fixedp.shape or not np.isfinite(fixedp).all() or not np.isfinite(q).all():raise ValueError('Invalid AC injections.')
    if abs(float(fixedp.sum()))>1e-7:raise ValueError('Pre-loss DC injection must be balanced.')
    buspos={str(b):i for i,b in enumerate(bus_ids)}
    active_ids=tuple(map(str,net.gen.index[net.gen.in_service.astype(bool)]))
    if tuple(generator_ids)!=active_ids:raise ValueError('Frozen source generator IDs/order changed.')
    if len(generator_p)!=len(generator_ids) or not np.isfinite(generator_p).all():raise ValueError('PV active-power dispatch is incomplete/nonfinite.')
    for index,p in zip(generator_ids,generator_p):
        net.gen.at[int(index),'p_mw']=float(p);fixedp[buspos[str(int(net.gen.at[int(index),'bus']))]]-=float(p)
    # Synthetic PQ elements encode supplied non-PV net injections only.
    # PV generators, their voltage targets/Q limits, shunts, taps, and slack
    # controls remain the source devices. Slack P adjusts for nonlinear losses.
    for i,bus in enumerate(bus_ids):pp.create_sgen(net,int(bus),p_mw=float(fixedp[i]),q_mvar=float(q[i]),name='FROZEN_NET_PQ')
    return net


def solve(net):
    import pandapower as pp
    attempts=[]
    for init in ('dc','flat'):
        try:
            pp.runpp(net,algorithm='nr',init=init,calculate_voltage_angles=True,
                     enforce_q_lims=True,check_connectivity=True,max_iteration=100,tolerance_mva=1e-8)
            if bool(net.converged):return True,attempts
            attempts.append({'init':init,'error':'nonconverged'})
        except Exception as exc:attempts.append({'init':init,'error':f'{type(exc).__name__}: {exc}'})
    return False,attempts


def reactive_delta(connection,request,x,directions,pf_case):
    if pf_case=='unity':return np.zeros(connection.shape[0])
    # Signed net-injection convention. The sensitivity cases apply the same
    # algebraic lagging/leading relation to every compared policy, not a claim
    # about actual project controls.
    tangent=np.tan(np.arccos(0.98));lag=1 if pf_case=='lagging_098' else -1
    if pf_case not in ('lagging_098','leading_098'):raise ValueError(pf_case)
    orientation=np.array([-1.0 if d=='import' else 1.0 for d in directions])
    return connection@(orientation*(np.asarray(request)-np.asarray(x))*tangent*lag)


# ============================================================================
# tpwrs_v3.redispatch_audit (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np
from scipy.optimize import linprog

from tpwrs.storage import ContractError, SolverFailure
from tpwrs.contracts import PhysicalCase


def common_redispatch_restore(case: PhysicalCase, x_mw: np.ndarray, R: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str,object]:
    """Selected-state common remedial robustness audit, outside the parent model."""
    R=np.asarray(R,float); lo=np.asarray(lower,float); hi=np.asarray(upper,float)
    if R.shape[0]!=case.n_buses or np.max(np.abs(R.sum(axis=0)))>1e-9: raise ContractError("Redispatch actions must be balanced.")
    K=R.shape[1]
    if lo.shape!=(K,) or hi.shape!=(K,) or np.any(lo>hi): raise ContractError("Redispatch bounds invalid.")
    p=case.post_state_mw(x_mw); H=case.flat_H; h=case.flat_h; rating=case.flat_ratings; GR=H@R
    total_before=0.; total_after=0.; actions=[]
    for t in range(case.n_time):
        base=H@p[t]+h; before=np.maximum(np.abs(base)-rating,0); total_before+=before.sum(); M=len(rating); n=K+2*M
        c=np.r_[np.zeros(K),np.ones(2*M)]; A=[]; b=[]
        for r in range(M):
            row=np.zeros(n); row[:K]=GR[r]; row[K+r]=-1; A.append(row); b.append(rating[r]-base[r])
            row=np.zeros(n); row[:K]=-GR[r]; row[K+M+r]=-1; A.append(row); b.append(rating[r]+base[r])
        res=linprog(c,A_ub=np.asarray(A),b_ub=np.asarray(b),bounds=list(zip(lo,hi))+[(0,None)]*(2*M),method="highs")
        if not res.success: raise SolverFailure(f"Redispatch audit failed at t={t}: {res.message}")
        actions.append(res.x[:K]); total_after+=float(res.fun)
    return {"actions_mw":np.asarray(actions),"pre_redispatch_residual_violation_mw":float(total_before),"post_redispatch_residual_violation_mw":float(total_after),"parent_policy_changed":False}
