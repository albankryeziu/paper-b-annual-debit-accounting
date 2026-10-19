from __future__ import annotations


# ============================================================================
# tpwrs_v3.metrics (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.contracts import PhysicalCase


def residual_index(case: PhysicalCase, x_mw: np.ndarray) -> tuple[float,float,float,int,int]:
    x=np.asarray(x_mw,float)
    if x.shape!=case.requests_mw.shape: raise ValueError("x_mw shape does not match physical case.")
    rating=case.flat_ratings[None,:]; S=0.0; Sn=0.0; maximum=0.0; insecure=0; pairs=0
    for start in range(0,case.n_time,256):
        stop=min(start+256,case.n_time); post=case.requested_state_mw[start:stop]+x[start:stop]@case.A.T; flows=post@case.flat_H.T+case.flat_h[None,:]; violation=np.maximum(np.abs(flows)-rating,0.0); dt=case.time_grid.delta[start:stop,None]
        S+=float(np.sum(dt*violation)); Sn+=float(np.sum(dt*np.divide(violation,rating,out=np.zeros_like(violation),where=rating>0))); maximum=max(maximum,float(np.max(violation,initial=0.0))); hit=violation>1e-9; insecure+=int(np.any(hit,axis=1).sum()); pairs+=int(hit.sum())
    return S,Sn,maximum,insecure,pairs


def reference_state_metrics(case: PhysicalCase, optimized_x_mw: np.ndarray|None=None) -> dict[str,float|int]:
    zero=np.zeros_like(case.requests_mw); S_background=0.0
    # Background reference bypasses requested transactions; chunking changes
    # memory use only and retains every registered state-element row.
    for start in range(0,case.n_time,256):
        stop=min(start+256,case.n_time); flows=case.p_background_mw[start:stop]@case.flat_H.T+case.flat_h[None,:]; violation=np.maximum(np.abs(flows)-case.flat_ratings[None,:],0.0); S_background+=float(np.sum(case.time_grid.delta[start:stop,None]*violation))
    result={"S_background_mw_h_row":S_background,"S_requested_mw_h_row":residual_index(case,zero)[0]}
    if optimized_x_mw is not None: result["S_policy_mw_h_row"]=residual_index(case,optimized_x_mw)[0]
    return result


def episode_metrics(x_mw: np.ndarray, delta_t_hours: np.ndarray, *, tol_mw:float=1e-7) -> dict[str,float|int]:
    x=np.asarray(x_mw,float); dt=np.asarray(delta_t_hours,float); active=x>tol_mw; count=0; longest=0.0; isolated=0
    for n in range(x.shape[1]):
        current=0.0; length=0
        for t in range(x.shape[0]):
            if active[t,n]: current+=dt[t]; length+=1; longest=max(longest,current)
            else:
                if length: count+=1; isolated+=int(length==1)
                current=0; length=0
        if length: count+=1; isolated+=int(length==1)
    return {"episode_count":count,"isolated_one_interval_episodes":isolated,"maximum_episode_duration_h":float(longest)}


# ============================================================================
# tpwrs_v3.result_aggregation (consolidated; see SOURCE_MAP.json)
# ============================================================================

import json
from pathlib import Path
from typing import Iterable

from tpwrs.storage import ContractError
from tpwrs.storage import sha256_file
from tpwrs.storage import atomic_write_json


def _metric_paths(root: Path, year: int, campaign: str) -> list[tuple[float,str,str,Path]]:
    rows=[]
    for rho in (0.15,0.03):
        if campaign=="full": annual=("M1_LP","M2","M3")
        elif rho==0.15: annual=("M1_LP","M2","M3") if campaign=="recommended" else ("M1_LP","M2")
        else: annual=("M2","M3")
        annual+=(("M2_MATCHED",) if rho==0.03 else ())
        for rule in annual:
            rows.append((rho,"annual",rule,root/f"year_{year}/rho_{rho:.2f}/annual/{rule}/metrics.json"))
        if campaign=="full": rolling=("M1","M2","M3")
        elif rho==0.15: rolling=("M1","M2","M3") if campaign=="recommended" else ("M1","M2")
        else: rolling=("M2","M3") if campaign=="recommended" else ()
        for rule in rolling:
            rows.append((rho,"rolling",rule,root/f"year_{year}/rho_{rho:.2f}/rolling/{rule}/rolling_result.json"))
    return rows


def _aggregate(root: Path, years: Iterable[int], label: str, campaign: str) -> dict[str,object]:
    years=tuple(map(int,years)); records=[]; missing=[]
    for year in years:
        for rho,mode,rule,path in _metric_paths(root,year,campaign):
            if not path.is_file():
                missing.append(str(path.relative_to(root))); continue
            value=json.loads(path.read_text(encoding="utf-8"))
            if not value.get("replay_passed"):
                raise ContractError(f"Cannot aggregate a policy without passing replay: {path}")
            records.append({
                "year":year,"rho":rho,"mode":mode,"rule":rule,
                "S_mw_h_row":float(value["S_mw_h_row"]),
                "E_restricted_mwh":float(value["E_restricted_mwh"]),
                "physical_fde_sum_h":float(value["physical_fde_sum_h"]),
                "source_path":str(path.relative_to(root)).replace("\\","/"),
                "source_sha256":sha256_file(path),
            })
    if missing:
        raise ContractError(f"Aggregation inputs are missing: {missing}")
    grouped={}
    for row in records:
        key=f"rho_{row['rho']:.2f}/{row['mode']}/{row['rule']}"
        item=grouped.setdefault(key,{"years":list(years),"S_mw_h_row":0.0,"E_restricted_mwh":0.0,"physical_fde_sum_h":0.0})
        item["S_mw_h_row"]+=row["S_mw_h_row"]
        item["E_restricted_mwh"]+=row["E_restricted_mwh"]
        item["physical_fde_sum_h"]+=row["physical_fde_sum_h"]
    return {
        "schema":"TPWRS_FINAL_RESULT_AGGREGATE_V1",
        "status":"PASS",
        "evaluation_role":label,
        "campaign":campaign,
        "years":years,
        "confirmation_pooled_with_selection":False,
        "groups":grouped,
        "records":records,
    }


def write_final_aggregates(results_root: str | Path, *, campaign: str="full") -> tuple[Path,Path]:
    if campaign not in {"minimum","recommended","full"}: raise ContractError(f"Unknown campaign {campaign}.")
    root=Path(results_root); summary=root/"summaries"; summary.mkdir(parents=True,exist_ok=True)
    heldout=summary/"HELDOUT_2029_2030.json"
    confirmation=summary/"CONFIRMATION_2031.json"
    atomic_write_json(heldout,_aggregate(root,(2029,2030),"HELD_OUT_2029_2030",campaign))
    atomic_write_json(confirmation,_aggregate(root,(2031,),"SEPARATE_CONFIRMATION_2031",campaign))
    return heldout,confirmation


# ============================================================================
# tpwrs_v3.block_resampling (consolidated; see SOURCE_MAP.json)
# ============================================================================
import numpy as np
import pandas as pd


def circular_block_sums(series: dict[str,np.ndarray], *, draws:int=1000, block_length:int=168, seed:int=118) -> pd.DataFrame:
    values={k:np.asarray(v,float).reshape(-1) for k,v in series.items()}; lengths={len(v) for v in values.values()}
    if len(lengths)!=1: raise ValueError("All resampled trajectories must have equal length.")
    n=next(iter(lengths)); rng=np.random.default_rng(seed); n_blocks=int(np.ceil(n/block_length)); rows=[]
    for d in range(draws):
        starts=rng.integers(0,n,size=n_blocks); idx=np.concatenate([np.mod(np.arange(s,s+block_length),n) for s in starts])[:n]
        row={"draw":d}; row.update({k:float(v[idx].sum()) for k,v in values.items()}); rows.append(row)
    out=pd.DataFrame(rows); out.attrs["interpretation"]="FIXED_POLICY_TEMPORAL_RESAMPLING_NOT_OPTIMIZATION_GAP_CI"
    return out


# ============================================================================
# tpwrs_v3.claim_contract (consolidated; see SOURCE_MAP.json)
# ============================================================================


CLAIMS = {
    "m1_m2_strict_certificate": {
        "required": (
            "replay-feasible M2 + certified annual M1-LP Stage-I lower bound"
        ),
        "pass_wording": (
            "M2 is certified to attain a strictly lower Stage-I residual "
            "thermal-row index than every feasible annual M1 schedule."
        ),
        "fail_wording": "No strict M1 certificate is claimed for this case.",
    },
    "m3_matched_exposure": {
        "required": (
            "M2, corrected frozen-card M3, and M2-MATCHED on one identical "
            "physical case and allowance lineage"
        ),
        "pass_wording": (
            "M3 results are decomposed into participant-specific exposure "
            "and state-shape effects."
        ),
        "fail_wording": (
            "M3 is reported only nominally; no matched-exposure efficiency "
            "claim is made."
        ),
    },
    "m3_ordering_advantage": {
        "required": (
            "selected M3 compared with PTDF, inverse-order, and all frozen "
            "within-direction shuffle controls"
        ),
        "pass_wording": (
            "The selected KKT-informed ordering is compared with transparent, "
            "inverse, and frozen-shuffle alternatives."
        ),
        "fail_wording": (
            "No unique KKT-ordering advantage is claimed without the complete "
            "control family."
        ),
    },
    "m3_gamma_path": {
        "required": (
            "post-freeze gamma path with exact M3 and M2-MATCHED solves and "
            "physical-FDE reporting"
        ),
        "pass_wording": (
            "The frozen card is located on an exact post-freeze "
            "security-versus-physical-exposure path."
        ),
        "fail_wording": "No gamma-path robustness claim is made.",
    },
    "multibus_primary": {
        "required": "all 35 multibus embeddings pass the frozen all-or-none gate",
        "pass_wording": "The primary case contains all 114 positions.",
        "fail_wording": (
            "The primary case contains 79 direct mappings; multibus positions "
            "are sensitivity only."
        ),
    },
    "independent_dc": {
        "required": (
            "complete one-to-one selected-state/row mapping and independent "
            "DC software replay with no false-safe classifications"
        ),
        "pass_wording": (
            "The frozen affine evaluator is reproduced by an independent "
            "selected-state DC software path."
        ),
        "fail_wording": "No independent selected-state DC reproduction claim is made.",
    },
    "selected_ac": {
        "required": (
            "mapped selected-state AC consistency/stress audit reporting "
            "convergence, voltage, thermal loading, reactive limits, and "
            "AC/DC active-flow error separately"
        ),
        "pass_wording": (
            "Selected policy states were subjected to nonlinear consistency "
            "and stress checks."
        ),
        "fail_wording": "No nonlinear engineering-robustness claim is made.",
    },
}


def claim_record(name: str, passed: bool) -> dict[str, str | bool]:
    if name not in CLAIMS:
        raise KeyError(f"Unknown claim {name!r}.")
    item = CLAIMS[name]
    return {
        "claim": name,
        "passed": bool(passed),
        "required_evidence": item["required"],
        "allowed_wording": (
            item["pass_wording"] if passed else item["fail_wording"]
        ),
    }


# ============================================================================
# tpwrs_v3.component_audit (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Validate supplied chronological components without imputing or rescaling them."""
import numpy as np
import pandas as pd
COLUMNS=('prov_load_MW','prov_solar_MW','prov_wind_on_MW','prov_wind_off_MW')

def validate_component_panel(frame: pd.DataFrame):
    needed={'timestamp_utc','province',*COLUMNS}
    if not needed.issubset(frame): raise ValueError(f'Missing component columns: {sorted(needed-set(frame))}')
    f=frame.copy();f['timestamp_utc']=pd.to_datetime(f.timestamp_utc,utc=True,errors='raise')
    if f[['timestamp_utc','province']].duplicated().any(): raise ValueError('Duplicate time/province')
    v=f[list(COLUMNS)].to_numpy(float)
    if not np.isfinite(v).all() or (v<0).any(): raise ValueError('Demand and renewable components must be finite and nonnegative; residual load may have either sign.')
    times=pd.DatetimeIndex(sorted(f.timestamp_utc.unique()))
    if len(times)>1 and not np.all(np.diff(times.asi8)==3600*10**9): raise ValueError('Chronology is not consecutive hourly UTC')
    nprovince=f.province.nunique()
    if len(f)!=len(times)*nprovince or not f.groupby('timestamp_utc').size().eq(nprovince).all(): raise ValueError('Incomplete province panel')
    rl=v[:,0]-v[:,1]-v[:,2]-v[:,3]
    if 'RL_MW' in f and not np.allclose(rl,f.RL_MW.to_numpy(float),rtol=1e-12,atol=1e-7): raise ValueError('Residual load identity failed')
    return {'status':'PASS','hours':len(times),'provinces':nprovince,'rows':len(f),
            'negative_residual_rows':int((rl<0).sum()),'imputation_performed':False}
