from __future__ import annotations


# ============================================================================
# scripts.run_selected_ac_validation_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Selected-state AC stress audit using explicit frozen reactive inputs."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from tpwrs.engineering import load as load_ac
from tpwrs.engineering import prepare, solve, reactive_delta
from tpwrs.config import source_path, scientific_fingerprint
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.storage import atomic_write_json, atomic_write_csv
from tpwrs.persistence import load_physical_case
from tpwrs.execution import utc_now
from jobs.common import year_case, load_policy_array
from jobs.selection import _policy_catalog
from jobs.dc import _load_network, _extract_rows

AC_ORDERING_TIE_TOLERANCE = 1e-4
AC_INPUT_SOURCE_KEYS = (
    "q_fixed",
    "generator_dispatch",
    "participant_connection_weights",
)


def ac_input_preflight(config_path: Path) -> dict:
    """Validate that Q can freeze three explicit, hash-bound AC sources."""
    path = Path(config_path)
    if not path.is_file():
        return {
            "status": "AC_INPUTS_BLOCKED",
            "config": str(path),
            "reason": "local/AC_INPUTS.json is absent",
            "required_sources": list(AC_INPUT_SOURCE_KEYS),
        }
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "status": "AC_INPUTS_BLOCKED",
            "config": str(path),
            "reason": f"invalid AC input JSON: {exc}",
            "required_sources": list(AC_INPUT_SOURCE_KEYS),
        }
    records = {}
    errors = []
    for key in AC_INPUT_SOURCE_KEYS:
        item = config.get(key)
        if not isinstance(item, dict) or not item.get("path"):
            errors.append(f"{key}: missing path")
            continue
        source = Path(item["path"])
        if not source.is_absolute():
            source = path.parent / source
        expected = item.get("sha256")
        if not source.is_file():
            errors.append(f"{key}: source absent: {source}")
            continue
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"{key}: frozen SHA-256 is absent")
            continue
        actual = sha256_file(source)
        if actual != expected:
            errors.append(f"{key}: SHA-256 mismatch")
            continue
        records[key] = {"path": str(source), "sha256": actual}
    return {
        "status": "PASS" if not errors else "AC_INPUTS_BLOCKED",
        "config": str(path),
        "required_sources": list(AC_INPUT_SOURCE_KEYS),
        "sources": records,
        "errors": errors,
    }


def require_ac_input_preflight(config_path: Path) -> dict:
    result = ac_input_preflight(config_path)
    marker = ROOT / "artifacts/run_control/AC_INPUTS_PREFLIGHT.json"
    atomic_write_json(marker, {**result, "created_utc": utc_now()})
    if result["status"] != "PASS":
        raise RuntimeError("AC_INPUTS_BLOCKED: " + "; ".join(result.get("errors") or [result.get("reason", "missing frozen AC inputs")]))
    return result


def dc_normalized_thermal_exceedance(
    active_power_mw: np.ndarray,
    ratings_mw: np.ndarray,
) -> float:
    p = np.asarray(active_power_mw, dtype=float)
    rating = np.asarray(ratings_mw, dtype=float)
    if p.shape != rating.shape or not np.isfinite(p).all() or not np.isfinite(rating).all() or np.any(rating <= 0.0):
        raise ValueError("DC ordering inputs must be finite, aligned, and positively rated.")
    return float(np.sum(np.maximum(np.abs(p) / rating - 1.0, 0.0)))


def ac_normalized_thermal_exceedance(loading_percent: np.ndarray) -> float:
    loading = np.asarray(loading_percent, dtype=float)
    if not np.isfinite(loading).all() or np.any(loading < 0.0):
        raise ValueError("AC apparent-power loading must be finite and nonnegative.")
    return float(np.sum(np.maximum(loading / 100.0 - 1.0, 0.0)))


def classify_normalized_ordering(
    left_dc: float,
    right_dc: float,
    left_ac: float,
    right_ac: float,
    *,
    tie_tolerance: float = AC_ORDERING_TIE_TOLERANCE,
) -> dict:
    values = np.asarray([left_dc, right_dc, left_ac, right_ac, tie_tolerance], dtype=float)
    if not np.isfinite(values).all() or tie_tolerance < 0.0:
        raise ValueError("Ordering inputs must be finite and tie tolerance nonnegative.")
    dc_difference = float(left_dc - right_dc)
    ac_difference = float(left_ac - right_ac)
    if abs(dc_difference) <= tie_tolerance:
        classification = "DC_TIE"
    elif abs(ac_difference) <= tie_tolerance:
        classification = "AC_TIE_LOSS_OF_DISCRIMINATION"
    elif np.sign(dc_difference) == np.sign(ac_difference):
        classification = "ORDER_PRESERVED"
    else:
        classification = "ORDER_REVERSED"
    return {
        "dc_difference_normalized": dc_difference,
        "ac_difference_normalized": ac_difference,
        "tie_tolerance": float(tie_tolerance),
        "classification": classification,
        "ordering_preserved": classification in {"DC_TIE", "ORDER_PRESERVED"},
    }


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.bool_):return bool(value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,default=ROOT/'artifacts/real_case/physical_case/FINAL_DUTCH_CASE')
    parser.add_argument('--output',type=Path,default=ROOT/'results_engineering_validation/selected_ac')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    regroot=ROOT/'results_engineering_validation/selected_registry';regpath=regroot/'SELECTED_POLICY_STATE_CASES.csv';frpath=regroot/'SELECTED_STATE_REGISTRY_FREEZE.json'
    statuspath=args.output/'STATUS.json';started=time.monotonic()
    status={'status':'RUNNING','created_utc':utc_now(),'scientific_fingerprint':scientific_fingerprint(),'selected_state_only':True,'annual_ac_feasibility_claim':False}
    atomic_write_json(statuspath,status)
    try:
        case=load_physical_case(args.case);source=source_path('reduced_network');net=_load_network(source)
        freeze=json.loads(frpath.read_text());assert freeze['status']=='PASS'
        if freeze['physical_case_hash']!=case.physical_case_hash or sha256_file(regpath)!=freeze['outputs']['cases']['sha256']:raise ValueError('Selected registry hash/physical case mismatch.')
        meta,arrays=load_ac(ROOT/'artifacts/ac_inputs',case,source)
        registry=pd.read_csv(regpath);catalogs={};policies={};years={}
        for year in sorted(registry.year.unique()):
            years[int(year)]=year_case(case,int(year))
            catalogs[int(year)]=_policy_catalog(int(year),years[int(year)],ROOT/'results_m1_m2_final',ROOT/'results_m3_final')
        time_to_global={str(t):i for i,t in enumerate(case.time_grid.timestamps_utc)}
        cases=[];branches=[];voltages=[];loadings=[]
        pf_cases=('unity','lagging_098','leading_098')
        native_states={s.state_id:s for s in case.network_states}
        for row in registry.itertuples(index=False):
            year=int(row.year);ti=int(row.time_index);yc=years[year];record=catalogs[year][row.policy_state]
            if record['source_hash']!=row.policy_source_hash:raise ValueError('A selected policy source changed after selection.')
            p=np.asarray(record['state'][ti]);state=native_states[row.state_id]
            timestamp=pd.Timestamp(row.timestamp_utc).isoformat()
            matches=np.flatnonzero(case.time_grid.index==pd.Timestamp(timestamp))
            if len(matches)!=1:raise ValueError('Selected timestamp absent from complete physical time axis.')
            gi=int(matches[0])
            if row.policy_state=='BACKGROUND':req=np.zeros(case.n_participants);x=np.zeros_like(req)
            elif row.policy_state=='REQUEST':req=yc.requests_mw[ti];x=np.zeros_like(req)
            else:
                path=Path(record['source'])
                if str(path) not in policies:policies[str(path)]=load_policy_array(path,rolling='ROLLING' in row.policy_state)
                req=yc.requests_mw[ti];x=policies[str(path)][ti]
            for pf in pf_cases:
                key=sha256_json({'row':row._asdict(),'pf':pf,'q_inputs':meta['identity_hash'],'code':status['scientific_fingerprint']})
                cp=args.output/'cases'/f'{key}.json'
                if cp.is_file():item=json.loads(cp.read_text())
                else:
                    print(f'AC CASE={row.case_id} PF={pf}',flush=True)
                    q=arrays['q_fixed_mvar'][gi]+reactive_delta(arrays['participant_q_connection'],req,x,yc.participant_directions,pf)
                    testnet=prepare(net,row.state_id,case.bus_ids,p,q,meta['generator_ids'],arrays['generator_p_mw'][gi])
                    converged,attempts=solve(testnet)
                    base={'case_id':row.case_id,'pf_case':pf,'year':year,'time_index':ti,'timestamp_utc':row.timestamp_utc,
                          'state_id':row.state_id,'policy_state':row.policy_state,'converged':converged,'attempts':attempts}
                    br=[];vr=[];lr=[];dc=state.H@p+state.h_mw
                    if converged:
                        ac=_extract_rows(testnet,state)
                        for rid,acp,dcp,rate in zip(state.row_ids,ac,dc,state.ratings_mw):
                            kind,idx=rid.split('::',1);idx=int(idx);result=testnet['res_'+kind]
                            loading=float(result.at[idx,'loading_percent'])
                            br.append({'case_id':row.case_id,'pf_case':pf,'row_id':rid,'ac_p_mw':float(acp),'dc_p_mw':float(dcp),'rating_mw':float(rate)})
                            lr.append({'case_id':row.case_id,'pf_case':pf,'row_id':rid,'loading_percent':loading})
                        vr=[{'case_id':row.case_id,'pf_case':pf,'bus_id':str(i),'vm_pu':float(v)} for i,v in testnet.res_bus.vm_pu.items()]
                        vm=np.asarray(testnet.res_bus.vm_pu,float);load=np.array([v['loading_percent'] for v in lr])
                        qviol=[]
                        for name in ('gen','ext_grid'):
                            table=testnet[name];res=testnet['res_'+name]
                            for idx in table.index[table.in_service.astype(bool)]:
                                val=float(res.at[idx,'q_mvar']);qviol.append(max(float(table.at[idx,'min_q_mvar'])-val,val-float(table.at[idx,'max_q_mvar']),0))
                        finite=bool(np.isfinite(vm).all() and np.isfinite(load).all())
                        base.update(min_vm_pu=float(vm.min()),max_vm_pu=float(vm.max()),maximum_loading_percent=float(load.max()),
                            voltage_violations=int(np.sum((vm<0.95)|(vm>1.05))),thermal_violations=int(np.sum(load>100)),
                            max_q_limit_violation_mvar=max(qviol,default=0.0),
                            ac_feasible_under_declared_checks=bool(finite and np.all((vm>=0.95)&(vm<=1.05)) and np.all(load<=100+1e-6) and max(qviol,default=0)<=1e-6),
                            ac_apparent_normalized_thermal_exceedance=ac_normalized_thermal_exceedance(load),
                            dc_active_normalized_thermal_exceedance=dc_normalized_thermal_exceedance(dc,state.ratings_mw),
                            slack_p_mw=float(testnet.res_ext_grid.p_mw.sum()))
                    else:base.update(ac_feasible_under_declared_checks=False)
                    item=clean({'case':base,'branches':br,'voltages':vr,'loadings':lr,'input_hash':key})
                    atomic_write_json(cp,item)
                if item.get('input_hash')!=key:raise ValueError('AC case cache key mismatch.')
                cases.append(item['case']);branches+=item['branches'];voltages+=item['voltages'];loadings+=item['loadings']
                atomic_write_json(args.output/'HEARTBEAT.json',{'status':'RUNNING','completed_cases':len(cases),'total_cases':len(registry)*len(pf_cases),'updated_utc':utc_now()})
        cf=pd.DataFrame(cases)
        # Frozen ordering compares normalized DC active-power exceedance with
        # normalized AC apparent-power thermal loading.
        pairs=[]
        for keys,group in cf.groupby(['year','time_index','state_id','pf_case']):
            by=group.set_index('policy_state')
            for left,right in [('RHO15_ROLLING_M1','RHO15_ROLLING_M2'),('RHO03_ANNUAL_M2','RHO03_ANNUAL_M3')]:
                usable=left in by.index and right in by.index and bool(by.at[left,'converged']) and bool(by.at[right,'converged'])
                rowpair=dict(zip(('year','time_index','state_id','pf_case'),keys));rowpair.update(left=left,right=right,usable=usable)
                if usable:
                    rowpair.update(classify_normalized_ordering(
                        float(by.at[left,'dc_active_normalized_thermal_exceedance']),
                        float(by.at[right,'dc_active_normalized_thermal_exceedance']),
                        float(by.at[left,'ac_apparent_normalized_thermal_exceedance']),
                        float(by.at[right,'ac_apparent_normalized_thermal_exceedance']),
                    ))
                pairs.append(rowpair)
        tables={'cases':cf,'branches':pd.DataFrame(branches,columns=['case_id','pf_case','row_id','ac_p_mw','dc_p_mw','rating_mw']),
                'voltages':pd.DataFrame(voltages,columns=['case_id','pf_case','bus_id','vm_pu']),
                'loading':pd.DataFrame(loadings,columns=['case_id','pf_case','row_id','loading_percent']), 'ordering':pd.DataFrame(pairs)}
        outputs={}
        for name,frame in tables.items():
            pth=args.output/(name.upper()+'.csv');atomic_write_csv(pth,frame);outputs[name]={'path':pth.name,'sha256':sha256_file(pth)}
        usable=sum(p['usable'] for p in pairs);reversals=sum(p.get('classification')=='ORDER_REVERSED' for p in pairs)
        lost=sum(p.get('classification')=='AC_TIE_LOSS_OF_DISCRIMINATION' for p in pairs)
        ordering_complete=usable==len(pairs) and len(pairs)>0 and all(p.get('ordering_preserved') is True for p in pairs)
        status.update(status='PASS',execution_complete=True,completed_utc=utc_now(),elapsed_s=time.monotonic()-started,
            registry_freeze_sha256=sha256_file(frpath),physical_case_hash=case.physical_case_hash,ac_input_identity_hash=meta['identity_hash'],
            summary={'selected_case_count':len(cf),'converged_count':int(cf.converged.sum()),
                'nonconverged_count':int((~cf.converged).sum()),'feasible_checked_case_count':int(cf.ac_feasible_under_declared_checks.sum()),
                'usable_ordering_pairs':usable,'unusable_ordering_pairs':len(pairs)-usable,'ordering_reversals':reversals,
                'ac_tie_loss_of_discrimination_count':lost,'ordering_tie_tolerance':AC_ORDERING_TIE_TOLERANCE,
                'complete_selected_order_preservation':ordering_complete},outputs=outputs,
            claim_boundary='PASS means the audit completed, not that every selected state is AC feasible. Nonconvergent, voltage, thermal and Q-limit adverse cases are retained.')
        atomic_write_json(statuspath,clean(status));print(json.dumps(clean(status),indent=2));return 0
    except BaseException as exc:
        status.update(status='BLOCKED',error=f'{type(exc).__name__}: {exc}',ended_utc=utc_now());atomic_write_json(statuspath,status);raise


# ============================================================================
# scripts.freeze_ac_inputs (consolidated; see SOURCE_MAP.json)
# ============================================================================
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from tpwrs.engineering import freeze
from tpwrs.persistence import load_physical_case
from tpwrs.config import source_path
from jobs.dc import _load_network

def freeze_main():
    parser=argparse.ArgumentParser(description='Freeze explicitly documented Q, generator P and participant connection inputs; never backfill Q=0.')
    parser.add_argument('--config',type=Path,default=ROOT/'local/AC_INPUTS.json')
    args=parser.parse_args()
    require_ac_input_preflight(args.config)
    c=load_physical_case(ROOT/'artifacts/real_case/physical_case/FINAL_DUTCH_CASE');p=source_path('reduced_network')
    out=freeze(args.config,c,ROOT/'artifacts/ac_inputs',p,_load_network(p));print(json.dumps(out,indent=2));return 0
