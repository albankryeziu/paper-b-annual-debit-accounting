from __future__ import annotations


# ============================================================================
# scripts.aggregate_final_evidence (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Extract verified absolute results; missing evidence never becomes submission readiness."""
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from tpwrs.execution import validate_policy_solution, utc_now
from tpwrs.contracts import AllowanceSpec
from tpwrs.storage import sha256_file, sha256_array
from tpwrs.persistence import load_policy_solution
from tpwrs.storage import atomic_write_json, atomic_write_csv, atomic_write_text
from tpwrs.persistence import load_physical_case
from tpwrs.reporting import residual_index, episode_metrics
from tpwrs.config import scientific_fingerprint
from jobs.common import frozen_matched_decomposition, year_case, require_m1_m2_freeze, solution_metrics
from jobs.primary_freeze import _experiment_from_file
from tpwrs.provenance import compatibility_entry


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def status_if_valid(path, *, compatibility_use=None):
    path=Path(path)
    if not path.is_file():return {'status':'MISSING'}
    obj=read(path)
    if obj.get('scientific_fingerprint') is not None and obj.get('scientific_fingerprint')!=scientific_fingerprint():
        if compatibility_use is None:
            raise ValueError(f'Changed scientific identity: {path}')
        compatibility_entry(path,use=compatibility_use,payload=obj)
    for rec in obj.get('outputs',{}).values():
        p=path.parent/rec['path']
        if not p.is_file() or sha256_file(p)!=rec['sha256']:raise ValueError(f'Changed evidence: {p}')
    return obj


def annual(folder,case):
    exp=_experiment_from_file(folder/'EXPERIMENT_CASE.json')
    sol=load_policy_solution(folder,expected_experiment_hash=exp.hash,expected_physical_case_hash=case.physical_case_hash)
    validate_policy_solution(case,exp,sol,require_stage3=False)
    stored=read(folder/'metrics.json');log=read(folder/'case.log.json')
    if stored.get('solution_hash')!=sol.solution_hash or log.get('status')!='PASS':raise ValueError(f'Annual acceptance changed: {folder}')
    metric=solution_metrics(case,sol)
    for k in ('S_mw_h_row','E_restricted_mwh','physical_fde_sum_h'):
        if abs(float(stored[k])-float(metric[k]))>2e-6:raise ValueError(f'Metric differs from replay: {folder}:{k}')
    m=sol.stage2.metadata
    metric.update(wall_seconds=log.get('runtime_s'),stage1_seconds=sol.stage1.runtime_s,stage2_seconds=sol.stage2.runtime_s,
                  model_rows=m.get('model_rows'),model_columns=m.get('model_columns'),model_nonzeros=m.get('model_nonzeros'),
                  stage1_lower_bound=sol.stage1.best_bound,peak_private_gb=m.get('peak_private_gb'))
    return metric,sol.x_mw,sol


def rolling(folder,case):
    exp=_experiment_from_file(folder/'EXPERIMENT_CASE.json');meta=read(folder/'rolling_result.json');log=read(folder/'case.log.json')
    if meta.get('experiment_hash')!=exp.hash or meta.get('replay_passed') is not True or log.get('status')!='PASS':raise ValueError('Rolling acceptance failed.')
    ap=folder/meta['arrays_file']
    if sha256_file(ap)!=meta['arrays_sha256']:raise ValueError('Rolling arrays changed.')
    with np.load(ap,allow_pickle=False) as z:x=z['x_mw'].copy();u=z['u'].copy() if 'u' in z.files else None;terminal=z['final_balance_hours'].copy()
    if x.shape!=case.requests_mw.shape or not np.isfinite(x).all() or np.min(x)<-1e-7 or np.max(x-case.xbar_mw)>1e-7:raise ValueError('Rolling action bounds invalid.')
    balance=exp.allowance.values.copy();q=np.ones_like(x);step=round(exp.implementation_hours/case.time_grid.uniform_delta);look=round(exp.lookahead_hours/case.time_grid.uniform_delta)
    hashes=meta.get('epoch_solution_hashes',[])
    if len(hashes)!=len(range(0,case.n_time,step)):raise ValueError('Rolling epoch count mismatch.')
    stage1_s=stage2_s=0.0
    for start,expected in zip(range(0,case.n_time,step),hashes):
        sub=case.slice_time(start,min(start+look,case.n_time));n=min(step,case.n_time-start)
        epoch=load_policy_solution(folder/'epochs'/f'epoch_{start:06d}')
        if epoch.solution_hash!=expected or not np.allclose(x[start:start+n],epoch.x_mw[:n],rtol=0,atol=1e-7):raise ValueError('Rolling block differs from accepted epoch.')
        allowance=AllowanceSpec.explicit(case.participant_ids,balance,source='explicit',nominal_rho=exp.allowance.nominal_rho)
        ee=replace(exp,case_id=f'{exp.case_id}__epoch_{start:06d}',physical_case_id=sub.physical_case_id,physical_case_hash=sub.physical_case_hash,
                   time_grid_id=sub.time_grid.grid_id,allowance=allowance,mode='annual',lookahead_hours=None,implementation_hours=None,
                   result_root_id=f'{exp.result_root_id}/epochs/epoch_{start:06d}',rate_matrix_hash=sha256_array(epoch.q) if epoch.q is not None else None)
        if epoch.experiment_hash!=ee.hash:raise ValueError('Rolling epoch allowance, time or configuration identity mismatch.')
        validate_policy_solution(sub,ee,epoch,require_stage3=False)
        dt=case.time_grid.delta[start:start+n,None]
        if exp.rule=='M1':
            if u is None or epoch.u is None or not np.allclose(u[start:start+n],epoch.u[:n],rtol=0,atol=1e-7):raise ValueError('M1 binary chronology differs.')
            debit=np.sum(dt*u[start:start+n],axis=0)
        else:
            if exp.rule=='M3':q[start:start+n]=epoch.q[:n]
            debit=np.sum(dt*q[start:start+n]*x[start:start+n]/case.capacities_mw,axis=0)
        balance-=debit
        if balance.min()<-1e-7:raise ValueError('Rolling allowance exceeded.')
        balance=np.maximum(balance,0);stage1_s+=epoch.stage1.runtime_s;stage2_s+=epoch.stage2.runtime_s
    if exp.rule=='M3' and exp.rate_matrix_hash is not None and sha256_array(q)!=exp.rate_matrix_hash:raise ValueError('Reconstructed rolling q differs from the frozen full card matrix.')
    if not np.allclose(balance,terminal,rtol=0,atol=2e-6):raise ValueError('Terminal ledger mismatch.')
    dt=case.time_grid.delta[:,None];debit=np.sum(dt*(u if exp.rule=='M1' else q*x/case.capacities_mw),axis=0)
    if np.max(debit-exp.allowance.values)>2e-6:raise ValueError('Annual rolling debit exceeds allowance.')
    S,Sn,maxo,ih,vr=residual_index(case,x);E=float(np.sum(dt*x))
    if abs(S-meta['S_mw_h_row'])>2e-6 or abs(E-meta['E_restricted_mwh'])>2e-6:raise ValueError('Rolling metrics differ from full replay.')
    metric={**meta,'S_mw_h_row':S,'E_restricted_mwh':E,'stage1_optimum_mw_h_row':None,'stage1_lower_bound':None,
            'wall_seconds':log.get('runtime_s'),'stage1_seconds':stage1_s,'stage2_seconds':stage2_s,
            'physical_fde_sum_h':float(np.sum(dt*x/case.capacities_mw))}
    return metric,x,None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results_final_evidence');args=parser.parse_args()
    case=load_physical_case(ROOT/'artifacts/real_case/physical_case/FINAL_DUTCH_CASE');baseline=ROOT/'results_m1_m2_final';m3=ROOT/'results_m3_final'
    primary= require_m1_m2_freeze(baseline,case);core=status_if_valid(
        m3/'M3_CORE_STATUS.json',
        compatibility_use='UPSTREAM_STAGE05_CORE_NUMERICAL_EVIDENCE',
    )
    if core.get('status')!='COMPLETE':raise RuntimeError('M3 core is incomplete.')
    b03_status=read(baseline/'r10_certificates/B03_STATUS.json')
    if b03_status.get('status')!='PASS' or b03_status.get('scientific_fingerprint')!=scientific_fingerprint():
        raise RuntimeError('Corrected B03 Stage-04 recertification is absent or not PASS.')
    b03_records={int(row['year']):row for row in b03_status.get('records',[])}
    if set(b03_records)!={2029,2030,2031}:
        raise RuntimeError('Corrected B03 recertification does not contain exactly 2029-2031.')
    for row in b03_records.values():
        target=ROOT/row['path']
        if not target.is_file() or sha256_file(target)!=row['sha256']:
            raise RuntimeError(f"Corrected B03 certificate changed: {target}")
    optional_price=None;pf=ROOT/'artifacts/prices/PRICE_FREEZE.json'
    if pf.is_file():
        pm=read(pf);path=pf.parent/'prices.csv'
        if pm.get('physical_case_hash')!=case.physical_case_hash or sha256_file(path)!=pm['prices_sha256']:raise ValueError('Price freeze mismatch.')
        optional_price=pd.read_csv(path);optional_price.index=pd.to_datetime(optional_price.timestamp_utc,utc=True)
    mainrows=[];certrows=[];matchedrows=[];sources={}
    definitions=[(0.15,'annual','M1_LP',baseline),(0.15,'annual','M2',baseline),(0.15,'rolling','M1',baseline),(0.15,'rolling','M2',baseline),
                 (0.15,'annual','M3',m3),(0.15,'rolling','M3',m3),
                 (0.03,'annual','M2',m3),(0.03,'annual','M3',m3),(0.03,'annual','M2_MATCHED',m3),
                 (0.03,'rolling','M2',m3),(0.03,'rolling','M3',m3)]
    scalar=('S_mw_h_row','E_restricted_mwh','stage1_optimum_mw_h_row','stage1_lower_bound','physical_fde_sum_h','touched_duration_sum_h','maximum_overload_mw','insecure_intervals','violated_row_state_pairs','affected_participants','wall_seconds','stage1_seconds','stage2_seconds','closure_rounds','active_row_count','model_rows','model_columns','model_nonzeros','peak_private_gb')
    for year in (2029,2030,2031):
        yc=year_case(case,year);records={}
        for rho,mode,rule,root in definitions:
            folder=root/f'year_{year}/rho_{rho:.2f}/{mode}/{rule}';print(f'REPLAY REPORT {year} {rho} {mode} {rule}',flush=True)
            metric,x,sol=annual(folder,yc) if mode=='annual' else rolling(folder,yc)
            value={k:metric.get(k) for k in scalar};value.update(year=year,rho=rho,mode=mode,rule=rule,replay_passed=True)
            value['price_weighted_burden_eur']=None if optional_price is None else float(np.sum(yc.time_grid.delta[:,None]*x*optional_price.loc[yc.time_grid.index,'price_eur_mwh'].to_numpy()[:,None]))
            mainrows.append(value);records[(rho,mode,rule)]=(metric,sol)
            for p in folder.glob('*.json'):sources[str(p.relative_to(ROOT))]=sha256_file(p)
        m2=records[(0.15,'annual','M2')][0];lp=records[(0.15,'annual','M1_LP')][1];roll=records[(0.15,'rolling','M1')][0]
        lower=lp.stage1.best_bound
        if lower is None:raise RuntimeError('M1-LP lower bound is absent.')
        certificate_path=ROOT/f'results_m1_m2_final/r10_certificates/year_{year}/B03_M1_M2_CERTIFICATE.json'
        if certificate_path != ROOT/b03_records[year]['path']:
            raise RuntimeError(f'Corrected B03 certificate registry path changed for {year}.')
        certificate=read(certificate_path)
        if certificate.get('status')!='PASS' or certificate.get('scientific_fingerprint')!=scientific_fingerprint():
            raise RuntimeError(f'Corrected B03 certificate is absent or not PASS: {certificate_path}')
        if certificate.get('source_m2_solution_hash')!=records[(0.15,'annual','M2')][1].solution_hash:
            raise RuntimeError(f'Corrected B03 certificate source changed for {year}.')
        if abs(float(certificate['m1lp_certified_lower_bound'])-float(lower))>0.0:
            raise RuntimeError(f'Corrected B03 lower bound changed for {year}.')
        if lower>roll['S_mw_h_row']+2e-6:raise RuntimeError('Annual M1 bracket is inconsistent.')
        certrows.append({'year':year,'selected_S_M2':certificate['selected_replay_S_mw_h_row'],
                        'witness_S_M2':certificate['witness_S_mw_h_row'],'M1_LP_certified_lower_bound':lower,
                        'S_rolling_M1':roll['S_mw_h_row'],'strict_certificate_passed':certificate['status']=='PASS',
                        'certificate_epsilon':certificate['certificate_epsilon'],
                        'guaranteed_margin_mw_h_row':certificate['guaranteed_margin_mw_h_row'],
                        'witness_hash':certificate['witness_hash'],'certificate_sha256':sha256_file(certificate_path)})
        s2=records[(0.03,'annual','M2')][0]['S_mw_h_row'];s3=records[(0.03,'annual','M3')][0]['S_mw_h_row'];sm=records[(0.03,'annual','M2_MATCHED')][1].stage1.objective_value
        decomposition=frozen_matched_decomposition(s2,s3,sm)
        matchedrows.append({'year':year,'S_M2':s2,'S_M3':s3,'S_M2_MATCHED':sm,
                           'exposure_effect':decomposition['exposure_allowance_effect'],
                           'state_shape_penalty':decomposition['shape_penalty'],
                           'raw_shape_residual':decomposition['raw_shape_residual'],
                           'comparison_tolerance':decomposition['comparison_tolerance'],
                           'realized_M3_gain':decomposition['realized_m3_gain'],
                           'shape_penalty_nonnegative':decomposition['shape_penalty']>=0.0,
                           'comparison_semantics':'M2/M3 replayed selected trajectories; matched Stage-I optimum; frozen absolute-plus-relative numerical envelope.'})
    args.output.mkdir(parents=True,exist_ok=True)
    tables={'MAIN_POLICY_RESULTS.csv':pd.DataFrame(mainrows),'M1_CERTIFICATE_RESULTS.csv':pd.DataFrame(certrows),'M3_MATCHED_DECOMPOSITION.csv':pd.DataFrame(matchedrows)}
    for name,df in tables.items():atomic_write_csv(args.output/name,df)
    dc=status_if_valid(
        ROOT/'results_engineering_validation/independent_dc/STATUS.json',
        compatibility_use='UPSTREAM_STAGE06_DC_EVIDENCE',
    );ac=status_if_valid(
        ROOT/'results_engineering_validation/selected_ac/STATUS.json',
        compatibility_use='UPSTREAM_STAGE06_AC_EVIDENCE',
    )
    controls=status_if_valid(ROOT/'results_m3_controls/M3_CONTROL_STATUS.json',compatibility_use='UPSTREAM_STAGE05B_CONTROL_EVIDENCE')
    gamma=status_if_valid(
        ROOT/'results_m3_sensitivity_r10/gamma/M3_GAMMA_STATUS.json',
        compatibility_use='UPSTREAM_STAGE05C_GAMMA_EVIDENCE',
    )
    bounds=status_if_valid(
        ROOT/'results_m3_sensitivity_r10/rate_bounds/RATE_BOUND_STATUS.json',
        compatibility_use='UPSTREAM_STAGE05C_RATE_BOUND_EVIDENCE',
    )
    if gamma.get('status')!='COMPLETE' or bounds.get('status')!='COMPLETE':
        raise RuntimeError('Required post-freeze sensitivity evidence is incomplete.')
    if (gamma.get('completed_case_count')!=gamma.get('expected_case_count') or gamma.get('heavy_solve_count')!=0
        or gamma.get('all_cases_action')!='REUSE_EXACT_EQUIVALENT_CORE' or bounds.get('rows')!=bounds.get('expected_rows')
        or bounds.get('heavy_solve_count')!=0 or bounds.get('all_cases_action')!='REUSE_EXACT_EQUIVALENT_CORE'):
        raise RuntimeError('R10 exact-equivalent Stage-05C evidence is incomplete or not exact-reuse certified.')
    dc_acceptance=dc.get('acceptance',{})
    ac_summary=ac.get('summary',{})
    dc_gate_passed=(dc.get('status')=='PASS' and dc_acceptance.get('passed') is True
        and dc_acceptance.get('per_row_tolerance_formula')=='max(1e-4 MW, 1e-8*rating_mw)'
        and dc_acceptance.get('flow_sign_materiality_mw')==1e-6
        and dc_acceptance.get('required_material_sign_agreement')==1.0
        and dc_acceptance.get('material_sign_agreement_share')==1.0
        and dc_acceptance.get('false_safe_count_beyond_tolerance')==0
        and dc_acceptance.get('false_overload_count_beyond_tolerance')==0)
    ac_ordering_passed=(ac.get('status')=='PASS' and ac.get('execution_complete') is True
        and ac_summary.get('ordering_tie_tolerance')==1e-4
        and ac_summary.get('complete_selected_order_preservation') is True)
    claims={'m1_m2_strict_certificate':{'passed':all(r['strict_certificate_passed'] for r in certrows)},
            'm1_m2_certificate_source':{'passed':True,'legacy_boolean_authorizes_claim':False,'required_schema':'TPWRS_B03_CONSERVATIVE_M2_WITNESS_CERTIFICATE_V1'},
            'm3_matched_exposure':{'passed':all(r['shape_penalty_nonnegative'] for r in matchedrows)},
            'independent_dc':{'passed':dc_gate_passed},
            'selected_ac_completed':{'passed':ac.get('status')=='PASS' and ac.get('execution_complete') is True},
            'selected_ac_order_preservation':{'passed':ac_ordering_passed},
            'annual_ac_feasibility':{'passed':False,'allowed':False},'price_burden':{'passed':optional_price is not None},
            'm3_ordering_specificity_controls':{'passed':controls.get('status') in ('COMPLETE','PASS')},
            'm3_gamma_sensitivity_completed':{'passed':gamma.get('status')=='COMPLETE'},
            'm3_rate_bound_sensitivity_completed':{'passed':bounds.get('status')=='COMPLETE'}}
    atomic_write_json(args.output/'CLAIM_GATE.json',claims)
    manifest={'schema':'TPWRS_VERIFIED_EVIDENCE_V2','status':'COMPLETE','created_utc':utc_now(),
              'physical_case_hash':case.physical_case_hash,'scientific_fingerprint':scientific_fingerprint(),
              'claims':claims,'independent_dc':dc,'selected_ac':ac,'gamma_sensitivity':gamma,'rate_bound_sensitivity':bounds,
              'input_hashes':sources,'outputs':{name:{'path':name,'sha256':sha256_file(args.output/name)} for name in tables},
              'submission_ready':False,'note':'Evidence extraction is not a manuscript sign-off; B03/DC/AC/R10 sensitivity claim gates are enforced.'}
    atomic_write_json(args.output/'FINAL_EVIDENCE_MANIFEST.json',manifest)
    export_review_bundle(ROOT,args.output)
    print(f"EVIDENCE_EXPORT=PASS output={args.output}",flush=True)
    return 0


def export_review_bundle(root,output):
    """Collect verified result tables/statuses for review, not draft conclusions."""
    import zipfile
    root=Path(root);output=Path(output)
    paths=[]
    folders=(output,root/"results_m1_m2_final",root/"results_m3_final",
        root/"results_m3_controls",root/"results_m3_sensitivity_r10",
        root/"results_m1_m2_final/r10_certificates",
        root/"results_engineering_validation",root/"results_2028_performance_regression",
        root/"results_numerical_reproduction",root/"artifacts/performance",root/"artifacts/cards")
    raw_references={}
    for folder in folders:
        if folder.is_dir():
            for p in folder.rglob("*"):
                if not p.is_file() or p.name.startswith('.') or '__pycache__' in p.parts:continue
                if p.suffix in {".csv",".json"}:paths.append(p)
                elif p.suffix in {".npz",".parquet",".joblib",".log"}:
                    raw_references[p.relative_to(root).as_posix()]={"sha256":sha256_file(p),"bytes":p.stat().st_size}
    for relative in ("RELEASE_MANIFEST.json","BUILD_VALIDATION.json","artifacts/FINAL_PHYSICAL_READINESS.json","artifacts/features/FEATURE_GATE.json","artifacts/features/F0_SCHEMA.json"):
        if (root/relative).is_file():paths.append(root/relative)
    atomic_write_json(output/"RAW_OUTPUT_REFERENCES.json",raw_references)
    paths.append(output/"RAW_OUTPUT_REFERENCES.json")
    paths=sorted(set(paths))
    manifest={p.relative_to(root).as_posix():sha256_file(p) for p in paths}
    target=output/"RESULTS_FOR_REVIEW.zip"
    temporary=target.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED) as archive:
            for path in paths:archive.write(path,path.relative_to(root).as_posix())
            archive.writestr("BUNDLE_MANIFEST.json",json.dumps({"sha256":manifest,
                "scope":"RESULT_EVIDENCE_ONLY_NO_EXPERIMENTAL_PROSE","submission_ready":False},indent=2))
        import os
        os.replace(temporary,target)
    finally:
        if temporary.exists():temporary.unlink()
    return target
