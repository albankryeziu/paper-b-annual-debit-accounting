from __future__ import annotations


# ============================================================================
# scripts.run_2028_performance_regression (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Bounded, visible validation-year qualification; never evaluates held-out years."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from tpwrs.annual import _screen_geometry, load_closed_row_seed, build_geometry_audit
from tpwrs.contracts import nominal_experiment
from tpwrs.execution import CampaignLock, utc_now, validate_policy_solution
from tpwrs.storage import sha256_file, sha256_json
from tpwrs.persistence import load_policy_solution, save_policy_solution
from tpwrs.storage import atomic_write_json
from tpwrs.persistence import load_physical_case
from tpwrs.model import solve_policy
from tpwrs.config import scientific_fingerprint
from tpwrs.execution import annual_solver_fingerprint, validation_solver_profile
from tpwrs.execution import memory_snapshot
from jobs.common import frozen_matched_decomposition, solution_metrics


def profile(seconds=3600):
    return validation_solver_profile(seconds)


def result_root_for_fingerprint(output: Path, fingerprint: str) -> Path:
    return output/'solver_fingerprints'/fingerprint


def accepted(root,case,exp):
    p=root/'STATUS.json'
    if not p.is_file(): return None
    meta=json.loads(p.read_text(encoding='utf-8'))
    if meta.get('status')!='PASS': return None
    if meta.get('experiment_hash')!=exp.hash:
        raise RuntimeError(f'Accepted cache belongs to different inputs/code: {root}. Preserve it; choose a new result root.')
    for name,checksum in meta['output_hashes'].items():
        if sha256_file(root/name)!=checksum: raise RuntimeError(f'Accepted output changed: {root/name}')
    sol=load_policy_solution(root,expected_experiment_hash=exp.hash,expected_physical_case_hash=case.physical_case_hash)
    validate_policy_solution(case,exp,sol,require_stage3=False)
    return sol


def run_method(case,rule,folder,fingerprint,*,generic=False,prior_rows=None,seconds=3600):
    folder.mkdir(parents=True,exist_ok=True)
    exp=nominal_experiment(case,case_id=f'QUALIFY2028_{folder.name}_{rule}',rule=rule,rho=0.15,
        solver_profile=profile(seconds),implementation_hash=fingerprint,result_root_id=str(folder.relative_to(ROOT)))
    old=accepted(folder,case,exp)
    if old is not None:
        print(f'REUSE {folder}: accepted hash-verified {rule}',flush=True);return old,exp
    began=time.monotonic();old_job=os.environ.get('TPWRS_CURRENT_JOB_ROOT')
    os.environ['TPWRS_CURRENT_JOB_ROOT']=str(folder)
    status={'status':'RUNNING','started_utc':utc_now(),'rule':rule,'generic_reference':generic,
            'experiment_hash':exp.hash,'physical_case_hash':case.physical_case_hash}
    atomic_write_json(folder/'STATUS.json',status);atomic_write_json(folder/'EXPERIMENT_CASE.json',asdict(exp))
    try:
        print(f'\nMETHOD={rule} REFERENCE={generic} HOURS={case.n_time} OUTPUT={folder}',flush=True)
        sol=solve_policy(case,exp,use_stage3=False,force_generic=generic,
            checkpoint_dir=folder,initial_directed_rows=prior_rows)
        replay=validate_policy_solution(case,exp,sol,require_stage3=False)
        save_policy_solution(sol,folder)
        metrics=solution_metrics(case,sol)
        metrics.update(wall_seconds=time.monotonic()-began,**memory_snapshot())
        atomic_write_json(folder/'metrics.json',metrics)
        atomic_write_json(folder/'STAGE1_CERTIFICATE.json',asdict(sol.stage1))
        atomic_write_json(folder/'STAGE2_CERTIFICATE.json',asdict(sol.stage2))
        atomic_write_json(folder/'REPLAY_STATUS.json',replay)
        outputs=('solution.json','solution_arrays.npz','metrics.json','STAGE1_CERTIFICATE.json','STAGE2_CERTIFICATE.json','REPLAY_STATUS.json')
        # Use the actual array filename from the serializer, not an assumed name.
        serialized=json.loads((folder/'solution.json').read_text(encoding='utf-8'))
        outputs=tuple(serialized['arrays_file'] if n=='solution_arrays.npz' else n for n in outputs)
        status.update(status='PASS',completed_utc=utc_now(),wall_seconds=time.monotonic()-began,
                      output_hashes={n:sha256_file(folder/n) for n in outputs})
        atomic_write_json(folder/'STATUS.json',status);return sol,exp
    except BaseException as exc:
        status.update(status='INCOMPLETE',ended_utc=utc_now(),wall_seconds=time.monotonic()-began,error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(folder/'STATUS.json',status);raise
    finally:
        if old_job is None: os.environ.pop('TPWRS_CURRENT_JOB_ROOT',None)
        else: os.environ['TPWRS_CURRENT_JOB_ROOT']=old_job


def compare(a,b,exp):
    s_tol=exp.stage1_tolerance(max(abs(a.stage1.objective_value),abs(b.stage1.objective_value)))
    e_tol=exp.stage2_tolerance(max(abs(a.stage2.objective_value),abs(b.stage2.objective_value)))
    ds=abs(a.stage1.objective_value-b.stage1.objective_value);de=abs(a.stage2.objective_value-b.stage2.objective_value)
    return {'status':'PASS' if ds<=s_tol and de<=e_tol else 'FAIL',
        'S_stage1_abs_difference':ds,'E_stage2_abs_difference':de,'S_tolerance':s_tol,'E_tolerance':e_tol,
        'x_max_abs_difference_mw':float(np.max(np.abs(a.x_mw-b.x_mw))),
        'x_equality_required':False,'reason':'Stage-II optima need not be unique.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,default=ROOT/'artifacts/real_case/physical_case/FINAL_DUTCH_CASE')
    parser.add_argument('--output',type=Path,default=ROOT/'results_2028_performance_regression')
    parser.add_argument('--week-only',action='store_true',help='Stop after A/B qualification; never report full-year readiness.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    release_fp=scientific_fingerprint(); fp=annual_solver_fingerprint(ROOT); started=time.monotonic()
    run_root=result_root_for_fingerprint(args.output,fp); run_root.mkdir(parents=True,exist_ok=True)
    state={'status':'RUNNING','created_utc':utc_now(),'year':2028,'scientific_fingerprint':release_fp,
           'annual_solver_fingerprint':fp,'active_result_root':str(run_root.relative_to(ROOT)),
           'completed_methods':[],'full_year_completed':False,'scientific_optimization_executed':True}
    status=args.output/'STATUS.json'
    try:
        with CampaignLock(args.output/'.campaign.lock',fp):
            atomic_write_json(status,state)
            print('Loading immutable case; selecting 2028 only.',flush=True)
            from tpwrs.persistence import load_fitting_case
            parent,partition=load_fitting_case(args.case)
            parent_identity=partition['physical_case_hash']
            indexes=np.flatnonzero(parent.time_grid.index.year==2028)
            if len(indexes)!=8784: raise RuntimeError('Expected complete leap-year 2028: 8,784 hours.')
            case=parent.slice_time(int(indexes[0]),int(indexes[-1])+1);del parent
            week=case.slice_time(0,168)
            state['physical_case_hash']=case.physical_case_hash
            for rule in ('M1_LP','M2'):
                state.update(phase='WEEK_AB',method=rule);atomic_write_json(status,state)
                ref,exp=run_method(week,rule,run_root/'qualification'/f'{rule}_reference',fp,generic=True,seconds=600)
                new,_=run_method(week,rule,run_root/'qualification'/f'{rule}_support_oa',fp,seconds=600)
                agreement=compare(ref,new,exp)
                atomic_write_json(run_root/'qualification'/f'{rule}_AGREEMENT.json',agreement)
                if agreement['status']!='PASS': raise RuntimeError(f'{rule}: week A/B objectives disagree; no annual solve launched.')
            if args.week_only:
                state.update(status='PASS_WEEK_ONLY',completed_utc=utc_now());atomic_write_json(status,state);return 0
            solutions={}; prior=None
            for rule in ('M1_LP','M2'):
                state.update(phase='FULL_YEAR',method=rule);atomic_write_json(status,state)
                exp0=nominal_experiment(case,case_id='GEOMETRY',rule=rule,rho=0.15,solver_profile=profile())
                geo=_screen_geometry(case,exp0)
                if rule=='M2':
                    prior=load_closed_row_seed(run_root/'M1_LP',geometry_hash=geo.geometry_hash,
                        n_rows=case.n_flat_rows,n_hours=len(geo.active_hours))
                sol,exp=run_method(case,rule,run_root/rule,fp,prior_rows=prior)
                solutions[rule]=sol;state['completed_methods'].append(rule);atomic_write_json(status,state)
            m1=solutions['M1_LP'];m2=solutions['M2']
            tolerance=max(2e-6,exp.stage1_tolerance(max(abs(m1.stage1.objective_value),abs(m2.stage1.objective_value))))
            hierarchy=m2.stage1.objective_value<=m1.stage1.objective_value+tolerance
            if not hierarchy: raise RuntimeError('2028 M2/M1-LP Stage-I hierarchy failed.')
            selected={'status':'PASS','profile':asdict(profile()),'scientific_fingerprint':release_fp,'annual_solver_fingerprint':fp,
                'physical_case_hash':case.physical_case_hash,'parent_physical_case_hash':parent_identity,'year':2028,
                'basis':'One qualified structural profile; not a broad solver-parameter speed tournament.',
                'full_year_reference_completed':False,'speedup_claim':None}
            atomic_write_json(ROOT/'artifacts/performance/SELECTED_SOLVER_PROFILE.json',selected)
            state.update(status='PASS',full_year_completed=True,completed_utc=utc_now(),elapsed_s=time.monotonic()-started,
                hierarchy_passed=True,comparison_tolerance=tolerance,selected_profile_sha256=sha256_file(ROOT/'artifacts/performance/SELECTED_SOLVER_PROFILE.json'))
            atomic_write_json(status,state);print('2028_PERFORMANCE_GATE=PASS',flush=True);return 0
    except BaseException as exc:
        state.update(status='INCOMPLETE',ended_utc=utc_now(),elapsed_s=time.monotonic()-started,error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(status,state);traceback.print_exc();print('2028_PERFORMANCE_GATE=INCOMPLETE; no held-out authorization.',flush=True);return 1


# ============================================================================
# scripts.run_real_smoke (consolidated; see SOURCE_MAP.json)
# ============================================================================

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tpwrs.contracts import nominal_experiment
from tpwrs.contracts import AllowanceSpec, ExperimentCase, SolverProfile
from tpwrs.persistence import load_card
from tpwrs.storage import sha256_array, sha256_file
from tpwrs.persistence import load_policy_solution, save_policy_solution
from tpwrs.storage import atomic_write_json
from tpwrs.persistence import load_physical_case
from tpwrs.cards import build_teacher_rows, matched_allowance, q_from_card, verify_m2_matched_contains_m3
from tpwrs.model import solve_policy
from tpwrs.rolling import solve_rolling


def smoke_main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--case",type=Path,default=ROOT/"artifacts/real_case/physical_case/FINAL_DUTCH_CASE"); parser.add_argument("--cards",type=Path,default=ROOT/"artifacts/cards"); parser.add_argument("--features",type=Path,default=ROOT/"artifacts/features"); parser.add_argument("--dc-audit",type=Path,default=ROOT/"artifacts/real_case/network/REFERENCE_2018_AUDIT.json"); parser.add_argument("--output",type=Path,default=ROOT/"artifacts/smoke"); args=parser.parse_args(); case=load_physical_case(args.case); idx=case.time_grid.index; year_rows=np.flatnonzero(idx.year==2028); year_case=case.slice_time(int(year_rows[0]),int(year_rows[-1])+1); smoke=year_case.slice_time(0,48); annual_allowance=AllowanceSpec.from_nominal(year_case.participant_ids,year_case.contract_mask,year_case.time_grid,0.15); gp=SolverProfile(name="real_smoke_gurobi",backend="gurobi",time_limit_s=1800,mip_gap=1e-8,feasibility_tol=1e-8,presolve=True,output_flag=False)
    card=load_card(args.cards/"RHO15_SELECTED_CARD.json"); card_freeze=__import__("json").loads((args.cards/"RHO15_CARD_FREEZE.json").read_text(encoding="utf-8")); freeze_metadata=card_freeze["freeze_metadata"]
    if card.allowance_hash!=freeze_metadata["allowance_contract_hash"] or freeze_metadata["annual_allowance_hashes"]["2028"]!=annual_allowance.hash: raise RuntimeError("Smoke card does not match the frozen 2028 annual allowance specification.")
    model_path=args.cards/f"rho_0.15/MODEL_{card.model_family}.joblib"
    if sha256_file(model_path)!=card.source_hashes.get("model_bundle"): raise RuntimeError("Smoke model bundle hash differs from the frozen card.")
    feature_path=args.features/"F0_2028.parquet"; feature_gate=__import__("json").loads((args.features/"FEATURE_GATE.json").read_text(encoding="utf-8"))
    if feature_gate.get("status")!="PASS" or sha256_file(feature_path)!=feature_gate["outputs"]["2028"]["sha256"]: raise RuntimeError("Smoke F0 feature hash mismatch.")
    bundle=joblib.load(model_path); feature=pd.read_parquet(feature_path); selected_times=set(smoke.time_grid.timestamps_utc); feature["timestamp_utc"]=pd.to_datetime(feature.timestamp_utc,utc=True).map(lambda x:x.isoformat()); feature=feature.loc[feature.timestamp_utc.isin(selected_times)].copy(); expected=pd.MultiIndex.from_product([smoke.time_grid.timestamps_utc,smoke.participant_ids]); actual=pd.MultiIndex.from_frame(feature[["timestamp_utc","participant_id"]]);
    if not actual.equals(expected): raise RuntimeError("Smoke feature rows are not exact time-major case coordinates.")
    scores=bundle.predict(feature).reshape(smoke.n_time,smoke.n_participants); q_card=q_from_card(card,smoke.participant_directions,scores)
    experiments={
        "M2":ExperimentCase("REAL_SMOKE_M2",smoke.physical_case_id,smoke.physical_case_hash,smoke.time_grid.grid_id,"M2",annual_allowance,solver_profile=gp,result_root_id="artifacts/smoke/solutions/M2"),
        "M1_LP":ExperimentCase("REAL_SMOKE_M1LP",smoke.physical_case_id,smoke.physical_case_hash,smoke.time_grid.grid_id,"M1_LP",annual_allowance,solver_profile=gp,result_root_id="artifacts/smoke/solutions/M1_LP"),
        "M3":ExperimentCase("REAL_SMOKE_M3",smoke.physical_case_id,smoke.physical_case_hash,smoke.time_grid.grid_id,"M3",annual_allowance,card_id=card.card_id,card_hash=card.hash,feature_set_id=card.feature_set_id,information_level="FROZEN_F0",solver_profile=gp,result_root_id="artifacts/smoke/solutions/M3",rate_matrix_hash=sha256_array(q_card)),
    }
    solutions={"M2":solve_policy(smoke,experiments["M2"],use_stage3=False),"M1_LP":solve_policy(smoke,experiments["M1_LP"],use_stage3=False)}
    solutions["M3"]=solve_policy(smoke,experiments["M3"],q=q_card,use_stage3=False)
    teacher_experiment=ExperimentCase("REAL_SMOKE_M2_TEACHER_STAGE2",smoke.physical_case_id,smoke.physical_case_hash,smoke.time_grid.grid_id,"M2",annual_allowance,solver_profile=gp,information_level="M2_EXACT_UNPERTURBED_STAGE_II_PRIMAL_DUAL_TEACHER",result_root_id="artifacts/smoke/solutions/M2_TEACHER_STAGE2")
    teacher_solution=solve_policy(smoke,teacher_experiment,use_stage3=False)
    matched=matched_allowance(smoke,solutions["M3"]); matched_exp=ExperimentCase("REAL_SMOKE_M2_MATCHED",smoke.physical_case_id,smoke.physical_case_hash,smoke.time_grid.grid_id,"M2_MATCHED",matched,"annual",card.card_id,card.hash,card.feature_set_id,"POST_M3_MATCHED_EXPOSURE",solver_profile=gp,result_root_id="artifacts/smoke/solutions/M2_MATCHED"); solutions["M2_MATCHED"]=solve_policy(smoke,matched_exp,use_stage3=False); decomposition=frozen_matched_decomposition(solutions["M2"].s_value_mw_h_row,solutions["M3"].s_value_mw_h_row,solutions["M2_MATCHED"].stage1.objective_value)
    teacher=build_teacher_rows(smoke,teacher_solution,year=2028)
    restart_ok=True
    for key,value in {**solutions,"M2_TEACHER_STAGE2":teacher_solution}.items():
        path=args.output/"solutions"/key; save_policy_solution(value,path); restart_ok=restart_ok and load_policy_solution(path,expected_experiment_hash=value.experiment_hash,expected_physical_case_hash=smoke.physical_case_hash).solution_hash==value.solution_hash
    m1e=ExperimentCase("REAL_SMOKE_ROLL_M1",smoke.physical_case_id,smoke.physical_case_hash,smoke.time_grid.grid_id,"M1",annual_allowance,"rolling",information_level="FROZEN_INPUT",lookahead_hours=48,implementation_hours=24,solver_profile=gp,result_root_id="artifacts/smoke/rolling/M1"); m2e=ExperimentCase("REAL_SMOKE_ROLL_M2",smoke.physical_case_id,smoke.physical_case_hash,smoke.time_grid.grid_id,"M2",annual_allowance,"rolling",information_level="FROZEN_INPUT",lookahead_hours=48,implementation_hours=24,solver_profile=gp,result_root_id="artifacts/smoke/rolling/M2"); m3e=ExperimentCase("REAL_SMOKE_ROLL_M3",smoke.physical_case_id,smoke.physical_case_hash,smoke.time_grid.grid_id,"M3",annual_allowance,"rolling",card.card_id,card.hash,card.feature_set_id,"FROZEN_F0",48,24,gp,result_root_id="artifacts/smoke/rolling/M3",rate_matrix_hash=sha256_array(q_card)); roll1=solve_rolling(smoke,m1e,use_stage3=False,checkpoint_root=args.output/"rolling/M1"); roll2=solve_rolling(smoke,m2e,use_stage3=False,checkpoint_root=args.output/"rolling/M2"); roll3=solve_rolling(smoke,m3e,q=q_card,use_stage3=False,checkpoint_root=args.output/"rolling/M3"); roll2_resumed=solve_rolling(smoke,m2e,use_stage3=False,checkpoint_root=args.output/"rolling/M2")
    tiny=smoke.slice_time(0,12); gurobi_e=nominal_experiment(tiny,case_id="REAL_PARITY_GUROBI",rule="M2",rho=0.15,solver_profile=gp); scipy_e=nominal_experiment(tiny,case_id="REAL_PARITY_SCIPY",rule="M2",rho=0.15,solver_profile=SolverProfile(name="real_smoke_scipy",backend="scipy",feasibility_tol=1e-8,presolve=True)); gs=solve_policy(tiny,gurobi_e,use_stage3=False); ss=solve_policy(tiny,scipy_e,use_stage3=False)
    def rolling_replay(result,experiment,q=None):
        dt=smoke.time_grid.delta[:,None]; x=result.x_mw; debit=np.sum(dt*result.u,axis=0) if experiment.rule=="M1" else np.sum(dt*(q if q is not None else 1.0)*x/smoke.capacities_mw[None,:],axis=0); return bool(result.complete_coverage and all(s.replay_summary.get("passed") for s in result.epoch_solutions) and np.min(result.final_balance_hours)>=-1e-7 and np.max(debit-experiment.allowance.values)<=2e-6 and np.max(np.abs(smoke.post_state_mw(x).sum(axis=1)))<=1e-7)
    dc_audit=__import__("json").loads(args.dc_audit.read_text(encoding="utf-8")); checks={
        "all_replays_pass":all(bool(x.replay_summary.get("passed")) for x in solutions.values()),
        "all_annual_stages_optimal":all(x.stage1.status=="OPTIMAL" and x.stage2.status=="OPTIMAL" and (x.stage3 is None or x.stage3.status=="OPTIMAL") for x in solutions.values()),
        "same_scale_annual_allowance_exact":all(experiments[key].allowance.hash==annual_allowance.hash and solutions[key].allowance_hash==annual_allowance.hash for key in ("M1_LP","M2","M3")),
        "m2_teacher_has_unperturbed_stage2_primal_duals":teacher_solution.stage3 is None and teacher_solution.stage1.status=="OPTIMAL" and teacher_solution.stage2.status=="OPTIMAL" and len(teacher_solution.m2_budget_duals)==smoke.n_participants and len(teacher)>0,
        "m2_superset_m1lp_stage1":solutions["M2"].s_value_mw_h_row<=solutions["M1_LP"].s_value_mw_h_row+1e-5,
        "m2_matched_contains_m3":verify_m2_matched_contains_m3(smoke,solutions["M3"],matched),
        "matched_shape_penalty_nonnegative":decomposition["shape_penalty"]>=0.0,
        "rolling_m1_complete_replay":rolling_replay(roll1,m1e),
        "rolling_m2_complete_replay":rolling_replay(roll2,m2e),
        "rolling_m3_complete_replay":rolling_replay(roll3,m3e,q_card),
        "gurobi_scipy_S_parity":abs(gs.s_value_mw_h_row-ss.s_value_mw_h_row)<=1e-5,
        "gurobi_scipy_E_parity":abs(gs.e_value_mwh-ss.e_value_mwh)<=1e-4,
        "card_hash_exact":experiments["M3"].card_hash==card.hash,
        "solution_restart_hashes":restart_ok,
        "rolling_checkpoint_restart_hashes":np.array_equal(roll2.x_mw,roll2_resumed.x_mw) and [x.solution_hash for x in roll2.epoch_solutions]==[x.solution_hash for x in roll2_resumed.epoch_solutions],
        "independent_2018_dc_mapping_pass":dc_audit.get("status")=="PASS",
    }
    payload={"schema":"TPWRS_REAL_SMOKE_STATUS_V1","status":"PASS" if all(checks.values()) else "FAIL","created_utc":datetime.now(timezone.utc).isoformat(),"physical_case_id":case.physical_case_id,"physical_case_hash":case.physical_case_hash,"card_id":card.card_id,"card_hash":card.hash,"window":{"start":smoke.time_grid.timestamps_utc[0],"hours":48,"participants":smoke.n_participants,"flat_rows":smoke.n_flat_rows},"checks":checks,"objectives":{key:{"S":value.s_value_mw_h_row,"E":value.e_value_mwh,"closure_rounds":value.closure_rounds,"active_rows":value.active_row_count,"stage1":value.stage1.status,"stage2":value.stage2.status,"stage3":None if value.stage3 is None else value.stage3.status} for key,value in solutions.items()},"matched_decomposition":decomposition,"parity":{"gurobi":{"S":gs.s_value_mw_h_row,"E":gs.e_value_mwh},"scipy":{"S":ss.s_value_mw_h_row,"E":ss.e_value_mwh}},"full_campaign_executed":False}; atomic_write_json(args.output/"REAL_SMOKE_STATUS.json",payload); atomic_write_json(ROOT/"artifacts/SMOKE_STATUS.json",payload); print(payload); return 0 if payload["status"]=="PASS" else 2


def _read_frozen_experiment(path):
    from tpwrs.contracts import AllowanceSpec, ExperimentCase, SolverProfile
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    fields=dict(obj.get("experiment",obj))
    allowance=dict(fields["allowance"])
    allowance["participant_ids"]=tuple(allowance["participant_ids"])
    allowance["values_hours"]=tuple(allowance["values_hours"])
    fields["allowance"]=AllowanceSpec(**allowance)
    fields["solver_profile"]=SolverProfile(**fields["solver_profile"])
    experiment=ExperimentCase(**fields)
    if obj.get("experiment_hash",experiment.hash)!=experiment.hash:
        raise RuntimeError("Frozen historical experiment hash is inconsistent.")
    return experiment


def _reconstruct_historical_epoch(case, parent, epoch_root, target):
    """Read-only replay of an old ledger. This does NOT migrate accepted policies."""
    from tpwrs.contracts import AllowanceSpec, ExperimentCase
    from tpwrs.replay import replay_solution
    from tpwrs.rolling import _hours_to_intervals
    if parent.mode!="rolling" or parent.rule!="M1":
        raise RuntimeError("Historical reproducer supports the reported rolling M1 failure only.")
    case.verify_experiment_identity(parent.physical_case_id,parent.physical_case_hash,parent.time_grid_id)
    impl=_hours_to_intervals(case.time_grid,parent.implementation_hours)
    look=_hours_to_intervals(case.time_grid,parent.lookahead_hours)
    if target<0 or target>=case.n_time or target%impl:
        raise ValueError("Historical target must be an implemented-block boundary.")
    balance=parent.allowance.values.copy()
    previous=None; lineage=[]
    for start in range(0,target+1,impl):
        subcase=case.slice_time(start,min(start+look,case.n_time))
        allowance=AllowanceSpec.explicit(case.participant_ids,balance,source="explicit",nominal_rho=parent.allowance.nominal_rho)
        subexp=ExperimentCase(case_id=f"{parent.case_id}__epoch_{start:06d}",
            physical_case_id=subcase.physical_case_id,physical_case_hash=subcase.physical_case_hash,
            time_grid_id=subcase.time_grid.grid_id,rule=parent.rule,allowance=allowance,
            mode="annual",card_id=parent.card_id,card_hash=parent.card_hash,
            feature_set_id=parent.feature_set_id,information_level=parent.information_level,
            solver_profile=parent.solver_profile,epsilon_s_abs=parent.epsilon_s_abs,
            epsilon_s_rel=parent.epsilon_s_rel,epsilon_e_abs=parent.epsilon_e_abs,
            epsilon_e_rel=parent.epsilon_e_rel,source_root_id=parent.source_root_id,
            result_root_id=f"{parent.result_root_id}/epochs/epoch_{start:06d}",
            implementation_hash=parent.implementation_hash)
        if start==target:
            return subcase,subexp,previous,lineage
        folder=Path(epoch_root)/f"epoch_{start:06d}"
        sol=load_policy_solution(folder,expected_experiment_hash=subexp.hash,
                                 expected_physical_case_hash=subcase.physical_case_hash)
        report=replay_solution(subcase,subexp,sol,raise_on_failure=True)
        if sol.stage1.status!="OPTIMAL" or sol.stage2.status!="OPTIMAL" or sol.stage3 is not None:
            raise RuntimeError("Historical epoch lacks its declared Stage-I/II status.")
        # Validate the old contract AS OLD, never silently relabel it current.
        labels=[c.metadata.get("face_numeric_contract") for c in (sol.stage1,sol.stage2)]
        if labels != ["GENERIC_EXPLICIT_FACE_V1"]*2:
            raise RuntimeError("Historical epoch is not an R9 explicit generic certificate.")
        old_numeric=max(1e-6,100*parent.solver_profile.feasibility_tol)
        for certificate in (sol.stage1,sol.stage2):
            if certificate.metadata.get("generic_face_certification_tolerance")!=old_numeric:
                raise RuntimeError("Historical numerical certificate has been altered.")
        if report.s_value_mw_h_row>sol.stage1.objective_value+subexp.stage1_tolerance(sol.stage1.objective_value)+old_numeric:
            raise RuntimeError("Historical epoch fails its own original Stage-I face.")
        if report.e_value_mwh>sol.stage2.objective_value+subexp.stage2_tolerance(sol.stage2.objective_value)+old_numeric:
            raise RuntimeError("Historical epoch fails its own original Stage-II face.")
        k=min(impl,case.n_time-start)
        balance-=np.sum(case.time_grid.delta[start:start+k,None]*sol.u[:k],axis=0)
        if balance.min() < -10*parent.solver_profile.feasibility_tol:
            raise RuntimeError("Historical allowance ledger is negative.")
        balance=np.maximum(balance,0.0)
        previous=sol
        lineage.append({"start":start,"solution_hash":sol.solution_hash,"experiment_hash":subexp.hash})
    raise AssertionError("Unreachable historical target.")


def reproduce_main():
    """Cheap native diagnostic after card freeze, not a production cache migration."""
    from dataclasses import replace
    from tpwrs.config import settings
    from tpwrs.storage import sha256_json
    from tpwrs.persistence import save_policy_solution
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root",type=Path,default=Path(settings().get("previous_results_workspace",r"C:\TPWRS_V3_FINAL\TPWRS_V3_RELEASE_20260906_R9")))
    parser.add_argument("--epoch",type=int,default=5160)
    args=parser.parse_args()
    source=args.previous_root
    required=[source/"artifacts/real_case/physical_case/FINAL_DUTCH_CASE/physical_case.json",
              source/"results_m1_m2_final/year_2029/rho_0.15/rolling/M1/EXPERIMENT_CASE.json"]
    required += [source/f"results_m1_m2_final/year_2029/rho_0.15/rolling/M1/epochs/epoch_{start:06d}/{name}"
                 for start in range(0,args.epoch,24) for name in ("solution.json","solution_arrays.npz")]
    missing=[str(path) for path in required if not path.is_file()]
    if missing:
        atomic_write_json(ROOT/"results_numerical_reproduction/STATUS.json",{
            "status":"BLOCKED_MISSING_HISTORY","scientific_fingerprint":scientific_fingerprint(),
            "missing":missing})
        print("REPRO=BLOCKED_MISSING_HISTORY",flush=True)
        return 2
    case=load_physical_case(source/"artifacts/real_case/physical_case/FINAL_DUTCH_CASE")
    indexes=np.flatnonzero(case.time_grid.index.year==2029)
    if len(indexes)!=8760:
        raise RuntimeError("The R9 historical case must contain complete 2029 inputs.")
    case=case.slice_time(int(indexes[0]),int(indexes[-1])+1)
    oldfolder=source/"results_m1_m2_final/year_2029/rho_0.15/rolling/M1"
    oldparent=_read_frozen_experiment(oldfolder/"EXPERIMENT_CASE.json")
    subcase,oldexp,previous,lineage=_reconstruct_historical_epoch(case,oldparent,oldfolder/"epochs",args.epoch)
    del case
    identity=sha256_json({"old_experiment":oldexp.hash,"history":lineage,"code":scientific_fingerprint()})
    output=ROOT/"results_numerical_reproduction"/identity
    output.mkdir(parents=True,exist_ok=True)
    exp=replace(oldexp,case_id=f"DIAGNOSTIC_R9_EPOCH_{args.epoch}",
        implementation_hash=scientific_fingerprint(),result_root_id=output.relative_to(ROOT).as_posix())
    warm_x=warm_u=None
    if previous is not None:
        impl=24
        overlap=min(previous.x_mw.shape[0]-impl,subcase.n_time)
        warm_x=np.full(subcase.requests_mw.shape,np.nan)
        warm_u=np.full(subcase.requests_mw.shape,np.nan)
        if overlap>0:
            warm_x[:overlap]=np.minimum(previous.x_mw[impl:impl+overlap],subcase.xbar_mw[:overlap])
            warm_u[:overlap]=previous.u[impl:impl+overlap]
    # No old solution is rebound, edited, moved or copied into a new accepted cache.
    atomic_write_json(output/"LINEAGE.json",{"old_experiment":asdict(oldexp),"old_history":lineage,"scope":"DIAGNOSTIC_ONLY_NOT_PRODUCTION"})
    solution=solve_policy(subcase,exp,force_generic=True,checkpoint_dir=output,
                          initial_x=warm_x,initial_u=warm_u)
    report=validate_policy_solution(subcase,exp,solution,require_stage3=False)
    save_policy_solution(solution,output)
    atomic_write_json(output/"STATUS.json",{"status":"PASS","hours_solved":subcase.n_time,
        "epoch":args.epoch,"scientific_fingerprint":scientific_fingerprint(),"replay":report,
        "scope":"DIAGNOSTIC_ONLY_NOT_PRODUCTION","old_workspace_modified":False})
    atomic_write_json(ROOT/"results_numerical_reproduction/STATUS.json",{
        "status":"PASS","scientific_fingerprint":scientific_fingerprint(),
        "diagnostic_status_sha256":sha256_file(output/"STATUS.json"),
        "diagnostic_path":output.relative_to(ROOT).as_posix()})
    print(f"REPRODUCTION=PASS epoch={args.epoch} hours={subcase.n_time} output={output}",flush=True)
    return 0
