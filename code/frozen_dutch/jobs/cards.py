from __future__ import annotations


# ============================================================================
# scripts.build_f0_features (consolidated; see SOURCE_MAP.json)
# ============================================================================

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tpwrs.features import F0_CORE, FeatureProvenanceRecord, build_feature_schema
from tpwrs.storage import sha256_file
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.persistence import load_physical_case, load_fitting_case


def features_atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp.parquet",dir=path.parent); os.close(fd); tmp=Path(name)
    try:
        frame.to_parquet(tmp,index=False); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def features_main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--deployment",action="store_true"); parser.add_argument("--case",type=Path,default=ROOT/"artifacts/real_case/physical_case/FINAL_DUTCH_CASE"); parser.add_argument("--source-registry",type=Path,default=ROOT/"artifacts/real_case/SOURCE_REGISTRY.json"); parser.add_argument("--local-source-config",type=Path,default=ROOT/"artifacts/real_case/LOCAL_SOURCE_CONFIG.json"); parser.add_argument("--output",type=Path,default=ROOT/"artifacts/features"); args=parser.parse_args()
    if args.deployment:
        status=json.loads((ROOT/"artifacts/cards/CARD_BUILD_STATUS.json").read_text(encoding="utf-8"))
        from tpwrs.config import scientific_fingerprint
        if status.get("status")!="PASS" or status.get("scientific_fingerprint")!=scientific_fingerprint():
            raise RuntimeError("Deployment features require current frozen cards.")
        case=load_physical_case(args.case)
    else:
        case,_=load_fitting_case(args.case)
    registry=json.loads(args.source_registry.read_text(encoding="utf-8")); local=json.loads(args.local_source_config.read_text(encoding="utf-8"))
    if local.get("registry_hash")!=registry.get("registry_hash"): raise RuntimeError("Local source config does not match the frozen logical registry.")
    participant_path=Path(local["records"]["participants"]["path"])
    if sha256_file(participant_path)!=registry["records"]["participants"]["sha256"]: raise RuntimeError("Participant source hash mismatch.")
    participants=pd.read_parquet(participant_path); participants["participant_id"]=participants.participant_id.astype(str); participants=participants.set_index("participant_id").loc[list(case.participant_ids)]
    source_for={
        "hour_sin":("DETERMINISTIC_CALENDAR",sha256_file(Path(__file__))),"hour_cos":("DETERMINISTIC_CALENDAR",sha256_file(Path(__file__))),"dow_sin":("DETERMINISTIC_CALENDAR",sha256_file(Path(__file__))),"dow_cos":("DETERMINISTIC_CALENDAR",sha256_file(Path(__file__))),
        "capacity_mw":("FROZEN_PARTICIPANT_CONTRACT",registry["records"]["participants"]["sha256"]),"request_fraction":("FROZEN_CONTROLLED_REQUEST_SCENARIO",registry["records"]["hourly_request_manifest"]["sha256"]),"zone":("FROZEN_PARTICIPANT_CROSSWALK",registry["records"]["participants"]["sha256"]),"technology":("FROZEN_PARTICIPANT_CROSSWALK",registry["records"]["participants"]["sha256"]),
    }
    # No fictitious historical timestamp is manufactured.  The F0 family is
    # intentionally non-vintaged: calendar features are deterministic, contract
    # metadata are static, and request_fraction is a predeclared controlled input.
    # Their D-1 admissibility is recorded by availability_class, not by inventing
    # an archived forecast issue time.
    calendar={"hour_sin","hour_cos","dow_sin","dow_cos"}
    controlled={"request_fraction"}
    records=[]
    for feature in F0_CORE:
        if feature in calendar:
            availability_class="DETERMINISTIC_CALENDAR"
        elif feature in controlled:
            availability_class="PREDECLARED_CONTROLLED_INPUT"
        else:
            availability_class="STATIC_CONTRACT"
        records.append(FeatureProvenanceRecord(
            feature=feature,
            source_id=source_for[feature][0],
            source_hash=source_for[feature][1],
            issue_timestamp=None,
            delivery_period="2025-2031",
            timezone="Europe/Amsterdam",
            forecast_or_realization="predeclared_controlled_input_not_archived_forecast",
            revision_policy="immutable_after_physical_freeze",
            d1_pass=True,
            role="deployable",
            availability_class=availability_class,
        ))
    schema=build_feature_schema("F0",records,F0_CORE); args.output.mkdir(parents=True,exist_ok=True); atomic_write_csv(args.output/"FEATURE_PROVENANCE.csv",pd.DataFrame([asdict(x) for x in records])); atomic_write_json(args.output/"F0_SCHEMA.json",{"schema":"TPWRS_FEATURE_SCHEMA_FREEZE_V1","status":"PASS","feature_set_id":"F0","feature_schema_hash":schema.hash,"columns":schema.columns,"provenance_hash":schema.provenance_hash,"claim_boundary":"CONTROLLED EX-ANTE INPUTS; NO CLAIM OF ARCHIVED DUTCH D-1 FORECAST VINTAGES"})
    idx=case.time_grid.index; N=case.n_participants; zone=participants.area_id.astype(str).to_numpy(); technology=participants.technology.astype(str).to_numpy(); direction=np.asarray(case.participant_directions,dtype=str)
    outputs={}
    existing_gate=args.output/"FEATURE_GATE.json"
    if args.deployment and existing_gate.is_file():
        outputs=json.loads(existing_gate.read_text(encoding="utf-8"))["outputs"]
    for year in (range(2029,2032) if args.deployment else range(2025,2029)):
        rows=np.flatnonzero(idx.year==year); local=idx[rows].tz_convert("Europe/Amsterdam"); hour=local.hour.to_numpy(); dow=local.dayofweek.to_numpy(); T=len(rows)
        frame=pd.DataFrame({
            "year":year,"time_index":np.repeat(rows,N),"timestamp_utc":np.repeat(idx[rows].astype(str).to_numpy(),N),"participant_id":np.tile(np.asarray(case.participant_ids,dtype=str),T),"direction":np.tile(direction,T),
            "hour_sin":np.repeat(np.sin(2*np.pi*hour/24.0),N),"hour_cos":np.repeat(np.cos(2*np.pi*hour/24.0),N),"dow_sin":np.repeat(np.sin(2*np.pi*dow/7.0),N),"dow_cos":np.repeat(np.cos(2*np.pi*dow/7.0),N),
            "capacity_mw":np.tile(case.capacities_mw,T),"request_fraction":(case.requests_mw[rows]/case.capacities_mw[None,:]).reshape(-1),"zone":np.tile(zone,T),"technology":np.tile(technology,T),
        })
        path=args.output/f"F0_{year}.parquet"; features_atomic_parquet(path,frame); outputs[str(year)]={"path":path.name,"bytes":path.stat().st_size,"sha256":sha256_file(path),"rows":len(frame)}
    gate={"schema":"TPWRS_FEATURE_FAMILY_GATE_V1","status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"families":{"F0":{"status":"PASS","columns":list(schema.columns)},"F1":{"status":"OMITTED_PROVENANCE_FAILURE","reason":"Provincial residual-load source is a controlled scenario without archived issue/vintage timestamps."},"F2":{"status":"OMITTED_PROVENANCE_FAILURE","reason":"F1 provenance failed; no forecast-dependent residual-load transformation is deployable."},"PUBLIC_RESIDUAL_SCORE":{"status":"OMITTED_PROVENANCE_FAILURE","reason":"Requires provenance-passing F1."}},"outputs":outputs,"heldout_feature_note":"Fitting loads only partitioned 2025-2028 physical inputs and features. Deployment features are generated after card freeze."}; atomic_write_json(args.output/"FEATURE_GATE.json",gate); print(f"FEATURES=PASS family=F0 columns={len(schema.columns)} rows={sum(x['rows'] for x in outputs.values())}"); return 0


# ============================================================================
# scripts.fit_freeze_cards (consolidated; see SOURCE_MAP.json)
# ============================================================================

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tpwrs.contracts import nominal_experiment
from tpwrs.execution import validate_policy_solution
from tpwrs.cards import ValidationCandidate, freeze_selected_card, select_validation_card
from tpwrs.cards import frozen_shuffle_scores, inverse_order, make_ordering_control_card, ptdf_net_relief_scores
from tpwrs.contracts import CardSpec, SolverProfile
from tpwrs.storage import sha256_array, sha256_file, sha256_json
from tpwrs.cards import validate_preheldout_audit
from tpwrs.persistence import save_card
from tpwrs.persistence import load_policy_solution, save_policy_solution
from tpwrs.storage import atomic_write_json
from tpwrs.persistence import load_physical_case, load_fitting_case
from tpwrs.cards import average_tie_percentile, build_teacher_rows, make_card, q_from_card
from tpwrs.model import solve_policy
from tpwrs.features import fit_predictor
from tpwrs.cards import compare_teacher_attributions
from tpwrs.config import scientific_fingerprint


FIT_YEARS=(2025,2026,2027)
VALIDATION_YEAR=2028
FAMILIES=("ridge","histgb")
SHUFFLE_SEEDS=tuple(range(118001,118033))
LEX_ABS=1e-6
LEX_REL=1e-9
REBUILD_TOL=2e-6


# RUN_03 numerical artifacts are bound to the complete scientific source tree.
# No numerical solution is migrated or rebound across release identities.
PRODUCTION_IMPLEMENTATION_HASH=scientific_fingerprint()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp.parquet",dir=path.parent); os.close(fd); tmp=Path(name)
    try: frame.to_parquet(tmp,index=False); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def _atomic_joblib(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp.joblib",dir=path.parent); os.close(fd); tmp=Path(name)
    try: joblib.dump(value,tmp); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def _year_case(case: object, year: int) -> object:
    rows=np.flatnonzero(case.time_grid.index.year==year)
    if not len(rows): raise RuntimeError(f"Physical case omits year {year}.")
    return case.slice_time(int(rows[0]),int(rows[-1])+1)


def _features(path: Path, year: int, columns: tuple[str,...]) -> pd.DataFrame:
    if year>VALIDATION_YEAR: raise RuntimeError("Held-out feature access attempted during card fitting.")
    gate=json.loads((path/"FEATURE_GATE.json").read_text(encoding="utf-8"))
    feature_path=path/f"F0_{year}.parquet"
    if gate.get("status")!="PASS" or sha256_file(feature_path)!=gate["outputs"][str(year)]["sha256"]:
        raise RuntimeError("Fitting feature hash mismatch.")
    frame=pd.read_parquet(feature_path); required={"timestamp_utc","participant_id","direction",*columns}
    if not required.issubset(frame.columns): raise RuntimeError(f"F0_{year} missing {sorted(required-set(frame.columns))}")
    frame=frame.rename(columns={"timestamp_utc":"timestamp"}); frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True).astype(str); frame["participant_id"]=frame.participant_id.astype(str)
    return frame


def _teacher(case: object,rho: float,year: int,root: Path,profile: SolverProfile) -> tuple[pd.DataFrame,pd.DataFrame,str,str,list[float],dict[str,object]]:
    yc=_year_case(case,year)
    namespace=root/f"rho_{rho:.2f}"/"teacher_experiments"/PRODUCTION_IMPLEMENTATION_HASH
    cache=namespace/f"teacher_solution_{year}"
    alternate_cache=namespace/f"teacher_rebuild_solution_{year}"
    result_id=str(cache.relative_to(ROOT)).replace("\\","/")
    exp=nominal_experiment(yc,case_id=f"CARD_TEACHER_RHO{int(round(rho*100)):02d}_{year}",rule="M2",rho=rho,solver_profile=profile,result_root_id=result_id,implementation_hash=PRODUCTION_IMPLEMENTATION_HASH)
    exp=replace(exp,information_level="M2_EXACT_UNPERTURBED_STAGE_II_PRIMAL_DUAL_TEACHER")

    _freeze_experiment(cache,exp)
    if (cache/"solution.json").is_file():
        solution=load_policy_solution(cache,expected_experiment_hash=exp.hash,expected_physical_case_hash=yc.physical_case_hash)
        validate_policy_solution(yc,exp,solution,require_stage3=False)
        cache_mode="REUSED"
    else:
        solution=solve_policy(yc,exp,use_stage3=False,checkpoint_dir=cache)
        validate_policy_solution(yc,exp,solution,require_stage3=False)
        save_policy_solution(solution,cache)
        cache_mode="BUILT"

    alternate_exp=replace(exp,result_root_id=str(alternate_cache.relative_to(ROOT)).replace("\\","/"))
    _freeze_experiment(alternate_cache,alternate_exp)
    if (alternate_cache/"solution.json").is_file():
        alternate=load_policy_solution(alternate_cache,expected_experiment_hash=alternate_exp.hash,expected_physical_case_hash=yc.physical_case_hash)
        validate_policy_solution(yc,alternate_exp,alternate,require_stage3=False)
        alternate_mode="REUSED"
    else:
        alternate=solve_policy(yc,alternate_exp,use_stage3=False,checkpoint_dir=alternate_cache)
        validate_policy_solution(yc,alternate_exp,alternate,require_stage3=False)
        save_policy_solution(alternate,alternate_cache)
        alternate_mode="BUILT"

    frame=build_teacher_rows(yc,solution,year=year); alternate_frame=build_teacher_rows(yc,alternate,year=year)
    s_tol=LEX_ABS+LEX_REL*max(1.0,abs(solution.stage1.objective_value),abs(alternate.stage1.objective_value)); e_tol=LEX_ABS+LEX_REL*max(1.0,abs(solution.stage2.objective_value),abs(alternate.stage2.objective_value)); dual_diff=max(abs(float(solution.m2_budget_duals[p])-float(alternate.m2_budget_duals[p])) for p in yc.participant_ids)
    audit={"schema":"TPWRS_EXACT_TEACHER_REBUILD_V1","status":"PASS","rho":rho,"year":year,"reference_solution_hash":solution.solution_hash,"alternate_solution_hash":alternate.solution_hash,"stage1_abs_difference":abs(solution.stage1.objective_value-alternate.stage1.objective_value),"stage2_abs_difference":abs(solution.stage2.objective_value-alternate.stage2.objective_value),"stage1_lexicographic_tolerance":s_tol,"stage2_lexicographic_tolerance":e_tol,"maximum_restriction_difference_mw":float(np.max(np.abs(solution.x_mw-alternate.x_mw))),"maximum_selected_dual_difference":float(dual_diff),"selected_dual_tolerance":REBUILD_TOL,"attribution_stability":compare_teacher_attributions(frame,alternate_frame)}
    if audit["stage1_abs_difference"]>s_tol or audit["stage2_abs_difference"]>e_tol or audit["maximum_restriction_difference_mw"]>REBUILD_TOL or audit["maximum_selected_dual_difference"]>REBUILD_TOL: audit["status"]="FAIL_FLAT_CARD_REQUIRED"
    atomic_write_json(root/f"rho_{rho:.2f}"/f"EXACT_REBUILD_{year}.json",audit); _atomic_parquet(root/f"rho_{rho:.2f}"/f"teacher_rows_{year}.parquet",frame); print(f"CARD_TEACHER rho={rho:.2f} year={year} cache={cache_mode}/{alternate_mode} rebuild={audit['status']} S={solution.s_value_mw_h_row:.9g} E={solution.e_value_mwh:.9g}",flush=True); return frame,alternate_frame,solution.solution_hash,exp.allowance.hash,exp.allowance.values.tolist(),audit


def _sample_fit(frame: pd.DataFrame) -> pd.DataFrame:
    rng=np.random.default_rng(118); picked=[]
    for _,indices in frame.groupby("direction").groups.items():
        indices=np.asarray(list(indices),int); picked.extend((indices if len(indices)<=200000 else np.sort(rng.choice(indices,200000,replace=False))).tolist())
    return frame.loc[np.sort(np.asarray(picked,int))]


def _flat_fallback(card: CardSpec, reason: str) -> CardSpec:
    directions=tuple(sorted(card.rates))
    return CardSpec(card.card_id,card.allowance_hash,card.feature_set_id,card.model_family,{d:(-np.inf,np.inf) for d in directions},{d:(1.0,1.0,1.0) for d in directions},card.q_bounds,card.lambda_q,{d:reason for d in directions},card.source_hashes,card.fit_years,card.validation_year,card.heldout_access_audit)


def _preprocessing_spec(family: str,columns: tuple[str,...]) -> dict[str,object]:
    common={"feature_order":columns,"numeric":"median imputation then StandardScaler","categorical_missing":"most_frequent"}
    if family=="ridge": return {**common,"categorical_encoding":"OneHotEncoder(handle_unknown=ignore)","estimator":"Ridge","alpha":2.0}
    return {**common,"categorical_encoding":"OrdinalEncoder(handle_unknown=use_encoded_value,unknown_value=-1)","estimator":"HistGradientBoostingRegressor","max_iter":80,"max_leaf_nodes":15,"max_depth":3,"min_samples_leaf":40,"l2_regularization":1.0,"learning_rate":0.08,"random_state":118}


def _card_difference(a: CardSpec,b: CardSpec) -> dict[str,object]:
    directions=tuple(sorted(a.rates)); rate=max(abs(a.rates[d][i]-b.rates[d][i]) for d in directions for i in range(3)); finite_thresholds=[abs(a.thresholds[d][i]-b.thresholds[d][i]) for d in directions for i in range(2) if np.isfinite(a.thresholds[d][i]) and np.isfinite(b.thresholds[d][i])]; threshold=max(finite_thresholds,default=0.0)
    return {"maximum_rate_difference":float(rate),"maximum_finite_threshold_difference":float(threshold),"fallbacks_identical":a.direction_fallbacks==b.direction_fallbacks,"within_tolerance":bool(rate<=REBUILD_TOL and threshold<=REBUILD_TOL and a.direction_fallbacks==b.direction_fallbacks)}


def _freeze_controls(case: object,fit: pd.DataFrame,selected: ValidationCandidate,output: Path,heldout_audit_path: Path) -> dict[str,object]:
    selected_score=f"score_{selected.card.model_family}"
    if selected_score not in fit: raise RuntimeError("Selected fitting score is absent while freezing controls.")
    controls=output/"controls"; controls.mkdir(parents=True,exist_ok=True); common={"selected_card":selected.card.hash,"physical_case":case.physical_case_hash,"controls_code":sha256_file(ROOT/"tpwrs/cards.py")}; records={}

    # PTDF fitting scores are computed solely on 2025--2027 and are never a
    # learned feature.  Merge by exact stable coordinates, not array shape.
    ptdf_parts=[]; ptdf_summary={}
    for year in FIT_YEARS:
        yc=_year_case(case,year); gross,induced,net=ptdf_net_relief_scores(yc); ptdf_summary[str(year)]={"gross_relief_sum":float(gross.sum()),"induced_overload_sum":float(induced.sum()),"net_relief_sum":float(net.sum()),"score_sha256":sha256_array(net)}
        ptdf_parts.append(pd.DataFrame({"year":year,"time_index":np.repeat(np.arange(yc.n_time),yc.n_participants),"participant_id":np.tile(np.asarray(yc.participant_ids,dtype=str),yc.n_time),"score_ptdf":net.reshape(-1)}))
    control_fit=fit.merge(pd.concat(ptdf_parts,ignore_index=True),on=["year","time_index","participant_id"],validate="one_to_one")
    ptdf_card=make_ordering_control_card(card_id="RHO03_CONTROL_PTDF",base_card=selected.card,fit_frame=control_fit,score_column="score_ptdf",feature_set_id="CONTROL_PTDF_NOT_LEARNED",source_hashes={**common,"score_contract":sha256_json(ptdf_summary)}); save_card(ptdf_card,controls/"PTDF_CARD.json"); records["PTDF"]={"card_id":ptdf_card.card_id,"card_hash":ptdf_card.hash,"path":"controls/PTDF_CARD.json","fitting_score_summary":ptdf_summary}

    inverse_frame=fit.copy(); inverse_frame["score_inverse"]=inverse_order(inverse_frame[selected_score].to_numpy(float)); inverse_card=make_ordering_control_card(card_id="RHO03_CONTROL_INVERSE",base_card=selected.card,fit_frame=inverse_frame,score_column="score_inverse",feature_set_id="CONTROL_INVERSE_SELECTED_ORDER",source_hashes=common); save_card(inverse_card,controls/"INVERSE_CARD.json"); records["INVERSE"]={"card_id":inverse_card.card_id,"card_hash":inverse_card.hash,"path":"controls/INVERSE_CARD.json"}

    direction=np.asarray(case.participant_directions,dtype=str)
    for seed in SHUFFLE_SEEDS:
        values=np.empty(len(fit),float)
        for year in FIT_YEARS:
            part=fit.loc[fit.year.eq(year)]; matrix=part.pivot(index="time_index",columns="participant_index",values=selected_score).reindex(index=range(_year_case(case,year).n_time),columns=range(case.n_participants)).to_numpy(float)
            shuffled=frozen_shuffle_scores(matrix,direction,(seed,))[seed]; values[part.index.to_numpy(int)]=shuffled[part.time_index.to_numpy(int),part.participant_index.to_numpy(int)]
        column=f"score_shuffle_{seed}"; shuffle_frame=fit.copy(); shuffle_frame[column]=values; card=make_ordering_control_card(card_id=f"RHO03_CONTROL_SHUFFLE_{seed}",base_card=selected.card,fit_frame=shuffle_frame,score_column=column,feature_set_id="CONTROL_FROZEN_WITHIN_DIRECTION_SHUFFLE",source_hashes={**common,"shuffle_seed":sha256_json(seed)}); path=controls/f"SHUFFLE_{seed}_CARD.json"; save_card(card,path); records[f"SHUFFLE_{seed}"]={"card_id":card.card_id,"card_hash":card.hash,"path":f"controls/{path.name}","seed":seed}
    audit_payload,audit_hash=validate_preheldout_audit(heldout_audit_path,expected_validation_year=VALIDATION_YEAR); payload={"schema":"TPWRS_RHO03_CONTROL_CARD_FREEZE_V2","status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"fit_years":FIT_YEARS,"heldout_access_audit":"PASS","heldout_access_audit_path":Path(heldout_audit_path).name,"heldout_access_audit_sha256":audit_hash,"heldout_access_audit_schema":audit_payload.get("schema"),"scientific_fingerprint":PRODUCTION_IMPLEMENTATION_HASH,"selected_card_id":selected.card.card_id,"selected_card_hash":selected.card.hash,"flat_control":"M2_Q_EQUALS_ONE","public_control":"OMITTED_PROVENANCE_FAILURE","ptdf_control":"ONE_MW_ADMISSIBLE_PROBE_NOT_A_LEARNED_FEATURE","shuffle_seeds":SHUFFLE_SEEDS,"records":records}; atomic_write_json(output/"RHO03_CONTROL_CARD_FREEZE.json",payload); return payload


def _freeze_experiment(cache,exp):
    from jobs.common import freeze_experiment
    freeze_experiment(cache,exp)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--case",type=Path,default=ROOT/"artifacts/real_case/physical_case/FINAL_DUTCH_CASE"); parser.add_argument("--features",type=Path,default=ROOT/"artifacts/features"); parser.add_argument("--output",type=Path,default=ROOT/"artifacts/cards"); args=parser.parse_args(); case,partition=load_fitting_case(args.case); feature_schema=json.loads((args.features/"F0_SCHEMA.json").read_text(encoding="utf-8")); columns=tuple(feature_schema["columns"]); args.output.mkdir(parents=True,exist_ok=True)
    from tpwrs.cards import freeze_preheldout_inventory
    heldout=freeze_preheldout_inventory(ROOT,args.output)
    atomic_write_json(args.output/"CARD_BUILD_STATUS.json",{"status":"RUNNING","created_utc":datetime.now(timezone.utc).isoformat()})

    from tpwrs.config import qualified_solver_profile
    heldout_path=args.output/"PRE_FREEZE_HELDOUT_ACCESS_AUDIT.json"; heldout_payload,heldout_hash=validate_preheldout_audit(heldout_path,expected_validation_year=VALIDATION_YEAR); profile=qualified_solver_profile(); summary={"schema":"TPWRS_ALLOWANCE_SPECIFIC_CARD_BUILD_V3","status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"scientific_fingerprint":PRODUCTION_IMPLEMENTATION_HASH,"heldout_access_audit":"PASS","heldout_access_audit_path":heldout_path.name,"heldout_access_audit_sha256":heldout_hash,"heldout_access_audit_schema":heldout_payload.get("schema"),"feature_set_id":"F0","feature_schema_hash":feature_schema["feature_schema_hash"],"rho":{}}
    for rho in (0.15,0.03):
        teachers=[]; alternate_teachers=[]; teacher_hashes=[]; allowance_hashes=[]; allowance_vectors={}; annual_allowance_hashes={}; rebuild_audits=[]
        # Allowances depend only on the frozen contract-duration mask and time
        # grid, so all deployment-year vectors are lawfully frozen before any
        # held-out policy outcome is opened.
        for allowance_year in range(2025,2032):
            allowance=partition["allowances"][f"{rho:.2f}"][str(allowance_year)]; allowance_vectors[str(allowance_year)]=allowance["values_hours"]; annual_allowance_hashes[str(allowance_year)]=allowance["hash"]
        for year in FIT_YEARS:
            frame,alternate_frame,solution_hash,allowance_hash,allowance_values,rebuild=_teacher(case,rho,year,args.output,profile); teachers.append(frame); alternate_teachers.append(alternate_frame); teacher_hashes.append(solution_hash); allowance_hashes.append(allowance_hash); rebuild_audits.append(rebuild)
            if allowance_hash!=annual_allowance_hashes[str(year)] or not np.array_equal(np.asarray(allowance_values),np.asarray(allowance_vectors[str(year)])): raise RuntimeError("Teacher allowance differs from the pre-frozen annual vector.")
        fit=pd.concat(teachers,ignore_index=True); fit_alt=pd.concat(alternate_teachers,ignore_index=True)
        for target in (fit,fit_alt): target["target_percentile"]=target.groupby("direction",dropna=False)["alpha"].transform(lambda x:pd.Series(average_tie_percentile(x.to_numpy()),index=x.index)); target["timestamp"]=pd.to_datetime(target.timestamp,utc=True).astype(str)
        raw_fit_features=pd.concat([_features(args.features,year,columns) for year in FIT_YEARS],ignore_index=True); keys=["timestamp","participant_id","direction"]; add=[c for c in columns if c not in fit.columns]; fit_features=raw_fit_features[keys+add]
        fit=fit.merge(fit_features,on=keys,validate="one_to_one"); fit_alt=fit_alt.merge(fit_features,on=keys,validate="one_to_one")
        candidates=[]; candidate_rows=[]; validation_case=_year_case(case,VALIDATION_YEAR); validation_features=_features(args.features,VALIDATION_YEAR,columns); expected_times=pd.DatetimeIndex(pd.to_datetime(validation_case.time_grid.timestamps_utc,utc=True)).astype(str); expected_order=pd.MultiIndex.from_product([expected_times,validation_case.participant_ids],names=["timestamp","participant_id"]); actual=pd.MultiIndex.from_frame(validation_features[["timestamp","participant_id"]]);
        if not actual.equals(expected_order): raise RuntimeError("Validation F0 row order is not exact time-major physical-case order.")
        validation_allowance=nominal_experiment(validation_case,case_id="CARD_VALIDATION_ALLOWANCE_ONLY",rule="M2",rho=rho,solver_profile=profile).allowance
        if validation_allowance.hash!=annual_allowance_hashes[str(VALIDATION_YEAR)]: raise RuntimeError("Validation allowance differs from the pre-frozen annual vector.")
        allowance_contract_hash=sha256_json({"rho":rho,"annual_allowance_hashes":annual_allowance_hashes,"participants":case.participant_ids,"physical_case_hash":partition["physical_case_hash"],"feature_schema_hash":feature_schema["feature_schema_hash"],"uniform_grid_rule":"K=floor(rho*N_contract);H=K*Delta"})
        for family in FAMILIES:
            training=_sample_fit(fit); bundle=fit_predictor(family,training,columns,training.target_percentile.to_numpy(float)); model_path=args.output/f"rho_{rho:.2f}"/f"MODEL_{family}.joblib"; _atomic_joblib(model_path,bundle); fit[f"score_{family}"]=bundle.predict(fit); validation_score=bundle.predict(validation_features).reshape(validation_case.n_time,validation_case.n_participants)
            card=make_card(card_id=f"RHO{int(round(rho*100)):02d}_F0_{family.upper()}",allowance_hash=allowance_contract_hash,feature_set_id="F0",model_family=family,fit_frame=fit,score_column=f"score_{family}",q_bounds=(0.5,1.5),lambda_q=0.001,source_hashes={"feature_schema":feature_schema["feature_schema_hash"],"model_bundle":sha256_file(model_path),"m3_code":sha256_file(ROOT/"tpwrs/cards.py"),"predictor_code":sha256_file(ROOT/"tpwrs/features.py"),"physical_case":partition["physical_case_hash"],"validation_allowance":validation_allowance.hash,**{f"teacher_table_{year}":sha256_file(args.output/f"rho_{rho:.2f}"/f"teacher_rows_{year}.parquet") for year in FIT_YEARS},**{f"feature_table_{year}":sha256_file(args.features/f"F0_{year}.parquet") for year in FIT_YEARS},**{f"teacher_{year}":value for year,value in zip(FIT_YEARS,teacher_hashes)}}); candidate_card_path=args.output/f"rho_{rho:.2f}"/f"CARD_{family}.json"; save_card(card,candidate_card_path)
            alternate_training=_sample_fit(fit_alt); alternate_bundle=fit_predictor(family,alternate_training,columns,alternate_training.target_percentile.to_numpy(float)); fit_alt[f"score_{family}"]=alternate_bundle.predict(fit_alt); alternate_card=make_card(card_id=card.card_id,allowance_hash=allowance_contract_hash,feature_set_id="F0",model_family=family,fit_frame=fit_alt,score_column=f"score_{family}",q_bounds=(0.5,1.5),lambda_q=0.001,source_hashes=card.source_hashes); card_stability=_card_difference(card,alternate_card); exact_rebuild_pass=all(x["status"]=="PASS" for x in rebuild_audits) and card_stability["within_tolerance"]
            if not exact_rebuild_pass: card=_flat_fallback(card,"FLAT_EXACT_REBUILD_STABILITY_FAILURE"); save_card(card,candidate_card_path); validation_score=np.zeros_like(validation_score)
            q=q_from_card(card,validation_case.participant_directions,validation_score)
            cache=args.output/f"rho_{rho:.2f}"/"validation_experiments"/PRODUCTION_IMPLEMENTATION_HASH/card.hash/f"validation_{family}"
            result_id=str(cache.relative_to(ROOT)).replace("\\","/")
            exp=nominal_experiment(validation_case,case_id=f"CARD_VALIDATION_RHO{int(round(rho*100)):02d}_{family}",rule="M3",rho=rho,card_id=card.card_id,card_hash=card.hash,feature_set_id="F0",solver_profile=profile,result_root_id=result_id,implementation_hash=PRODUCTION_IMPLEMENTATION_HASH)
            exp=replace(exp,rate_matrix_hash=sha256_array(q)); _freeze_experiment(cache,exp)
            if (cache/"solution.json").is_file():
                solution=load_policy_solution(cache,expected_experiment_hash=exp.hash,expected_physical_case_hash=validation_case.physical_case_hash)
                validate_policy_solution(validation_case,exp,solution,require_stage3=False)
                cache_mode="REUSED"
            else:
                solution=solve_policy(validation_case,exp,q=q,use_stage3=False,checkpoint_dir=cache)
                validate_policy_solution(validation_case,exp,solution,require_stage3=False)
                save_policy_solution(solution,cache)
                cache_mode="BUILT"

            if not solution.replay_summary.get("passed") or solution.stage1.status!="OPTIMAL" or solution.stage2.status!="OPTIMAL" or solution.stage3 is not None: raise RuntimeError(f"{family}: validation solution failed Stage-I/II/replay gate.")
            candidate=ValidationCandidate(card,VALIDATION_YEAR,solution.stage1.objective_value,solution.stage2.objective_value,f"F0::{family}"); candidates.append(candidate); candidate_rows.append({"family":family,"card_id":card.card_id,"card_hash":card.hash,"validation_solution_hash":solution.solution_hash,"S":solution.stage1.objective_value,"E":solution.stage2.objective_value,"replayed_selected_S":solution.s_value_mw_h_row,"replayed_selected_E":solution.e_value_mwh,"cache":cache_mode,"exact_rebuild_pass":exact_rebuild_pass,"card_stability":card_stability}); print(f"CARD_VALIDATION rho={rho:.2f} family={family} rebuild={exact_rebuild_pass} S={solution.stage1.objective_value:.9g} E={solution.stage2.objective_value:.9g}",flush=True)
        selected=select_validation_card(candidates,validation_year=VALIDATION_YEAR); selected_row=next(x for x in candidate_rows if x["card_hash"]==selected.card.hash); selected_path=args.output/f"RHO{int(round(rho*100)):02d}_SELECTED_CARD.json"; save_card(selected.card,selected_path); freeze_path=args.output/f"RHO{int(round(rho*100)):02d}_CARD_FREEZE.json"; freeze_selected_card(selected,freeze_path,heldout_access_audit="PASS",heldout_audit_path=heldout_path,freeze_metadata={"physical_case_id":partition["physical_case_id"],"physical_case_hash":partition["physical_case_hash"],"participant_ids":case.participant_ids,"allowance_contract_hash":allowance_contract_hash,"annual_allowance_hashes":annual_allowance_hashes,"allowance_vectors_hours":allowance_vectors,"allowance_vector_hashes":{year:sha256_array(np.asarray(values,float)) for year,values in allowance_vectors.items()},"feature_schema":feature_schema,"model_preprocessing":{**_preprocessing_spec(selected.card.model_family,columns),"bundle_file":f"rho_{rho:.2f}/MODEL_{selected.card.model_family}.joblib"},"exact_teacher_rebuild_audits":rebuild_audits,"selected_card_stability":selected_row["card_stability"],"validation_outcome":{"S":selected.s_value,"E":selected.e_value,"tie_break_key":selected.tie_break_key},"code_hashes":{"card_builder":sha256_file(Path(__file__)),"m3":sha256_file(ROOT/"tpwrs/cards.py"),"predictors":sha256_file(ROOT/"tpwrs/features.py"),"model":sha256_file(ROOT/"tpwrs/model.py"),"linear_solver":sha256_file(ROOT/"tpwrs/linear.py"),"annual_engine":sha256_file(ROOT/"tpwrs/annual.py"),"kkt_stability":sha256_file(ROOT/"tpwrs/cards.py"),"card_io":sha256_file(ROOT/"tpwrs/persistence.py")}}); summary["rho"][f"{rho:.2f}"]={"allowance_contract_hash":allowance_contract_hash,"candidates":candidate_rows,"selected_card_id":selected.card.card_id,"selected_card_hash":selected.card.hash,"selected_card_path":selected_path.name,"freeze_path":freeze_path.name}
        if rho==0.03: summary["controls"]=_freeze_controls(case,fit,selected,args.output,heldout_path)
    atomic_write_json(args.output/"CARD_BUILD_STATUS.json",summary); print("CARDS=PASS rho=0.15,0.03",flush=True); return 0
