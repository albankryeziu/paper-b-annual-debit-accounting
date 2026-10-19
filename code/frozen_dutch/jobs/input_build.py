from __future__ import annotations


# ============================================================================
# scripts.build_real_physical_case (consolidated; see SOURCE_MAP.json)
# ============================================================================

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tpwrs.network import validate_and_build_participant_actions
from tpwrs.inputs import build_component_state_from_provincial_panel, security_adjust_background_squared_reference
from tpwrs.network import extract_affine_n1_states, load_pandapower_network
from tpwrs.storage import sha256_array, sha256_file, sha256_json
from tpwrs.storage import atomic_write_csv, atomic_write_json
from tpwrs.persistence import load_physical_case, save_physical_case
from tpwrs.contracts import PhysicalCase
from tpwrs.contracts import TimeGrid


def _source(path: Path, role: str, provenance: str) -> tuple[dict[str,Any],dict[str,Any]]:
    path=path.resolve()
    if not path.is_file(): raise FileNotFoundError(f"Required real source is absent: {path}")
    digest=sha256_file(path); common={"bytes":path.stat().st_size,"sha256":digest,"role":role,"provenance":provenance}
    # Publication artifacts deliberately carry only a logical identifier and
    # immutable content metadata.  The machine path lives in the explicitly
    # local companion config and is never folded into a scientific hash.
    return ({**common,"file_name":path.name},{"path":str(path),"sha256":digest})


def discover_sources() -> dict[str,Path]:
    from tpwrs.config import source_records, source_path
    return {key:source_path(key) for key in source_records()}


def build_registry(paths: dict[str,Path], output: Path) -> dict[str,Any]:
    roles={
        "reduced_network":("45-bus policy network","PAPER_A_ZOMERDIJK_CANDIDATE"),"detailed_network":("383-bus localization/validation network","PAPER_A_ZOMERDIJK_CANDIDATE"),
        "ratings":("central branch ratings","PAPER_A_ENGINEERING_ENSEMBLE"),"component_panel":("provincial D/PV/onshore/offshore scenario panel","CONTROLLED_SCENARIO_NOT_D1_FORECAST"),
        "component_weights":("technology-specific province-to-bus weights","PAPER_A_PUBLIC_STATE_PRIOR"),"paper_a_component_allocations":("renewable-curtailment reference","PAPER_A_PUBLIC_INFERENCE"),
        "paper_a_dispatch":("device dispatch reference","PAPER_A_PUBLIC_INFERENCE"),"paper_a_inference_manifest":("public-inference validation manifest","PAPER_A_MACHINE_AUDIT"),
        "paper_a_participant_crosswalk":("114-position public crosswalk","PAPER_A_PRE_OUTCOME_CONSTRUCTED_CROSSWALK"),"participants":("position contract and electrical metadata","PRE_OUTCOME_REPOSITORY_ARTIFACT"),
        "prior_action_metadata":("predeclared direct/multibus supports and priors","LEGACY_PRE_OUTCOME_SUPPORT_ONLY"),"prior_actions":("direct-column balancing-key recovery","LEGACY_PRE_OUTCOME_BALANCING_KEY_ONLY"),
        "prior_network_matrices":("legacy action bus-axis labels","LEGACY_AXIS_LABELS_ONLY"),"hourly_request_manifest":("hourly request partition audit","PAPER_B_CONTROLLED_SCENARIO"),
        "dutch2018_manifest":("2018 benchmark gate","PAPER_A_MACHINE_AUDIT"),"dutch2018_states":("2018 hidden nodal truth","PAPER_A_ZOMERDIJK_2018_BENCHMARK"),"dutch2018_flows":("2018 hidden branch truth","PAPER_A_ZOMERDIJK_2018_BENCHMARK"),
    }
    records={}; local_records={}
    for key,path in paths.items():
        role,provenance=roles.get(key,("hourly participant requests","PAPER_B_CONTROLLED_SCENARIO_NATIVE_HOURLY"))
        records[key],local_records[key]=_source(path,role,provenance)
    payload={"schema":"TPWRS_REAL_SOURCE_REGISTRY_V1","status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"records":records,"forecast_claim_boundary":"The provincial component panel and participant requests are controlled scenario trajectories, not authenticated D-1 forecasts.","registry_hash":sha256_json({k:v["sha256"] for k,v in sorted(records.items())})}
    local={"schema":"TPWRS_LOCAL_SOURCE_CONFIG_V1","status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"registry_hash":payload["registry_hash"],"records":local_records,"publication_exclusion":"MACHINE-SPECIFIC; DO NOT DISTRIBUTE AS A FROZEN SCIENTIFIC ARTIFACT"}
    atomic_write_json(output,payload); atomic_write_json(output.with_name("LOCAL_SOURCE_CONFIG.json"),local); return payload


def _ratings(frame: pd.DataFrame, candidate: str, net: Any) -> np.ndarray:
    q=frame.loc[frame.candidate_id.astype(str).eq(candidate)]
    values=[]
    for kind,table in (("line",net.line),("trafo",net.trafo)):
        for index in table.index:
            hit=q.loc[q.element_type.astype(str).eq(kind)&q.element_index.astype(int).eq(int(index))]
            if len(hit)!=1: raise RuntimeError(f"{candidate}: rating row for {kind}::{index} is not unique.")
            values.append(float(hit.iloc[0].central_rating_mw))
    out=np.asarray(values,float)
    if np.any(~np.isfinite(out)) or np.any(out<=0): raise RuntimeError(f"{candidate}: invalid central ratings.")
    return out


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp.npz",dir=path.parent); os.close(fd); tmp=Path(name)
    try:
        np.savez_compressed(tmp,**arrays); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def validate_2018_reference(paths: dict[str,Path],bus_ids: tuple[str,...],intact: Any) -> dict[str,Any]:
    state=pd.read_parquet(paths["dutch2018_states"],columns=["hour","bus","net_injection_mw"])
    injections=state.pivot(index="hour",columns="bus",values="net_injection_mw").reindex(columns=list(map(int,bus_ids))).to_numpy(float)
    flow=pd.read_parquet(paths["dutch2018_flows"],columns=["hour","element_type","element_index","dc_flow_mw","rating_mw"]); flow["row_id"]=flow.element_type.astype(str)+"::"+flow.element_index.astype(str)
    reference=flow.pivot(index="hour",columns="row_id",values="dc_flow_mw").reindex(columns=list(intact.row_ids)).to_numpy(float); ratings=flow.pivot(index="hour",columns="row_id",values="rating_mw").reindex(columns=list(intact.row_ids)).to_numpy(float)
    rebuilt=injections@intact.H.T+intact.h_mw[None,:]; error=rebuilt-reference; tolerance=np.maximum(1e-4,1e-8*ratings)
    material=np.abs(reference)>1e-6; false_safe=(np.abs(rebuilt)<=ratings-tolerance)&(np.abs(reference)>ratings+tolerance); false_over=(np.abs(rebuilt)>ratings+tolerance)&(np.abs(reference)<=ratings-tolerance)
    limiting_reference=np.argmax(np.abs(reference)/ratings,axis=1); limiting_rebuilt=np.argmax(np.abs(rebuilt)/ratings,axis=1)
    payload={"schema":"TPWRS_2018_MAPPED_DC_REFERENCE_AUDIT_V1","status":"PASS" if np.all(np.abs(error)<=tolerance) and not np.any(false_safe|false_over) else "FAIL","hours":len(injections),"buses":injections.shape[1],"mapped_rows":reference.shape[1],"balance_max_abs_mw":float(np.max(np.abs(injections.sum(axis=1)))),"mae_mw":float(np.mean(np.abs(error))),"rmse_mw":float(np.sqrt(np.mean(error*error))),"max_abs_error_mw":float(np.max(np.abs(error))),"max_acceptance_tolerance_mw":float(np.max(tolerance)),"sign_agreement_material":float(np.mean(np.sign(rebuilt[material])==np.sign(reference[material]))),"limiting_element_agreement":float(np.mean(limiting_reference==limiting_rebuilt)),"false_safe_count":int(false_safe.sum()),"false_overload_count":int(false_over.sum()),"claim":"EXACT INTACT MAPPED-ROW DC RECONSTRUCTION AGAINST THE INDEPENDENT 2018 PAPER-A TRUTH BENCHMARK","source_hashes":{"states":sha256_file(paths["dutch2018_states"]),"flows":sha256_file(paths["dutch2018_flows"]),"manifest":sha256_file(paths["dutch2018_manifest"])}}
    if payload["status"]!="PASS": raise RuntimeError(f"2018 mapped DC reference gate failed: {payload}")
    return payload


def build_network_and_reduction(paths: dict[str,Path], registry: dict[str,Any], root: Path, *, write_artifacts: bool=True) -> tuple[Any,tuple[Any,...],np.ndarray,dict[str,Any],pd.DataFrame]:
    reduced,_=load_pandapower_network(paths["reduced_network"]); detailed,_=load_pandapower_network(paths["detailed_network"]); ratings=pd.read_csv(paths["ratings"])
    rr=_ratings(ratings,"AGGREGATED_EHV",reduced); dr=_ratings(ratings,"FULL_383",detailed)
    bus_ids,states,contingencies,network_audit=extract_affine_n1_states(reduced,rr)
    network_audit.update({"status":"PASS","source_registry_hash":registry["registry_hash"],"reduced_source_sha256":registry["records"]["reduced_network"]["sha256"],"rating_source_sha256":registry["records"]["ratings"]["sha256"],"state_hashes":{s.state_id:s.hash for s in states}})
    if write_artifacts:
        by_state={r.get("state_id"):r for r in contingencies if r.get("state_id")}
        state_registry=[]
        for state in states:
            outage=by_state.get(state.state_id,{})
            state_registry.append({"state_id":state.state_id,"state_kind":state.state_kind,"outaged_element_id":state.outaged_element_id,"element_type":outage.get("element_type"),"from_bus":outage.get("from_bus"),"to_bus":outage.get("to_bus"),"orientation":outage.get("orientation"),"outaged_element_rating_mw":outage.get("rating_mw"),"exact_row_ids":state.row_ids,"row_count":len(state.row_ids),"H_sha256":sha256_array(state.H),"h_sha256":sha256_array(state.h_mw),"ratings_sha256":sha256_array(state.ratings_mw),"state_source_hash":state.source_hash,"network_source_sha256":registry["records"]["reduced_network"]["sha256"],"ratings_source_sha256":registry["records"]["ratings"]["sha256"]})
        monitored=[{key:r.get(key) for key in ("row_id","element_id","element_type","element_index","from_bus","to_bus","orientation","rating_mw","ppc_branch_index")} for r in contingencies]
        atomic_write_json(root/"network/NETWORK_AUDIT.json",network_audit); atomic_write_csv(root/"network/CONTINGENCY_REGISTRY.csv",pd.DataFrame(contingencies)); atomic_write_json(root/"network/NETWORK_STATE_REGISTRY.json",{"schema":"TPWRS_NETWORK_STATE_REGISTRY_V1","status":"PASS","claim_boundary":"RETAINED NON-ISLANDING SINGLE-ELEMENT CONTINGENCIES REPRESENTED BY THE REDUCED EHV MODEL; NOT COMPREHENSIVE DUTCH N-1","monitored_elements":monitored,"states":state_registry}); atomic_write_json(root/"network/REFERENCE_2018_AUDIT.json",validate_2018_reference(paths,bus_ids,states[0]))
    participants=pd.read_parquet(paths["participants"]); prior=pd.read_parquet(paths["prior_action_metadata"])
    with np.load(paths["prior_actions"],allow_pickle=False) as z: old_A=np.asarray(z["bus_action"],float)
    with np.load(paths["prior_network_matrices"],allow_pickle=False) as z: old_bus=np.asarray(z["bus_ids"],int)
    reduction=validate_and_build_participant_actions(reduced,detailed,rr,dr,participants,prior,old_A,old_bus)
    reduction["gate"].update({"threshold_source":"specification/METHODOLOGY_FREEZE_V1.json::dutch_testbed.multibus_validation_gate","threshold_source_sha256":sha256_file(ROOT/"specification/METHODOLOGY_FREEZE_V1.json"),"direct_positions_are_empirical_benchmark":True})
    if write_artifacts:
        atomic_write_json(root/"reduction/PRIMARY_PARTICIPANT_GATE.json",reduction["gate"]); atomic_write_json(root/"reduction/BALANCING_KEY_AUDIT.json",reduction["balancing_audit"]); atomic_write_json(root/"reduction/REDUCTION_GROUP_SUMMARY.json",reduction["group_summary"])
        atomic_write_csv(root/"reduction/COMMON_INTERFACE_MAP.csv",pd.DataFrame(reduction["mapping_audit"])); atomic_write_csv(root/"reduction/COMMON_CONTINGENCY_AUDIT.csv",pd.DataFrame(reduction["contingency_audit"])); atomic_write_csv(root/"reduction/PARTICIPANT_ACTIONS.csv",pd.DataFrame(reduction["participant_records"])); atomic_write_csv(root/"reduction/PARTICIPANT_REDUCTION_METRICS.csv",pd.DataFrame(reduction["metrics"])); atomic_write_csv(root/"reduction/PARTICIPANT_STATE_METRICS.csv",pd.DataFrame(reduction["state_metrics"])); _atomic_npz(root/"reduction/REDUCTION_EFFECTS.npz",A=reduction["A"],balancing_key=reduction["balancing_key"],detailed_effects=reduction["detailed_effects"],reduced_effects=reduction["reduced_effects"])
    return reduced,states,rr,reduction,participants


def _load_panel(path: Path, years: tuple[int,...]=tuple(range(2025,2032))) -> pd.DataFrame:
    columns=["timestamp_utc","year","province","prov_load_MW","nat_load_MW","prov_solar_MW","nat_solar_MW","prov_wind_on_MW","nat_wind_on_MW","prov_wind_off_MW","nat_wind_off_MW","template_year_res"]
    frame=pd.read_csv(path,usecols=columns); frame=frame.loc[frame.year.isin(years)].copy(); frame["timestamp_utc"]=pd.to_datetime(frame.timestamp_utc,utc=True)
    for provincial,national in (("prov_load_MW","nat_load_MW"),("prov_solar_MW","nat_solar_MW"),("prov_wind_on_MW","nat_wind_on_MW"),("prov_wind_off_MW","nat_wind_off_MW")):
        grouped=frame.groupby("timestamp_utc",sort=True); error=(grouped[provincial].sum()-grouped[national].first()).abs().max()
        if float(error)>1e-6: raise RuntimeError(f"{provincial}: national conservation failure {error} MW.")
    return frame


def build_devices(net: Any,bus_ids: tuple[str,...],timestamps: pd.DatetimeIndex,nodal: dict[str,np.ndarray],dispatch: pd.DataFrame,allocations: pd.DataFrame) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,list[dict[str,Any]],dict[str,Any]]:
    bus_int=tuple(map(int,bus_ids)); col={b:i for i,b in enumerate(bus_int)}; T=len(timestamps)
    dispatch=dispatch.copy(); dispatch["timestamp"]=pd.to_datetime(dispatch.timestamp,utc=True); dispatch=dispatch.set_index("timestamp").reindex(timestamps)
    allocations=allocations.copy(); allocations["timestamp"]=pd.to_datetime(allocations.timestamp,utc=True); allocations=allocations.set_index("timestamp").reindex(timestamps)
    if dispatch.isna().any().any() or allocations.isna().any().any(): raise RuntimeError("Public dispatch/curtailment reference does not align with the frozen time grid.")
    C=[]; lo=[]; hi=[]; ref=[]; weights=[]; records=[]
    external_gen=[int(i) for i,r in net.gen.iterrows() if str(r.get("type",""))=="External Grid"]
    internal=[int(i) for i in net.gen.index if int(i) not in external_gen]
    for i in internal:
        row=net.gen.loc[i]; lower=float(row.min_p_mw); upper=float(row.max_p_mw)
        if not np.isfinite(lower+upper) or lower>upper: raise RuntimeError(f"gen::{i}: invalid public bounds.")
        vector=np.zeros(len(bus_ids)); vector[col[int(row.bus)]]=1.0; reference=dispatch[f"dispatch__gen_{i:04d}"].to_numpy(float)
        C.append(vector); lo.append(np.full(T,lower)); hi.append(np.full(T,upper)); ref.append(reference); weights.append(1.0/max(upper-lower,1.0)**2); records.append({"device_id":f"gen::{i}","kind":"INTERNAL_GENERATOR","bus_ids":[int(row.bus)],"lower_source":"ZOMERDIJK_GEN_MIN","upper_source":"ZOMERDIJK_GEN_MAX"})
    async_indices=[i for i in external_gen if any(tag in str(net.gen.loc[i,"name"]) for tag in ("BritNed","COBRA","NorNed"))]
    sync_members=[("ext_grid",int(i)) for i in net.ext_grid.index]+[("gen",i) for i in external_gen if i not in async_indices]
    groups=[("SYNC_AC",sync_members)]+[(f"ASYNC_{i}",[("gen",i)]) for i in async_indices]
    for group,members in groups:
        caps=[]; total=np.zeros(T); vector=np.zeros(len(bus_ids)); member_rows=[]
        for kind,i in members:
            table=getattr(net,kind); row=table.loc[i]; cap=max(abs(float(row.min_p_mw)),abs(float(row.max_p_mw))); caps.append(cap)
            total+=dispatch[f"dispatch__{kind}_{i:04d}"].to_numpy(float); member_rows.append({"type":kind,"index":i,"bus":int(row.bus),"capacity_weight_source_mw":cap})
        beta=np.asarray(caps,float); beta/=beta.sum()
        for share,member in zip(beta,member_rows): vector[col[member["bus"]]]+=float(share)
        C.append(vector); lo.append(total.copy()); hi.append(total.copy()); ref.append(total.copy()); weights.append(1.0/max(sum(caps),1.0)**2); records.append({"device_id":f"boundary::{group}","kind":"FIXED_SIGNED_NET_INTERCHANGE_GROUP","members":member_rows,"spatial_weights":beta.tolist(),"hourly_total_and_sign_frozen_to_public_scenario":True})
    fit=np.asarray([x.year<=2027 for x in timestamps])
    for key,prefix in (("pv","solar"),("wind_on","onshore_wind"),("wind_off","offshore_wind")):
        availability=np.asarray(nodal[key],float)
        for b,bus in enumerate(bus_int):
            upper=availability[:,b]
            if float(upper.max())<=1e-10: continue
            name=str(net.bus.loc[bus,"name"]); reference=allocations[f"curtailment_{prefix}__{name}"].to_numpy(float)
            if np.any(reference< -1e-8) or np.any(reference>upper+1e-6): raise RuntimeError(f"curtail::{key}::{bus}: public reference outside availability.")
            vector=np.zeros(len(bus_ids)); vector[b]=-1.0; C.append(vector); lo.append(np.zeros(T)); hi.append(upper); ref.append(np.maximum(reference,0.0)); scale=max(float(upper[fit].max()),1.0); weights.append(1.0/scale**2); records.append({"device_id":f"curtail::{key}::{bus}","kind":"RENEWABLE_DOWNWARD_FLEXIBILITY","bus_ids":[bus],"upper_source":"SEPARATE_COMPONENT_AVAILABILITY","reference_source":"PAPER_A_PUBLIC_INFERENCE_CURTAILMENT"})
    C=np.column_stack(C); lower=np.column_stack(lo); upper=np.column_stack(hi); reference=np.column_stack(ref); w=np.asarray(weights,float)
    audit={"schema":"TPWRS_BACKGROUND_DEVICE_AUDIT_V1","status":"PASS","devices":len(records),"internal_generators":len(internal),"boundary_groups":len(groups),"renewable_downward_devices":sum(r["kind"]=="RENEWABLE_DOWNWARD_FLEXIBILITY" for r in records),"synchronous_boundary_has_single_signed_quantity":True,"public_scenario_boundary_total_and_sign_retained":True,"reference_bound_violation_mw":float(max(np.max(lower-reference),np.max(reference-upper),0.0)),"device_matrix_hash":sha256_array(C),"reference_weight_hash":sha256_array(w)}
    return C,lower,upper,reference,w,records,audit


def build_background(paths: dict[str,Path],net: Any,states: tuple[Any,...],bus_ids: tuple[str,...],root: Path,years: tuple[int,...]=tuple(range(2025,2032))) -> tuple[pd.DatetimeIndex,np.ndarray]:
    construction_years=tuple(sorted(set(years)|{2025,2026,2027})); panel=_load_panel(paths["component_panel"],construction_years); weights=pd.read_csv(paths["component_weights"]); pnd,nodal,component_audit=build_component_state_from_provincial_panel(panel,weights,tuple(map(int,bus_ids))); timestamps=pd.DatetimeIndex(pd.to_datetime(nodal.pop("timestamps_utc"),utc=True))
    atomic_write_json(root/"components"/("COMPONENT_ALLOCATION_AUDIT.json" if len(years)==7 else f"COMPONENT_ALLOCATION_AUDIT_WORKER_{years[0]}.json"),component_audit)
    if len(years)==7:
        matrix_path=root/"components/TECHNOLOGY_ALLOCATION_MATRICES.npz"; _atomic_npz(matrix_path,W_D=nodal["W_D"],W_PV=nodal["W_PV"],W_ON=nodal["W_ON"],W_OFF=nodal["W_OFF"],bus_ids=np.asarray(bus_ids),province_ids=np.asarray(sorted(panel.province.astype(str).unique()))); component_audit["frozen_matrix_file"]="TECHNOLOGY_ALLOCATION_MATRICES.npz"; component_audit["frozen_matrix_sha256"]=sha256_file(matrix_path); atomic_write_json(root/"components/COMPONENT_ALLOCATION_AUDIT.json",component_audit)
    dispatch=pd.read_parquet(paths["paper_a_dispatch"]); allocations=pd.read_parquet(paths["paper_a_component_allocations"])
    C,lo,hi,ref,w,records,device_audit=build_devices(net,bus_ids,timestamps,nodal,dispatch,allocations); atomic_write_json(root/"background"/("DEVICE_AUDIT.json" if len(years)==7 else f"DEVICE_AUDIT_WORKER_{years[0]}.json"),device_audit)
    if len(years)==7: atomic_write_json(root/"background/DEVICE_REGISTRY.json",records)
    pbg=np.zeros_like(pnd); year_audits={}
    for year in years:
        cache=root/f"background/background_{year}.npz"; status=root/f"background/BACKGROUND_AUDIT_{year}.json"; mask=np.asarray(timestamps.year==year); expected_timestamps=np.asarray([x.isoformat() for x in timestamps[mask]]); legacy_inputs={"year":year,"pnd":sha256_array(pnd[mask]),"C":sha256_array(C),"lo":sha256_array(lo[mask]),"hi":sha256_array(hi[mask]),"ref":sha256_array(ref[mask]),"states":{s.state_id:s.hash for s in states}}; legacy_signature=sha256_json(legacy_inputs); signature=sha256_json({**legacy_inputs,"timestamps":sha256_array(expected_timestamps),"reference_weights":sha256_array(w),"builder_contract_version":2})
        if cache.is_file() and status.is_file():
            audit=json.loads(status.read_text(encoding="utf-8"))
            signature_ok=audit.get("builder_signature")==signature or (audit.get("builder_signature")==legacy_signature and device_audit["reference_weight_hash"]==sha256_array(w))
            if signature_ok and audit.get("cache_sha256")==sha256_file(cache):
                with np.load(cache,allow_pickle=False) as z:
                    if not np.array_equal(z["timestamps_utc"],expected_timestamps): raise RuntimeError(f"{year}: background cache timestamp IDs differ from the frozen grid.")
                    pbg[mask]=z["p_background_mw"]
                if audit.get("builder_signature")==legacy_signature:
                    audit.update({"builder_signature":signature,"signature_migration":"PENDING_FINAL_FULL_ROW_REPLAY","reference_weight_hash":sha256_array(w)}); atomic_write_json(status,audit)
                year_audits[str(year)]=audit; print(f"BACKGROUND_YEAR={year} CACHE=REUSED",flush=True); continue
        counter={"last":0}
        def progress(done:int,total:int,state:str,slack:float,violation:float) -> None:
            if done==total or done-counter["last"]>=250:
                counter["last"]=done; print(f"BACKGROUND_YEAR={year} HOURS={done}/{total} LAST={state} SLACK={slack:.6g} VIOL={violation:.6g}",flush=True)
        built,device_dispatch,audit=security_adjust_background_squared_reference(pnd[mask],C,lo[mask],hi[mask],ref[mask],w,states,progress=progress); pbg[mask]=built; _atomic_npz(cache,p_background_mw=built,device_dispatch_mw=device_dispatch,timestamps_utc=np.asarray([x.isoformat() for x in timestamps[mask]])); audit.update({"year":year,"builder_signature":signature,"cache_sha256":sha256_file(cache)}); atomic_write_json(status,audit); year_audits[str(year)]=audit
    if len(years)==7:
        H=np.vstack([s.H for s in states]); h=np.concatenate([s.h_mw for s in states]); rating=np.concatenate([s.ratings_mw for s in states]); replay={}
        for year in years:
            mask=np.asarray(timestamps.year==year); total=0.0; maximum=0.0
            rows=pbg[mask]
            for start in range(0,len(rows),128):
                flow=rows[start:start+128]@H.T+h[None,:]; violation=np.maximum(np.abs(flow)-rating[None,:],0.0); maximum=max(maximum,float(violation.max(initial=0.0))); violation[violation<=1e-6]=0.0; total+=float(violation.sum())
            solved=float(year_audits[str(year)]["aggregate_minimum_stage1_slack_mw"]); error=abs(total-solved); tolerance=max(1e-3,1e-8*max(1.0,solved))
            if error>tolerance: raise RuntimeError(f"{year}: cached background Stage-I full-row replay differs by {error} MW.")
            replay[str(year)]={"replayed_aggregate_stage1_slack_mw":total,"solved_aggregate_stage1_slack_mw":solved,"absolute_difference_mw":error,"acceptance_tolerance_mw":tolerance,"maximum_row_violation_mw":maximum,"status":"PASS"}
            if year_audits[str(year)].get("signature_migration")=="PENDING_FINAL_FULL_ROW_REPLAY": year_audits[str(year)]["signature_migration"]="LEGACY_V1_SIGNATURE_REVALIDATED_WITH_REFERENCE_WEIGHT_HASH_TIMESTAMP_IDENTITY_AND_FINAL_FULL_ROW_REPLAY"; atomic_write_json(root/f"background/BACKGROUND_AUDIT_{year}.json",year_audits[str(year)])
    else: replay={}
    aggregate={"schema":"TPWRS_BACKGROUND_ALL_YEARS_V1" if len(years)==7 else "TPWRS_BACKGROUND_PARTIAL_V1","status":"PASS","scientific_label":"SECURITY-ADJUSTED PUBLIC-DATA BACKGROUND OPERATING STATE","observed_tennet_dispatch_claim":False,"artificial_border_countertrade_allowed":False,"positive_minimum_stage1_slack_is_reported_not_hidden":True,"aggregate_stage1_slack_units":"MW-h-row on the frozen uniform one-hour grid","years":year_audits,"max_balance_residual_mw":float(np.max(np.abs(pbg.sum(axis=1)))),"aggregate_stage1_slack_mw":float(sum(x["aggregate_minimum_stage1_slack_mw"] for x in year_audits.values())),"independent_full_row_stage1_replay":replay}; atomic_write_json(root/"background"/("BACKGROUND_AUDIT_ALL.json" if len(years)==7 else f"BACKGROUND_AUDIT_PARTIAL_{years[0]}.json"),aggregate)
    return timestamps,pbg


def load_requests(paths: dict[str,Path],participants: pd.DataFrame,primary_ids: list[str],timestamps: pd.DatetimeIndex) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    position={pid:i for i,pid in enumerate(primary_ids)}; blocks=[]
    for year in range(2025,2032):
        frame=pd.read_parquet(paths[f"requests_{year}"],columns=["timestamp","participant_id","requested_mw"]); frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True); frame["participant_id"]=frame.participant_id.astype(str); frame=frame.loc[frame.participant_id.isin(position)].copy()
        if frame.duplicated(["timestamp","participant_id"]).any(): raise RuntimeError(f"{year}: duplicate request coordinates.")
        pivot=frame.pivot(index="timestamp",columns="participant_id",values="requested_mw").reindex(columns=primary_ids)
        # Arrow preserves the request timestamp column at microsecond
        # resolution while the CSV-derived panel is nanosecond-backed.  Compare
        # the exact UTC instants at one canonical storage resolution; this is
        # an identity check, not resampling, rounding, or interpolation.
        actual_index=pd.DatetimeIndex(pivot.index).as_unit("ns")
        expected=pd.DatetimeIndex(timestamps[timestamps.year==year]).as_unit("ns")
        if not np.array_equal(actual_index.asi8,expected.asi8) or pivot.isna().any().any(): raise RuntimeError(f"{year}: requests do not exactly align with the native hourly grid.")
        pivot.index=actual_index
        blocks.append(pivot.to_numpy(float))
    requests=np.vstack(blocks); meta=participants.set_index(participants.participant_id.astype(str)).loc[primary_ids]
    ready=meta.ready_year.to_numpy(int); contract=timestamps.year.to_numpy()[:,None]>=ready[None,:]; eligibility=(requests>1e-7)&contract
    if np.any(eligibility&~contract): raise RuntimeError("Eligibility is not contained in contract duration.")
    return requests,contract,eligibility


def freeze_case(paths: dict[str,Path],registry: dict[str,Any],states: tuple[Any,...],reduction: dict[str,Any],participants: pd.DataFrame,timestamps: pd.DatetimeIndex,pbg: np.ndarray,root: Path) -> dict[str,Any]:
    primary_ids=list(reduction["gate"]["primary_participant_ids"]); order={str(pid):i for i,pid in enumerate(participants.participant_id.astype(str))}; indices=[order[pid] for pid in primary_ids]; meta=participants.iloc[indices].reset_index(drop=True)
    requests,contract,eligibility=load_requests(paths,participants,primary_ids,timestamps); capacities=meta.capacity_mw.to_numpy(float); directions=tuple(meta.direction.astype(str)); A=np.asarray(reduction["A"],float)[:,indices]
    grid=TimeGrid.from_index(timestamps,1.0,timezone="Europe/Amsterdam")
    source_hashes={key:value["sha256"] for key,value in registry["records"].items()}; participant_set_id=("ALL114_" if len(primary_ids)==114 else "DIRECT79_")+sha256_json(primary_ids)[:16]; contingency_set_id="REDUCED_N1_"+sha256_json([s.state_id for s in states])[:16]
    case=PhysicalCase(grid,tuple(primary_ids),tuple(reduction["bus_ids"]),capacities,requests,contract,eligibility,pbg,A,states,directions,participant_set_id,contingency_set_id,reduction["balancing_audit"]["balancing_key_hash"],source_hashes)
    target=root/"physical_case"/"FINAL_DUTCH_CASE"; metadata=save_physical_case(case,target); loaded=load_physical_case(target)
    # Freeze human/audit-facing state and participant metadata beside the
    # numerical arrays.  These are logical records only; local source paths
    # remain exclusively in LOCAL_SOURCE_CONFIG.json.
    state_registry=json.loads((root/"network/NETWORK_STATE_REGISTRY.json").read_text(encoding="utf-8"))
    mapping={"schema":"TPWRS_PRIMARY_PARTICIPANT_MAPPING_V1","status":"PASS","participant_set_id":participant_set_id,"records":[r for r in reduction["participant_records"] if r["participant_id"] in set(primary_ids)],"balancing_key_hash":reduction["balancing_audit"]["balancing_key_hash"]}
    atomic_write_json(target/"NETWORK_STATE_REGISTRY.json",state_registry); atomic_write_json(target/"PARTICIPANT_MAPPING.json",mapping)
    G=np.vstack([state.H@loaded.A for state in loaded.network_states]); requested=loaded.requested_state_mw
    network_audit=json.loads((root/"network/NETWORK_AUDIT.json").read_text(encoding="utf-8"))
    physical_validation={"schema":"TPWRS_PHYSICAL_VALIDATION_V1","status":"PASS","background_balance_max_abs_mw":float(np.max(np.abs(loaded.p_background_mw.sum(axis=1)))),"participant_action_balance_max_abs_mw":float(np.max(np.abs(loaded.A.sum(axis=0)))),"requested_state_balance_max_abs_mw":float(np.max(np.abs(requested.sum(axis=1)))),"H_A_consistency":{"status":"PASS","definition":"G_c=H_c@A using exact frozen state/row order","flat_row_ids_hash":sha256_json(loaded.flat_row_ids),"G_sha256":sha256_array(G),"finite":bool(np.all(np.isfinite(G))),"shape":list(G.shape)},"phase_shift":{"status":"PASS","fixed_phase_shift_transformer_count":network_audit["fixed_phase_shift_transformer_count"],"fixed_phase_shift_max_abs_degree":network_audit["fixed_phase_shift_max_abs_degree"],"affine_h_is_explicit":True,"flat_h_sha256":sha256_array(loaded.flat_h),"max_abs_h_mw":float(np.max(np.abs(loaded.flat_h))),"conclusion":"NO_NONZERO_FIXED_PHASE_SHIFT_PRESENT" if network_audit["fixed_phase_shift_transformer_count"]==0 else "NONZERO_PHASE_SHIFT_RETAINED_IN_AFFINE_OFFSETS"},"round_trip":{"status":"PASS","physical_case_hash":loaded.physical_case_hash,"xbar_exact":bool(np.array_equal(loaded.xbar_mw,case.xbar_mw)),"state_ids_exact":bool(tuple(s.state_id for s in loaded.network_states)==tuple(s.state_id for s in case.network_states)),"row_ids_exact":bool(loaded.flat_row_ids==case.flat_row_ids)},"supplement_hashes":{"network_state_registry":sha256_file(target/"NETWORK_STATE_REGISTRY.json"),"participant_mapping":sha256_file(target/"PARTICIPANT_MAPPING.json")}}
    if max(physical_validation["background_balance_max_abs_mw"],physical_validation["participant_action_balance_max_abs_mw"],physical_validation["requested_state_balance_max_abs_mw"])>loaded.balance_tolerance_mw or not physical_validation["H_A_consistency"]["finite"]: raise RuntimeError("Frozen physical validation failed.")
    atomic_write_json(target/"PHYSICAL_VALIDATION.json",physical_validation)
    request_audit={"schema":"TPWRS_REQUEST_MASK_AUDIT_V1","status":"PASS","native_resolution":"HOURLY","timestamps":len(timestamps),"participants":len(primary_ids),"contract_true_coordinates":int(contract.sum()),"eligible_coordinates":int(eligibility.sum()),"positive_request_coordinates":int(np.sum(requests>1e-7)),"eligibility_source":"EXPLICIT_SCENARIO_ASSUMPTION_ELIGIBLE_WHEN_POSITIVE_REQUEST_AND_WITHIN_CONTRACT_DURATION","chi_subset_kappa":bool(np.all(~eligibility|contract)),"request_hash":sha256_array(requests),"contract_mask_hash":sha256_array(contract.astype(np.uint8)),"eligibility_mask_hash":sha256_array(eligibility.astype(np.uint8))}; atomic_write_json(root/"physical_case/REQUEST_MASK_AUDIT.json",request_audit)
    derived_hashes={"technology_allocation_matrices":sha256_file(root/"components/TECHNOLOGY_ALLOCATION_MATRICES.npz"),"background_audit":sha256_file(root/"background/BACKGROUND_AUDIT_ALL.json"),"network_state_registry":sha256_file(root/"network/NETWORK_STATE_REGISTRY.json"),"reduction_effects":sha256_file(root/"reduction/REDUCTION_EFFECTS.npz"),"reduction_group_summary":sha256_file(root/"reduction/REDUCTION_GROUP_SUMMARY.json")}
    code_hashes={"case_builder":sha256_file(Path(__file__)),"state_builder":sha256_file(ROOT/"tpwrs/inputs.py"),"network_extractor":sha256_file(ROOT/"tpwrs/network.py"),"participant_reduction":sha256_file(ROOT/"tpwrs/network.py"),"physical_case":sha256_file(ROOT/"tpwrs/contracts.py"),"physical_io":sha256_file(ROOT/"tpwrs/persistence.py")}
    freeze={"schema":"TPWRS_PHYSICAL_CASE_FREEZE_V1","status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"physical_case_id":loaded.physical_case_id,"physical_case_hash":loaded.physical_case_hash,"participant_set_id":participant_set_id,"participant_count":loaded.n_participants,"multibus_sensitivity_count":len(reduction["gate"]["sensitivity_only_multibus_ids"]),"contingency_set_id":contingency_set_id,"network_state_count":len(states),"flat_row_count":loaded.n_flat_rows,"time_grid_id":grid.grid_id,"time_count":loaded.n_time,"source_registry_hash":registry["registry_hash"],"derived_artifact_hashes":derived_hashes,"code_hashes":code_hashes,"case_metadata":metadata,"request_mask_audit":request_audit,"physical_validation":physical_validation}; atomic_write_json(root/"physical_case/PHYSICAL_CASE_FREEZE.json",freeze); atomic_write_json(target/"PHYSICAL_CASE_FREEZE.json",freeze); return freeze


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=("registry","network","reduction","background","freeze","all"),default="all"); parser.add_argument("--output",type=Path,default=ROOT/"artifacts/reconstructed_real_case"); parser.add_argument("--background-year",type=int,choices=range(2025,2032)); args=parser.parse_args(); output=args.output.resolve(); paths=discover_sources()
    if output==(ROOT/"artifacts/real_case").resolve() and (output/"physical_case/FINAL_DUTCH_CASE/PHYSICAL_CASE_FREEZE.json").exists():
        raise RuntimeError("The accepted physical case will not be overwritten. Rebuild to artifacts/reconstructed_real_case and audit it as a new identity.")
    if args.background_year is not None and (output/"SOURCE_REGISTRY.json").is_file(): registry=json.loads((output/"SOURCE_REGISTRY.json").read_text(encoding="utf-8")); print(f"SOURCE_REGISTRY=REUSED files={len(registry['records'])}",flush=True)
    else: registry=build_registry(paths,output/"SOURCE_REGISTRY.json"); print(f"SOURCE_REGISTRY=PASS files={len(registry['records'])}",flush=True)
    if args.phase=="registry": return 0
    net,states,_,reduction,participants=build_network_and_reduction(paths,registry,output,write_artifacts=args.background_year is None); print(f"NETWORK=PASS states={len(states)} PARTICIPANT_GATE=PASS primary={len(reduction['gate']['primary_participant_ids'])} sensitivity={len(reduction['gate']['sensitivity_only_multibus_ids'])}",flush=True)
    if args.phase in {"network","reduction"}: return 0
    years=(args.background_year,) if args.background_year is not None else tuple(range(2025,2032))
    if args.background_year is not None and args.phase not in {"background"}: raise RuntimeError("--background-year is valid only with --phase background.")
    timestamps,pbg=build_background(paths,net,states,reduction["bus_ids"],output,years); print(f"BACKGROUND=PASS hours={len(timestamps)}",flush=True)
    if args.phase=="background": return 0
    freeze=freeze_case(paths,registry,states,reduction,participants,timestamps,pbg,output); print(f"PHYSICAL_CASE=PASS id={freeze['physical_case_id']} hash={freeze['physical_case_hash']}",flush=True); return 0
