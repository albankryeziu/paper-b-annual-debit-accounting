from __future__ import annotations


# ============================================================================
# tpwrs_v3.dutch.dc_network (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np

from tpwrs.storage import ContractError
from tpwrs.storage import sha256_json
from tpwrs.contracts import NetworkState


@dataclass(frozen=True)
class Branch:
    element_id: str
    from_bus: str
    to_bus: str
    susceptance_pu: float
    rating_mw: float
    phase_shift_rad: float = 0.0
    element_type: str = "line"
    in_service: bool = True
    physical_circuit_count: int = 1
    outage_semantics: str = "single_element"
    outage_supported: bool = True

    def __post_init__(self) -> None:
        if self.from_bus==self.to_bus or self.rating_mw<=0 or not np.isfinite(self.susceptance_pu) or abs(self.susceptance_pu)<=0:
            raise ContractError(f"Invalid branch {self.element_id}.")
        if self.physical_circuit_count<1: raise ContractError("physical_circuit_count must be positive.")


def _incidence(bus_ids: tuple[str,...], branches: list[Branch]) -> np.ndarray:
    index={b:i for i,b in enumerate(bus_ids)}; C=np.zeros((len(branches),len(bus_ids)))
    for r,br in enumerate(branches):
        if br.from_bus not in index or br.to_bus not in index: raise ContractError(f"{br.element_id}: bus not in retained bus set.")
        C[r,index[br.from_bus]]=1; C[r,index[br.to_bus]]=-1
    return C


def build_affine_dc_state(bus_ids: Iterable[str], branches: Iterable[Branch], *, state_id: str="INTACT", slack_bus: str|None=None, state_kind: str="INTACT", outaged_element_id: str|None=None) -> NetworkState:
    buses=tuple(map(str,bus_ids)); active=[b for b in branches if b.in_service]
    if len(buses)<2 or not active: raise ContractError("DC state requires buses and active branches.")
    slack=slack_bus or buses[0]
    if slack not in buses: raise ContractError("Slack bus not in bus IDs.")
    C=_incidence(buses,active); b=np.asarray([br.susceptance_pu for br in active],float); phi=np.asarray([br.phase_shift_rad for br in active],float)
    B=C.T@(b[:,None]*C); s=buses.index(slack); keep=[i for i in range(len(buses)) if i!=s]; Br=B[np.ix_(keep,keep)]
    if np.linalg.matrix_rank(Br)<len(keep): raise ContractError(f"{state_id}: topology is islanded/singular after slack removal.")
    inv=np.linalg.inv(Br); map_theta=np.zeros((len(buses),len(buses))); map_theta[np.ix_(keep,keep)]=inv
    shiftinj=C.T@(b*phi); theta_shift=np.zeros(len(buses)); theta_shift[keep]=inv@shiftinj[keep]
    H=(b[:,None]*C)@map_theta; h=b*(C@theta_shift-phi)
    return NetworkState(state_id,tuple(br.element_id for br in active),H,h,np.asarray([br.rating_mw for br in active]),state_kind,outaged_element_id,sha256_json({"branches":[asdict(br) for br in active],"slack":slack}))


def enumerate_retained_contingencies(bus_ids: Iterable[str], branches: Iterable[Branch], *, slack_bus: str|None=None) -> tuple[tuple[NetworkState,...],list[dict[str,str]]]:
    branches=list(branches); states=[build_affine_dc_state(bus_ids,branches,state_id="INTACT",slack_bus=slack_bus)]; excluded=[]
    for br in branches:
        if not br.in_service: continue
        if not br.outage_supported:
            excluded.append({"element_id":br.element_id,"reason":"UNSUPPORTED_OUTAGE"}); continue
        if br.physical_circuit_count>1 and br.outage_semantics!="single_element":
            excluded.append({"element_id":br.element_id,"reason":"UNRESOLVED_OR_EQUIVALENT_PARALLEL_CIRCUIT"}); continue
        trial=[b for b in branches if b.element_id!=br.element_id]
        try:
            states.append(build_affine_dc_state(bus_ids,trial,state_id=f"OUT::{br.element_id}",slack_bus=slack_bus,state_kind="SINGLE_ELEMENT_OUTAGE",outaged_element_id=br.element_id))
        except ContractError as exc:
            excluded.append({"element_id":br.element_id,"reason":"ISLANDING_OR_SINGULAR","detail":str(exc)})
    return tuple(states),excluded


# ============================================================================
# tpwrs_v3.dutch.participant_embeddings (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
from scipy.optimize import minimize

from tpwrs.storage import ContractError, SolverFailure


@dataclass(frozen=True)
class EmbeddingMetrics:
    participant_id: str
    mae: float
    rmse: float
    max_abs_error: float
    e2: float
    einf: float
    sign_agreement: float
    topk_overlap: float
    passed: bool
    state_count: int = 1


def calibrate_simplex_embedding(target: np.ndarray, support_effects: np.ndarray, *, prior: np.ndarray|None=None, row_weight: np.ndarray|None=None, lambda_w: float=1e-6) -> np.ndarray:
    target=np.asarray(target,float).reshape(-1); G=np.asarray(support_effects,float)
    if G.ndim!=2 or G.shape[0]!=len(target): raise ContractError("Support effects must have shape (row,state, support) after flattening.")
    m=G.shape[1]; w0=np.full(m,1/m) if prior is None else np.asarray(prior,float)
    if w0.shape!=(m,) or np.any(w0<0) or w0.sum()<=0: raise ContractError("Invalid embedding prior.")
    w0=w0/w0.sum(); omega=np.ones(len(target)) if row_weight is None else np.asarray(row_weight,float)
    if omega.shape!=(len(target),) or np.any(omega<0): raise ContractError("Invalid row weights.")
    def obj(w):
        e=G@w-target; return float(np.sum(omega*e*e)+lambda_w*np.sum((w-w0)**2))
    res=minimize(obj,w0,method="SLSQP",bounds=[(0,1)]*m,constraints=[{"type":"eq","fun":lambda w:float(np.sum(w)-1)}],options={"ftol":1e-13,"maxiter":1000})
    if not res.success: raise SolverFailure(f"Embedding calibration failed: {res.message}")
    w=np.asarray(res.x,float)
    if np.min(w)<-1e-8 or abs(w.sum()-1)>1e-8: raise SolverFailure("Embedding weights violate simplex.")
    return np.maximum(w,0)/np.maximum(w,0).sum()


def _one_state_metrics(d: np.ndarray, r: np.ndarray, top_k: int) -> tuple[float,float,float,float,float,float,float]:
    diff=r-d; mae=float(np.mean(np.abs(diff))); rmse=float(np.sqrt(np.mean(diff*diff))); max_abs=float(np.max(np.abs(diff)))
    e2=float(np.linalg.norm(diff)/max(np.linalg.norm(d),1e-9)); einf=float(max_abs/max(np.max(np.abs(d)),1e-9))
    material=np.abs(d)>=0.05*max(np.max(np.abs(d)),1e-12)
    sign=float(np.mean(np.sign(r[material])==np.sign(d[material]))) if np.any(material) else 1.0
    k=min(int(top_k),len(d)); td=set(np.argsort(np.abs(d))[-k:]); tr=set(np.argsort(np.abs(r))[-k:]); overlap=len(td&tr)/max(k,1)
    return mae,rmse,max_abs,e2,einf,sign,float(overlap)


def embedding_state_metrics(detailed: np.ndarray, reduced: np.ndarray, state_slices: Mapping[str,slice], *, top_k:int=10) -> list[dict[str,float|str]]:
    d=np.asarray(detailed,float).reshape(-1); r=np.asarray(reduced,float).reshape(-1)
    if d.shape!=r.shape or len(d)==0: raise ContractError("Detailed/reduced sensitivity vectors must align exactly.")
    rows=[]
    covered=np.zeros(len(d),dtype=bool)
    for state_id,part in state_slices.items():
        if not isinstance(part,slice): raise ContractError("state_slices values must be slices.")
        ds,rs=d[part],r[part]
        if len(ds)==0: raise ContractError(f"{state_id}: validation state has no common rows.")
        covered[part]=True; mae,rmse,max_abs,e2,einf,sign,overlap=_one_state_metrics(ds,rs,top_k)
        rows.append({"state_id":str(state_id),"mae":mae,"rmse":rmse,"max_abs_error":max_abs,"e2":e2,"einf":einf,"sign_agreement":sign,"topk_overlap":overlap})
    if not rows or not np.all(covered): raise ContractError("state_slices must cover every detailed/reduced coordinate exactly.")
    return rows


def embedding_metrics(participant_id: str, detailed: np.ndarray, reduced: np.ndarray, *, top_k:int=10, thresholds=(0.10,0.25,0.95,0.80), state_slices: Mapping[str,slice]|None=None) -> EmbeddingMetrics:
    d=np.asarray(detailed,float).reshape(-1); r=np.asarray(reduced,float).reshape(-1)
    if d.shape!=r.shape or len(d)==0: raise ContractError("Detailed/reduced sensitivity vectors must align exactly.")
    state_slices={"ALL":slice(0,len(d))} if state_slices is None else state_slices
    per_state=embedding_state_metrics(d,r,state_slices,top_k=top_k)
    diff=r-d; mae=float(np.mean(np.abs(diff))); rmse=float(np.sqrt(np.mean(diff*diff))); max_abs=float(np.max(np.abs(diff)))
    e2=max(float(x["e2"]) for x in per_state); einf=max(float(x["einf"]) for x in per_state)
    sign=min(float(x["sign_agreement"]) for x in per_state); overlap=min(float(x["topk_overlap"]) for x in per_state)
    t2,ti,ts,tk=thresholds; passed=e2<=t2 and einf<=ti and sign>=ts and overlap>=tk
    return EmbeddingMetrics(str(participant_id),mae,rmse,max_abs,e2,einf,sign,float(overlap),bool(passed),len(per_state))


def validate_embeddings(participant_ids: Sequence[str], detailed_effects: np.ndarray, reduced_effects: np.ndarray, *, thresholds=(0.10,0.25,0.95,0.80)) -> list[EmbeddingMetrics]:
    D=np.asarray(detailed_effects,float); R=np.asarray(reduced_effects,float)
    if D.shape!=R.shape or D.ndim!=2 or D.shape[0]!=len(participant_ids): raise ContractError("Effect matrices must align by exact participant and row ordering.")
    return [embedding_metrics(pid,D[i],R[i],thresholds=thresholds) for i,pid in enumerate(participant_ids)]


def empirical_embedding_thresholds(direct_metrics):
    rows=list(direct_metrics)
    if not rows:raise ContractError("Direct-position benchmark is empty.")
    values=np.array([[m.e2,m.einf,m.sign_agreement,m.topk_overlap] for m in rows],float)
    if not np.isfinite(values).all():raise ContractError("Direct-position benchmark is nonfinite.")
    return tuple(float(np.quantile(values[:,i],quantile)) for i,quantile in enumerate((.95,.95,.05,.05)))


def primary_participant_gate(direct_ids: Sequence[str], multibus_ids: Sequence[str], multibus_metrics: Sequence[EmbeddingMetrics]) -> dict[str,object]:
    metric_by={m.participant_id:m for m in multibus_metrics}
    missing=[pid for pid in multibus_ids if pid not in metric_by]
    if missing: raise ContractError(f"Missing multibus validation metrics: {missing}")
    all_pass=all(metric_by[pid].passed for pid in multibus_ids)
    primary=tuple(direct_ids)+tuple(multibus_ids) if all_pass else tuple(direct_ids)
    sensitivity=tuple() if all_pass else tuple(multibus_ids)
    return {"all_multibus_pass":all_pass,"primary_participant_ids":primary,"sensitivity_only_multibus_ids":sensitivity,"partial_post_outcome_inclusion_allowed":False}


# ============================================================================
# tpwrs_v3.dutch.zomerdijk_extract (consolidated; see SOURCE_MAP.json)
# ============================================================================
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from tpwrs.storage import MethodologyContractConflict
from tpwrs.storage import sha256_array, sha256_file, sha256_json
from tpwrs.contracts import NetworkState


def load_pandapower_network(path: str|Path) -> tuple[Any,dict[str,str]]:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    try:
        import pandapower as pp
    except Exception as exc:
        raise MethodologyContractConflict("pandapower is required locally to extract the Zomerdijk network; NOT EXECUTED IN CHAT ENVIRONMENT") from exc
    if p.suffix.lower()==".json": net=pp.from_json(str(p))
    elif p.suffix.lower() in {".p",".pickle",".pkl"}: net=pp.from_pickle(str(p))
    else: raise MethodologyContractConflict(f"Unsupported Zomerdijk network serialization: {p.suffix}")
    return net,{"source_path":p.name,"sha256":sha256_file(p)}


def network_inventory(net: Any) -> dict[str,int]:
    names=("bus","line","trafo","trafo3w","gen","sgen","load","ext_grid","shunt")
    return {name:int(len(getattr(net,name,[]))) for name in names}


def _ppc_bus_order(net: Any, ppc: dict[str,Any]) -> tuple[int,...]:
    """Map internal PYPOWER columns back to immutable pandapower bus IDs."""
    from pandapower.pypower.idx_bus import BUS_I
    internal=ppc["bus"][:,BUS_I].astype(int)
    lookup=getattr(net,"_pd2ppc_lookups",{}).get("bus")
    out=np.full(len(internal),-1,dtype=int)
    if lookup is not None:
        lookup=np.asarray(lookup)
        for external in net.bus.index:
            if int(external)<len(lookup):
                position=int(lookup[int(external)])
                if 0<=position<len(out): out[position]=int(external)
    if np.any(out<0):
        candidate=tuple(map(int,internal))
        if set(candidate)!=set(map(int,net.bus.index)):
            raise MethodologyContractConflict("Could not map PYPOWER bus columns to pandapower bus IDs.")
        out=np.asarray(candidate,dtype=int)
    return tuple(map(int,out))


def _ppc_branch_map(net: Any, n_branch: int) -> tuple[tuple[str,int],...]:
    mapping:list[tuple[str,int]]=[("ppc_branch",i) for i in range(n_branch)]
    lookup=getattr(net,"_pd2ppc_lookups",{}).get("branch",{})
    if isinstance(lookup,dict):
        for kind,span in lookup.items():
            if span is None: continue
            start,stop=map(int,span)
            table=getattr(net,kind,None)
            indices=list(table.index) if table is not None else list(range(stop-start))
            for offset,row in enumerate(range(start,min(stop,n_branch))):
                mapping[row]=(str(kind),int(indices[offset] if offset<len(indices) else offset))
    return tuple(mapping)


def _connected(ppc: dict[str,Any], branch: np.ndarray) -> bool:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from pandapower.pypower.idx_brch import BR_STATUS, F_BUS, T_BUS
    from pandapower.pypower.idx_bus import BUS_I
    ids=ppc["bus"][:,BUS_I].astype(int); pos={v:i for i,v in enumerate(ids)}
    rows:list[int]=[]; cols:list[int]=[]
    for row in np.flatnonzero(branch[:,BR_STATUS]>0.5):
        f,t=int(branch[row,F_BUS]),int(branch[row,T_BUS])
        if f in pos and t in pos: rows.extend((pos[f],pos[t])); cols.extend((pos[t],pos[f]))
    graph=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(ids),len(ids)))
    return int(connected_components(graph,directed=False,return_labels=False))==1


def _affine_state_from_ppc(
    ppc: dict[str,Any], branch: np.ndarray, *, state_id: str,
    bus_ids: tuple[int,...], row_ids: tuple[str,...], ratings_mw: np.ndarray,
    state_kind: str="INTACT", outaged_element_id: str|None=None,
) -> NetworkState:
    """Build f = H p + h exactly from PYPOWER's B matrices.

    Unlike makePTDF alone, this retains the Pfinj/Pbusinj affine terms created
    by phase shifts and transformer taps.  H is valid on the balanced-injection
    subspace used by PhysicalCase.
    """
    from pandapower.pypower.makeBdc import makeBdc
    from pandapower.pypower.idx_bus import BUS_TYPE, REF
    from pandapower.pypower.idx_brch import BR_STATUS
    active=np.flatnonzero(branch[:,BR_STATUS]>0.5)
    b=branch[active].copy(); rate=np.asarray(ratings_mw,float)[active]
    rows=tuple(row_ids[i] for i in active)
    try:
        # pandapower >=3 follows the current PYPOWER signature.
        built=makeBdc(ppc["bus"],b,return_csr=False)
        Bbus,Bf,Pbusinj,Pfinj=built[:4]
    except TypeError:
        # pandapower 2.x bundled the historical MATPOWER-style signature.
        Bbus,Bf,Pbusinj,Pfinj=makeBdc(float(ppc["baseMVA"]),ppc["bus"],b)
    Bbus=np.asarray(Bbus.toarray() if hasattr(Bbus,"toarray") else Bbus,float)
    Bf=np.asarray(Bf.toarray() if hasattr(Bf,"toarray") else Bf,float)
    Pbusinj=np.asarray(Pbusinj,float).reshape(-1); Pfinj=np.asarray(Pfinj,float).reshape(-1)
    refs=np.flatnonzero(ppc["bus"][:,BUS_TYPE].astype(int)==REF); slack=int(refs[0] if len(refs) else 0)
    keep=np.asarray([i for i in range(len(bus_ids)) if i!=slack],dtype=int)
    reduced=Bbus[np.ix_(keep,keep)]
    if np.linalg.matrix_rank(reduced)<len(keep):
        raise MethodologyContractConflict(f"{state_id}: islanded/singular DC state.")
    inv=np.linalg.inv(reduced)
    theta_map=np.zeros((len(bus_ids),len(bus_ids))); theta_map[np.ix_(keep,keep)]=inv
    theta_offset=np.zeros(len(bus_ids)); theta_offset[keep]=-inv@Pbusinj[keep]
    H=Bf@theta_map
    h=float(ppc["baseMVA"])*(Bf@theta_offset+Pfinj)
    source_hash=sha256_json({"state_id":state_id,"branch":sha256_array(b),"bus":sha256_array(ppc["bus"]),"baseMVA":float(ppc["baseMVA"])})
    return NetworkState(state_id,rows,H,h,rate,state_kind,outaged_element_id,source_hash)


def extract_affine_n1_states(
    net: Any, ratings_mw: Iterable[float], *, include_types: tuple[str,...]=("line","trafo"),
) -> tuple[tuple[str,...],tuple[NetworkState,...],list[dict[str,Any]],dict[str,Any]]:
    """Extract intact plus every credible non-islanding single-element outage."""
    from pandapower.converter import to_ppc
    from pandapower.pypower.idx_brch import BR_STATUS
    ppc=to_ppc(net,calculate_voltage_angles=True,init="flat")
    branch=np.asarray(ppc["branch"],float); mapping=_ppc_branch_map(net,len(branch))
    bus_ids_int=_ppc_bus_order(net,ppc); bus_ids=tuple(map(str,bus_ids_int))
    rating=np.asarray(tuple(ratings_mw),float)
    if rating.shape!=(len(branch),) or np.any(~np.isfinite(rating)) or np.any(rating<=0):
        raise MethodologyContractConflict("Ratings must be positive and align with every PYPOWER branch.")
    element_ids=tuple(f"{kind}::{index}" for kind,index in mapping)
    states=[_affine_state_from_ppc(ppc,branch,state_id="INTACT",bus_ids=bus_ids_int,row_ids=element_ids,ratings_mw=rating)]
    records:list[dict[str,Any]]=[]
    for row,(kind,index) in enumerate(mapping):
        if branch[row,BR_STATUS]<=0.5 or kind not in include_types: continue
        element_id=element_ids[row]; trial=branch.copy(); trial[row,BR_STATUS]=0.0
        table=getattr(net,kind); element=table.loc[index]
        if kind=="line": from_bus=int(element.from_bus); to_bus=int(element.to_bus)
        else: from_bus=int(element.hv_bus); to_bus=int(element.lv_bus)
        record={"element_id":element_id,"element_type":kind,"element_index":index,"ppc_branch_index":row,"from_bus":from_bus,"to_bus":to_bus,"orientation":f"{from_bus}->{to_bus}","rating_mw":float(rating[row]),"row_id":element_id}
        parallel=float(element.get("parallel",1.0))
        if parallel>1.0:
            records.append({**record,"status":"EXCLUDED_EQUIVALENT_PARALLEL_CORRIDOR","parallel_count":parallel}); continue
        if not _connected(ppc,trial):
            records.append({**record,"status":"EXCLUDED_ISLANDING"}); continue
        sid=f"OUT::{element_id}"
        states.append(_affine_state_from_ppc(ppc,trial,state_id=sid,bus_ids=bus_ids_int,row_ids=element_ids,ratings_mw=rating,state_kind="SINGLE_ELEMENT_OUTAGE",outaged_element_id=element_id))
        records.append({**record,"status":"RETAINED","state_id":sid})
    shifts=np.asarray(net.trafo.get("shift_degree",np.zeros(len(net.trafo))),float) if len(net.trafo) else np.zeros(0); taps=np.asarray(net.trafo.get("tap_pos",np.full(len(net.trafo),np.nan)),float) if len(net.trafo) else np.zeros(0); neutral=np.asarray(net.trafo.get("tap_neutral",np.full(len(net.trafo),np.nan)),float) if len(net.trafo) else np.zeros(0)
    unsupported={name:int(len(getattr(net,name))) for name in ("trafo3w","impedance","dcline","xward") if hasattr(net,name)}
    audit={
        "schema":"TPWRS_AFFINE_REDUCED_N1_V1","bus_count":len(bus_ids),
        "ppc_branch_count":len(branch),"candidate_outages":len(records),
        "retained_outages":sum(r["status"]=="RETAINED" for r in records),
        "excluded_islanding":sum(r["status"]=="EXCLUDED_ISLANDING" for r in records),
        "excluded_equivalent_parallel_corridors":sum(r["status"]=="EXCLUDED_EQUIVALENT_PARALLEL_CORRIDOR" for r in records),
        "separately_named_single_circuit_line_rows":int(sum(float(net.line.loc[i].get("parallel",1.0))==1.0 for i in net.line.index)),
        "separately_named_single_transformer_rows":int(sum(float(net.trafo.loc[i].get("parallel",1.0))==1.0 for i in net.trafo.index)),
        "fixed_phase_shift_transformer_count":int(np.sum(np.abs(shifts)>1e-12)),
        "fixed_phase_shift_max_abs_degree":float(np.max(np.abs(shifts),initial=0.0)),
        "transformers_with_explicit_tap_position":int(np.sum(np.isfinite(taps))),
        "transformers_off_neutral_tap":int(np.sum(np.isfinite(taps)&np.isfinite(neutral)&(np.abs(taps-neutral)>1e-12))),
        "unsupported_branch_device_counts":unsupported,
        "unsupported_branch_devices_absent":not any(unsupported.values()),
        "line_rating_min_max_mw":[float(np.min(rating[:len(net.line)])),float(np.max(rating[:len(net.line)]))] if len(net.line) else None,
        "transformer_rating_min_max_mw":[float(np.min(rating[len(net.line):])),float(np.max(rating[len(net.line):]))] if len(net.trafo) else None,
        "phase_offset_max_abs_mw":max(float(np.max(np.abs(s.h_mw))) for s in states),
        "affine_offsets_explicit":True,"outage_semantics":"ONE_PANDAPOWER_LINE_OR_TWO_WINDING_TRANSFORMER_TABLE_ELEMENT",
    }
    if not audit["unsupported_branch_devices_absent"]: raise MethodologyContractConflict(f"Reduced evaluator contains unsupported branch devices: {unsupported}.")
    return bus_ids,tuple(states),records,audit


# ============================================================================
# tpwrs_v3.dutch.network_reduction (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import asdict, dataclass
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from tpwrs.storage import ContractError, MethodologyContractConflict
from tpwrs.storage import sha256_array, sha256_json


@dataclass(frozen=True)
class CommonInterface:
    common_row_id: str
    reduced_branch_row: int
    detailed_branch_row: int
    reduced_element_id: str
    detailed_element_id: str
    reduced_element_type: str
    reduced_element_index: int
    detailed_element_type: str
    detailed_element_index: int
    orientation: int
    rating_mw: float


def _bus_name(net: Any, index: int) -> str:
    return str(net.bus.loc[int(index), "name"])


def _ends(net: Any, kind: str, index: int) -> tuple[str,str]:
    row=getattr(net,kind).loc[int(index)]
    if kind=="line": return _bus_name(net,int(row.from_bus)),_bus_name(net,int(row.to_bus))
    if kind=="trafo": return _bus_name(net,int(row.hv_bus)),_bus_name(net,int(row.lv_bus))
    raise ContractError(f"Unsupported common-interface element type: {kind}")


def map_common_interfaces(
    reduced_net: Any, detailed_net: Any, reduced_ratings_mw: Iterable[float],
) -> tuple[list[CommonInterface],list[dict[str,Any]]]:
    """Map only objectively identical EHV lines/transformers across layers."""
    from pandapower.converter import to_ppc
    rppc=to_ppc(reduced_net,calculate_voltage_angles=True,init="flat")
    dppc=to_ppc(detailed_net,calculate_voltage_angles=True,init="flat")
    rmap=_ppc_branch_map(reduced_net,len(rppc["branch"])); dmap=_ppc_branch_map(detailed_net,len(dppc["branch"]))
    rrow={item:i for i,item in enumerate(rmap)}; drow={item:i for i,item in enumerate(dmap)}
    ratings=np.asarray(tuple(reduced_ratings_mw),float)
    if ratings.shape!=(len(rmap),): raise ContractError("Reduced ratings do not align with PYPOWER branches.")
    detailed_by_name:dict[tuple[str,str],list[int]]={}
    for kind in ("line",):
        for idx,row in getattr(detailed_net,kind).iterrows():
            detailed_by_name.setdefault((kind,str(row.get("name",""))),[]).append(int(idx))
    mapped:list[CommonInterface]=[]; excluded:list[dict[str,Any]]=[]
    for kind in ("line","trafo"):
        for ridx,r in getattr(reduced_net,kind).iterrows():
            ridx=int(ridx); r_from,r_to=_ends(reduced_net,kind,ridx); candidates:list[int]=[]
            method=""
            if kind=="line":
                candidates=detailed_by_name.get((kind,str(r.get("name",""))),[]); method="EXACT_ELEMENT_NAME"
            else:
                target={r_from,r_to}
                candidates=[int(i) for i in detailed_net.trafo.index if set(_ends(detailed_net,"trafo",int(i)))==target]
                method="EXACT_ENDPOINT_PAIR"
            candidates=[i for i in candidates if set(_ends(detailed_net,kind,i))=={r_from,r_to}]
            if len(candidates)!=1:
                excluded.append({"reduced_element_id":f"{kind}::{ridx}","name":str(r.get("name","")),"reason":"NO_UNIQUE_OBJECTIVE_DETAILED_MATCH","candidate_count":len(candidates)})
                continue
            didx=candidates[0]; d_from,d_to=_ends(detailed_net,kind,didx)
            orientation=1 if (d_from,d_to)==(r_from,r_to) else -1
            rr=rrow[(kind,ridx)]; dr=drow[(kind,didx)]
            name=str(r.get("name",f"{r_from}-{r_to}")); common=f"COMMON::{kind}::{ridx}::{name}"
            mapped.append(CommonInterface(common,rr,dr,f"{kind}::{ridx}",f"{kind}::{didx}",kind,ridx,kind,didx,orientation,float(ratings[rr])))
            excluded.append({"reduced_element_id":f"{kind}::{ridx}","detailed_element_id":f"{kind}::{didx}","name":name,"status":"MAPPED","method":method,"orientation":orientation})
    if not mapped: raise MethodologyContractConflict("No objective detailed/reduced common EHV interfaces were mapped.")
    return mapped,excluded


def _json_vector(value: Any, *, dtype: type=float) -> np.ndarray:
    if isinstance(value,str): value=json.loads(value)
    out=np.asarray(value,dtype=dtype).reshape(-1)
    if len(out)==0: raise ContractError("Empty connection support vector.")
    return out


def _recover_balancing_key(
    prior_actions: pd.DataFrame, old_A: np.ndarray, old_bus_ids: np.ndarray,
) -> tuple[np.ndarray,dict[str,Any]]:
    A=np.asarray(old_A,float); buses=np.asarray(old_bus_ids,int).reshape(-1)
    if A.shape!=(len(buses),len(prior_actions)): raise ContractError("Prior action matrix axes do not align with its metadata.")
    estimates=[]
    for j,row in prior_actions.reset_index(drop=True).iterrows():
        if bool(row["connection_is_multibus"]): continue
        bus=int(row["bus"]); locations=np.flatnonzero(buses==bus)
        if len(locations)!=1: raise ContractError(f"Direct participant bus {bus} is not unique in prior action axes.")
        sign=1.0 if str(row["direction"])=="import" else -1.0
        local=np.zeros(len(buses)); local[locations[0]]=1.0
        estimates.append(local-A[:,j]/sign)
    if not estimates: raise ContractError("No direct actions are available to recover the frozen balancing key.")
    stack=np.vstack(estimates); pi=np.mean(stack,axis=0)
    spread=float(np.max(np.abs(stack-pi[None,:])))
    if spread>1e-10 or np.min(pi)<-1e-10 or abs(pi.sum()-1.0)>1e-10:
        raise MethodologyContractConflict(f"Legacy pre-outcome actions do not encode one reproducible simplex balancing key (spread={spread}).")
    pi=np.maximum(pi,0.0); pi/=pi.sum()
    return pi,{"schema":"TPWRS_BALANCING_KEY_RECOVERY_V1","source":"PRE_OUTCOME_PARTICIPANT_ACTION_COLUMNS_DIRECT_SUBSET","direct_columns":len(estimates),"cross_column_max_abs_spread":spread,"simplex_residual":float(abs(pi.sum()-1.0)),"balancing_key_hash":sha256_array(pi)}


def validate_and_build_participant_actions(
    reduced_net: Any,
    detailed_net: Any,
    reduced_ratings_mw: Iterable[float],
    detailed_ratings_mw: Iterable[float],
    participants: pd.DataFrame,
    prior_actions: pd.DataFrame,
    old_A: np.ndarray,
    old_bus_ids: np.ndarray,
    *,
    lambda_w: float=1e-6,
) -> dict[str,Any]:
    """Rebuild A and execute the frozen 383-vs-45 participant gate.

    The legacy multibus vectors are used only as pre-outcome support/prior
    declarations.  Every final weight is recalibrated and judged here against
    freshly extracted detailed sensitivities; no policy outcome is read.
    """
    from pandapower.converter import to_ppc
    from pandapower.pypower.idx_brch import BR_STATUS

    required={"participant_id","station_name","direction","capacity_mw"}
    if not required.issubset(participants.columns): raise ContractError(f"Participant table missing {sorted(required-set(participants.columns))}")
    if set(participants.participant_id.astype(str))!=set(prior_actions.participant_id.astype(str)):
        raise ContractError("Participant and prior-action identifiers differ.")
    prior=prior_actions.copy(); prior["participant_id"]=prior.participant_id.astype(str); prior=prior.set_index("participant_id",drop=False)
    p=participants.copy(); p["participant_id"]=p.participant_id.astype(str)
    if p.participant_id.duplicated().any(): raise ContractError("Duplicate participant IDs.")

    rppc=to_ppc(reduced_net,calculate_voltage_angles=True,init="flat"); dppc=to_ppc(detailed_net,calculate_voltage_angles=True,init="flat")
    rbranch=np.asarray(rppc["branch"],float); dbranch=np.asarray(dppc["branch"],float)
    rratings=np.asarray(tuple(reduced_ratings_mw),float); dratings=np.asarray(tuple(detailed_ratings_mw),float)
    rmap=_ppc_branch_map(reduced_net,len(rbranch)); dmap=_ppc_branch_map(detailed_net,len(dbranch))
    rids=tuple(f"{k}::{i}" for k,i in rmap); dids=tuple(f"{k}::{i}" for k,i in dmap)
    rbuses=_ppc_bus_order(reduced_net,rppc); dbuses=_ppc_bus_order(detailed_net,dppc)
    rcol={int(v):i for i,v in enumerate(rbuses)}; dcol={int(v):i for i,v in enumerate(dbuses)}
    rname={str(reduced_net.bus.loc[i,"name"]):int(i) for i in reduced_net.bus.index}
    dname={str(detailed_net.bus.loc[i,"name"]):int(i) for i in detailed_net.bus.index}
    if len(rname)!=len(reduced_net.bus) or len(dname)!=len(detailed_net.bus): raise MethodologyContractConflict("Network bus names are not unique.")

    pi_old,balancing_audit=_recover_balancing_key(prior_actions,np.asarray(old_A,float),np.asarray(old_bus_ids,int))
    if set(map(int,old_bus_ids))!=set(map(int,reduced_net.bus.index)):
        raise MethodologyContractConflict("Prior balancing-key bus IDs do not match the reduced Zomerdijk source.")
    pi_r=np.zeros(len(rbuses))
    for old_pos,bus in enumerate(np.asarray(old_bus_ids,int)): pi_r[rcol[int(bus)]]=pi_old[old_pos]
    pi_d=np.zeros(len(dbuses))
    for bus,value in zip(rbuses,pi_r):
        name=str(reduced_net.bus.loc[int(bus),"name"])
        if name not in dname: raise MethodologyContractConflict(f"Reduced balancing bus {name} is absent from the detailed model.")
        pi_d[dcol[dname[name]]]+=value
    if abs(pi_d.sum()-1.0)>1e-10: raise ContractError("Detailed balancing key does not sum to one.")

    common,mapping_audit=map_common_interfaces(reduced_net,detailed_net,rratings)
    state_specs:list[tuple[str,CommonInterface|None]]=[("INTACT",None)]
    contingency_audit=[]
    for interface in common:
        rt=rbranch.copy(); dt=dbranch.copy(); rt[interface.reduced_branch_row,BR_STATUS]=0.0; dt[interface.detailed_branch_row,BR_STATUS]=0.0
        if not _connected(rppc,rt) or not _connected(dppc,dt):
            contingency_audit.append({"common_row_id":interface.common_row_id,"status":"EXCLUDED_NOT_COMMON_NONISLANDING"}); continue
        state_specs.append((f"OUT::{interface.common_row_id}",interface)); contingency_audit.append({"common_row_id":interface.common_row_id,"status":"RETAINED"})

    red_blocks=[]; det_blocks=[]; state_slices:dict[str,slice]={}; cursor=0; state_row_ids:dict[str,list[str]]={}
    for state_id,outage in state_specs:
        rt=rbranch.copy(); dt=dbranch.copy()
        if outage is not None:
            rt[outage.reduced_branch_row,BR_STATUS]=0.0; dt[outage.detailed_branch_row,BR_STATUS]=0.0
        sr=_affine_state_from_ppc(rppc,rt,state_id=state_id,bus_ids=rbuses,row_ids=rids,ratings_mw=rratings)
        sd=_affine_state_from_ppc(dppc,dt,state_id=state_id,bus_ids=dbuses,row_ids=dids,ratings_mw=dratings)
        sri={rid:i for i,rid in enumerate(sr.row_ids)}; sdi={rid:i for i,rid in enumerate(sd.row_ids)}
        retained=[x for x in common if x.reduced_element_id in sri and x.detailed_element_id in sdi]
        if not retained: raise MethodologyContractConflict(f"{state_id}: no aligned common monitored rows.")
        ratings=np.asarray([x.rating_mw for x in retained])
        Hr=np.vstack([sr.H[sri[x.reduced_element_id]] for x in retained])/ratings[:,None]
        Hd=np.vstack([x.orientation*sd.H[sdi[x.detailed_element_id]] for x in retained])/ratings[:,None]
        red_blocks.append(Hr); det_blocks.append(Hd); state_slices[state_id]=slice(cursor,cursor+len(retained)); cursor+=len(retained)
        state_row_ids[state_id]=[x.common_row_id for x in retained]
    HR=np.vstack(red_blocks); HD=np.vstack(det_blocks)

    N=len(p); A=np.zeros((len(rbuses),N)); detailed_effect=np.zeros((N,len(HD))); reduced_effect=np.zeros_like(detailed_effect)
    records=[]; metric_records=[]; state_metric_records=[]; metric_objects=[]; cache:dict[str,tuple[np.ndarray,np.ndarray,np.ndarray]]={}
    for j,row in p.reset_index(drop=True).iterrows():
        pid=str(row.participant_id); direction=str(row.direction)
        if direction not in {"import","export"}: raise ContractError(f"{pid}: invalid direction {direction}")
        station=str(row.station_name)
        if station not in dname: raise MethodologyContractConflict(f"{pid}: detailed station {station} is absent.")
        pr=prior.loc[pid]; direct=not bool(pr.connection_is_multibus)
        if direct:
            # A direct position is defined by the predeclared exact reduced
            # operational row, not by requiring its mixed-voltage physical
            # station name itself to survive EHV reduction.
            supports=np.asarray([int(pr.bus)],int); w0=np.asarray([1.0]); source="EXACT_PREDECLARED_OPERATIONAL_ROW"
        else:
            supports=_json_vector(pr.connection_operational_buses,dtype=int); w0=_json_vector(pr.connection_weights,dtype=float); source="PRE_OUTCOME_STRUCTURAL_KRON_SUPPORT_PRIOR"
        if len(supports)!=len(w0) or np.any(w0<0) or w0.sum()<=0: raise ContractError(f"{pid}: invalid support/prior vectors.")
        if any(int(b) not in rcol for b in supports): raise MethodologyContractConflict(f"{pid}: support includes a bus absent from current reduced model.")
        w0=w0/w0.sum(); local_d=np.zeros(len(dbuses)); local_d[dcol[dname[station]]]=1.0; target=HD@(local_d-pi_d)
        support_actions=[]
        for bus in supports:
            probe=np.zeros(len(rbuses)); probe[rcol[int(bus)]]=1.0
            support_actions.append(HR@(probe-pi_r))
        G=np.column_stack(support_actions)
        key=sha256_json({"station":station,"supports":supports.tolist(),"prior":w0.tolist()})
        if key in cache: weights,target_cached,reduced_cached=cache[key]; target=target_cached; reduced=reduced_cached
        else:
            weights=w0 if direct else calibrate_simplex_embedding(target,G,prior=w0,lambda_w=lambda_w)
            reduced=G@weights; cache[key]=(weights,target,reduced)
        sign=1.0 if direction=="import" else -1.0
        local_r=np.zeros(len(rbuses))
        for bus,weight in zip(supports,weights): local_r[rcol[int(bus)]]+=float(weight)
        A[:,j]=sign*(local_r-pi_r); detailed_effect[j]=sign*target; reduced_effect[j]=sign*reduced
        metric=embedding_metrics(pid,sign*target,sign*reduced,state_slices=state_slices); metric_objects.append(metric)
        metric_records.append({**asdict(metric),"mapping_type":"DIRECT" if direct else "MULTIBUS"})
        for item in embedding_state_metrics(sign*target,sign*reduced,state_slices): state_metric_records.append({"participant_id":pid,**item})
        records.append({"participant_id":pid,"direction":direction,"capacity_mw":float(row.capacity_mw),"station_name":station,"connection_is_multibus":not direct,"support_bus_ids":[int(x) for x in supports],"weights":[float(x) for x in weights],"prior_weights":[float(x) for x in w0],"support_source":source,"weight_sum":float(weights.sum()),"action_balance_residual":float(A[:,j].sum()),"passed_frozen_reduction_gate":bool(metric.passed)})
    if np.max(np.abs(A.sum(axis=0)))>1e-10: raise ContractError("Rebuilt participant actions are not balanced.")
    direct_ids=[str(p.iloc[i].participant_id) for i,r in enumerate(records) if not r["connection_is_multibus"]]
    multi_ids=[str(p.iloc[i].participant_id) for i,r in enumerate(records) if r["connection_is_multibus"]]
    thresholds=empirical_embedding_thresholds(m for m in metric_objects if m.participant_id in set(direct_ids))
    metric_objects=[embedding_metrics(pid,detailed_effect[i],reduced_effect[i],
                    thresholds=thresholds,state_slices=state_slices) for i,pid in enumerate(p.participant_id.astype(str))]
    for i,metric in enumerate(metric_objects):
        metric_records[i].update(asdict(metric))
        records[i]["passed_frozen_reduction_gate"]=bool(metric.passed)
    multi_metrics=[m for m in metric_objects if m.participant_id in set(multi_ids)]
    gate=primary_participant_gate(direct_ids,multi_ids,multi_metrics)
    gate.update({"direct_count":len(direct_ids),"multibus_count":len(multi_ids),"common_interface_count":len(common),"common_state_count":len(state_specs),"thresholds":dict(zip(("max_e2","max_einf","min_sign_agreement","min_top10_overlap"),thresholds)),"threshold_basis":"DIRECT_WORST_STATE_EMPIRICAL_95_95_05_05_PERCENTILES","status":"PASS"})
    group_summary={}
    for label,indices in (("DIRECT",[i for i,r in enumerate(records) if not r["connection_is_multibus"]]),("MULTIBUS",[i for i,r in enumerate(records) if r["connection_is_multibus"]])):
        diff=reduced_effect[indices]-detailed_effect[indices]; metrics=[metric_objects[i] for i in indices]
        group_summary[label]={
            "participant_count":len(indices),
            "selection_role":"EMPIRICAL_UNAVOIDABLE_REDUCTION_BENCHMARK" if label=="DIRECT" else "ALL_OR_NONE_PRIMARY_PARTICIPANT_GATE",
            "participant_selection_gate_applied":label=="MULTIBUS",
            "error_units":"FRACTION_OF_COMMON_ROW_RATING_PER_MW_BALANCED_TRANSACTION",
            "pooled_mae":float(np.mean(np.abs(diff))),
            "pooled_rmse":float(np.sqrt(np.mean(diff*diff))),
            "maximum_abs_error":float(np.max(np.abs(diff))),
            "participant_worst_state_sign_agreement_mean":float(np.mean([m.sign_agreement for m in metrics])),
            "participant_worst_state_sign_agreement_minimum":float(np.min([m.sign_agreement for m in metrics])),
            "participant_worst_state_top10_overlap_mean":float(np.mean([m.topk_overlap for m in metrics])),
            "participant_worst_state_top10_overlap_minimum":float(np.min([m.topk_overlap for m in metrics])),
            "maximum_worst_state_e2":float(np.max([m.e2 for m in metrics])),
            "maximum_worst_state_einf":float(np.max([m.einf for m in metrics])),
            "threshold_diagnostic_pass_count":int(sum(m.passed for m in metrics)),
        }
    return {
        "bus_ids":tuple(map(str,rbuses)),"A":A,"participant_records":records,"metrics":metric_records,"state_metrics":state_metric_records,
        "group_summary":{"schema":"TPWRS_REDUCTION_GROUP_SUMMARY_V1","status":"PASS","definitions":{"pooled_mae":"mean absolute reduced-minus-detailed sensitivity error over all participants, common states, and common interfaces in the group","pooled_rmse":"root mean squared reduced-minus-detailed sensitivity error over the same coordinates","maximum_abs_error":"maximum absolute reduced-minus-detailed sensitivity error over the same coordinates"},"groups":group_summary},
        "gate":gate,"balancing_key":pi_r,"balancing_audit":balancing_audit,
        "common_interfaces":[asdict(x) for x in common],"mapping_audit":mapping_audit,"contingency_audit":contingency_audit,
        "state_slices":{k:(v.start,v.stop) for k,v in state_slices.items()},"state_row_ids":state_row_ids,
        "detailed_effects":detailed_effect,"reduced_effects":reduced_effect,
        "effect_identity_hash":sha256_json({"detailed":sha256_array(detailed_effect),"reduced":sha256_array(reduced_effect)}),
    }
