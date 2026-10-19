from __future__ import annotations


# ============================================================================
# tpwrs_v3.m3_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from tpwrs.storage import ContractError, SolverFailure
from tpwrs.contracts import AllowanceSpec, CardSpec
from tpwrs.contracts import PhysicalCase
from tpwrs.contracts import PolicySolution


def average_tie_percentile(values: Sequence[float]) -> np.ndarray:
    s=pd.Series(np.asarray(values,float)); return (s.rank(method="average",pct=True)).to_numpy(float)


def build_teacher_rows(case: PhysicalCase, solution: PolicySolution, *, year: int) -> pd.DataFrame:
    if solution.rule!="M2": raise ContractError("M3 teacher must come from an M2 solution.")
    if solution.stage3 is not None:
        raise ContractError("M3 teacher requires the unperturbed Stage-II primal; a Stage-III-selected solution is not admissible.")
    if solution.stage1.status!="OPTIMAL" or solution.stage2.status!="OPTIMAL" or not solution.replay_summary.get("passed"):
        raise ContractError("M3 teacher requires proved Stage-I/II optima and a passing independent replay.")
    if set(solution.m2_budget_duals)!=set(case.participant_ids): raise ContractError("M2 teacher requires every participant Stage-II budget dual.")
    if solution.physical_case_hash!=case.physical_case_hash: raise ContractError("M2 teacher physical case mismatch.")
    rows=[]; dt=case.time_grid.delta
    directions=case.participant_directions or tuple("unknown" for _ in case.participant_ids)
    for n,pid in enumerate(case.participant_ids):
        mu=float(solution.m2_budget_duals.get(pid,0.0))
        for t,ts in enumerate(case.time_grid.timestamps_utc):
            x=float(solution.x_mw[t,n]); alpha=mu*float(dt[t])*x/float(case.capacities_mw[n])
            rows.append({"year":int(year),"time_index":t,"timestamp":ts,"participant_id":pid,"participant_index":n,"direction":directions[n],"capacity_mw":float(case.capacities_mw[n]),"request_mw":float(case.requests_mw[t,n]),"x_m2_mw":x,"delta_t_hours":float(dt[t]),"mu_m2_stage2":mu,"alpha":alpha})
    frame=pd.DataFrame(rows)
    frame["target_percentile"]=frame.groupby("direction",dropna=False)["alpha"].transform(lambda x: pd.Series(average_tie_percentile(x.to_numpy()),index=x.index))
    return frame


def score_thresholds(scores: Sequence[float]) -> tuple[float,float] | None:
    a=np.asarray(scores,float)
    if len(a)==0 or np.all(a==a[0]): return None
    q=np.quantile(a,[1/3,2/3]); return float(q[0]),float(q[1])


def assign_buckets(scores: Sequence[float], thresholds: tuple[float,float] | None) -> np.ndarray:
    a=np.asarray(scores,float)
    if thresholds is None: return np.ones(len(a),dtype=int)
    lo,hi=thresholds
    return np.where(a<=lo,1,np.where(a<=hi,2,3)).astype(int)


def solve_three_rate_convex_qp(
    linear: Sequence[float],
    quadratic_weights: Sequence[float],
    center: Sequence[float],
    neutrality_weights: Sequence[float],
    *,
    q_bounds: tuple[float,float]=(0.5,1.5),
) -> np.ndarray:
    """Globally solve a convex three-rate QP by complete active-set enumeration.

    This solves

        min_q  linear^T q + 0.5 * sum_i quadratic_weights_i * (q_i-center_i)^2
        s.t.   neutrality_weights^T q = sum_i neutrality_weights_i,
               qlo <= q_i <= qhi,
               q_1 >= q_2 >= q_3.

    The equality reduces the feasible set to a compact polygon in an affine
    two-dimensional plane.  A convex differentiable objective attains its
    global minimum either in the relative interior (no active inequality), on
    an edge (one independent active inequality), or at a vertex (two or more
    active inequalities).  The routine enumerates all three cases, so no
    iterative nonlinear optimizer or scale-sensitive feasibility detection is
    involved.
    """
    linear=np.asarray(linear,float); quad=np.asarray(quadratic_weights,float); center=np.asarray(center,float); neutral=np.asarray(neutrality_weights,float)
    if linear.shape!=(3,) or quad.shape!=(3,) or center.shape!=(3,) or neutral.shape!=(3,):
        raise ContractError("Three-rate QP expects length-three coefficient vectors.")
    if np.any(~np.isfinite(linear)) or np.any(~np.isfinite(quad)) or np.any(~np.isfinite(center)) or np.any(~np.isfinite(neutral)) or np.any(quad<0) or np.any(neutral<0):
        raise ContractError("Three-rate QP coefficients must be finite and weights nonnegative.")
    nsum=float(neutral.sum())
    if nsum<=0: raise ContractError("Three-rate QP requires positive total neutrality weight.")
    qlo,qhi=map(float,q_bounds)
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qlo>1.0 or qhi<1.0 or qlo>qhi:
        raise ContractError("Three-rate QP bounds must be finite, ordered, and contain the neutral rate 1.")

    wn=neutral/nsum
    D=[]; d=[]
    for i in range(3):
        row=np.zeros(3); row[i]=1.0; D.append(row); d.append(qhi)
        row=np.zeros(3); row[i]=-1.0; D.append(row); d.append(-qlo)
    D.extend((np.array([-1.0,1.0,0.0]),np.array([0.0,-1.0,1.0]))); d.extend((0.0,0.0))
    D=np.asarray(D,float); d=np.asarray(d,float)

    H=np.diag(quad); f=linear-quad*center
    scale=max(1.0,float(np.max(np.abs(f))),float(np.max(quad)))
    Hs=H/scale; fs=f/scale

    def feasible(q, tol=3e-9):
        q=np.asarray(q,float)
        return bool(np.all(np.isfinite(q)) and abs(float(wn@q)-1.0)<=tol and np.all(D@q<=d+tol))

    def objective(q):
        q=np.asarray(q,float)
        return float(linear@q + 0.5*np.sum(quad*(q-center)**2))

    candidates=[np.ones(3,float)]

    def add_stationary(active_rows):
        C=np.vstack([wn]+[D[i] for i in active_rows])
        rhs_eq=np.asarray([1.0]+[float(d[i]) for i in active_rows],float)
        K=np.block([[Hs,C.T],[C,np.zeros((C.shape[0],C.shape[0]))]])
        rhs=np.concatenate((-fs,rhs_eq))
        sol=np.linalg.lstsq(K,rhs,rcond=1e-13)[0]
        q=np.asarray(sol[:3],float)
        if np.max(np.abs(C@q-rhs_eq))<=3e-9 and feasible(q): candidates.append(q)

    # Relative-interior and edge stationary points.
    add_stationary(())
    for i in range(len(d)): add_stationary((i,))

    # All vertices.  Avoid determinant thresholds: solve every nonsingular
    # three-equation system and accept it only after explicit feasibility checks.
    for i in range(len(d)):
        for j in range(i+1,len(d)):
            M=np.vstack((wn,D[i],D[j])); rhs=np.asarray([1.0,d[i],d[j]],float)
            try: q=np.linalg.solve(M,rhs)
            except np.linalg.LinAlgError: continue
            if feasible(q): candidates.append(np.asarray(q,float))

    values=np.asarray([objective(q) for q in candidates],float)
    if np.any(~np.isfinite(values)):
        raise SolverFailure("Three-rate active-set solver produced a nonfinite objective candidate.")
    q=np.asarray(candidates[int(np.argmin(values))],float)
    # A semidefinite primary QP can leave a continuum of optimal rates.
    # Its exact optimal face fixes H(q-q*)=0 and f.T(q-q*)=0.
    # Solve a second convex QP on that face, minimizing distance to one.
    if np.any(quad==0):
        import itertools
        base=[wn];base_rhs=[1.0]
        for i in np.flatnonzero(quad>0):
            row=np.zeros(3);row[i]=1.;base.append(row);base_rhs.append(float(q[i]))
        free_linear=np.where(quad==0,linear,0.)
        if np.any(free_linear!=0):
            row=free_linear/float(np.max(np.abs(free_linear)))
            base.append(row);base_rhs.append(float(row@q))
        secondary=[]
        for count in range(4):
            for active in itertools.combinations(range(len(d)),count):
                C=np.vstack(base+[D[i] for i in active])
                b=np.asarray(base_rhs+[float(d[i]) for i in active])
                K=np.block([[np.eye(3),C.T],[C,np.zeros((len(b),len(b)))]] )
                candidate=np.linalg.lstsq(K,np.r_[np.ones(3),b],rcond=1e-13)[0][:3]
                if np.max(np.abs(C@candidate-b))<=3e-9 and feasible(candidate):
                    secondary.append(candidate)
        if not secondary:raise SolverFailure("Minimum-norm optimal-rate face is unresolved.")
        q=min(secondary,key=lambda value:float(np.sum((value-1.)**2)))
    equality_tol=1e-8*max(1.0,abs(nsum))
    if abs(float(neutral@q-nsum))>equality_tol or q[0]+1e-8<q[1] or q[1]+1e-8<q[2] or min(q)<qlo-1e-8 or max(q)>qhi+1e-8:
        raise SolverFailure("Three-rate active-set solver returned a contract-infeasible vector.")
    return q


def rate_card_qp(A: Sequence[float], W: Sequence[float], *, q_bounds: tuple[float,float]=(0.5,1.5), lambda_q: float=1e-3) -> tuple[np.ndarray,str]:
    """Solve the frozen three-bucket rate-card QP without iterative NLP calls."""
    A=np.asarray(A,float); W=np.asarray(W,float)
    if A.shape!=(3,) or W.shape!=(3,) or np.any(~np.isfinite(A)) or np.any(~np.isfinite(W)) or np.any(W<0):
        raise ContractError("Rate QP expects finite length-three A and nonnegative finite W.")
    if not np.isfinite(lambda_q) or lambda_q<0:
        raise ContractError("Rate QP requires a finite nonnegative lambda_q.")
    if np.sum(W>0.0)<2 or W.sum()<=0: return np.ones(3),"FLAT_SPARSE_REFERENCE_FDE"
    q=solve_three_rate_convex_qp(A,lambda_q*W,np.ones(3),W,q_bounds=q_bounds)
    return q,"OPTIMAL_QP"


def bucket_inputs(frame: pd.DataFrame, bucket_column: str="bucket") -> dict[str,dict[str,np.ndarray]]:
    required={"direction","alpha","delta_t_hours","x_m2_mw","capacity_mw",bucket_column}
    if not required.issubset(frame.columns): raise ContractError(f"Teacher rows missing {required-set(frame.columns)}")
    out={}
    for direction,part in frame.groupby("direction"):
        A=[]; W=[]
        for b in (1,2,3):
            hit=part.loc[part[bucket_column].eq(b)]
            A.append(float(hit["alpha"].sum()))
            W.append(float((hit["delta_t_hours"]*hit["x_m2_mw"]/hit["capacity_mw"]).sum()))
        out[str(direction)]={"A":np.asarray(A),"W":np.asarray(W)}
    return out


def make_card(*, card_id: str, allowance_hash: str, feature_set_id: str, model_family: str, fit_frame: pd.DataFrame, score_column: str, q_bounds=(0.5,1.5), lambda_q=1e-3, source_hashes: Mapping[str,str]|None=None) -> CardSpec:
    thresholds={}; rates={}; fallbacks={}; work=fit_frame.copy(); work["bucket"]=1
    for direction,idx in work.groupby("direction").groups.items():
        th=score_thresholds(work.loc[idx,score_column]); thresholds[str(direction)]=(-np.inf,np.inf) if th is None else th
        work.loc[idx,"bucket"]=assign_buckets(work.loc[idx,score_column],th)
    for direction,values in bucket_inputs(work).items():
        q,mode=rate_card_qp(values["A"],values["W"],q_bounds=q_bounds,lambda_q=lambda_q); rates[direction]=tuple(map(float,q))
        if mode.startswith("FLAT"): fallbacks[direction]=mode
    return CardSpec(card_id,allowance_hash,feature_set_id,model_family,thresholds,rates,tuple(q_bounds),float(lambda_q),fallbacks,source_hashes or {})


def q_from_card(card: CardSpec, directions: Sequence[str], scores: np.ndarray) -> np.ndarray:
    scores=np.asarray(scores,float)
    if scores.ndim!=2 or scores.shape[1]!=len(directions): raise ContractError("Score matrix must have shape (time, participant).")
    q=np.ones_like(scores)
    for n,d in enumerate(directions):
        if d not in card.rates: raise ContractError(f"Card has no rates for direction {d!r}.")
        th=card.thresholds[d]; rates=card.rates[d]
        if np.isneginf(th[0]) and np.isposinf(th[1]): buckets=np.ones(scores.shape[0],int)
        else: buckets=assign_buckets(scores[:,n],th)
        q[:,n]=np.asarray([rates[b-1] for b in buckets])
    return q


def matched_allowance(case: PhysicalCase, m3_solution: PolicySolution) -> AllowanceSpec:
    if m3_solution.rule!="M3": raise ContractError("Matched allowance requires a completed M3 solution.")
    if (m3_solution.stage1.status!='OPTIMAL' or m3_solution.stage2.status!='OPTIMAL'
        or not m3_solution.replay_summary.get('passed')
        or m3_solution.physical_case_hash!=case.physical_case_hash):
        raise ContractError('Matched allowance requires a certified, case-aligned M3 trajectory.')
    dt=case.time_grid.delta[:,None]; fde=np.sum(dt*m3_solution.x_mw/case.capacities_mw[None,:],axis=0)
    return AllowanceSpec.explicit(case.participant_ids,fde,source="m3_realized_physical_fde",source_solution_hash=m3_solution.solution_hash)


def verify_m2_matched_contains_m3(case: PhysicalCase, m3_solution: PolicySolution, matched: AllowanceSpec, tol: float=1e-8) -> bool:
    dt=case.time_grid.delta[:,None]; fde=np.sum(dt*m3_solution.x_mw/case.capacities_mw[None,:],axis=0)
    return bool(matched.participant_ids==case.participant_ids
        and matched.source=='m3_realized_physical_fde'
        and matched.source_solution_hash==m3_solution.solution_hash
        and np.array_equal(fde,matched.values))


def matched_decomposition(
    S_m2: float,
    S_m3: float,
    S_matched: float,
    tol: float = 1e-7,
    rel_tol: float = 1e-9,
) -> dict[str, float]:
    """Return the M2/M3 matched-exposure decomposition with numerical guard.

    Mathematically, M2-MATCHED contains the completed M3 trajectory, hence
    S_matched <= S_m3.  Independently solved LPs can reverse that inequality by
    solver-scale noise.  Only a violation larger than the declared absolute plus
    relative numerical envelope is treated as a scientific contract failure.
    """
    values = np.asarray([S_m2, S_m3, S_matched], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ContractError("Matched-exposure decomposition requires finite Stage-I values.")
    if tol < 0 or rel_tol < 0:
        raise ContractError("Matched-exposure tolerances must be nonnegative.")

    effective_tol = float(tol) + float(rel_tol) * max(1.0, abs(float(S_m3)), abs(float(S_matched)))
    raw_shape = float(S_m3 - S_matched)
    if raw_shape < -effective_tol:
        raise ContractError(
            "M2-MATCHED Stage-I value exceeds M3 beyond the numerical tolerance: "
            f"S_m3={S_m3:.12g}, S_matched={S_matched:.12g}, "
            f"difference={raw_shape:.12g}, tolerance={effective_tol:.12g}."
        )

    return {
        "exposure_allowance_effect": float(S_m2 - S_matched),
        "shape_penalty": max(raw_shape, 0.0),
        "realized_m3_gain": float(S_m2 - S_m3),
        "raw_shape_residual": raw_shape,
        "comparison_tolerance": effective_tol,
    }


def exposure_sandwich_holds(physical_fde: np.ndarray, weighted_debit: np.ndarray, q_min: float, q_max: float, tol: float=1e-9) -> bool:
    physical=np.asarray(physical_fde,float); weighted=np.asarray(weighted_debit,float)
    return bool(np.all(weighted>=q_min*physical-tol) and np.all(weighted<=q_max*physical+tol))


# ============================================================================
# tpwrs_v3.controls_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

from typing import Sequence
import numpy as np

from tpwrs.contracts import PhysicalCase
from tpwrs.storage import ContractError, SolverFailure
from tpwrs.contracts import CardSpec


def ptdf_net_relief_scores(case: PhysicalCase) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    H=case.flat_H; h=case.flat_h; G=H@case.A; xbar=case.xbar_mw; rating=case.flat_ratings
    T,N=case.n_time,case.n_participants; gross=np.zeros((T,N)); induced=np.zeros((T,N)); max_effect=np.max(np.abs(G),axis=1)
    # Evaluate the exact one-MW probe while keeping memory bounded.  For a row
    # with |f|+max_n|G_n| < rating, neither the baseline nor any admissible
    # probe can contribute, so omitting it is a proof-based screen rather than
    # an approximation.
    block=128
    for start in range(0,T,block):
        stop=min(start+block,T); base=case.requested_state_mw[start:stop]@H.T+h[None,:]
        for local,t in enumerate(range(start,stop)):
            f=base[local]; baseline=np.maximum(np.abs(f)-rating,0.0); active=(baseline>0.0)|(np.abs(f)+max_effect>=rating)
            if not np.any(active): continue
            delta=np.minimum(1.0,xbar[t]); eligible=delta>0.0
            after=f[active,None]+G[active,:]*delta[None,:]; violation=np.maximum(np.abs(after)-rating[active,None],0.0); before=baseline[active,None]
            denom=np.where(eligible,delta,1.0)
            gross[t]=np.where(eligible,np.sum(np.maximum(before-violation,0.0),axis=0)/denom,0.0)
            induced[t]=np.where(eligible,np.sum(np.maximum(violation-before,0.0),axis=0)/denom,0.0)
    return gross,induced,gross-induced


def inverse_order(scores: np.ndarray) -> np.ndarray:
    return -np.asarray(scores,float)


def frozen_shuffle_scores(scores: np.ndarray, directions: Sequence[str], seeds: Sequence[int]) -> dict[int,np.ndarray]:
    values=np.asarray(scores,float); directions=np.asarray(directions)
    out={}
    for seed in seeds:
        rng=np.random.default_rng(int(seed)); shuffled=values.copy()
        for d in np.unique(directions):
            cols=np.flatnonzero(directions==d)
            for t in range(values.shape[0]): shuffled[t,cols]=values[t,rng.permutation(cols)]
        out[int(seed)]=shuffled
    return out



def transparent_public_score(request_fraction: np.ndarray, provincial_directional: np.ndarray, national_directional: np.ndarray, *, directions: Sequence[str], fitting_mask: np.ndarray) -> np.ndarray:
    """Frozen fitting-reference, within-direction public percentile score."""
    arrays=[np.asarray(request_fraction,float),np.asarray(provincial_directional,float),np.asarray(national_directional,float)]
    if not all(a.shape==arrays[0].shape for a in arrays): raise ValueError("Public-score arrays must align.")
    if arrays[0].ndim!=2: raise ValueError("Public-score arrays must have shape (time, participant).")
    group=np.asarray(tuple(map(str,directions)))
    fit=np.asarray(fitting_mask,bool).reshape(-1)
    if group.shape!=(arrays[0].shape[1],) or fit.shape!=(arrays[0].shape[0],): raise ValueError("Direction/fitting axes do not align with public-score arrays.")
    def pct(a):
        result=np.empty_like(a,dtype=float)
        for direction in np.unique(group):
            cols=np.flatnonzero(group==direction); reference=np.sort(a[np.ix_(fit,cols)].reshape(-1))
            if len(reference)==0: raise ValueError(f"Direction {direction!r} has no fitting reference rows.")
            values=a[:,cols]; left=np.searchsorted(reference,values,side="left"); right=np.searchsorted(reference,values,side="right")
            result[:,cols]=(left+right)/(2.0*len(reference))
        return result
    return sum(pct(a) for a in arrays)/3.0


def project_control_rates(selected_rates: Sequence[float], exposure_weights: Sequence[float], *, q_bounds: tuple[float,float]=(0.5,1.5)) -> tuple[np.ndarray,str]:
    qstar=np.asarray(selected_rates,float); W=np.asarray(exposure_weights,float)
    if qstar.shape!=(3,) or W.shape!=(3,) or np.any(~np.isfinite(qstar)) or np.any(~np.isfinite(W)) or np.any(W<0):
        raise ContractError("Control projection requires finite three-rate and nonnegative exposure-weight vectors.")
    if np.sum(W>0.0)<2 or W.sum()<=0: return np.ones(3),"FLAT_SPARSE_REFERENCE_FDE"
    # min sum W_i(q_i-qstar_i)^2: omit the constant term.
    q=solve_three_rate_convex_qp(np.zeros(3),2.0*W,qstar,W,q_bounds=q_bounds)
    return q,"SELECTED_AMPLITUDE_NEUTRAL_PROJECTION"


def make_ordering_control_card(*, card_id: str, base_card: CardSpec, fit_frame: "object", score_column: str, feature_set_id: str, source_hashes: dict[str,str]|None=None) -> CardSpec:
    import pandas as pd
    if not isinstance(fit_frame,pd.DataFrame): raise ContractError("Control-card fitting rows must be a DataFrame.")
    required={"direction","delta_t_hours","x_m2_mw","capacity_mw",score_column}
    if not required.issubset(fit_frame.columns): raise ContractError(f"Control-card rows missing {sorted(required-set(fit_frame.columns))}")
    thresholds={}; rates={}; fallbacks={}
    for direction,part in fit_frame.groupby("direction",sort=True):
        d=str(direction); threshold=score_thresholds(part[score_column]); thresholds[d]=(-np.inf,np.inf) if threshold is None else threshold; buckets=assign_buckets(part[score_column],threshold); exposure=part.delta_t_hours.to_numpy(float)*part.x_m2_mw.to_numpy(float)/part.capacity_mw.to_numpy(float); W=np.asarray([float(exposure[buckets==b].sum()) for b in (1,2,3)]); q,mode=project_control_rates(base_card.rates[d],W,q_bounds=base_card.q_bounds); rates[d]=tuple(map(float,q))
        if mode.startswith("FLAT"): fallbacks[d]=mode
    return CardSpec(card_id,base_card.allowance_hash,feature_set_id,"ordering_control",thresholds,rates,base_card.q_bounds,base_card.lambda_q,fallbacks,source_hashes or {},base_card.fit_years,base_card.validation_year,base_card.heldout_access_audit)


# ============================================================================
# tpwrs_v3.gamma_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.storage import ContractError, SolverFailure


def _validate_rates(
    q: np.ndarray,
    reference_weight: np.ndarray,
    bounds: tuple[float, float],
    *,
    label: str,
    equality_tolerance: float = 1e-7,
    inequality_tolerance: float = 1e-8,
) -> np.ndarray:
    values = np.asarray(q, dtype=float).reshape(-1)
    weight = np.asarray(reference_weight, dtype=float).reshape(-1)
    if values.shape != (3,) or weight.shape != (3,):
        raise ContractError(f"{label}: expected length-three rate and weight vectors.")
    if not np.all(np.isfinite(values)):
        raise SolverFailure(f"{label}: projected rates are not finite.")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
        raise ContractError(f"{label}: reference weights must be finite and nonnegative.")

    lower, upper = map(float, bounds)
    if not (0.0 < lower <= 1.0 <= upper):
        raise ContractError("Gamma rate bounds must be positive and contain one.")

    if values[0] + inequality_tolerance < values[1]:
        raise SolverFailure(f"{label}: q1 < q2.")
    if values[1] + inequality_tolerance < values[2]:
        raise SolverFailure(f"{label}: q2 < q3.")
    if np.min(values) < lower - inequality_tolerance:
        raise SolverFailure(f"{label}: a rate lies below the lower bound.")
    if np.max(values) > upper + inequality_tolerance:
        raise SolverFailure(f"{label}: a rate lies above the upper bound.")

    if weight.sum() > 0.0:
        residual = abs(float(weight @ values - weight.sum()))
        if residual > equality_tolerance:
            raise SolverFailure(
                f"{label}: reference-mean-one residual {residual:.6g} exceeds "
                f"{equality_tolerance:.6g}."
            )
    return values


def project_gamma_rates(
    selected: np.ndarray,
    reference_weight: np.ndarray,
    gamma: float,
    bounds: tuple[float, float] = (0.25, 2.0),
) -> tuple[np.ndarray, str]:
    """Scale a frozen card around one and project it back to the frozen geometry.

    The endpoints are exact scientific objects:
    ``gamma=0`` is the flat M2 card and ``gamma=1`` is the serialized selected
    card.  Non-endpoint values are projected to the closest ordered, bounded,
    reference-mean-one vector.
    """
    qstar = np.asarray(selected, dtype=float).reshape(-1)
    weight = np.asarray(reference_weight, dtype=float).reshape(-1)
    if qstar.shape != (3,) or weight.shape != (3,):
        raise ContractError("Gamma projection requires length-three vectors.")
    if not np.isfinite(gamma):
        raise ContractError("gamma must be finite.")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
        raise ContractError("Gamma reference weights must be finite and nonnegative.")

    lower, upper = map(float, bounds)
    if not (0.0 < lower <= 1.0 <= upper):
        raise ContractError("Gamma rate bounds must be positive and contain one.")

    effective_weight = weight.copy()
    if effective_weight.sum() <= 0.0:
        effective_weight = np.ones(3, dtype=float)

    # Validate the selected endpoint under the global sensitivity bounds and
    # frozen normalization before it can anchor a path.
    _validate_rates(
        qstar,
        effective_weight,
        (lower, upper),
        label="frozen selected gamma endpoint",
    )

    if abs(float(gamma)) < 1e-12:
        flat = np.ones(3, dtype=float)
        return (
            _validate_rates(
                flat,
                effective_weight,
                (lower, upper),
                label="flat gamma endpoint",
            ),
            "EXACT_FLAT_M2_ENDPOINT",
        )

    if abs(float(gamma) - 1.0) < 1e-12:
        return qstar.copy(), "EXACT_FROZEN_SELECTED_ENDPOINT"

    raw = 1.0 + float(gamma) * (qstar - 1.0)
    objective_weight=effective_weight
    # min sum objective_weight_i (q_i-raw_i)^2 under the frozen
    # neutrality/order/bound geometry.  Use the exact three-rate active-set QP.
    values=solve_three_rate_convex_qp(
        np.zeros(3),
        2.0*objective_weight,
        raw,
        effective_weight,
        q_bounds=(lower,upper),
    )
    values = _validate_rates(
        values,
        effective_weight,
        (lower, upper),
        label=f"gamma={float(gamma):.12g} projection",
    )
    return values, "WEIGHTED_REFERENCE_NEUTRAL_PROJECTION"


# ============================================================================
# tpwrs_v3.kkt_stability (consolidated; see SOURCE_MAP.json)
# ============================================================================
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tpwrs.storage import ContractError


def compare_teacher_attributions(reference: pd.DataFrame, alternate: pd.DataFrame, *, keys=("year","time_index","participant_id"), alpha_col="alpha") -> dict[str,float]:
    r=reference[list(keys)+[alpha_col,"direction"]].copy(); a=alternate[list(keys)+[alpha_col]].copy(); m=r.merge(a,on=list(keys),suffixes=("_ref","_alt"),validate="one_to_one")
    if len(m)!=len(r) or len(m)!=len(a): raise ContractError("KKT stability comparison requires identical teacher keys.")
    correlations=[]
    for _,p in m.groupby("direction"):
        if len(p)>=2:
            left=p[f"{alpha_col}_ref"].to_numpy(float); right=p[f"{alpha_col}_alt"].to_numpy(float)
            # Identical constant arrays are exactly reproducible even though
            # their rank correlation is mathematically undefined.
            if np.array_equal(left,right): value=1.0
            else:
                value=float(spearmanr(left,right).statistic)
                if np.isnan(value): value=0.0
            correlations.append(value)
    difference=np.abs(m[f"{alpha_col}_ref"].to_numpy(float)-m[f"{alpha_col}_alt"].to_numpy(float))
    return {"row_count":float(len(m)),"max_abs_alpha_difference":float(difference.max()) if difference.size else 0.0,"minimum_within_direction_spearman":float(min(correlations)) if correlations else 1.0}


# ============================================================================
# tpwrs_v3.heldout_protocol (consolidated; see SOURCE_MAP.json)
# ============================================================================

import json
from pathlib import Path
from typing import Any

from tpwrs.storage import ContractError
from tpwrs.storage import sha256_file

HELDOUT_AUDIT_SCHEMA = "TPWRS_PRE_FREEZE_HELDOUT_ACCESS_AUDIT_V3"
FIT_YEARS = (2025, 2026, 2027)
VALIDATION_YEAR = 2028
PROHIBITED_OUTCOME_YEARS = (2029, 2030, 2031)


def validate_preheldout_audit(
    path: str | Path,
    *,
    expected_validation_year: int = VALIDATION_YEAR,
) -> tuple[dict[str, Any], str]:
    """Validate the single pre-freeze held-out-access contract used by RUN_03+.

    The audit is intentionally a machine-checkable release-local protocol record.
    It certifies that no held-out policy-outcome artifacts exist in the current
    release output roots before card selection.  It does not claim anything about
    historical human access outside this release.
    """
    audit_path = Path(path)
    if not audit_path.is_file():
        raise ContractError(f"Held-out access audit is missing: {audit_path}")
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError(f"Held-out access audit is unreadable: {audit_path}") from exc

    if payload.get("schema") != HELDOUT_AUDIT_SCHEMA:
        raise ContractError("Held-out access audit schema is not the current protocol.")
    if payload.get("status") != "PASS":
        raise ContractError("Held-out access audit must have status PASS.")
    if tuple(map(int, payload.get("fit_years", ()))) != FIT_YEARS:
        raise ContractError("Held-out audit fitting years differ from protocol.")
    if int(payload.get("validation_year", -1)) != int(expected_validation_year):
        raise ContractError("Held-out audit validation year differs from card selection.")
    if tuple(map(int, payload.get("prohibited_outcome_years", ()))) != PROHIBITED_OUTCOME_YEARS:
        raise ContractError("Held-out audit prohibited years differ from protocol.")
    if payload.get("policy_outcome_files_present") not in ([], (), None):
        raise ContractError("Held-out policy-outcome files existed before card freeze.")
    if payload.get("policy_outcome_files_read") not in ([], (), None):
        raise ContractError("Held-out audit records prohibited policy-outcome access.")
    if payload.get("loader_scope") != "Fitting/selection code reads only 2025-2028 policy/feature outcomes":
        raise ContractError("Held-out audit loader scope differs from protocol.")
    return payload, sha256_file(audit_path)


# ============================================================================
# tpwrs_v3.card_selection_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from tpwrs.storage import ContractError
from tpwrs.contracts import CardSpec
from tpwrs.persistence import card_to_json_dict
from tpwrs.storage import atomic_write_json


@dataclass(frozen=True)
class ValidationCandidate:
    card: CardSpec
    year: int
    s_value: float
    e_value: float
    tie_break_key: str


def select_validation_card(
    candidates: Iterable[ValidationCandidate],
    *,
    validation_year: int = 2028,
    s_tolerance: float | None = None,
    e_tolerance: float | None = None,
) -> ValidationCandidate:
    """Select a card lexicographically on the frozen validation year only."""
    rows = list(candidates)
    if not rows:
        raise ContractError("No validation cards supplied.")
    if any(value is not None and (not np.isfinite(value) or value<0) for value in (s_tolerance,e_tolerance)):
        raise ContractError("Validation tolerances must be nonnegative.")
    if any(row.year != validation_year for row in rows):
        raise ContractError("Card selection may use only the frozen validation year.")
    if any(not row.tie_break_key for row in rows):
        raise ContractError("Every validation candidate requires a deterministic tie key.")

    minimum_s = min(row.s_value for row in rows)
    if s_tolerance is None:s_tolerance=1e-6+1e-9*max(1.,abs(minimum_s))
    stage = [row for row in rows if row.s_value <= minimum_s + s_tolerance]
    minimum_e = min(row.e_value for row in stage)
    if e_tolerance is None:e_tolerance=1e-6+1e-9*max(1.,abs(minimum_e))
    stage = [row for row in stage if row.e_value <= minimum_e + e_tolerance]
    complexity={'ridge':0,'histgb':1,'shallow_histogram_boosting':1}
    return min(stage, key=lambda row: (
        float(np.std([q for rates in row.card.rates.values() for q in rates])),
        complexity.get(row.card.model_family,2),row.card.card_id,row.tie_break_key))


def freeze_selected_card(
    selected: ValidationCandidate,
    path: str | Path,
    *,
    heldout_access_audit: str | None = None,
    heldout_audit_path: str | Path | None = None,
    freeze_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Freeze a selected card only against an explicit, hash-recorded access audit.

    A bare default ``PASS`` is intentionally impossible.  The caller must pass
    the explicit marker and the actual audit artifact that established it.
    """
    if heldout_access_audit != "PASS":
        raise ContractError("Held-out access audit must be explicitly PASS.")
    if heldout_audit_path is None:
        raise ContractError("Card freeze requires the held-out audit artifact path.")

    audit_payload, audit_hash = validate_preheldout_audit(
        heldout_audit_path,
        expected_validation_year=selected.year,
    )

    metadata = dict(freeze_metadata or {})
    metadata["heldout_access_audit_path"] = Path(heldout_audit_path).name
    metadata["heldout_access_audit_sha256"] = audit_hash
    metadata["heldout_access_audit_schema"] = audit_payload.get("schema")

    payload = {
        "schema": "TPWRS_V3_CARD_FREEZE_V2",
        "selected_card_hash": selected.card.hash,
        "card_id": selected.card.card_id,
        "card_spec": card_to_json_dict(selected.card),
        "validation_year": selected.year,
        "validation_S": selected.s_value,
        "validation_E": selected.e_value,
        "tie_break_key": selected.tie_break_key,
        "heldout_access_audit": "PASS",
        "heldout_access_audit_sha256": audit_hash,
        "freeze_metadata": metadata,
    }
    return atomic_write_json(path, payload)


# ============================================================================
# tpwrs_v3.provenance_gate (consolidated; see SOURCE_MAP.json)
# ============================================================================
from pathlib import Path
import json
from tpwrs.storage import atomic_write_json
from tpwrs.execution import utc_now
from tpwrs.storage import sha256_file

def freeze_preheldout_inventory(root: Path, output: Path):
    target=output/'PRE_FREEZE_HELDOUT_ACCESS_AUDIT.json'
    # Inspect only this clean release's derived policy-output roots, not raw frozen
    # physical inputs. This is a release-local machine audit, not a claim about
    # historical human access outside the current release.
    outcomes=[]
    scan_roots=[path for path in root.glob('results_*') if path.is_dir()]
    history=root/'artifacts'/'stage_history'
    if history.is_dir(): scan_roots.append(history)
    for base in scan_roots:
        if not base.is_dir(): continue
        for path in base.rglob('solution.json'):
            if any(f'year_{year}' in path.parts for year in PROHIBITED_OUTCOME_YEARS):
                outcomes.append({'path':str(path.relative_to(root)),'sha256':sha256_file(path)})
    if outcomes:
        raise RuntimeError('Held-out policy outcomes already exist before the card freeze. They will not be labelled unseen. Use an explicitly revised protocol.')
    if target.is_file():
        # Existing state is reusable only if it validates under the exact current
        # protocol. Never silently accept a legacy schema.
        payload,_=validate_preheldout_audit(target,expected_validation_year=VALIDATION_YEAR)
        return payload
    d={
        'schema':HELDOUT_AUDIT_SCHEMA,
        'status':'PASS',
        'created_utc':utc_now(),
        'fit_years':list(FIT_YEARS),
        'validation_year':VALIDATION_YEAR,
        'prohibited_outcome_years':list(PROHIBITED_OUTCOME_YEARS),
        'policy_outcome_files_present':[],
        'policy_outcome_files_read':[],
        'loader_scope':'Fitting/selection code reads only 2025-2028 policy/feature outcomes',
        'limitation':'Does not certify historical human access to outputs in other folders or conversations.',
        'physical_scenario_contains_all_years':True,
    }
    atomic_write_json(target,d)
    validate_preheldout_audit(target,expected_validation_year=VALIDATION_YEAR)
    return d
