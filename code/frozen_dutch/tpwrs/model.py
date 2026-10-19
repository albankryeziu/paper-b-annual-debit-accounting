from __future__ import annotations


# ============================================================================
# tpwrs_v3.model_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import asdict, dataclass, replace
import gc
import os
import time
from typing import Mapping

import numpy as np

from tpwrs.storage import ContractError, SolverFailure
from tpwrs.contracts import ExperimentCase
from tpwrs.linear import ModelBuilder, RawSolveResult, solve_linear_model
from tpwrs.contracts import PhysicalCase
from tpwrs.replay import physical_objectives, replay_solution, physical_support_cut, PhysicalSupportCut
from tpwrs.contracts import PolicySolution, StageCertificate
from tpwrs.replay import exact_action_box_screen, hidden_violations, deterministic_violation_seed
from tpwrs.storage import sha256_array
from tpwrs.replay import GENERIC_EXPLICIT_FACE_CONTRACT, generic_face_numeric_tolerance


@dataclass(frozen=True)
class VariableLayout:
    x: np.ndarray
    u: np.ndarray | None


def _validate_inputs(case: PhysicalCase, experiment: ExperimentCase, q: np.ndarray | None) -> None:
    case.verify_experiment_identity(experiment.physical_case_id, experiment.physical_case_hash, experiment.time_grid_id)
    if tuple(experiment.allowance.participant_ids) != case.participant_ids:
        raise ContractError("Allowance participant IDs differ from the physical case.")
    if experiment.rule == "M3":
        if q is None or np.asarray(q).shape != case.requests_mw.shape or (np.any(~np.isfinite(np.asarray(q))) or np.any(np.asarray(q) <= 0)):
            raise ContractError("M3 requires a positive q matrix aligned with the case.")
        if experiment.rate_matrix_hash is not None and experiment.rate_matrix_hash!=sha256_array(np.asarray(q,float)):
            raise ContractError("M3 q matrix differs from the immutable ExperimentCase hash.")
    elif q is not None and experiment.rule not in {"M2_MATCHED"}:
        raise ContractError("q may only be supplied for M3 or matched provenance diagnostics.")


def _transaction_factor(A: np.ndarray) -> tuple[np.ndarray,np.ndarray,float]:
    """Group collinear transaction columns into deterministic sparse directions."""
    matrix=np.asarray(A,dtype=float)
    directions: list[np.ndarray]=[]; assignments: list[tuple[int,float]|None]=[]
    for n in range(matrix.shape[1]):
        column=matrix[:,n]; column_scale=max(1.0,float(np.max(np.abs(column),initial=0.0)))
        if np.max(np.abs(column),initial=0.0)<=1e-14:
            assignments.append(None); continue
        match=None
        for k,direction in enumerate(directions):
            pivot=int(np.argmax(np.abs(direction)))
            scale=float(column[pivot]/direction[pivot])
            if float(np.max(np.abs(column-scale*direction),initial=0.0))<=1e-12*column_scale:
                match=(k,scale); break
        if match is None:
            directions.append(column.copy()); match=(len(directions)-1,1.0)
        assignments.append(match)
    left=np.column_stack(directions) if directions else np.empty((matrix.shape[0],0),float)
    right=np.zeros((len(directions),matrix.shape[1]),float)
    for n,match in enumerate(assignments):
        if match is not None: right[match[0],n]=match[1]
    error=float(np.max(np.abs(left@right-matrix),initial=0.0))
    tolerance=1e-12*max(1.0,float(np.max(np.abs(matrix),initial=0.0)))
    if error>tolerance:
        raise ContractError(f"Transaction factorization is not exact to machine precision ({error} > {tolerance}).")
    return left,right,error


def _build(case: PhysicalCase, experiment: ExperimentCase, active: np.ndarray, *, objective: str, q: np.ndarray | None, stage1_limit: float | None = None, stage2_limit: float | None = None, physical_supports: tuple[PhysicalSupportCut, ...] = ()) -> tuple[object, VariableLayout]:
    T,N,M = case.n_time,case.n_participants,case.n_flat_rows
    dt=case.time_grid.delta; xbar=case.xbar_mw; floor=case.materiality_floor_mw
    builder=ModelBuilder()
    x=np.empty((T,N),np.int32)
    for t in range(T):
        for n in range(N):
            obj = dt[t] if objective=="E" else (dt[t]*(1.0 + (t*N+n+1)/(1000*T*N)) if objective=="TIE" else 0.0)
            x[t,n]=builder.add_var(lb=0,ub=float(xbar[t,n]),obj=obj,name=f"x::{t}::{n}")
    slack=np.full((T,M),-1,np.int32); flow=np.full((T,M),-1,np.int32)
    for t in range(T):
        for r in np.flatnonzero(active[t]):
            coeff=dt[t] if objective=="S" else 0.0
            slack[t,r]=builder.add_var(lb=0,obj=coeff,name=f"sv::{t}::{r}")
            flow[t,r]=builder.add_var(lb=-np.inf,ub=np.inf,obj=0.0,name=f"flow::{t}::{r}")
    u=None
    if experiment.rule=="M1":
        u=np.empty((T,N),np.int32)
        for t in range(T):
            for n in range(N):
                ub=1.0 if xbar[t,n]>0 else 0.0
                u[t,n]=builder.add_var(lb=0,ub=ub,obj=0,integer=True,name=f"u::{t}::{n}")
                builder.add_ub({x[t,n]:1.0,u[t,n]:-float(xbar[t,n])},0.0,f"m1_upper::{t}::{n}")
                builder.add_ub({u[t,n]:float(floor[t,n]),x[t,n]:-1.0},0.0,f"m1_lower::{t}::{n}")
    # The Dutch participant columns occupy only a small set of collinear
    # transaction directions.  The extended formulation is algebraically
    # equivalent to Gx=(H A)x and its linking rows remain extremely sparse.
    factor_left,factor_right,factor_error=_transaction_factor(case.A)
    factor_effect=case.flat_H @ factor_left
    K=factor_right.shape[0]
    latent=np.empty((T,K),np.int32)
    for t in range(T):
        for k in range(K):
            latent[t,k]=builder.add_var(lb=-np.inf,ub=np.inf,obj=0.0,name=f"transaction_factor::{t}::{k}")
            equality={int(latent[t,k]):1.0}
            equality.update({int(x[t,n]):-float(factor_right[k,n]) for n in range(N)})
            builder.add_eq(equality,0.0,f"transaction_factor_definition::{t}::{k}")
    rating=case.flat_ratings
    for start in range(0,T,512):
        stop=min(start+512,T); base=case.requested_state_mw[start:stop]@case.flat_H.T+case.flat_h[None,:]
        for t in range(start,stop):
            for r in np.flatnonzero(active[t]):
                equality={int(flow[t,r]):1.0}; equality.update({int(latent[t,k]):-float(factor_effect[r,k]) for k in range(K)})
                builder.add_eq(equality,float(base[t-start,r]),f"flow_definition::{t}::{r}")
                builder.add_ub({int(flow[t,r]):1.0,int(slack[t,r]):-1.0},float(rating[r]),f"thermal_upper::{t}::{r}")
                builder.add_ub({int(flow[t,r]):-1.0,int(slack[t,r]):-1.0},float(rating[r]),f"thermal_lower::{t}::{r}")
    H=experiment.allowance.values
    for n,pid in enumerate(case.participant_ids):
        coeff: dict[int,float]={}
        if experiment.rule=="M1":
            assert u is not None
            coeff={u[t,n]:float(dt[t]) for t in range(T)}
        elif experiment.rule=="M1_LP":
            for t in range(T):
                if xbar[t,n]>0: coeff[x[t,n]]=float(dt[t]/xbar[t,n])
        elif experiment.rule in {"M2","M2_MATCHED"}:
            coeff={x[t,n]:float(dt[t]/case.capacities_mw[n]) for t in range(T)}
        elif experiment.rule=="M3":
            assert q is not None
            coeff={x[t,n]:float(dt[t]*q[t,n]/case.capacities_mw[n]) for t in range(T)}
        builder.add_ub(coeff,float(H[n]),f"budget::{pid}")
    if stage1_limit is not None:
        coeff={}
        for t in range(T):
            for r in np.flatnonzero(active[t]):
                coeff[int(slack[t,r])]=float(dt[t])
        builder.add_ub(coeff,float(stage1_limit),"lex_stage1_limit")
    if stage2_limit is not None:
        coeff={x[t,n]:float(dt[t]) for t in range(T) for n in range(N)}
        builder.add_ub(coeff,float(stage2_limit),"lex_stage2_limit")
    # These cuts are implied by S(x)<=stage1_limit in exact arithmetic.
    # In Stage I they strengthen the physical epigraph directly. In later
    # stages they avoid accumulating thousands of local auxiliary residuals.
    for index, cut in enumerate(physical_supports):
        coefficients = {int(x[t,n]):float(cut.gradient[t,n])
                        for t in range(T) for n in range(N)
                        if cut.gradient[t,n] != 0.0}
        if objective == "S":
            for t in range(T):
                for r in np.flatnonzero(active[t]):
                    coefficients[int(slack[t,r])] = -float(dt[t])
            rhs = -float(cut.intercept)
        else:
            if stage1_limit is None:
                raise ContractError("Physical face cuts require a fixed Stage-I cap.")
            rhs = float(stage1_limit) - float(cut.intercept)
        builder.add_ub(coefficients, rhs, f"physical_support::{index}")
    # Only x/u are required to decode the policy.  Do not retain the dense
    # (T,M) flow/slack index maps while the solver owns its sparse matrix.
    return builder.build({"transaction_factor_rank":K,"transaction_factor_max_abs_error":factor_error}), VariableLayout(x,u)


def _cert(
    stage: str,
    raw: RawSolveResult,
    *,
    generic_face_tolerance: float | None = None,
) -> StageCertificate:
    metadata=dict(raw.metadata)
    if generic_face_tolerance is not None and stage in {"I","II"}:
        metadata["face_numeric_contract"]=GENERIC_EXPLICIT_FACE_CONTRACT
        metadata["generic_face_certification_tolerance"]=float(generic_face_tolerance)
    return StageCertificate(stage,raw.status,raw.objective_value,raw.runtime_s,raw.best_bound,raw.relative_gap,raw.solver_message,metadata)


def _decode(raw: RawSolveResult, layout: VariableLayout, case: PhysicalCase, q: np.ndarray | None) -> tuple[np.ndarray,np.ndarray|None]:
    x=raw.primal[layout.x]
    u=None if layout.u is None else raw.primal[layout.u]
    return x,u


def _budget_duals(raw: RawSolveResult, names: tuple[str,...], case: PhysicalCase) -> dict[str,float]:
    if raw.inequality_duals is None: return {}
    result={}
    for i,name in enumerate(names):
        if name.startswith("budget::"):
            result[name.split("::",1)[1]]=max(0.0,float(-raw.inequality_duals[i]))
    return result


def _require_exact_optimum(raw: RawSolveResult, stage: str) -> None:
    if raw.status!="OPTIMAL":
        raise SolverFailure(f"{stage} requires a proved optimum; solver returned {raw.status}.")


def solve_policy(case: PhysicalCase, experiment: ExperimentCase, *,
                 q: np.ndarray | None = None, use_stage3: bool = False,
                 initial_active_mask: np.ndarray | None = None,
                 maximum_closure_rounds: int = 100,
                 progress_callback=None, checkpoint_dir=None,
                 initial_directed_rows=None, force_generic: bool = False,
                 initial_x: np.ndarray | None = None,
                 initial_u: np.ndarray | None = None) -> PolicySolution:
    _validate_inputs(case,experiment,q)
    if (not force_generic and not use_stage3 and initial_active_mask is None
        and experiment.mode == "annual"
        and experiment.rule in {"M1_LP", "M2", "M3", "M2_MATCHED"}
        and experiment.solver_profile.backend in {"gurobi", "auto"}):
        from tpwrs.linear import _gurobi_available
        if experiment.solver_profile.backend == "gurobi" or _gurobi_available():
            from tpwrs.annual import solve_annual_gurobi
            return solve_annual_gurobi(case, experiment, q=q,
                progress_callback=progress_callback, checkpoint_dir=checkpoint_dir,
                initial_directed_rows=initial_directed_rows,
                maximum_closure_rounds=maximum_closure_rounds)

    return _solve_generic_guarded(
        case, experiment, q=q, use_stage3=use_stage3,
        initial_active_mask=initial_active_mask,
        maximum_closure_rounds=maximum_closure_rounds,
        initial_x=initial_x, initial_u=initial_u, checkpoint_dir=checkpoint_dir)


def _dump_failure(case, experiment, q, active, checkpoint_dir, error, attempts):
    """A small epoch reproducer. Never overwrite an accepted numerical policy."""
    if checkpoint_dir is None:
        return
    from pathlib import Path
    from tpwrs.persistence import save_physical_case
    from tpwrs.storage import atomic_write_json
    root = Path(checkpoint_dir) / "failure_reproducer"
    root.mkdir(parents=True, exist_ok=True)
    save_physical_case(case, root / "physical_case")
    atomic_write_json(root / "experiment.json", asdict(experiment))
    np.savez_compressed(root / "diagnostic_arrays.npz", active=active,
                        q=np.empty(0) if q is None else q)
    atomic_write_json(root / "FAILURE.json", {
        "schema":"TPWRS_NUMERICAL_REPRODUCER_V1", "status":"FAIL",
        "error":str(error), "attempts":attempts,
        "accepted_as_scientific_result":False})


def _solve_generic_guarded(case, experiment, *, q, use_stage3,
                           initial_active_mask, maximum_closure_rounds,
                           initial_x, initial_u, checkpoint_dir):
    """Exact row/physical-face separation, with bounded numerical recovery.

    No policy coordinate is clipped or projected. Every return passes the same
    independent audit that campaign/rolling consumers subsequently execute.
    The caller's scientific epsilons and solver acceptance envelope never grow.
    """
    q_arr = None if q is None else np.asarray(q, float)
    started = time.monotonic()
    case_budget = float(os.environ.get("TPWRS_MAX_CASE_SECONDS", "14400"))
    if not np.isfinite(case_budget) or case_budget <= 0:
        raise ContractError("Case execution budget must be positive and finite.")
    numeric = generic_face_numeric_tolerance(experiment.solver_profile)
    active = (deterministic_violation_seed(case) if initial_active_mask is None
              else np.asarray(initial_active_mask, bool).copy())
    if active.shape != (case.n_time, case.n_flat_rows):
        raise ContractError("active row mask shape mismatch.")
    if maximum_closure_rounds < 0:
        raise ContractError("maximum_closure_rounds must be nonnegative.")
    if initial_x is not None and np.asarray(initial_x).shape != case.requests_mw.shape:
        raise ContractError("Warm-start x dimensions differ from the epoch.")
    if initial_u is not None and np.asarray(initial_u).shape != case.requests_mw.shape:
        raise ContractError("Warm-start u dimensions differ from the epoch.")
    supports = []
    seen = set()
    attempts = []
    recovery = False
    r1 = None

    def inspect(raw, layout, stage1):
        x, u = _decode(raw, layout, case, q_arr)
        S, E = physical_objectives(case, x)
        temp = PolicySolution(experiment.case_id,experiment.hash,
            case.physical_case_id,case.physical_case_hash,experiment.rule,
            experiment.allowance.hash,x,u,q_arr,
            _cert("I",stage1,generic_face_tolerance=numeric),
            _cert("II",raw,generic_face_tolerance=numeric),None,
            S,E,{},0,int(active.sum()),{})
        report = replay_solution(case,experiment,temp)
        raw_integer_error = (0.0 if layout.u is None else
            float(np.max(np.abs(raw.primal[layout.u]-np.rint(raw.primal[layout.u])),initial=0.0)))
        return x, u, report, raw_integer_error

    def add_support(x):
        cut = physical_support_cut(case,x)
        if cut.selector_hash in seen:
            raise SolverFailure("Physical-face separation stalled on an existing support. "
                                "The numerical policy remains uncertified.")
        supports.append(cut); seen.add(cut.selector_hash)

    def run_stage(objective, name, prior, s_cap=None, e_cap=None):
        nonlocal recovery
        warm_source = prior
        # A bound on corrective iterations is not a permission to accept the
        # final iterate. Exhaustion always raises and saves a reproducer.
        for repair in range(33):
            remaining = case_budget - (time.monotonic()-started)
            if remaining <= 0:
                raise SolverFailure("CASE_TIME_BUDGET: generic solve remains uncertified.")
            effective_profile = replace(experiment.solver_profile,time_limit_s=min(
                remaining,experiment.solver_profile.time_limit_s or remaining))
            matrix, layout = _build(case,experiment,active,objective=objective,
                q=q_arr,stage1_limit=s_cap,stage2_limit=e_cap,
                physical_supports=tuple(supports))
            warm = None
            if warm_source is not None and warm_source.primal.shape == matrix.c.shape:
                warm = warm_source.primal
            elif objective == "S" and initial_x is not None:
                warm = np.full(len(matrix.c),np.nan)
                warm[layout.x] = np.asarray(initial_x,float)
                if layout.u is not None and initial_u is not None:
                    warm[layout.u] = np.asarray(initial_u,float)
            raw = solve_linear_model(matrix,effective_profile,
                stage=name,initial_primal=warm,
                basis_start=None if recovery else warm_source,
                numeric_recovery=recovery)
            _require_exact_optimum(raw,name)
            x,u,report,int_error = inspect(raw,layout,raw if objective=="S" else r1)
            names = matrix.inequality_names
            del matrix,layout
            hidden = hidden_violations(report.violated_mask,active)
            if hidden.any():
                active[:] |= hidden
                attempts.append({"stage":name,"action":"ADD_PHYSICAL_ROWS",
                                 "added":int(hidden.sum()),"S":report.s_value_mw_h_row})
                return None
            limit = raw.objective_value if objective == "S" else s_cap
            face_ok = report.s_value_mw_h_row <= float(limit)+numeric
            energy_ok = (e_cap is None or report.e_value_mwh <= e_cap+numeric)
            if objective == "E":
                energy_ok &= abs(report.e_value_mwh-raw.objective_value) <= numeric
            integer_tol = float(raw.metadata.get(
                "effective_integrality_tol",
                max(1e-6, 10*experiment.solver_profile.feasibility_tol)))
            feasible = report.passed and int_error <= integer_tol
            attempts.append({"stage":name,"numeric_recovery":recovery,
                "S":report.s_value_mw_h_row,"S_limit":float(limit),
                "S_excess":report.s_value_mw_h_row-float(limit),
                "E":report.e_value_mwh,"supports":len(supports),
                "raw_integrality_error":int_error,"physical_replay_passed":bool(feasible),
                "solver_primal_violation":raw.metadata.get("primal_violation")})
            if feasible and face_ok and energy_ok:
                return raw,x,u,report,names
            if not recovery:
                # First retry the identical mathematical model more carefully.
                recovery=True; warm_source=raw
                continue
            if not feasible or not energy_ok:
                raise SolverFailure(f"{name}: physical/non-row replay failed after numerical recovery; "
                    f"budget={report.max_budget_violation_h}, x={report.max_x_bound_violation_mw}, "
                    f"integrality={int_error}, E={report.e_value_mwh}.")
            add_support(x)
            warm_source=raw
        raise SolverFailure(f"{name}: physical-face guard exhausted its separation limit.")

    try:
        for closure_round in range(maximum_closure_rounds+1):
            first=run_stage("S","STAGE_I_SECURITY_RESTORATION",None)
            if first is None:
                continue
            r1,x1,u1,report1,_=first
            s_star=float(r1.objective_value)
            s_cap=s_star+experiment.stage1_tolerance(s_star)
            second=run_stage("E","STAGE_II_MINIMUM_RESTRICTION",r1,s_cap)
            if second is None:
                continue
            r2,x2,u2,report2,names=second
            duals=_budget_duals(r2,names,case) if experiment.rule=="M2" else {}
            selected=second
            third=None
            if use_stage3:
                e_cap=float(r2.objective_value)+experiment.stage2_tolerance(r2.objective_value)
                selected=run_stage("TIE","STAGE_III_DETERMINISTIC_SELECTOR",r2,s_cap,e_cap)
                if selected is None:
                    continue
                third=_cert("III",selected[0])
            raw,x,u,report,_=selected
            metadata={"physical_face_guard":"GENERIC_PHYSICAL_FACE_GUARD_V3",
                      "physical_support_count":len(supports),
                      "numerical_recovery_used":recovery,
                      "stage1_physical_replay_S":report1.s_value_mw_h_row,
                      "stage2_physical_replay_S":report2.s_value_mw_h_row,
                      "stage1_scientific_cap":s_cap,"guard_attempts":attempts}
            c1=_cert("I",r1,generic_face_tolerance=numeric)
            c2=_cert("II",r2,generic_face_tolerance=numeric)
            c1=replace(c1,metadata={**c1.metadata,**metadata})
            c2=replace(c2,metadata={**c2.metadata,**metadata})
            final=PolicySolution(experiment.case_id,experiment.hash,
                case.physical_case_id,case.physical_case_hash,experiment.rule,
                experiment.allowance.hash,x,u,q_arr,c1,c2,third,
                report.s_value_mw_h_row,report.e_value_mwh,duals,
                closure_round,int(active.sum()),{
                    "passed":report.passed,"max_balance_residual_mw":report.max_balance_residual_mw,
                    "max_budget_violation_h":report.max_budget_violation_h,
                    "max_x_bound_violation_mw":report.max_x_bound_violation_mw,
                    "max_row_violation_mw":report.max_row_violation_mw,
                    "physical_support_count":len(supports),"numerical_recovery_used":recovery})
            from tpwrs.execution import validate_policy_solution
            validate_policy_solution(case,experiment,final,require_stage3=use_stage3)
            return final
        raise SolverFailure("Exact row closure exceeded maximum_closure_rounds.")
    except Exception as exc:
        _dump_failure(case,experiment,q_arr,active,checkpoint_dir,exc,attempts)
        raise
