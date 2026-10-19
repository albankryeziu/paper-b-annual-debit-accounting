from __future__ import annotations


# ============================================================================
# tpwrs_v3.linear_model (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass, field
import gc
import time
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, vstack

from tpwrs.storage import SolverFailure
from tpwrs.contracts import SolverProfile
from tpwrs.compression import ActiveCoordinateMap, add_active_mvar


def _fixed_integer_bound_certificate(objective: float, bound: float,
                                     profile: SolverProfile) -> float:
    """Numerical allowance for proving a polished incumbent against its MIP bound."""
    scale=max(1.0,abs(float(objective)),abs(float(bound)))
    return max(2e-6,100*float(profile.feasibility_tol),
               2*float(profile.mip_gap)*scale)


def _fixed_integer_assignment(value: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Return an unambiguous integer assignment for certified LP polishing."""
    integer_value=np.asarray(value,float)[np.asarray(mask,bool)]
    rounded=np.rint(integer_value)
    error=float(np.max(np.abs(integer_value-rounded),initial=0.0))
    if not np.all(np.isfinite(integer_value)) or error>=0.5:
        raise SolverFailure("Native MIP assignment is not a finite unambiguous integer assignment.")
    return rounded,error


@dataclass
class LinearModel:
    c: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    integrality: np.ndarray
    A_ub: csr_matrix
    b_ub: np.ndarray
    inequality_names: tuple[str, ...]
    A_eq: csr_matrix
    b_eq: np.ndarray
    equality_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelBuilder:
    def __init__(self) -> None:
        self.c: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integrality: list[int] = []
        # Store each sparse row in compact numeric buffers immediately.  The
        # former list-of-dicts representation was exact but exhausted memory
        # when it was duplicated into Python COO lists for annual models.
        self.rows: list[tuple[np.ndarray, np.ndarray]] = []
        self.rhs: list[float] = []
        self.names: list[str] = []
        self.eq_rows: list[tuple[np.ndarray, np.ndarray]] = []
        self.eq_rhs: list[float] = []
        self.eq_names: list[str] = []

    def add_var(self, *, lb: float = 0.0, ub: float = np.inf, obj: float = 0.0, integer: bool = False, name: str = "") -> int:
        idx = len(self.c)
        self.c.append(float(obj)); self.lb.append(float(lb)); self.ub.append(float(ub))
        # Variable names are deliberately not retained.  Annual exact models
        # contain millions of variables and their generated diagnostic labels
        # have no role in the mathematics, immutable identifiers, or replay.
        self.integrality.append(1 if integer else 0)
        return idx

    def add_ub(self, coefficients: dict[int, float], rhs: float, name: str) -> int:
        kept=[(int(k),float(v)) for k,v in coefficients.items() if abs(v)>0.0]
        columns=np.fromiter((x[0] for x in kept),dtype=np.int32,count=len(kept))
        values=np.fromiter((x[1] for x in kept),dtype=np.float64,count=len(kept))
        self.rows.append((columns,values))
        self.rhs.append(float(rhs)); self.names.append(str(name) if str(name).startswith(("budget::", "lex_", "physical_support::")) else "")
        return len(self.rows) - 1

    def add_eq(self, coefficients: dict[int, float], rhs: float, name: str) -> int:
        kept=[(int(k),float(v)) for k,v in coefficients.items() if abs(v)>0.0]
        columns=np.fromiter((x[0] for x in kept),dtype=np.int32,count=len(kept))
        values=np.fromiter((x[1] for x in kept),dtype=np.float64,count=len(kept))
        self.eq_rows.append((columns,values)); self.eq_rhs.append(float(rhs)); self.eq_names.append("")
        return len(self.eq_rows)-1

    def build(self, metadata: dict[str, Any] | None = None) -> LinearModel:
        n = len(self.c)
        def compact(rows: list[tuple[np.ndarray,np.ndarray]]) -> csr_matrix:
            lengths=np.fromiter((len(columns) for columns,_ in rows),dtype=np.int64,count=len(rows))
            indptr=np.empty(len(rows)+1,dtype=np.int64); indptr[0]=0; np.cumsum(lengths,out=indptr[1:])
            indices=np.empty(int(indptr[-1]),dtype=np.int32); data=np.empty(int(indptr[-1]),dtype=np.float64)
            for i,(columns,values) in enumerate(rows):
                start,end=int(indptr[i]),int(indptr[i+1]); indices[start:end]=columns; data[start:end]=values
            return csr_matrix((data,indices,indptr),shape=(len(rows),n),dtype=float,copy=False)
        A=compact(self.rows); Aeq=compact(self.eq_rows)
        return LinearModel(np.asarray(self.c,float),np.asarray(self.lb,float),np.asarray(self.ub,float),np.asarray(self.integrality,int),A,np.asarray(self.rhs,float),tuple(self.names),Aeq,np.asarray(self.eq_rhs,float),tuple(self.eq_names),metadata or {})


@dataclass(frozen=True)
class RawSolveResult:
    status: str
    objective_value: float
    primal: np.ndarray
    runtime_s: float
    best_bound: float | None
    relative_gap: float | None
    inequality_duals: np.ndarray | None
    solver_message: str
    metadata: dict[str, Any]
    variable_basis: np.ndarray | None = None
    inequality_basis: np.ndarray | None = None
    equality_basis: np.ndarray | None = None


_STATUS_LP = {0:"OPTIMAL",1:"LIMIT",2:"INFEASIBLE",3:"UNBOUNDED",4:"NUMERICAL"}
_STATUS_MIP = {0:"OPTIMAL",1:"LIMIT",2:"INFEASIBLE",3:"UNBOUNDED",4:"OTHER"}


def _gurobi_available() -> bool:
    try:
        import gurobipy  # noqa: F401
        return True
    except Exception:
        return False


def _solve_gurobi(model: LinearModel, profile: SolverProfile, *, stage: str,
                  initial_primal: np.ndarray | None = None,
                  basis_start: RawSolveResult | None = None,
                  numeric_recovery: bool = False) -> RawSolveResult:
    import os
    from pathlib import Path
    import gurobipy as gp
    from gurobipy import GRB
    from tpwrs.execution import ExecutionBudget
    started=time.perf_counter()
    root=os.environ.get("TPWRS_CURRENT_JOB_ROOT")
    budget=ExecutionBudget(None if root is None else Path(root)/"generic_solver")
    budget.set_stage(stage)
    m=None
    try:
        m=gp.Model(stage)
        is_mip=bool(np.any(model.integrality!=0))
        m.Params.OutputFlag=int(profile.output_flag or bool(root))
        m.Params.Presolve=2 if profile.presolve else 0
        m.Params.FeasibilityTol=min(max(profile.feasibility_tol,1e-9),1e-2)
        m.Params.OptimalityTol=min(max(profile.feasibility_tol,1e-9),1e-2)
        m.Params.MIPGap=max(profile.mip_gap,0.0)
        m.Params.IntFeasTol=min(max(profile.feasibility_tol,1e-9),1e-2)
        if numeric_recovery:
            m.Params.FeasibilityTol=1e-9
            m.Params.OptimalityTol=1e-9
            m.Params.IntFeasTol=1e-9
            m.Params.NumericFocus=3
            m.Params.Quad=1
        m.Params.Threads=int(profile.threads);m.Params.Seed=int(profile.seed)
        if profile.time_limit_s is not None:m.Params.TimeLimit=profile.time_limit_s
        if not is_mip:
            m.Params.Method={"auto":-1,"primal_simplex":0,"dual_simplex":1,"barrier":2}[profile.lp_method]
        if root:
            logroot=Path(root)/"generic_solver";logroot.mkdir(parents=True,exist_ok=True)
            m.Params.LogFile=str(logroot/(stage+".log"))
        vt=np.where(model.integrality==0,GRB.CONTINUOUS,
                    np.where((model.lower_bounds>=0)&(model.upper_bounds<=1),GRB.BINARY,GRB.INTEGER))
        lb=np.where(np.isneginf(model.lower_bounds),-GRB.INFINITY,model.lower_bounds)
        ub=np.where(np.isposinf(model.upper_bounds),GRB.INFINITY,model.upper_bounds)
        v=add_active_mvar(m,len(model.c),lb=lb,ub=ub,vtype=vt)
        ine=m.addMConstr(model.A_ub,v,GRB.LESS_EQUAL,model.b_ub)
        eq=m.addMConstr(model.A_eq,v,GRB.EQUAL,model.b_eq) if model.A_eq.shape[0] else None
        m.setObjective(model.c@v,GRB.MINIMIZE);m.update()
        # Configure the memory guard only after Gurobi owns the complete model.
        # MemUsed is otherwise still near zero, so the derived SoftMemLimit can
        # be lower than memory already committed by the model and the optimizer
        # aborts immediately on machines with modest free RAM.
        budget.configure(m, model_size=(len(model.c), len(model.b_ub)+len(model.b_eq),
                                       model.A_ub.nnz+model.A_eq.nnz))
        basis_used=False;start_used=False
        if not is_mip and basis_start is not None:
            vb=basis_start.variable_basis;cb=basis_start.inequality_basis;eb=basis_start.equality_basis
            if vb is not None and cb is not None and eb is not None and len(vb)==len(model.c) and len(cb)<=model.A_ub.shape[0] and len(eb)==model.A_eq.shape[0]:
                v.VBasis=np.asarray(vb,dtype=np.int32)
                ine.CBasis=np.concatenate([cb,np.zeros(model.A_ub.shape[0]-len(cb),dtype=np.int32)])
                if eq is not None:eq.CBasis=eb
                basis_used=True
        if initial_primal is not None and not basis_used:
            value=np.asarray(initial_primal,float)
            if value.shape!=(len(model.c),):raise SolverFailure("Warm start dimension mismatch.")
            if is_mip:
                v.Start=np.where(np.isfinite(value),value,GRB.UNDEFINED);start_used=True
            elif np.all(np.isfinite(value)):
                v.PStart=value;start_used=True
        counts={"model_rows":int(m.NumConstrs),"model_columns":int(m.NumVars),"model_nonzeros":int(m.NumNZs)}
        if root:print(f"[{stage}] rows={m.NumConstrs} cols={m.NumVars} nnz={m.NumNZs}",flush=True)
        # The sparse representation stays intact for independent callers; the
        # production annual path uses the persistent engine, not this reference.
        m.optimize(lambda mm,where:budget.check_callback(mm))
        budget.check_status(m)
        value=np.asarray(v.X,float)
        if is_mip:
            mask=np.asarray(model.integrality)!=0
            rounded_integer,error=_fixed_integer_assignment(value,mask)
            if error>0.0:
                # Reading X after an OPTIMAL native MIP can expose a residual
                # slightly larger than IntFeasTol (observed in the 2031
                # rolling campaign). Do not accept or edit that continuous
                # policy. Fix the intended integer assignment, re-optimize
                # every continuous coordinate, and require the resulting LP
                # objective to certify the native MIP bound below.
                from dataclasses import replace
                fixed_lb=model.lower_bounds.copy();fixed_ub=model.upper_bounds.copy()
                fixed_lb[mask]=rounded_integer;fixed_ub[mask]=fixed_lb[mask]
                fixed_model=replace(model,lower_bounds=fixed_lb,upper_bounds=fixed_ub,
                                    integrality=np.zeros_like(model.integrality))
                polished=solve_linear_model(fixed_model,profile,stage=stage+"_FIXED_INTEGER",
                                            numeric_recovery=True)
                bound=float(m.ObjBound)
                tolerance=_fixed_integer_bound_certificate(
                    polished.objective_value,bound,profile)
                if not np.isfinite(bound) or polished.objective_value-bound>tolerance:
                    raise SolverFailure(
                        "Fixed-integer native solve does not certify the MIP bound; "
                        f"gap={polished.objective_value-bound:.3e}, cert={tolerance:.3e}.")
                return replace(polished,best_bound=bound,relative_gap=float(m.MIPGap),
                    runtime_s=time.perf_counter()-started,inequality_duals=None,
                    metadata={**polished.metadata,"fixed_integer_polish":True,
                              "fixed_integer_bound_certificate":tolerance,
                              "raw_integrality_error":error,"primal_warm_start":start_used})
        dual=None if is_mip else np.asarray(ine.Pi,float)
        vb=cb=eb=None
        if not is_mip:
            try:
                vb=np.asarray(v.VBasis,dtype=np.int32);cb=np.asarray(ine.CBasis,dtype=np.int32)
                eb=np.asarray(eq.CBasis,dtype=np.int32) if eq is not None else np.empty(0,dtype=np.int32)
            except gp.GurobiError:pass
        metadata={"backend":"gurobi_sparse_matrix","node_count":float(m.NodeCount),
                  "primal_warm_start":start_used,"basis_warm_start":basis_used,
                  "numeric_recovery":bool(numeric_recovery),
                  "effective_feasibility_tol":float(m.Params.FeasibilityTol),
                  "effective_integrality_tol":float(m.Params.IntFeasTol),
                  "numeric_focus":int(m.Params.NumericFocus),
                  "quad":int(m.Params.Quad),
                  "integer_violation":float(m.IntVio) if is_mip else 0.0,
                  "bound_kind":"solver_reported_bound" if is_mip else "solver_optimum_with_numeric_tolerances",
                  "primal_violation":max(float(m.ConstrVio),float(m.BoundVio)),**counts}
        return RawSolveResult("OPTIMAL",float(m.ObjVal),value,time.perf_counter()-started,
                  float(m.ObjBound) if is_mip else float(m.ObjVal),float(m.MIPGap) if is_mip else None,
                  dual,"Gurobi OPTIMAL",metadata,vb,cb,eb)
    finally:
        budget.close()
        if m is not None: m.dispose()


def solve_linear_model(model: LinearModel, profile: SolverProfile, *, stage: str, initial_primal: np.ndarray | None = None, basis_start: RawSolveResult | None = None,
                  numeric_recovery: bool = False) -> RawSolveResult:
    from dataclasses import replace
    mapping = ActiveCoordinateMap.from_bounds(len(model.c), model.lower_bounds, model.upper_bounds)
    if not mapping.is_identity and mapping.active_size:
        active = mapping.active_flat
        reduced = replace(model, c=model.c[active], lower_bounds=model.lower_bounds[active],
            upper_bounds=model.upper_bounds[active], integrality=model.integrality[active],
            A_ub=model.A_ub[:,active], A_eq=model.A_eq[:,active])
        basis = basis_start
        if basis is not None:
            basis = replace(basis, primal=basis.primal[active],
                variable_basis=None if basis.variable_basis is None else basis.variable_basis[active])
        raw = solve_linear_model(reduced, profile, stage=stage,
            initial_primal=None if initial_primal is None else np.asarray(initial_primal)[active],
            basis_start=basis, numeric_recovery=numeric_recovery)
        return replace(raw, primal=mapping.expand(raw.primal),
            variable_basis=None if raw.variable_basis is None else mapping.expand(raw.variable_basis, fill=-1),
            metadata={**raw.metadata, "full_coordinates":mapping.full_size,
                      "active_coordinates":mapping.active_size,
                      "dropped_fixed_zero_coordinates":mapping.dropped_size})
    if profile.backend == "gurobi" or (profile.backend == "auto" and _gurobi_available()):
        return _solve_gurobi(model, profile, stage=stage, initial_primal=initial_primal, basis_start=basis_start, numeric_recovery=numeric_recovery)
    started = time.perf_counter()
    if np.any(model.integrality != 0):
        matrix=vstack([model.A_ub,model.A_eq],format="csr"); lower=np.concatenate([np.full(len(model.b_ub),-np.inf),model.b_eq]); upper=np.concatenate([model.b_ub,model.b_eq]); constraints = LinearConstraint(matrix,lower,upper)
        options: dict[str, Any] = {"disp": bool(profile.output_flag), "presolve": bool(profile.presolve), "mip_rel_gap": float(profile.mip_gap)}
        if profile.time_limit_s is not None: options["time_limit"] = float(profile.time_limit_s)
        # scipy.optimize.milp 1.11 intentionally exposes only a small option
        # surface.  Do not rely on undocumented HiGHS tolerance keys.
        result = milp(c=model.c, integrality=model.integrality, bounds=Bounds(model.lower_bounds, model.upper_bounds), constraints=constraints, options=options)
        status = _STATUS_MIP.get(int(result.status), f"STATUS_{result.status}")
        if result.x is None or result.fun is None:
            raise SolverFailure(f"{stage}: SciPy MILP failed ({status}): {result.message}")
        if status == "LIMIT": status = "FEASIBLE_LIMIT"

        primal=np.asarray(result.x,float)
        integer_mask=np.asarray(model.integrality)!=0
        integer_error=float(np.max(np.abs(primal[integer_mask]-np.rint(primal[integer_mask])),initial=0.0))
        polished=False
        polish_delta=0.0
        # HiGHS may return a numerically integral incumbent with tiny nonzero
        # continuous variables linked to a binary that is within its MIP
        # integrality envelope.  Never repair x/u coordinate-wise.  Instead fix
        # the intended integer assignment and re-optimize the continuous variables
        # in the exact same linear model.  This path is a SciPy reference/polish
        # path; production M1 uses the native Gurobi backend.
        if integer_error > 0.0:
            if integer_error > 1e-5:
                raise SolverFailure(f"{stage}: SciPy MILP incumbent is not numerically integral ({integer_error:.3e}).")
            fixed_lb=np.asarray(model.lower_bounds,float).copy();fixed_ub=np.asarray(model.upper_bounds,float).copy()
            fixed=np.rint(primal[integer_mask])
            fixed_lb[integer_mask]=fixed;fixed_ub[integer_mask]=fixed
            lp_bounds=list(zip(fixed_lb,fixed_ub))
            tol=max(profile.feasibility_tol,1e-10)
            lp_options={"disp":bool(profile.output_flag),"presolve":bool(profile.presolve),
                        "primal_feasibility_tolerance":tol,"dual_feasibility_tolerance":tol}
            if profile.time_limit_s is not None: lp_options["time_limit"]=float(profile.time_limit_s)
            polish=linprog(model.c,A_ub=model.A_ub,b_ub=model.b_ub,
                A_eq=model.A_eq if model.A_eq.shape[0] else None,
                b_eq=model.b_eq if len(model.b_eq) else None,bounds=lp_bounds,
                method="highs",options=lp_options)
            if polish.x is None or polish.fun is None or int(polish.status)!=0:
                raise SolverFailure(f"{stage}: fixed-integer SciPy polish failed: {polish.message}")
            primal=np.asarray(polish.x,float);polished=True
            polish_delta=float(polish.fun)-float(result.fun)
            bound=getattr(result,"mip_dual_bound",None)
            if bound is None or not np.isfinite(bound):
                raise SolverFailure("Fixed-integer polish requires a finite MIP lower bound.")
            if bound is not None and np.isfinite(bound):
                cert=_fixed_integer_bound_certificate(float(polish.fun),float(bound),profile)
                if float(polish.fun)-float(bound)>cert:
                    raise SolverFailure(f"{stage}: fixed-integer polish is not certified against the MIP bound; "
                                        f"gap={float(polish.fun)-float(bound):.3e}, cert={cert:.3e}.")
            objective_value=float(polish.fun)
        else:
            objective_value=float(result.fun)

        return RawSolveResult(status, objective_value, primal, time.perf_counter()-started, getattr(result,"mip_dual_bound",None), getattr(result,"mip_gap",None), None, str(result.message), {
            "backend":"scipy.milp",
            "node_count":getattr(result,"mip_node_count",None),
            "raw_integrality_error":integer_error,
            "fixed_integer_polish":polished,
            "fixed_integer_polish_objective_delta":polish_delta,
            "effective_integrality_tol":1e-6,
            "numeric_recovery":bool(numeric_recovery),
        })
    bounds = list(zip(model.lower_bounds, model.upper_bounds))
    tol = 1e-9 if numeric_recovery else max(profile.feasibility_tol,1e-10)
    options = {"disp": bool(profile.output_flag), "presolve": bool(profile.presolve), "primal_feasibility_tolerance":tol, "dual_feasibility_tolerance":tol}
    if profile.time_limit_s is not None: options["time_limit"] = float(profile.time_limit_s)
    result = linprog(model.c,A_ub=model.A_ub,b_ub=model.b_ub,A_eq=model.A_eq if model.A_eq.shape[0] else None,b_eq=model.b_eq if len(model.b_eq) else None,bounds=bounds,method="highs",options=options)
    status = _STATUS_LP.get(int(result.status), f"STATUS_{result.status}")
    if result.x is None or result.fun is None:
        raise SolverFailure(f"{stage}: SciPy LP failed ({status}): {result.message}")
    duals = None
    try:
        duals = np.asarray(result.ineqlin.marginals, dtype=float)
    except Exception:
        pass
    return RawSolveResult(status, float(result.fun), np.asarray(result.x,float), time.perf_counter()-started, float(result.fun) if status=="OPTIMAL" else None, None, duals, str(result.message), {"backend":"scipy.linprog.highs"})
