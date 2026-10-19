from __future__ import annotations


# ============================================================================
# tpwrs_v3.screening_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.contracts import PhysicalCase


def action_box_bounds(case: PhysicalCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return requested flow, minimum and maximum flow over the complete x-box, and G."""
    base = case.requested_state_mw @ case.flat_H.T + case.flat_h[None, :]
    G = case.flat_H @ case.A
    xbar = case.xbar_mw
    low = base.copy(); high = base.copy()
    for n in range(case.n_participants):
        effect = G[:, n][None, :] * xbar[:, n][:, None]
        low += np.minimum(effect, 0.0)
        high += np.maximum(effect, 0.0)
    return base, low, high, G


def exact_action_box_screen(case: PhysicalCase, *, tolerance_mw: float = 1e-9) -> np.ndarray:
    """Conservative complete-box screen, no per-participant T-by-row temporaries."""
    G=case.flat_H @ case.A
    positive=np.maximum(G,0.0); negative=np.minimum(G,0.0)
    active=np.empty((case.n_time,case.n_flat_rows),dtype=bool)
    rating=case.flat_ratings[None,:]
    xbar=case.xbar_mw; H=case.flat_H; h=case.flat_h
    requested=case.requested_state_mw
    for start in range(0,case.n_time,128):
        stop=min(start+128,case.n_time)
        base=requested[start:stop]@H.T+h[None,:]
        low=base+xbar[start:stop]@negative.T
        high=base+xbar[start:stop]@positive.T
        active[start:stop]=(low<=-rating+tolerance_mw)|(high>=rating-tolerance_mw)
    return active


def deterministic_violation_seed(case: PhysicalCase, *, tolerance_mw: float=1e-9,
                                 potential_mask: np.ndarray | None=None) -> np.ndarray:
    seed=np.zeros((case.n_time,case.n_flat_rows),dtype=bool)
    H=case.flat_H; h=case.flat_h; requested=case.requested_state_mw
    for start in range(0,case.n_time,128):
        stop=min(start+128,case.n_time)
        f=requested[start:stop]@H.T+h[None,:]
        seed[start:stop]=np.abs(f)>=case.flat_ratings[None,:]-tolerance_mw
    if potential_mask is not None:
        if potential_mask.shape!=seed.shape:raise ValueError("potential mask dimension mismatch")
        seed &= potential_mask
    return seed


def hidden_violations(violated_mask: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    return np.asarray(violated_mask, dtype=bool) & ~np.asarray(active_mask, dtype=bool)


# ============================================================================
# tpwrs_v3.replay_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

import math
from dataclasses import dataclass

import numpy as np

from tpwrs.storage import ReplayFailure
from tpwrs.contracts import ExperimentCase
from tpwrs.contracts import PhysicalCase
from tpwrs.contracts import PolicySolution, ReplayReport
from tpwrs.storage import sha256_array


def physical_objectives(case: PhysicalCase, x: np.ndarray, *, block_size: int=256) -> tuple[float,float]:
    values=np.asarray(x,float)
    if values.shape!=case.requests_mw.shape: raise ReplayFailure("x shape mismatch.")
    S=0.0
    for start in range(0,case.n_time,block_size):
        stop=min(start+block_size,case.n_time)
        post=case.requested_state_mw[start:stop]+values[start:stop]@case.A.T
        flows=post@case.flat_H.T+case.flat_h[None,:]
        violation=np.maximum(np.abs(flows)-case.flat_ratings[None,:],0.0)
        S+=float(np.sum(case.time_grid.delta[start:stop,None]*violation))
    E=float(np.sum(case.time_grid.delta[:,None]*values))
    return S,E


def debit_vectors(case: PhysicalCase, experiment: ExperimentCase, x: np.ndarray, *, u: np.ndarray | None = None, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    dt = case.time_grid.delta[:, None]
    physical = np.sum(dt * x / case.capacities_mw[None, :], axis=0)
    weighted = None
    touched = None
    if experiment.rule == "M3":
        if q is None or q.shape != x.shape or not np.isfinite(q).all() or np.any(q <= 0):
            raise ReplayFailure("M3 replay requires a positive q matrix aligned with x.")
        weighted = np.sum(dt * q * x / case.capacities_mw[None, :], axis=0)
    if experiment.rule == "M1":
        if u is None or u.shape != x.shape:
            raise ReplayFailure("M1 replay requires the activation matrix u.")
        touched = np.sum(dt * u, axis=0)
    return physical, weighted, touched


def replay_solution(case: PhysicalCase, experiment: ExperimentCase, solution: PolicySolution, *, feasibility_tol: float | None = None, raise_on_failure: bool = False) -> ReplayReport:
    case.verify_experiment_identity(experiment.physical_case_id, experiment.physical_case_hash, experiment.time_grid_id)
    if solution.experiment_hash!=experiment.hash or solution.rule!=experiment.rule or solution.physical_case_hash != case.physical_case_hash or solution.allowance_hash != experiment.allowance.hash:
        raise ReplayFailure("Solution experiment/rule/case/allowance hash mismatch.")
    if experiment.rule=="M3" and experiment.rate_matrix_hash is not None and (solution.q is None or sha256_array(solution.q)!=experiment.rate_matrix_hash): raise ReplayFailure("Solution q matrix hash differs from the immutable M3 experiment.")
    tol = feasibility_tol or experiment.solver_profile.feasibility_tol
    x = np.asarray(solution.x_mw, float)
    if x.shape != case.requests_mw.shape or not np.isfinite(x).all(): raise ReplayFailure("x shape/finite-value mismatch.")
    xbar = case.xbar_mw
    x_bound_violation = np.maximum(-x, 0.0)
    x_bound_violation = np.maximum(x_bound_violation, np.maximum(x-xbar,0.0))
    post = case.post_state_mw(x)
    balance = np.abs(post.sum(axis=1))
    violated=np.empty((case.n_time,case.n_flat_rows),dtype=bool); S=0.0; max_row=0.0
    for start in range(0,case.n_time,256):
        stop=min(start+256,case.n_time)
        flows=post[start:stop]@case.flat_H.T+case.flat_h[None,:]
        violation=np.maximum(np.abs(flows)-case.flat_ratings[None,:],0.0)
        violated[start:stop]=violation>tol
        S+=float(np.sum(case.time_grid.delta[start:stop,None]*violation))
        max_row=max(max_row,float(np.max(violation,initial=0.0)))
    E = float(np.sum(case.time_grid.delta[:,None] * x))
    physical, weighted, touched = debit_vectors(case, experiment, x, u=solution.u, q=solution.q)
    if experiment.rule == "M1": debit = touched
    elif experiment.rule == "M1_LP":
        debit = np.sum(case.time_grid.delta[:,None] * np.divide(x, xbar, out=np.zeros_like(x), where=xbar>0), axis=0)
    elif experiment.rule in {"M2","M2_MATCHED"}: debit = physical
    else: debit = weighted
    budget_violation = np.maximum(np.asarray(debit,float) - experiment.allowance.values, 0.0)
    material_violation = 0.0
    if experiment.rule == "M1":
        u = np.asarray(solution.u,float)
        if not np.isfinite(u).all() or np.any(u < -10*tol) or np.any(u > 1+10*tol):raise ReplayFailure("M1 activation must lie in [0,1].")
        floor = case.materiality_floor_mw
        material_violation = float(max(np.max(np.maximum(x-xbar*u,0)), np.max(np.maximum(floor*u-x,0)), np.max(np.abs(u-np.rint(u)))))
    max_bal=float(balance.max(initial=0.0)); max_x=float(x_bound_violation.max(initial=0.0)); max_budget=float(budget_violation.max(initial=0.0))
    s_match = abs(S-solution.s_value_mw_h_row) <= max(10*tol, experiment.stage1_tolerance(solution.s_value_mw_h_row)*10)
    e_match = abs(E-solution.e_value_mwh) <= max(10*tol, experiment.stage2_tolerance(solution.e_value_mwh)*10)
    passed = max_bal <= 10*tol and max_x <= 10*tol and max_budget <= 10*tol and material_violation <= 10*tol and s_match and e_match
    report = ReplayReport(passed,S,E,max_row,max_bal,max_x,max_budget,tuple(map(float,physical)),None if weighted is None else tuple(map(float,weighted)),None if touched is None else tuple(map(float,touched)),violated,{"materiality_violation":material_violation,"s_matches":s_match,"e_matches":e_match,"row_count":case.n_flat_rows})
    if raise_on_failure and not passed:
        raise ReplayFailure(f"Independent replay failed: {report.details}, budget={max_budget}, balance={max_bal}, x={max_x}")
    return report


@dataclass(frozen=True)
class PhysicalSupportCut:
    """Affine minorant of the full, duration-weighted congestion objective.

    For each (t,r), choose 0, +1 or -1 according to the hinge active at x0.
    Each selected directed affine overload is <= its positive-part loss at
    every x. Summing proves intercept + <gradient,x> <= S(x) globally.
    Nothing in this object changes the policy after optimization.
    """
    gradient: np.ndarray
    intercept: float
    anchor_value: float
    selector_hash: str

    def value(self, x: np.ndarray) -> float:
        values=np.asarray(x,dtype=float)
        if values.shape != self.gradient.shape or not np.isfinite(values).all():
            raise ReplayFailure("Support evaluation has invalid coordinates.")
        return math.fsum([self.intercept,math.fsum((self.gradient*values).ravel().tolist())])


def physical_support_cut(case: PhysicalCase, x: np.ndarray) -> PhysicalSupportCut:
    values = np.asarray(x, dtype=float)
    if values.shape != case.requests_mw.shape or not np.isfinite(values).all():
        raise ReplayFailure("Cannot separate a nonfinite or misaligned policy.")
    gradient = np.empty_like(values)
    constants = []
    loss_terms = []
    import hashlib
    digest = hashlib.sha256()
    # Small blocks keep the complete row universe out of annual dense memory.
    for start in range(0, case.n_time, 64):
        stop = min(start + 64, case.n_time)
        base = case.requested_state_mw[start:stop] @ case.flat_H.T + case.flat_h
        post = case.requested_state_mw[start:stop] + values[start:stop] @ case.A.T
        flow = post @ case.flat_H.T + case.flat_h
        selector = np.where(flow > case.flat_ratings, 1,
                            np.where(flow < -case.flat_ratings, -1, 0)).astype(np.int8)
        digest.update(selector.tobytes())
        for j, t in enumerate(range(start, stop)):
            dt = float(case.time_grid.delta[t])
            chosen = selector[j].astype(float)
            # Use the actual nodal physical map, not the transaction factor.
            gradient[t] = dt * ((chosen @ case.flat_H) @ case.A)
            constants.append(dt * math.fsum(
                (chosen * base[j] - np.abs(chosen) * case.flat_ratings).tolist()))
            loss_terms.append(dt * math.fsum(
                np.maximum(np.abs(flow[j]) - case.flat_ratings, 0.0).tolist()))
    return PhysicalSupportCut(gradient, math.fsum(constants), math.fsum(loss_terms), digest.hexdigest())


# ============================================================================
# tpwrs_v3.certification (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass
import numpy as np

from tpwrs.storage import ContractError
from tpwrs.contracts import PhysicalCase
from tpwrs.contracts import PolicySolution


def m1lp_debit(case: PhysicalCase, x_mw: np.ndarray) -> np.ndarray:
    x=np.asarray(x_mw,float); xbar=case.xbar_mw
    return np.sum(case.time_grid.delta[:,None]*np.divide(x,xbar,out=np.zeros_like(x),where=xbar>0),axis=0)


def m2_debit(case: PhysicalCase, x_mw: np.ndarray) -> np.ndarray:
    x=np.asarray(x_mw,float)
    return np.sum(case.time_grid.delta[:,None]*x/case.capacities_mw[None,:],axis=0)


def debit_gap_identity(case: PhysicalCase, x_mw: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    x=np.asarray(x_mw,float); xbar=case.xbar_mw
    left=m1lp_debit(case,x)-m2_debit(case,x)
    inv_xbar=np.divide(1.0,xbar,out=np.zeros_like(xbar),where=xbar>0)
    right=np.sum(case.time_grid.delta[:,None]*x*(inv_xbar-1.0/case.capacities_mw[None,:]),axis=0)
    return left,right


def verify_debit_gap_identity(case: PhysicalCase, x_mw: np.ndarray, tol: float=1e-10) -> bool:
    left,right=debit_gap_identity(case,x_mw)
    return bool(np.max(np.abs(left-right))<=tol and np.min(left)>=-tol)


# Shared numerical face-certification contract.  The annual support-function
# solver and every downstream validator must use the same tolerance.  This is
# deliberately separate from the scientific lexicographic epsilon: it only
# absorbs the solver/oracle feasibility envelope used to certify exact support
# closure.
ANNUAL_SUPPORT_FACE_CONTRACT = "ANNUAL_SUPPORT_OA_FACE_V1"
ANNUAL_SUPPORT_FACE_ABS_TOL = 1e-6
ANNUAL_SUPPORT_FACE_FEAS_MULTIPLIER = 100.0
# Fresh generic solutions are independently face-checked inside the solver.
# This envelope is not an optimization constraint or a scientific epsilon.
GENERIC_EXPLICIT_FACE_CONTRACT = "GENERIC_PHYSICAL_FACE_GUARD_V3"
GENERIC_FACE_ABS_TOL = 1e-7
GENERIC_FACE_FEAS_MULTIPLIER = 10.0
LEGACY_GENERIC_FACE_FEAS_MULTIPLIER = 10.0


def annual_support_face_numeric_tolerance(solver_profile: object) -> float:
    feasibility_tol = float(getattr(solver_profile, "feasibility_tol"))
    if not np.isfinite(feasibility_tol) or feasibility_tol < 0:
        raise ContractError("Solver feasibility tolerance must be finite and nonnegative.")
    return max(ANNUAL_SUPPORT_FACE_ABS_TOL, ANNUAL_SUPPORT_FACE_FEAS_MULTIPLIER * feasibility_tol)


def generic_face_numeric_tolerance(solver_profile: object) -> float:
    feasibility_tol = float(getattr(solver_profile, "feasibility_tol"))
    if not np.isfinite(feasibility_tol) or feasibility_tol < 0:
        raise ContractError("Solver feasibility tolerance must be finite and nonnegative.")
    return max(GENERIC_FACE_ABS_TOL, GENERIC_FACE_FEAS_MULTIPLIER * feasibility_tol)


def certified_face_numeric_tolerance(solution: PolicySolution, solver_profile: object) -> float:
    """Return the numeric envelope certified by the solution's solver contract.

    Fresh generic LP/MILP solutions certify physical replay before returning.
    Annual support-OA certificates explicitly identify their stronger support
    closure contract and must record the exact shared tolerance.  Unknown or
    inconsistent certificate labels fail closed rather than widening a face.
    """
    feasibility_tol = float(getattr(solver_profile, "feasibility_tol"))
    if not np.isfinite(feasibility_tol) or feasibility_tol < 0:
        raise ContractError("Solver feasibility tolerance must be finite and nonnegative.")
    legacy_generic = LEGACY_GENERIC_FACE_FEAS_MULTIPLIER * feasibility_tol
    stages = (solution.stage1, solution.stage2)
    contracts = [str(stage.metadata.get("face_numeric_contract", "")) for stage in stages]
    # Legacy/cache artifacts that predate an explicit contract remain under the
    # historical narrow envelope.  This is intentionally fail-closed: a new
    # release cannot silently widen the certificate of an old numerical result.
    if not any(contracts):
        return legacy_generic

    if contracts == [ANNUAL_SUPPORT_FACE_CONTRACT, ANNUAL_SUPPORT_FACE_CONTRACT]:
        expected = annual_support_face_numeric_tolerance(solver_profile)
        field = "support_certification_tolerance"
        label = "Annual support"
    elif contracts == [GENERIC_EXPLICIT_FACE_CONTRACT, GENERIC_EXPLICIT_FACE_CONTRACT]:
        expected = generic_face_numeric_tolerance(solver_profile)
        field = "generic_face_certification_tolerance"
        label = "Generic explicit-face"
    else:
        raise ContractError(f"Inconsistent or unknown face numeric certificate: {contracts!r}.")

    for stage in stages:
        try:
            recorded = float(stage.metadata[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"{label} certificate omits its certification tolerance.") from exc
        check_tol = max(1e-15, 1e-12 * max(1.0, abs(expected)))
        if not np.isfinite(recorded) or abs(recorded - expected) > check_tol:
            raise ContractError(
                f"{label} certificate tolerance {recorded!r} disagrees with current contract {expected!r}."
            )
    return max(legacy_generic, expected)


@dataclass(frozen=True)
class StrictCertificate:
    q_replayed_feasible_s: float
    epsilon_cert: float
    m1lp_certified_lower_bound: float
    passed: bool
    guaranteed_margin: float


def strict_m1_certificate(q_replayed_feasible_s: float, m1lp_certified_lower_bound: float, epsilon_cert: float) -> StrictCertificate:
    if not np.all(np.isfinite([q_replayed_feasible_s, m1lp_certified_lower_bound, epsilon_cert])):
        raise ContractError("Certificate inputs must be finite.")
    if epsilon_cert < 0: raise ContractError("epsilon_cert cannot be negative.")
    margin=float(m1lp_certified_lower_bound-q_replayed_feasible_s-epsilon_cert)
    return StrictCertificate(float(q_replayed_feasible_s),float(epsilon_cert),float(m1lp_certified_lower_bound),margin>0,margin)


def annual_m1_scalar_bracket(m1lp_lower: float, rolling_m1_upper: float, *, tol: float=1e-8) -> dict[str,float|bool]:
    if not np.all(np.isfinite([m1lp_lower, rolling_m1_upper, tol])) or tol < 0:
        raise ContractError("Scalar bracket inputs must be finite and tolerance nonnegative.")
    if m1lp_lower > rolling_m1_upper + tol:
        raise ContractError("M1-LP lower endpoint exceeds feasible rolling-M1 upper endpoint.")
    return {"m1lp_lower":float(m1lp_lower),"rolling_m1_upper":float(rolling_m1_upper),"width":float(rolling_m1_upper-m1lp_lower),"pointwise_bound_claim":False}


# ============================================================================
# tpwrs_v3.lp_certificates (consolidated; see SOURCE_MAP.json)
# ============================================================================
"""Independent box-dual reconstruction for the continuous restriction LP.

The lower bound is evaluated from original action coefficients, not copied from
Stage II's primal value. All arithmetic certificates are numerical, with the
reported rounding guard, not exact rational proofs.
"""
import math
import numpy as np


def box_dual_bound(c, upper, budget_matrix, allowance, budget_pi, thermal_blocks,
                   *, constant_s=0.0, stage1_cap=None, cap_pi=None):
    """thermal_blocks yield (A, b, duration, solver_pi, coordinate_indices).

    Variables x satisfy 0<=x<=upper. Thermal slack is nonnegative. In Stage I
    its price is duration. In Stage II its price is eta*duration, eta=-cap_pi.
    Clipping multipliers to these sign/box requirements creates a valid dual
    lower bound even when the solver multipliers have small sign residuals.
    """
    c=np.asarray(c,float);upper=np.asarray(upper,float)
    lam=np.maximum(-np.asarray(budget_pi,float),0.0)
    eta=0.0 if stage1_cap is None else max(0.0,-float(cap_pi))
    r=c+np.asarray(budget_matrix.T@lam).reshape(-1)
    terms=[float(constant_s) if stage1_cap is None else eta*(float(constant_s)-float(stage1_cap)),
           -float(np.dot(np.asarray(allowance,float),lam))]
    magnitude=sum(map(abs,terms)); sign_correction=0.0
    for A,b,dt,pi,indices in thermal_blocks:
        raw=-np.asarray(pi,float).reshape(-1)
        cap=np.asarray(dt,float)*(1.0 if stage1_cap is None else eta)
        z=np.clip(raw,0.0,cap)
        sign_correction=max(sign_correction,float(np.max(np.abs(z-raw),initial=0.0)))
        contribution=np.asarray(A.T@z).reshape(-1)
        r[np.asarray(indices,int)]+=contribution
        term=-float(np.dot(np.asarray(b,float),z));terms.append(term)
        magnitude+=abs(term)+float(np.dot(np.abs(contribution),upper[np.asarray(indices,int)]))
    terms.append(float(np.dot(upper,np.minimum(r,0.0))))
    magnitude+=abs(terms[-1])
    raw_bound=math.fsum(terms)
    # This guard accompanies—not replaces—original-coordinate feasibility,
    # primal/dual gap checks and the paper's declared comparison tolerance.
    guard=64*np.finfo(float).eps*max(1.0,magnitude)
    return {"lower_bound":raw_bound-guard,"unadjusted_lower_bound":raw_bound,
            "floating_point_guard":guard,"dual_sign_projection_max":sign_correction,
            "bound_kind":"NUMERICAL_BOX_DUAL_LOWER_BOUND", "cap_multiplier":eta}
