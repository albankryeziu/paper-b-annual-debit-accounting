"""Three-bucket QP, rate lookup and final campaign compatibility helpers.

See docs/CODE_GUIDE.md for the retained function scope.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import json
import numpy as np
from tpwrs.contracts import CardSpec
from tpwrs.storage import ContractError, SolverFailure, sha256_file


def assign_buckets(scores: Sequence[float], thresholds: tuple[float, float] | None) -> np.ndarray:
    a = np.asarray(scores, float)
    if thresholds is None:
        return np.ones(len(a), dtype=int)
    lo, hi = thresholds
    return np.where(a <= lo, 1, np.where(a <= hi, 2, 3)).astype(int)


def solve_three_rate_convex_qp(
    linear: Sequence[float],
    quadratic_weights: Sequence[float],
    center: Sequence[float],
    neutrality_weights: Sequence[float],
    *,
    q_bounds: tuple[float, float] = (0.5, 1.5),
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
    linear = np.asarray(linear, float)
    quad = np.asarray(quadratic_weights, float)
    center = np.asarray(center, float)
    neutral = np.asarray(neutrality_weights, float)
    if linear.shape != (3,) or quad.shape != (3,) or center.shape != (3,) or neutral.shape != (3,):
        raise ContractError("Three-rate QP expects length-three coefficient vectors.")
    if (
        np.any(~np.isfinite(linear))
        or np.any(~np.isfinite(quad))
        or np.any(~np.isfinite(center))
        or np.any(~np.isfinite(neutral))
        or np.any(quad < 0)
        or np.any(neutral < 0)
    ):
        raise ContractError("Three-rate QP coefficients must be finite and weights nonnegative.")
    nsum = float(neutral.sum())
    if nsum <= 0:
        raise ContractError("Three-rate QP requires positive total neutrality weight.")
    qlo, qhi = map(float, q_bounds)
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qlo > 1.0 or qhi < 1.0 or qlo > qhi:
        raise ContractError(
            "Three-rate QP bounds must be finite, ordered, and contain the neutral rate 1."
        )

    wn = neutral / nsum
    D = []
    d = []
    for i in range(3):
        row = np.zeros(3)
        row[i] = 1.0
        D.append(row)
        d.append(qhi)
        row = np.zeros(3)
        row[i] = -1.0
        D.append(row)
        d.append(-qlo)
    D.extend((np.array([-1.0, 1.0, 0.0]), np.array([0.0, -1.0, 1.0])))
    d.extend((0.0, 0.0))
    D = np.asarray(D, float)
    d = np.asarray(d, float)

    H = np.diag(quad)
    f = linear - quad * center
    scale = max(1.0, float(np.max(np.abs(f))), float(np.max(quad)))
    Hs = H / scale
    fs = f / scale

    def feasible(q, tol=3e-9):
        q = np.asarray(q, float)
        return bool(
            np.all(np.isfinite(q)) and abs(float(wn @ q) - 1.0) <= tol and np.all(D @ q <= d + tol)
        )

    def objective(q):
        q = np.asarray(q, float)
        return float(linear @ q + 0.5 * np.sum(quad * (q - center) ** 2))

    candidates = [np.ones(3, float)]

    def add_stationary(active_rows):
        C = np.vstack([wn] + [D[i] for i in active_rows])
        rhs_eq = np.asarray([1.0] + [float(d[i]) for i in active_rows], float)
        K = np.block([[Hs, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
        rhs = np.concatenate((-fs, rhs_eq))
        sol = np.linalg.lstsq(K, rhs, rcond=1e-13)[0]
        q = np.asarray(sol[:3], float)
        if np.max(np.abs(C @ q - rhs_eq)) <= 3e-9 and feasible(q):
            candidates.append(q)

    # Relative-interior and edge stationary points.
    add_stationary(())
    for i in range(len(d)):
        add_stationary((i,))

    # All vertices.  Avoid determinant thresholds: solve every nonsingular
    # three-equation system and accept it only after explicit feasibility checks.
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            M = np.vstack((wn, D[i], D[j]))
            rhs = np.asarray([1.0, d[i], d[j]], float)
            try:
                q = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            if feasible(q):
                candidates.append(np.asarray(q, float))

    values = np.asarray([objective(q) for q in candidates], float)
    if np.any(~np.isfinite(values)):
        raise SolverFailure(
            "Three-rate active-set solver produced a nonfinite objective candidate."
        )
    q = np.asarray(candidates[int(np.argmin(values))], float)
    # A semidefinite primary QP can leave a continuum of optimal rates.
    # Its exact optimal face fixes H(q-q*)=0 and f.T(q-q*)=0.
    # Solve a second convex QP on that face, minimizing distance to one.
    if np.any(quad == 0):
        import itertools

        base = [wn]
        base_rhs = [1.0]
        for i in np.flatnonzero(quad > 0):
            row = np.zeros(3)
            row[i] = 1.0
            base.append(row)
            base_rhs.append(float(q[i]))
        free_linear = np.where(quad == 0, linear, 0.0)
        if np.any(free_linear != 0):
            row = free_linear / float(np.max(np.abs(free_linear)))
            base.append(row)
            base_rhs.append(float(row @ q))
        secondary = []
        for count in range(4):
            for active in itertools.combinations(range(len(d)), count):
                C = np.vstack(base + [D[i] for i in active])
                b = np.asarray(base_rhs + [float(d[i]) for i in active])
                K = np.block([[np.eye(3), C.T], [C, np.zeros((len(b), len(b)))]])
                candidate = np.linalg.lstsq(K, np.r_[np.ones(3), b], rcond=1e-13)[0][:3]
                if np.max(np.abs(C @ candidate - b)) <= 3e-9 and feasible(candidate):
                    secondary.append(candidate)
        if not secondary:
            raise SolverFailure("Minimum-norm optimal-rate face is unresolved.")
        q = min(secondary, key=lambda value: float(np.sum((value - 1.0) ** 2)))
    equality_tol = 1e-8 * max(1.0, abs(nsum))
    if (
        abs(float(neutral @ q - nsum)) > equality_tol
        or q[0] + 1e-8 < q[1]
        or q[1] + 1e-8 < q[2]
        or min(q) < qlo - 1e-8
        or max(q) > qhi + 1e-8
    ):
        raise SolverFailure("Three-rate active-set solver returned a contract-infeasible vector.")
    return q


def rate_card_qp(
    A: Sequence[float],
    W: Sequence[float],
    *,
    q_bounds: tuple[float, float] = (0.5, 1.5),
    lambda_q: float = 1e-3,
) -> tuple[np.ndarray, str]:
    """Solve the frozen three-bucket rate-card QP without iterative NLP calls."""
    A = np.asarray(A, float)
    W = np.asarray(W, float)
    if (
        A.shape != (3,)
        or W.shape != (3,)
        or np.any(~np.isfinite(A))
        or np.any(~np.isfinite(W))
        or np.any(W < 0)
    ):
        raise ContractError("Rate QP expects finite length-three A and nonnegative finite W.")
    if not np.isfinite(lambda_q) or lambda_q < 0:
        raise ContractError("Rate QP requires a finite nonnegative lambda_q.")
    if np.sum(W > 0.0) < 2 or W.sum() <= 0:
        return np.ones(3), "FLAT_SPARSE_REFERENCE_FDE"
    q = solve_three_rate_convex_qp(A, lambda_q * W, np.ones(3), W, q_bounds=q_bounds)
    return q, "OPTIMAL_QP"


def q_from_card(card: CardSpec, directions: Sequence[str], scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or scores.shape[1] != len(directions):
        raise ContractError("Score matrix must have shape (time, participant).")
    q = np.ones_like(scores)
    for n, d in enumerate(directions):
        if d not in card.rates:
            raise ContractError(f"Card has no rates for direction {d!r}.")
        th = card.thresholds[d]
        rates = card.rates[d]
        if np.isneginf(th[0]) and np.isposinf(th[1]):
            buckets = np.ones(scores.shape[0], int)
        else:
            buckets = assign_buckets(scores[:, n], th)
        q[:, n] = np.asarray([rates[b - 1] for b in buckets])
    return q


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
    if (
        payload.get("loader_scope")
        != "Fitting/selection code reads only 2025-2028 policy/feature outcomes"
    ):
        raise ContractError("Held-out audit loader scope differs from protocol.")
    return payload, sha256_file(audit_path)
