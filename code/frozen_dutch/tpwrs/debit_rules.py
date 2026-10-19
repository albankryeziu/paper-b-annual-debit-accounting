"""Debit-family extension: accounting and card design, with no network solver.

Positive exogenous weights use the existing M3 optimization interface. Hybrid
and progressive objects evaluate their exact laws; they are not misrepresented
as weighted-LP optimization implementations.
"""
from dataclasses import dataclass
import math
from fractions import Fraction
import numpy as np
from tpwrs.cards import solve_three_rate_convex_qp


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sums(matrix):
    a = np.asarray(matrix, float)
    require(a.ndim == 2 and np.isfinite(a).all(), 'Invalid debit coordinates')
    return np.array([math.fsum(a[:, n]) for n in range(a.shape[1])])


@dataclass(frozen=True)
class DebitRule:
    rule_id: str
    beta: float = .5
    alpha: float = .5
    kappa: float = 1.
    tau: float = .5

    def __post_init__(self):
        require(self.rule_id in {'R_BETA', 'H_ALPHA', 'P_KAPPA_TAU', 'WEIGHTED'}, 'Unknown debit rule')
        require(all(math.isfinite(v) for v in (self.beta, self.alpha, self.kappa, self.tau)), 'Nonfinite parameters')
        require(0 <= self.beta <= 1 and 0 <= self.alpha <= 1 and self.kappa >= 0 and 0 < self.tau < 1, 'Parameter domain')

    def weights(self, case, q=None):
        if self.rule_id == 'R_BETA':
            ratio = np.divide(case.capacities_mw, case.xbar_mw, out=np.ones_like(case.xbar_mw), where=case.xbar_mw > 0)
            return (1-self.beta)*ratio+self.beta
        require(self.rule_id == 'WEIGHTED', 'This debit is not an exogenous weighted LP')
        q = np.asarray(q, float)
        require(q.shape == case.requests_mw.shape and np.isfinite(q).all() and np.all(q > 0), 'Weights must be positive and aligned')
        return q

    def debit(self, case, x, u=None, q=None):
        x = np.asarray(x, float)
        require(x.shape == case.requests_mw.shape and np.isfinite(x).all(), 'Action shape/finite mismatch')
        z = x/case.capacities_mw
        if self.rule_id in {'R_BETA', 'WEIGHTED'}:
            charge = z*self.weights(case, q)
        elif self.rule_id == 'P_KAPPA_TAU':
            charge = z+self.kappa*np.maximum(z-self.tau, 0.)
        elif self.alpha == 0:
            charge = z
        else:
            u = np.asarray(u, float)
            require(u.shape == x.shape and np.isfinite(u).all(), 'Hybrid needs aligned activations')
            require(np.array_equal(u, np.rint(u)) and np.all((u == 0)|(u == 1)), 'Hybrid activations must be binary')
            require(np.all(x >= case.materiality_floor_mw*u) and np.all(x <= case.xbar_mw*u), 'Hybrid materiality links violated')
            charge = self.alpha*u+(1-self.alpha)*z
        return sums(case.time_grid.delta[:, None]*charge)


def hinge_relief_scores(case, block_size=64):
    """Right derivative at requested state, in residual units per FDE hour.

    No optimized actions enter this score. Exact boundary directional hinges
    are included explicitly. No row screen or finite-difference approximation.
    """
    H, A, ratings = case.flat_H, case.A, case.flat_ratings
    G = H@A
    scores = np.empty(case.requests_mw.shape)
    for start in range(0, case.n_time, block_size):
        stop = min(start+block_size, case.n_time)
        f = case.requested_state_mw[start:stop]@H.T+case.flat_h
        selector = (f > ratings).astype(float)-(f < -ratings).astype(float)
        derivative = selector@G
        for i in range(len(f)):
            pos, neg = f[i] == ratings, f[i] == -ratings
            if np.any(pos): derivative[i] += np.maximum(G[pos], 0).sum(axis=0)
            if np.any(neg): derivative[i] += np.maximum(-G[neg], 0).sum(axis=0)
        scores[start:stop] = -derivative*case.capacities_mw
    scores[case.xbar_mw == 0] = 0.
    require(np.isfinite(scores).all(), 'Nonfinite network score')
    return scores


def weighted_quantile(values, weights, quantiles):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    require(values.shape == weights.shape and np.isfinite(values).all() and np.isfinite(weights).all()
            and np.all(weights >= 0) and weights.sum() > 0, 'Invalid reference weights')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return np.array([values[order[min(np.searchsorted(cumulative, p*cumulative[-1]), len(order)-1)]] for p in quantiles])


def buckets(scores, thresholds):
    lo, hi = thresholds
    require(np.isfinite([lo, hi]).all() and lo <= hi, 'Invalid thresholds')
    return np.where(scores <= lo, 0, np.where(scores <= hi, 1, 2))


def premium_card(W, means, *, gamma=.5, xi=.25, penalty=1.):
    """Exact variable substitution into the established three-rate QP.

    Means are dimensionless fitting-scaled marginal relief. Constant signal or
    fewer than two supported buckets is retained as a uniform result.
    """
    W, means = np.asarray(W, float), np.asarray(means, float)
    require(W.shape == means.shape == (3,) and np.isfinite(W).all() and np.isfinite(means).all()
            and np.all(W >= 0) and W.sum() > 0, 'Invalid card data')
    require(math.isfinite(gamma) and math.isfinite(xi) and math.isfinite(penalty)
            and 0 <= xi <= gamma and penalty > 0, 'Invalid premium design')
    c = 1+xi
    if np.count_nonzero(W) < 2 or xi == 0 or np.ptp(means[W > 0]) == 0:
        q = np.full(3, c)
    else:
        r = solve_three_rate_convex_qp(c*W*means, penalty*c*c*W, np.ones(3), W,
                                      q_bounds=(1/c, (1+gamma)/c))
        # A solver-tolerated q<1 would defeat the conservative ceiling theorem.
        q = np.clip(c*r, 1., 1+gamma)
    require(np.all(q[:-1] >= q[1:]-1e-10), 'Rate order failed')
    residual = float(W@q-c*W.sum())
    require(abs(residual) <= 1e-8*max(1, W.sum()), 'Mean-premium equality failed')
    return q, {'reference_mean_rate': float(W@q/W.sum()), 'mean_equality_residual': residual,
               'rate_dispersion': float(np.ptp(q)), 'positive_reference_buckets': int(np.count_nonzero(W)),
               'objective': float(W@(means*q)+.5*penalty*np.sum(W*(q-c)**2))}


def admissible_probe_checks(case, scores, count=24):
    """Fitting-only finite differences restricted to an affine hinge segment."""
    H, G = case.flat_H, case.flat_H@case.A
    coords = np.argwhere(case.xbar_mw > 1e-4)
    records = []
    if not len(coords): return records
    for idx in np.unique(np.linspace(0, len(coords)-1, min(count*4, len(coords)), dtype=int)):
        t, n = coords[idx]
        f = case.requested_state_mw[t]@H.T+case.flat_h
        g = G[:, n]
        roots = np.r_[np.divide(case.flat_ratings-f, g, out=np.full_like(f, np.inf), where=g != 0),
                      np.divide(-case.flat_ratings-f, g, out=np.full_like(f, np.inf), where=g != 0)]
        positive = roots[roots > 0]
        eps = min(1., float(case.xbar_mw[t, n]), float(positive.min())*.25 if len(positive) else 1.)
        if eps < 1e-4: continue
        before = np.maximum(abs(f)-case.flat_ratings, 0.)
        after = np.maximum(abs(f+eps*g)-case.flat_ratings, 0.)
        measured = math.fsum((before-after).tolist())*case.capacities_mw[n]/eps
        # Covers subtraction, flow-update and hinge arithmetic, not an arbitrary
        # economic display tolerance. Derivative's matrix products add this scale.
        bound = 128*np.finfo(float).eps*math.fsum(abs(f)+abs(eps*g)+case.flat_ratings)*case.capacities_mw[n]/eps
        error = abs(measured-scores[t, n])
        require(error <= bound, 'Analytic marginal score failed admissible finite-difference check')
        records.append({'time': int(t), 'position': int(n), 'epsilon_MW': eps,
                        'analytic': float(scores[t, n]), 'probe': measured,
                        'absolute_difference': error, 'roundoff_enclosure': bound})
        if len(records) >= count: break
    return records


def weighted_budget_audit(case, x, q, limits):
    """Strict finite-coordinate audit, independent of solver tolerance."""
    x, q, limits = np.asarray(x, float), np.asarray(q, float), np.asarray(limits, float)
    DebitRule('WEIGHTED').weights(case, q)
    require(x.shape == q.shape and np.isfinite(x).all(), 'Invalid actions')
    require(limits.shape == case.capacities_mw.shape and np.isfinite(limits).all() and np.all(limits >= 0), 'Invalid limits')
    weighted, physical = [], []
    for n, p in enumerate(case.capacities_mw):
        e, d = Fraction(0), Fraction(0)
        for a, rate, dt in zip(x[:, n], q[:, n], case.time_grid.delta):
            if a != 0:
                term = Fraction(float(a))*Fraction(float(dt))
                e += term
                d += term*Fraction(float(rate))
        weighted.append(d/Fraction(float(p))-Fraction(float(limits[n])))
        physical.append(e/Fraction(float(p))-Fraction(float(limits[n])))
    box = max(0., float(np.max(-x)), float(np.max(x-case.xbar_mw)))
    return {'strict_pass': box == 0 and all(v <= 0 for v in weighted+physical),
            'max_box_violation_MW': box,
            'weighted_budget_residual_h': [float(v) for v in weighted],
            'physical_FDE_residual_h': [float(v) for v in physical],
            'weighted_violating_positions': sum(v > 0 for v in weighted),
            'physical_violating_positions': sum(v > 0 for v in physical),
            'max_weighted_violation_h': max(0., float(max(weighted))),
            'max_physical_violation_h': max(0., float(max(physical))),
            'arithmetic': 'Exact rational sums/products/division of saved binary64 coordinates; physical and weighted ceilings independently checked'}


def weighted_witness(case, raw, q, limits):
    """Original roundoff-guard formula, generalized to a positive weighted dot."""
    from jobs.certificates import _debit_guard
    boxed = np.clip(raw, 0., case.xbar_mw)
    debit = DebitRule('WEIGHTED').debit(case, boxed, q=q)
    guards = np.array([_debit_guard(h, case.n_time) for h in limits])
    target = np.maximum(0., limits-guards)
    factors = np.ones_like(limits)
    hit = debit > target
    factors[hit] = np.nextafter(target[hit]/debit[hit], 0.)
    x = boxed*factors
    audit = weighted_budget_audit(case, x, q, limits)
    require(audit['strict_pass'], 'Weighted witness failed exact audit')
    return x, {'max_box_projection_MW': float(np.max(abs(raw-boxed))),
               'max_total_change_MW': float(np.max(abs(raw-x))),
               'factors': factors.tolist(), 'guards_h': guards.tolist(), **audit}
