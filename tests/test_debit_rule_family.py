"""Final numerical certificate reconstruction and verification.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from tpwrs.debit_rules import (
    DebitRule,
    premium_card,
    hinge_relief_scores,
    admissible_probe_checks,
    weighted_budget_audit,
    weighted_witness,
    buckets,
    weighted_quantile,
)


def toy(b=(25.0, 25.0), P=100.0):
    b = np.asarray(b, float).reshape(-1, 1)
    return SimpleNamespace(
        requests_mw=b.copy(),
        xbar_mw=b,
        capacities_mw=np.array([P]),
        materiality_floor_mw=np.minimum(b, 1.0),
        time_grid=SimpleNamespace(delta=np.ones(len(b))),
        n_time=len(b),
        n_participants=1,
        flat_H=np.array([[1.0]]),
        A=np.array([[-1.0]]),
        flat_h=np.array([0.0]),
        flat_ratings=np.array([10.0]),
        requested_state_mw=np.full((len(b), 1), 20.0),
    )


@pytest.mark.parametrize("beta", [0.0, 0.2, 0.5, 1.0])
def test_request_interpolation_formula_and_zero(beta):
    c = toy((25.0, 0.0))
    x = np.array([[10.0], [0.0]])
    d = DebitRule("R_BETA", beta=beta)
    assert d.debit(c, x)[0] == pytest.approx(10 * ((1 - beta) / 25 + beta / 100))
    assert d.debit(c, 0 * x)[0] == 0


def test_request_nesting_and_strictness():
    c = toy((25.0,))
    x = np.array([[25.0]])
    d = [DebitRule("R_BETA", beta=b).debit(c, x)[0] for b in (0, 0.5, 1)]
    assert d == [1.0, 0.625, 0.25]
    assert d[2] <= 0.5 < d[1] < d[0]


def test_hybrid_integer_links_and_endpoint_discontinuity():
    c = toy((25.0,))
    x = np.array([[0.5]])
    assert DebitRule("H_ALPHA", alpha=0).debit(c, x)[0] == 0.005
    with pytest.raises(ValueError, match="materiality"):
        DebitRule("H_ALPHA", alpha=0.01).debit(c, x, u=np.ones_like(x))
    with pytest.raises(ValueError, match="binary"):
        DebitRule("H_ALPHA").debit(c, x, u=np.full_like(x, 0.5))
    x[0, 0] = 5
    assert DebitRule("H_ALPHA", alpha=1).debit(c, x, u=np.ones_like(x))[0] == 1


def test_hybrid_relaxation_exact_request_family_identity():
    c = toy()
    x = np.array([[10.0], [20.0]])
    alpha = 0.3
    relaxed = np.sum(alpha * x / c.xbar_mw + (1 - alpha) * x / c.capacities_mw)
    assert relaxed == pytest.approx(DebitRule("R_BETA", beta=1 - alpha).debit(c, x)[0])


def test_progressive_convex_hinge_and_strict_protection():
    c = toy((100.0,))
    rule = DebitRule("P_KAPPA_TAU", kappa=1, tau=0.5)
    a, b = np.array([[20.0]]), np.array([[100.0]])
    assert rule.debit(c, b)[0] == 1.5
    assert rule.debit(c, (a + b) / 2)[0] <= (rule.debit(c, a)[0] + rule.debit(c, b)[0]) / 2
    with pytest.raises(ValueError, match="not an exogenous"):
        rule.weights(c)


def test_mean_one_nonnegative_premiums_force_flat():
    q, d = premium_card([1.0, 2.0, 3.0], [-1.0, 0.0, 1.0], xi=0)
    assert np.array_equal(q, np.ones(3))
    assert d["reference_mean_rate"] == 1


def test_constant_signal_qp_flat_and_sparse_fallback():
    for W in ([1.0, 2.0, 3.0], [0.0, 0.0, 3.0]):
        q, d = premium_card(W, [2.0, 2.0, 2.0])
        assert np.array_equal(q, np.full(3, 1.25))


def test_differentiated_qp_bounded_ordered_and_neutral():
    W = np.array([1.0, 2.0, 3.0])
    means = np.array([-1.0, 0.0, 1.0])
    q, d = premium_card(W, means)
    assert q.min() >= 1 and q.max() <= 1.5
    assert np.all(np.diff(q) <= 1e-10)
    assert W @ q == pytest.approx(1.25 * W.sum())
    assert d["rate_dispersion"] > 0.01
    obj = lambda x: W @ (means * x) + 0.5 * np.sum(W * (x - 1.25) ** 2)
    assert obj(q) < obj(np.full(3, 1.25))


def test_qp_reference_mean_does_not_equal_realized_exposure():
    c = toy()
    x = np.array([[25.0], [0.0]])
    D = DebitRule("WEIGHTED")
    q = np.array([[1.5], [1.0]])
    assert q.mean() == 1.25
    assert D.debit(c, x, q=q)[0] != D.debit(c, x, q=np.full_like(q, 1.25))[0]


def test_right_derivative_and_admissible_fd():
    c = toy()
    score = hinge_relief_scores(c)
    assert np.array_equal(score, np.full((2, 1), 100.0))
    probes = admissible_probe_checks(c, score, count=2)
    assert len(probes) == 2 and all(
        p["absolute_difference"] <= p["roundoff_enclosure"] for p in probes
    )
    # At the positive-rating boundary a beneficial inward action has derivative 0.
    c.requested_state_mw[:] = 10
    assert np.array_equal(hinge_relief_scores(c), np.zeros((2, 1)))
    # The same action at the negative boundary moves outward and is harmful.
    c.requested_state_mw[:] = -10
    assert np.array_equal(hinge_relief_scores(c), np.full((2, 1), -100.0))


def test_scores_do_not_consume_optimized_actions():
    c = toy()
    before = hinge_relief_scores(c)
    c.future_optimized_x = np.full((2, 1), 1e9)
    assert np.array_equal(before, hinge_relief_scores(c))


def test_exact_audit_detects_subtolerance_excess_and_guard_is_separate():
    c = toy((100.0,))
    x = np.array([[np.nextafter(50.0, np.inf)]])
    q = np.array([[1.25]])
    H = np.array([0.625])
    audit = weighted_budget_audit(c, x, q, H)
    assert not audit["strict_pass"] and audit["max_weighted_violation_h"] > 0
    saved = x.copy()
    guard, ga = weighted_witness(c, x, q, H)
    assert ga["strict_pass"] and np.array_equal(x, saved) and guard[0, 0] < x[0, 0]


def test_zero_allowance_guard_and_action_box_projection():
    c = toy((25.0, 0.0))
    x = np.array([[26.0], [-1e-10]])
    g, a = weighted_witness(c, x, np.ones_like(x), np.array([0.0]))
    assert a["strict_pass"] and np.array_equal(g, np.zeros_like(x))


def test_bucketing_and_calibration_weights():
    th = weighted_quantile(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), (1 / 3, 2 / 3))
    assert np.array_equal(th, [1.0, 2.0])
    assert np.array_equal(buckets([1.0, 2.0, 3.0], th), [0, 1, 2])
    with pytest.raises(ValueError):
        weighted_quantile(np.ones(3), np.array([-1.0, 1.0, 1.0]), (0.5,))


@pytest.mark.parametrize("kwargs", [{"beta": -1}, {"alpha": 2}, {"kappa": -1}, {"tau": 1}])
def test_invalid_rule_parameters(kwargs):
    with pytest.raises(ValueError):
        DebitRule("R_BETA", **kwargs)


def test_rolling_commits_only_implemented_weighted_exposure(monkeypatch):
    import pandas as pd
    from tpwrs.contracts import AllowanceSpec, ExperimentCase, TimeGrid
    import tpwrs.rolling as rolling
    from tpwrs.storage import sha256_array

    calls = []

    class Case:
        def __init__(self, start=0, stop=72):
            self.start, self.n_time, self.n_participants = start, stop - start, 1
            self.requests_mw = np.full((self.n_time, 1), 100.0)
            self.xbar_mw = self.requests_mw.copy()
            self.capacities_mw = np.array([100.0])
            self.participant_ids = ("p",)
            self.time_grid = TimeGrid.from_index(
                pd.date_range("2028-01-01", periods=72, freq="h", tz="UTC")[start:stop], 1.0
            )
            self.physical_case_id, self.physical_case_hash = (
                f"toy{start}:{stop}",
                f"h{start}:{stop}",
            )

        def slice_time(self, start, stop):
            return Case(self.start + start, self.start + stop)

    c = Case()
    q = np.repeat([1.0, 1.5, 1.25], 24).reshape(72, 1)
    exp = ExperimentCase(
        "test",
        c.physical_case_id,
        c.physical_case_hash,
        c.time_grid.grid_id,
        "M3",
        AllowanceSpec.explicit(("p",), [100.0], source="explicit"),
        mode="rolling",
        lookahead_hours=48,
        implementation_hours=24,
        card_id="test",
        card_hash="test",
        rate_matrix_hash=sha256_array(q),
    )

    def solve(sub, se, **kwargs):
        calls.append((sub.start, se.allowance.values.copy(), kwargs["q"].copy()))
        return SimpleNamespace(x_mw=np.ones((sub.n_time, 1)), u=None)

    monkeypatch.setattr(rolling, "solve_policy", solve)
    monkeypatch.setattr(rolling, "validate_policy_solution", lambda *args, **kwargs: None)
    result = rolling.solve_rolling(c, exp, q=q)
    assert [a[0] for a in calls] == [0, 24, 48]
    assert calls[1][1][0] == pytest.approx(100 - 0.24)
    assert calls[2][1][0] == pytest.approx(100 - 0.24 - 0.36)
    assert result.final_balance_hours[0] == pytest.approx(100 - 0.24 - 0.36 - 0.30)
    assert result.complete_coverage
    with pytest.raises(Exception, match="identity mismatch"):
        rolling.solve_rolling(c, exp, q=q + 1e-6)
