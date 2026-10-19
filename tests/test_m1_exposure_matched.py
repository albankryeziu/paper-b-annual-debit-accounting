"""Final numerical certificate reconstruction and verification.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jobs.m1_exposure_matched import (
    Archive,
    ceiling_audit,
    exact_energy,
    exposure,
    guarded_witness,
    validate_identity,
)
from tpwrs.contracts import AllowanceSpec, ExperimentCase, TimeGrid
from tpwrs.replay import physical_objectives
from tpwrs.rolling import solve_rolling
from tpwrs.storage import sha256_file


def test_matched_arithmetic_nonuniform_unequal_capacities():
    x = np.array([[20.0, 30.0], [10.0, 60.0], [40.0, 0.0]])
    dt, p = np.array([0.5, 2.0, 0.25]), np.array([40.0, 120.0])
    d, e, bound = exposure(x, dt, p)
    np.testing.assert_array_equal(e, [40.0, 135.0])
    np.testing.assert_array_equal(d, [1.0, 1.125])
    assert np.all(abs(p * d - e) <= bound)


def test_zero_exposure_rejects_positive_action_exactly(simple_case):
    raw = np.array([[5e-10], [0.0]])
    source = exact_energy(np.zeros_like(raw), simple_case.time_grid.delta)
    audit = ceiling_audit(simple_case, raw, np.array([0.0]), source)
    assert not audit["strict_pass"] and audit["max_energy_excess_MWh"] > 0
    witness, guard = guarded_witness(simple_case, raw, np.array([0.0]), source)
    assert not np.any(witness) and guard["strict_pass"]


def test_positions_cannot_pool_unused_exposure():
    case = SimpleNamespace(
        time_grid=SimpleNamespace(delta=np.array([1.0])),
        capacities_mw=np.array([10.0, 100.0]),
        xbar_mw=np.array([[10.0, 100.0]]),
    )
    # Total FDE .5 is below the sum 1, but position 1 has no entitlement.
    audit = ceiling_audit(
        case, np.array([[5.0, 0.0]]), np.array([0.0, 1.0]), [Fraction(0), Fraction(100)]
    )
    assert not audit["strict_pass"] and audit["budget_violating_positions"] == 1


def test_valid_M1_in_annual_matched_domain(simple_case):
    x = np.array([[10.0], [0.0]])
    u = np.array([[1.0], [0.0]])
    assert np.all(simple_case.materiality_floor_mw * u <= x)
    assert np.all(x <= simple_case.xbar_mw * u)
    h, _, _ = exposure(x, simple_case.time_grid.delta, simple_case.capacities_mw)
    assert ceiling_audit(simple_case, x, h, exact_energy(x, simple_case.time_grid.delta))[
        "strict_pass"
    ]


def test_nearly_zero_budget_guard_recomputes_full_objective(simple_case):
    raw = np.full((2, 1), 5e-8)
    limits = np.array([1e-12])
    source = [Fraction(float(1e-10))]
    before = physical_objectives(simple_case, raw)[0]
    witness, audit = guarded_witness(simple_case, raw, limits, source)
    after = physical_objectives(simple_case, witness)[0]
    assert audit["strict_pass"] and np.max(witness) < np.max(raw)
    # Reducing a helpful restriction increases S: no monotonicity shortcut.
    assert after > before
    assert physical_objectives(simple_case, witness)[1] <= float(source[0])


def test_box_projection_is_separate_from_radial_guard(simple_case):
    raw = np.array([[-1e-9], [100.0 + 1e-9]])
    witness, audit = guarded_witness(simple_case, raw, np.array([2.0]), [Fraction(200)])
    assert audit["box_projection_max_change_MW"] > 0
    assert audit["strict_pass"] and witness[0, 0] == 0 and witness[1, 0] <= 100


def test_overlap_commits_only_24_hours_and_truncates_tail(simple_case, monkeypatch):
    import tpwrs.rolling as rolling

    t = 50
    c = replace(
        simple_case,
        time_grid=TimeGrid.from_index(
            pd.date_range("2029-01-01", periods=t, freq="1h", tz="UTC"), 1.0
        ),
        requests_mw=np.full((t, 1), 100.0),
        contract_mask=np.ones((t, 1), bool),
        eligibility_mask=np.ones((t, 1), bool),
        p_background_mw=np.zeros((t, 2)),
    )
    a = AllowanceSpec.explicit(c.participant_ids, [2.0])
    e = ExperimentCase(
        "roll",
        c.physical_case_id,
        c.physical_case_hash,
        c.time_grid.grid_id,
        "M2",
        a,
        "rolling",
        lookahead_hours=48,
        implementation_hours=24,
    )
    seen = []

    def solve(sub, exp, **kwargs):
        seen.append((sub.n_time, exp.allowance.values.copy()))
        # .01 FDE/hour; the overlap is proposed twice and implemented once.
        return SimpleNamespace(x_mw=np.ones((sub.n_time, 1)), u=None)

    monkeypatch.setattr(rolling, "solve_policy", solve)
    result = solve_rolling(c, e)
    assert [s[0] for s in seen] == [48, 26, 2]
    np.testing.assert_allclose([s[1][0] for s in seen], [2.0, 1.76, 1.52], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result.final_balance_hours, [1.5], atol=1e-14, rtol=0)
    assert result.covered_intervals == ((0, 24), (24, 48), (48, 50))
    assert result.complete_coverage
    # A separate annual invocation starts afresh, despite previous warm starts.
    seen.clear()
    e2 = replace(e, case_id="2030", allowance=AllowanceSpec.explicit(c.participant_ids, [1.0]))
    result2 = solve_rolling(c, e2)
    assert seen[0][1][0] == 1.0 and result2.final_balance_hours[0] == pytest.approx(0.5)
    assert e.allowance.values[0] == 2.0


@pytest.mark.parametrize("fault", ["rho", "year", "order", "network", "case", "rule"])
def test_source_identity_rejects_wrong_coordinates(simple_case, fault):
    t = 8760
    c = replace(
        simple_case,
        time_grid=TimeGrid.uniform_hourly(2029),
        requests_mw=np.full((t, 1), 100.0),
        contract_mask=np.ones((t, 1), bool),
        eligibility_mask=np.ones((t, 1), bool),
        p_background_mw=np.zeros((t, 2)),
    )
    a = AllowanceSpec.from_nominal(c.participant_ids, c.contract_mask, c.time_grid, 0.15)
    e = ExperimentCase(
        "source",
        c.physical_case_id,
        c.physical_case_hash,
        c.time_grid.grid_id,
        "M1",
        a,
        "rolling",
        lookahead_hours=48,
        implementation_hours=24,
    )
    year = 2029
    validate_identity(c, e, year)
    if fault == "rho":
        e = replace(e, allowance=replace(a, nominal_rho=0.03))
    elif fault == "year":
        year = 2030
    elif fault == "order":
        e = replace(e, allowance=replace(a, participant_ids=("wrong",)))
    elif fault == "network":
        e = replace(e, physical_case_hash="wrong")
    elif fault == "case":
        e = replace(e, physical_case_id="wrong")
    elif fault == "rule":
        e = replace(e, rule="M2")
    with pytest.raises((ValueError, RuntimeError)):
        validate_identity(c, e, year)


def test_archive_original_preservation_and_tamper_detection(tmp_path):
    original = tmp_path / "MAIN_POLICY_RESULTS.csv"
    original.write_text("archived evidence\n")
    digest = sha256_file(original)
    (tmp_path / "SHA256SUMS").write_text(digest + "  MAIN_POLICY_RESULTS.csv\n")
    archive = Archive(tmp_path)
    archive.verify(original)
    (tmp_path / "extension.json").write_text("{}")
    assert archive.preservation()["status"] == "PASS"
    assert sha256_file(original) == digest
    original.write_text("tampered")
    with pytest.raises(ValueError):
        archive.preservation()


@pytest.mark.parametrize("x", [np.array([[np.nan]]), np.array([[np.inf]])])
def test_nonfinite_sources_rejected(x):
    with pytest.raises(ValueError):
        exposure(x, [1.0], [100.0])
