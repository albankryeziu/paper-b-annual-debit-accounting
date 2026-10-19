from __future__ import annotations

import json

import numpy as np
import pytest

from jobs.gamma import (
    _case_states,
    _bootstrap_ratio_record,
    _commit_case_status,
    _ratio_columns,
    _ratio_frame_columns,
    _ratio_metric,
    _require_finite_array,
)
from tpwrs.storage import atomic_write_json
from jobs.rate_bounds import _commit_case as _commit_rate_bound_case


def test_finite_ratio_is_strict_json(tmp_path):
    metric = _ratio_metric(1.0, 4.0, denominator_name="reference")
    path = tmp_path / "finite.json"
    atomic_write_json(path, _ratio_columns("change_pct", metric))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["change_pct"] == 25.0
    assert payload["change_pct_status"] == "FINITE_RESULT"
    assert payload["change_pct_reason"] is None
    assert payload["change_pct_numerator"] == 1.0
    assert payload["change_pct_denominator"] == 4.0
    assert payload["change_pct_comparison_tolerance"] > 0.0
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_zero_denominator_is_explicitly_undefined():
    metric = _ratio_metric(1.0, 0.0, denominator_name="reference")
    assert metric["value"] is None
    assert metric["status"] == "NOT_IDENTIFIED_ZERO_DENOMINATOR"
    assert metric["claimable"] is False
    assert metric["numerator"] == 1.0
    assert metric["denominator"] == 0.0
    assert "comparison tolerance" in metric["reason"]


def test_zero_over_zero_is_not_zero_percent():
    metric = _ratio_metric(0.0, 0.0, denominator_name="reference")
    assert metric["value"] is None
    assert metric["status"] == "NOT_IDENTIFIED_ZERO_DENOMINATOR"


def test_denominator_within_declared_tolerance_is_not_claimable():
    metric = _ratio_metric(
        -1.7881393432617188e-7,
        -2.086162567138672e-7,
        denominator_name="matched gain",
        comparison_tolerance=0.13689507884898983,
    )
    columns = _ratio_columns(
        "matched_efficiency_pct",
        metric,
        claimable_name="matched_efficiency_claimable",
    )
    assert columns["matched_efficiency_pct"] is None
    assert columns["matched_efficiency_pct_status"] == (
        "NOT_IDENTIFIED_ZERO_DENOMINATOR"
    )
    assert columns["matched_efficiency_claimable"] is False
    assert columns["matched_efficiency_pct_numerator"] == pytest.approx(
        -1.7881393432617188e-7
    )
    assert columns["matched_efficiency_pct_denominator"] == pytest.approx(
        -2.086162567138672e-7
    )
    assert columns["matched_efficiency_pct_comparison_tolerance"] == pytest.approx(
        0.13689507884898983
    )


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nonfinite_required_ratio_input_fails_closed(value):
    with pytest.raises(ValueError, match="non-finite"):
        _ratio_metric(value, 1.0, denominator_name="reference")


def test_empty_subset_remains_empty_without_fabricated_statistic():
    columns = _ratio_frame_columns(
        "change_pct",
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        denominator_name="empty reference",
    )
    assert all(len(value) == 0 for value in columns.values())


def test_not_applicable_method_field_has_explicit_semantics():
    columns = _ratio_columns(
        "method_metric",
        {
            "value": None,
            "status": "NOT_APPLICABLE",
            "reason": "Metric is not defined for M2_MATCHED.",
        },
    )
    assert columns["method_metric"] is None
    assert columns["method_metric_status"] == "NOT_APPLICABLE"


def test_pending_gamma_case_is_not_a_nan_placeholder():
    states = _case_states((0.7,), (2029,))
    assert states == {"2029/gamma_0p7": {"status": "PENDING"}}


def test_partially_completed_grid_distinguishes_certified_and_pending():
    states = _case_states(
        (0.7, 1.0),
        (2029, 2030),
        ["2029/gamma_0p7"],
    )
    assert states["2029/gamma_0p7"]["status"] == "CERTIFIED"
    assert states["2029/gamma_1"]["status"] == "PENDING"
    assert states["2030/gamma_0p7"]["status"] == "PENDING"


def test_required_scientific_array_nan_fails_closed():
    with pytest.raises(ValueError, match="non-finite"):
        _require_finite_array(np.asarray([1.0, np.nan]), "policy")


def test_undefined_bootstrap_rows_have_null_and_status():
    columns = _ratio_frame_columns(
        "matched_efficiency_pct",
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 2.0]),
        denominator_name="matched gain",
    )
    assert columns["matched_efficiency_pct"].tolist() == [None, 50.0]
    assert columns["matched_efficiency_pct_status"].tolist() == [
        "NOT_IDENTIFIED_ZERO_DENOMINATOR",
        "FINITE_RESULT",
    ]


def test_partial_undefined_bootstrap_has_no_finite_only_ci():
    record = _bootstrap_ratio_record(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 0.0, -1.0]),
        comparison_tolerance=0.1,
    )
    assert record["total_draws"] == 3
    assert record["finite_ratio_draws"] == 1
    assert record["undefined_ratio_draws"] == 2
    assert record["undefined_zero_denominator_draws"] == 1
    assert record["undefined_nonpositive_denominator_draws"] == 1
    assert record["fraction_undefined"] == pytest.approx(2.0 / 3.0)
    assert record["status"] == "UNSTABLE_NOT_IDENTIFIED"
    assert record["matched_efficiency_claimable"] is False
    assert record["finite_only_ci_used"] is False
    assert record["matched_efficiency_percentile_ci"] is None
    json.dumps(record, allow_nan=False)


def test_aggregate_failure_cannot_remove_certified_case(tmp_path):
    case_root = tmp_path / "year_2029" / "gamma_0p7"
    case_root.mkdir(parents=True)
    _commit_case_status(
        case_root,
        implementation_hash="fingerprint",
        year=2029,
        gamma=0.7,
        summary_row={
            **_ratio_columns(
                "matched_efficiency_pct",
                _ratio_metric(0.0, 0.0, denominator_name="matched gain"),
            )
        },
        artifacts={},
    )
    with pytest.raises(RuntimeError, match="aggregate"):
        raise RuntimeError("aggregate writer failed")
    payload = json.loads((case_root / "STATUS.json").read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    assert payload["summary_row"]["matched_efficiency_pct"] is None


def test_rate_bound_certified_solve_survives_reporting_failure(tmp_path):
    case_root = tmp_path / "year_2029" / "tight"
    m3 = tmp_path / "m3_metrics.json"
    matched = tmp_path / "matched_metrics.json"
    atomic_write_json(m3, {"status": "PASS", "replay": "PASS"})
    atomic_write_json(matched, {"status": "PASS", "replay": "PASS"})
    metric = _ratio_metric(
        0.0,
        0.0,
        denominator_name="matched gain",
        comparison_tolerance=0.1,
    )
    _commit_rate_bound_case(
        case_root,
        implementation_hash="fingerprint",
        year=2029,
        label="tight",
        summary_row={
            **_ratio_columns(
                "matched_efficiency_pct",
                metric,
                claimable_name="matched_efficiency_claimable",
            )
        },
        method_artifacts={"M3_metrics": m3, "M2_MATCHED_metrics": matched},
    )
    with pytest.raises(RuntimeError, match="aggregate"):
        raise RuntimeError("aggregate writer failed")
    payload = json.loads((case_root / "STATUS.json").read_text(encoding="utf-8"))
    assert payload["status"] == "COMPLETE"
    assert payload["summary_row"]["matched_efficiency_pct"] is None
    assert payload["summary_row"]["matched_efficiency_claimable"] is False
