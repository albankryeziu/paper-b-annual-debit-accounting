from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jobs.ac import (
    ac_input_preflight,
    ac_normalized_thermal_exceedance,
    classify_normalized_ordering,
    dc_normalized_thermal_exceedance,
)
from jobs.certificates import corrected_m1_certificate
from jobs.common import frozen_matched_decomposition
from jobs.dc import independent_dc_acceptance
from jobs.gamma import DEFAULT_GAMMA_GRID, _commit_case_status, _gamma_card, _gamma_tag, _run_exact_flat_campaign
from jobs.sensitivity_equivalence import exact_equivalence_certificate, is_exact_flat_card, matched_stage1_value
from jobs.rate_bounds import _run_exact_flat_rate_bounds
from tpwrs.contracts import CardSpec, NetworkState, PhysicalCase, TimeGrid
from tpwrs.cards import matched_decomposition
from tpwrs.storage import ContractError, sha256_json


def test_report_accepts_only_explicit_hash_bound_compatible_status(tmp_path, monkeypatch):
    import jobs.report as report

    path = tmp_path / "STATUS.json"
    path.write_text(
        '{"status":"COMPLETE","scientific_fingerprint":"old","outputs":{}}\n',
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(report, "scientific_fingerprint", lambda: "new")
    monkeypatch.setattr(
        report,
        "compatibility_entry",
        lambda target, *, use, payload: calls.append((Path(target), use, payload)),
    )
    payload = report.status_if_valid(path, compatibility_use="DECLARED_USE")
    assert payload["status"] == "COMPLETE"
    assert calls == [(path, "DECLARED_USE", payload)]
    with pytest.raises(ValueError, match="Changed scientific identity"):
        report.status_if_valid(path)


def flat_card(*, card_id="flat", bounds=(0.5, 1.5)):
    return CardSpec(
        card_id=card_id,
        allowance_hash="allowance",
        feature_set_id="F0",
        model_family="test",
        thresholds={"import": (0.2, 0.8), "export": (0.3, 0.7)},
        rates={"import": (1.0, 1.0, 1.0), "export": (1.0, 1.0, 1.0)},
        q_bounds=bounds,
        source_hashes={"source": "hash"},
    )


def test_r101_frozen_matched_envelope_accepts_observed_solver_reversal():
    s_m2 = 140000000.0
    # Exact stored values reported as 136894978.849 and 136894978.986.
    s_m3 = 136894978.8489896
    s_matched = 136894978.98588556
    with pytest.raises(ContractError, match="exceeds M3"):
        matched_decomposition(s_m2, s_m3, s_matched, tol=1e-7, rel_tol=1e-9)
    accepted = frozen_matched_decomposition(s_m2, s_m3, s_matched)
    assert accepted["raw_shape_residual"] == pytest.approx(-0.13689595460891724, abs=1e-15)
    assert accepted["comparison_tolerance"] == pytest.approx(0.13689597898588557, abs=1e-15)
    assert accepted["shape_penalty"] == 0.0
    outside = s_m3 + accepted["comparison_tolerance"] + 1e-6
    with pytest.raises(ContractError, match="exceeds M3"):
        frozen_matched_decomposition(s_m2, s_m3, outside)


def test_exact_reuse_uses_matched_stage1_optimum_not_stage2_policy_replay():
    solution = SimpleNamespace(
        s_value_mw_h_row=124251361.1506221,
        stage1=SimpleNamespace(objective_value=124251361.02636959),
    )
    assert matched_stage1_value(solution) == 124251361.02636959


def test_all_frozen_gamma_cards_are_bitwise_flat(tmp_path):
    parent = flat_card()
    for gamma in DEFAULT_GAMMA_GRID:
        card = _gamma_card(parent, None, gamma, tmp_path)
        assert is_exact_flat_card(card)
        assert all(tuple(values) == (1.0, 1.0, 1.0) for values in card.rates.values())
        rates = json.loads((tmp_path / f"GAMMA_{_gamma_tag(gamma)}_RATES.json").read_text())
        assert set(rates["modes"].values()) == {"EXACT_FLAT_PARENT_EQUIVALENCE"}


def test_exact_flat_campaign_function_contains_no_run_annual_call():
    tree = ast.parse(inspect.getsource(_run_exact_flat_campaign))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_annual" not in called


def test_exact_flat_rate_bound_campaign_contains_no_run_annual_call():
    tree = ast.parse(inspect.getsource(_run_exact_flat_rate_bounds))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "run_annual" not in called


def test_nonflat_parent_is_not_exact_reuse_eligible():
    parent = flat_card()
    nonflat = CardSpec(**{**parent.__dict__, "card_id": "nonflat", "rates": {"import": (1.1, 1.0, 0.9), "export": (1.0, 1.0, 1.0)}})
    assert not is_exact_flat_card(nonflat)
    with pytest.raises(RuntimeError, match="bitwise"):
        exact_equivalence_certificate(
            year=2029,
            variant_kind="gamma",
            variant_value=0.5,
            final_card=nonflat,
            selected_parent_card=parent,
            core_bundle={},
            physical_case_hash="physical",
            scientific_fingerprint="fingerprint",
        )


def synthetic_bundle(parent_hash="m3", allowance_ok=True):
    allowance = {
        "participant_ids": ["p"],
        "values_hours": [1.0],
        "source": "m3_realized_physical_fde",
        "nominal_rho": None,
        "source_solution_hash": parent_hash if allowance_ok else "wrong",
        "contract_mask_hash": None,
    }
    m3 = SimpleNamespace(solution_hash=parent_hash, q=np.ones((1, 1)))
    matched = SimpleNamespace(solution_hash="matched", allowance_hash=sha256_json(allowance))
    return {
        "solutions": {"M3": m3, "M2_MATCHED": matched},
        "metrics": {"M3": {"solution_hash": parent_hash}, "M2_MATCHED": {"solution_hash": "matched"}},
        "matched_experiment": {"experiment": {"allowance": allowance}},
        "source_files": {},
    }


def test_wrong_core_or_allowance_lineage_blocks_exact_reuse():
    parent = flat_card()
    final = CardSpec(
        **{
            **parent.__dict__,
            "card_id": "tight",
            "q_bounds": (0.7, 1.3),
            "source_hashes": {**parent.source_hashes, "selected_parent_card": parent.hash},
        }
    )
    with pytest.raises(RuntimeError, match="Matched allowance"):
        exact_equivalence_certificate(
            year=2029,
            variant_kind="rate_bound",
            variant_value="tight",
            final_card=final,
            selected_parent_card=parent,
            core_bundle=synthetic_bundle(allowance_ok=False),
            physical_case_hash="physical",
            scientific_fingerprint="fingerprint",
        )


def test_gamma_case_status_binds_source_solution_and_certificate_hashes(tmp_path):
    _commit_case_status(
        tmp_path,
        implementation_hash="fingerprint",
        year=2029,
        gamma=0.7,
        summary_row={"action": "REUSE_EXACT_EQUIVALENT_CORE"},
        artifacts={},
        equivalence={
            "action": "REUSE_EXACT_EQUIVALENT_CORE",
            "source_m3_solution_hash": "m3",
            "source_matched_solution_hash": "matched",
            "exact_equivalence_certificate_hash": "certificate",
            "final_card_hash": "card",
        },
    )
    status = json.loads((tmp_path / "STATUS.json").read_text())
    assert status["source_m3_solution_hash"] == "m3"
    assert status["source_matched_solution_hash"] == "matched"
    assert status["exact_equivalence_certificate_hash"] == "certificate"


def test_b03_zero_allowance_uses_distinct_feasible_witness():
    grid = TimeGrid.from_index(pd.date_range("2028-01-01", periods=1, freq="1h", tz="UTC"), 1.0)
    state = NetworkState("INTACT", ("line",), np.array([[0.5, -0.5]]), np.array([0.0]), np.array([0.1]))
    case = PhysicalCase(
        time_grid=grid,
        participant_ids=("p",),
        bus_ids=("b1", "b2"),
        capacities_mw=np.array([100.0]),
        requests_mw=np.array([[1.0]]),
        contract_mask=np.ones((1, 1), bool),
        eligibility_mask=np.ones((1, 1), bool),
        p_background_mw=np.zeros((1, 2)),
        A=np.array([[1.0e7], [-1.0e7]]),
        network_states=(state,),
        participant_directions=("import",),
        participant_set_id="P",
        contingency_set_id="C",
        balancing_key_hash="B",
        source_hashes={"test": "hash"},
        declared_case_id="B03_ZERO",
    )
    selected = np.array([[5.0e-8]])
    provisional = corrected_m1_certificate(case, selected, np.array([0.0]), 1.0)
    selected_s = provisional["selected_replay_S_mw_h_row"]
    witness_s = provisional["witness_S_mw_h_row"]
    assert witness_s > selected_s
    lower = (selected_s + witness_s) / 2.0
    corrected = corrected_m1_certificate(case, selected, np.array([0.0]), lower)
    old_false_pass = selected_s + corrected["certificate_epsilon"] < lower
    assert old_false_pass
    assert corrected["status"] == "FAIL"
    assert corrected["witness_debit_hours"] == [0.0]
    assert corrected["witness_scaling_factors"] == [0.0]


def test_dc_material_sign_disagreement_fails_despite_small_magnitude_error():
    rows = pd.DataFrame({"frozen_flow_mw": [2e-6], "independent_flow_mw": [-2e-6], "rating_mw": [100.0]})
    checked, summary = independent_dc_acceptance(rows)
    assert checked.loc[0, "magnitude_within_tolerance"]
    assert not checked.loc[0, "material_sign_agreement"]
    assert summary["passed"] is False


def test_dc_correct_sign_and_rowwise_magnitude_passes():
    rows = pd.DataFrame({"frozen_flow_mw": [50.0, 0.0], "independent_flow_mw": [50.00005, 0.0], "rating_mw": [100.0, 1e6]})
    _, summary = independent_dc_acceptance(rows)
    assert summary["passed"] is True


def test_ac_normalized_apparent_power_counterexample():
    rating = np.array([100.0, 1000.0])
    dc_a = dc_normalized_thermal_exceedance(np.array([100.0, 1020.0]), rating)
    dc_b = dc_normalized_thermal_exceedance(np.array([105.0, 1000.0]), rating)
    assert dc_a < dc_b
    ac_a = ac_normalized_thermal_exceedance(np.array([100.0, 102.0]))
    ac_b = ac_normalized_thermal_exceedance(np.array([105.0, 100.0]))
    result = classify_normalized_ordering(dc_a, dc_b, ac_a, ac_b)
    assert result["classification"] == "ORDER_PRESERVED"


@pytest.mark.parametrize(
    "values,expected",
    [
        ((1.0, 1.00005, 2.0, 3.0), "DC_TIE"),
        ((1.0, 2.0, 3.0, 3.00005), "AC_TIE_LOSS_OF_DISCRIMINATION"),
        ((1.0, 2.0, 4.0, 3.0), "ORDER_REVERSED"),
    ],
)
def test_ac_ordering_classifications(values, expected):
    assert classify_normalized_ordering(*values)["classification"] == expected


def test_ac_preflight_requires_all_three_hash_bound_sources(tmp_path):
    assert ac_input_preflight(tmp_path / "missing.json")["status"] == "AC_INPUTS_BLOCKED"
    config = {"provenance": "test"}
    for key in ("q_fixed", "generator_dispatch", "participant_connection_weights"):
        source = tmp_path / f"{key}.csv"
        source.write_text("x\n1\n", encoding="utf-8")
        from tpwrs.storage import sha256_file
        config[key] = {"path": source.name, "sha256": sha256_file(source)}
    path = tmp_path / "AC_INPUTS.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert ac_input_preflight(path)["status"] == "PASS"
