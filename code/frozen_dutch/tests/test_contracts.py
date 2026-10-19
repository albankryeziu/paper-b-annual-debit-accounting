from __future__ import annotations


# ============================================================================
# tests.test_contracts (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from tpwrs.contracts import AllowanceSpec, ExperimentCase
from tpwrs.contracts import TimeGrid


def test_contracts__delta_and_floor_allowance():
    grid=TimeGrid.uniform_hourly(2028)
    mask=np.ones((grid.n_intervals,1),bool)
    allow=AllowanceSpec.from_nominal(("p",),mask,grid,0.15)
    assert allow.values[0] == 1317.0


def test_contracts__zero_request_does_not_change_contract_allowance(simple_case):
    a=AllowanceSpec.from_nominal(simple_case.participant_ids,simple_case.contract_mask,simple_case.time_grid,0.5)
    changed=simple_case.requests_mw.copy(); changed[:]=0
    b=AllowanceSpec.from_nominal(simple_case.participant_ids,simple_case.contract_mask,simple_case.time_grid,0.5)
    assert np.allclose(a.values,b.values)


def test_contracts__experiment_requires_card_for_m3(simple_case):
    allowance=AllowanceSpec.explicit(simple_case.participant_ids,[1.0])
    with pytest.raises(Exception):
        ExperimentCase("bad",simple_case.physical_case_id,simple_case.physical_case_hash,simple_case.time_grid.grid_id,"M3",allowance)


def test_contracts__time_grid_adjacency_fails():
    with pytest.raises(Exception):
        TimeGrid.from_index(pd.to_datetime(["2028-01-01T00:00Z","2028-01-01T02:00Z"]),1.0)


# ============================================================================
# tests.test_dutch (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.inputs import build_nondispatchable_state, balance_with_frozen_keys
from tpwrs.network import Branch, build_affine_dc_state, enumerate_retained_contingencies
from tpwrs.network import calibrate_simplex_embedding, embedding_metrics, primary_participant_gate


def test_dutch__component_allocation_and_balance():
    values=np.array([[100.0]])
    zero=np.array([[0.0]])
    W=np.array([[0.6],[0.4]])
    p,report=build_nondispatchable_state(values,zero,zero,zero,W,W,W,W)
    assert abs(p.sum()+100)<1e-9 and report["demand_conservation_max_mw"]<1e-9
    balanced,audit=balance_with_frozen_keys(p,np.array([1.,0.]),np.array([0.,1.]))
    assert audit["max_balance_residual_mw"]<1e-8


def test_dutch__phase_shift_creates_affine_offset():
    branches=[Branch("l12","b1","b2",10.0,100.0,phase_shift_rad=0.1), Branch("l23","b2","b3",10.0,100.0), Branch("l31","b3","b1",10.0,100.0)]
    state=build_affine_dc_state(("b1","b2","b3"),branches,slack_bus="b1")
    assert np.max(np.abs(state.h_mw))>1e-8


def test_dutch__contingency_excludes_islanding_and_parallel():
    branches=[Branch("l12a","b1","b2",10,100,physical_circuit_count=2,outage_semantics="equivalent_corridor"),Branch("l12b","b1","b2",10,100)]
    states,excluded=enumerate_retained_contingencies(("b1","b2"),branches)
    assert any(x["element_id"]=="l12a" for x in excluded)


def test_dutch__embedding_calibration_and_gate():
    G=np.array([[1.,0.],[0.,1.]])
    target=np.array([0.25,0.75])
    w=calibrate_simplex_embedding(target,G,lambda_w=0)
    assert np.allclose(w,[0.25,0.75],atol=1e-5)
    m=embedding_metrics("m",target,G@w)
    gate=primary_participant_gate(["d"],["m"],[m])
    assert gate["all_multibus_pass"]


# ============================================================================
# tests.test_m3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from tpwrs.features import FeatureProvenanceRecord, build_feature_schema, F0_BASE
from tpwrs.cards import average_tie_percentile, rate_card_qp, assign_buckets, score_thresholds
from tpwrs.cards import project_gamma_rates


def test_m3__average_tie_percentile():
    r=average_tie_percentile([1,1,3])
    assert np.allclose(r,[0.5,0.5,1.0])


def test_m3__rate_qp_and_sparse_fallback():
    q,mode=rate_card_qp([1,2,4],[1,1,1])
    assert mode=="OPTIMAL_QP" and q[0]>=q[1]>=q[2] and abs(np.mean(q)-1)<1e-7
    q2,mode2=rate_card_qp([1,2,3],[1,0,0])
    assert np.allclose(q2,1) and mode2.startswith("FLAT")


def test_m3__gamma_endpoints():
    q=np.array([1.2,1.0,0.8]); w=np.array([1,2,1],float)
    assert np.allclose(project_gamma_rates(q,w,0)[0],1)
    assert np.allclose(project_gamma_rates(q,w,1)[0],q)


def test_m3__feature_schema_fail_closed():
    records=[FeatureProvenanceRecord(f,"src","hash","2028-01-01T00:00:00+00:00","2028","UTC","forecast","frozen",True,"deployable") for f in F0_BASE]
    schema=build_feature_schema("F0",records,F0_BASE)
    assert schema.feature_set_id=="F0"
    bad=records.copy(); bad[0]=FeatureProvenanceRecord(F0_BASE[0],"src","hash",None,"2028","UTC","realization","final",False,"deployable")
    with pytest.raises(Exception): build_feature_schema("F0",bad,F0_BASE)


def test_m3__rate_qp_is_stable_under_extreme_linear_scaling():
    A=np.array([8.0e8,2.0e8,-5.0e8])
    W=np.array([2.5e5,7.5e5,4.0e5])
    q,mode=rate_card_qp(A,W,q_bounds=(0.5,1.5),lambda_q=1e-3)
    assert mode=="OPTIMAL_QP"
    assert 0.5-1e-10 <= q[2] <= q[1] <= q[0] <= 1.5+1e-10
    assert abs(float(W@q-W.sum())) <= 1e-8*W.sum()



def test_m3__rate_qp_large_scale_agrees_with_linear_limit_vertex():
    # With a negligible regularizer relative to A, this instance has the
    # unique monotone/neutral linear optimum at the extreme rate vector.
    A=np.array([1.0e10,2.0e10,4.0e10])
    W=np.ones(3)
    q,mode=rate_card_qp(A,W,q_bounds=(0.5,1.5),lambda_q=1e-3)
    assert mode=="OPTIMAL_QP"
    assert np.allclose(q,[1.5,1.0,0.5],atol=1e-10)


def test_m3__rate_qp_zero_weight_bucket_still_optimizes_linear_term():
    # Two positive reference weights keep the QP active.  The third bucket has
    # zero W, so neutrality/quadratic regularization do not involve q3, but its
    # linear A term still must be optimized subject to monotonicity and bounds.
    A=np.array([-1.19405662e-6,8.59593844e-7,2.96826644e-7])
    W=np.array([2252448.96029459,3999217.71625605,0.0])
    q,mode=rate_card_qp(A,W,q_bounds=(0.5,1.5),lambda_q=52037.12682321861)
    assert mode=="OPTIMAL_QP"
    assert np.allclose(q,[1.0,1.0,0.5],atol=1e-10)
    assert abs(float(W@q-W.sum())) <= 1e-8*W.sum()


# ============================================================================
# tests.test_gamma_v3_final (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.cards import project_gamma_rates


def test_gamma_v3_final__gamma_projection_certifies_order_bounds_and_reference_mean() -> None:
    selected = np.array([1.25, 1.0, 0.75])
    weight = np.array([2.0, 3.0, 2.0])
    for gamma in (0.0, 0.5, 1.0, 1.8):
        q, _ = project_gamma_rates(selected, weight, gamma)
        assert np.isfinite(q).all()
        assert q[0] + 1e-8 >= q[1] >= q[2] - 1e-8
        assert np.min(q) >= 0.25 - 1e-8
        assert np.max(q) <= 2.0 + 1e-8
        assert abs(float(weight @ q - weight.sum())) <= 1e-7
    assert np.array_equal(project_gamma_rates(selected, weight, 1.0)[0], selected)
    assert np.array_equal(project_gamma_rates(selected, weight, 0.0)[0], np.ones(3))


# ============================================================================
# tests.test_m3_teacher_stage_contract (consolidated; see SOURCE_MAP.json)
# ============================================================================

from types import SimpleNamespace

import pytest

from tpwrs.storage import ContractError
from tpwrs.cards import build_teacher_rows


def test_m3_teacher_stage_contract__teacher_rejects_stage3_selected_primal_before_using_case_data() -> None:
    solution = SimpleNamespace(
        rule="M2",
        stage1=SimpleNamespace(status="OPTIMAL"),
        stage2=SimpleNamespace(status="OPTIMAL"),
        stage3=SimpleNamespace(status="OPTIMAL"),
        m2_budget_duals={"p": 1.0},
    )
    case = SimpleNamespace(n_participants=1)
    with pytest.raises(ContractError, match="Stage-III-selected"):
        build_teacher_rows(case, solution, year=2025)


def test_m3_teacher_stage_contract__fresh_stage2_teacher_policy_passes_full_face_validation(simple_case) -> None:
    from tpwrs.execution import validate_policy_solution
    from tpwrs.contracts import nominal_experiment
    from tpwrs.contracts import SolverProfile
    from tpwrs.model import solve_policy

    experiment = nominal_experiment(
        simple_case,
        case_id="TEACHER_FACE_VALIDATION",
        rule="M2",
        rho=1.0,
        solver_profile=SolverProfile(backend="scipy", output_flag=False),
        implementation_hash="test-current-release",
        result_root_id="test/teacher",
    )
    solution = solve_policy(simple_case, experiment, use_stage3=False)
    from tpwrs.replay import GENERIC_EXPLICIT_FACE_CONTRACT
    assert solution.stage1.metadata["face_numeric_contract"] == GENERIC_EXPLICIT_FACE_CONTRACT
    assert solution.stage2.metadata["face_numeric_contract"] == GENERIC_EXPLICIT_FACE_CONTRACT
    assert solution.stage1.metadata["generic_face_certification_tolerance"] == pytest.approx(1e-7)
    assert solution.stage2.metadata["generic_face_certification_tolerance"] == pytest.approx(1e-7)
    report = validate_policy_solution(
        simple_case,
        experiment,
        solution,
        require_stage3=False,
    )
    assert report["passed"] is True
    assert solution.stage3 is None


def test_m3_teacher_stage_contract__annual_support_face_numeric_contract_matches_validator(monkeypatch) -> None:
    from types import SimpleNamespace

    from tpwrs.execution import validate_policy_solution
    from tpwrs.replay import ANNUAL_SUPPORT_FACE_CONTRACT, annual_support_face_numeric_tolerance
    from tpwrs.storage import ReplayFailure
    from tpwrs.contracts import SolverProfile
    from tpwrs.contracts import StageCertificate

    profile = SolverProfile(backend="scipy", feasibility_tol=1e-8, output_flag=False)
    certified = annual_support_face_numeric_tolerance(profile)
    assert certified == pytest.approx(1e-6)

    metadata = {
        "face_numeric_contract": ANNUAL_SUPPORT_FACE_CONTRACT,
        "support_certification_tolerance": certified,
    }
    solution = SimpleNamespace(
        stage1=StageCertificate("I", "OPTIMAL", 100.0, 0.0, metadata=metadata),
        stage2=StageCertificate("II", "OPTIMAL", 50.0, 0.0, metadata=metadata),
        stage3=None,
        replay_summary={"passed": True},
    )
    experiment = SimpleNamespace(
        solver_profile=profile,
        stage1_tolerance=lambda _value: 0.0,
        stage2_tolerance=lambda _value: 0.0,
    )
    case = object()

    def report(s_value: float, e_value: float = 50.0):
        return SimpleNamespace(
            s_value_mw_h_row=s_value,
            e_value_mwh=e_value,
            max_budget_violation_h=0.0,
            max_balance_residual_mw=0.0,
            max_x_bound_violation_mw=0.0,
        )

    # This is the exact mismatch class that occurred on the native 2026
    # teacher: the annual solver certifies up to 1e-6, whereas the former
    # validator allowed only 1e-7.  The shared contract must accept it.
    monkeypatch.setattr("tpwrs.execution.replay_solution", lambda *_a, **_k: report(100.0 + 0.5e-6))
    assert validate_policy_solution(case, experiment, solution, require_stage3=False)["passed"] is True

    # The contract is still fail-closed beyond the certified support envelope.
    monkeypatch.setattr("tpwrs.execution.replay_solution", lambda *_a, **_k: report(100.0 + 1.1e-6))
    with pytest.raises(ReplayFailure, match="outside the certified Stage-I face"):
        validate_policy_solution(case, experiment, solution, require_stage3=False)


def test_m3_teacher_stage_contract__generic_explicit_face_contract_matches_validator(monkeypatch) -> None:
    from types import SimpleNamespace

    from tpwrs.execution import validate_policy_solution
    from tpwrs.replay import GENERIC_EXPLICIT_FACE_CONTRACT, generic_face_numeric_tolerance
    from tpwrs.storage import ReplayFailure
    from tpwrs.contracts import SolverProfile
    from tpwrs.contracts import StageCertificate

    profile = SolverProfile(backend="scipy", feasibility_tol=1e-8, output_flag=False)
    certified = generic_face_numeric_tolerance(profile)
    assert certified == pytest.approx(1e-7)
    metadata = {
        "face_numeric_contract": GENERIC_EXPLICIT_FACE_CONTRACT,
        "generic_face_certification_tolerance": certified,
    }
    solution = SimpleNamespace(
        stage1=StageCertificate("I", "OPTIMAL", 100.0, 0.0, metadata=metadata),
        stage2=StageCertificate("II", "OPTIMAL", 50.0, 0.0, metadata=metadata),
        stage3=None,
        replay_summary={"passed": True},
    )
    experiment = SimpleNamespace(
        solver_profile=profile,
        stage1_tolerance=lambda _value: 0.0,
        stage2_tolerance=lambda _value: 0.0,
    )

    def report(s_value: float):
        return SimpleNamespace(
            s_value_mw_h_row=s_value,
            e_value_mwh=50.0,
            max_budget_violation_h=0.0,
            max_balance_residual_mw=0.0,
            max_x_bound_violation_mw=0.0,
        )

    monkeypatch.setattr("tpwrs.execution.replay_solution", lambda *_a, **_k: report(100.0 + 0.5e-7))
    assert validate_policy_solution(object(), experiment, solution, require_stage3=False)["passed"] is True

    monkeypatch.setattr("tpwrs.execution.replay_solution", lambda *_a, **_k: report(100.0 + 1.1e-7))
    with pytest.raises(ReplayFailure, match="outside the certified Stage-I face"):
        validate_policy_solution(object(), experiment, solution, require_stage3=False)


def test_m3_teacher_stage_contract__legacy_generic_solution_does_not_silently_inherit_r9_contract(monkeypatch) -> None:
    from types import SimpleNamespace

    from tpwrs.execution import validate_policy_solution
    from tpwrs.storage import ReplayFailure
    from tpwrs.contracts import SolverProfile
    from tpwrs.contracts import StageCertificate

    profile = SolverProfile(backend="scipy", feasibility_tol=1e-8, output_flag=False)
    solution = SimpleNamespace(
        stage1=StageCertificate("I", "OPTIMAL", 100.0, 0.0),
        stage2=StageCertificate("II", "OPTIMAL", 50.0, 0.0),
        stage3=None,
        replay_summary={"passed": True},
    )
    experiment = SimpleNamespace(
        solver_profile=profile,
        stage1_tolerance=lambda _value: 0.0,
        stage2_tolerance=lambda _value: 0.0,
    )
    report = SimpleNamespace(
        s_value_mw_h_row=100.0 + 0.5e-6,
        e_value_mwh=50.0,
        max_budget_violation_h=0.0,
        max_balance_residual_mw=0.0,
        max_x_bound_violation_mw=0.0,
    )
    monkeypatch.setattr("tpwrs.execution.replay_solution", lambda *_a, **_k: report)
    with pytest.raises(ReplayFailure, match="outside the certified Stage-I face"):
        validate_policy_solution(object(), experiment, solution, require_stage3=False)


def test_m3_teacher_stage_contract__old_native_violation_is_not_accepted_by_a_wider_certificate(monkeypatch) -> None:
    from types import SimpleNamespace

    from tpwrs.execution import validate_policy_solution
    from tpwrs.replay import GENERIC_EXPLICIT_FACE_CONTRACT, generic_face_numeric_tolerance
    from tpwrs.contracts import SolverProfile
    from tpwrs.contracts import StageCertificate

    profile = SolverProfile(backend="scipy", feasibility_tol=1e-8, output_flag=False)
    numeric = generic_face_numeric_tolerance(profile)
    metadata = {
        "face_numeric_contract": GENERIC_EXPLICIT_FACE_CONTRACT,
        "generic_face_certification_tolerance": numeric,
    }
    stage1_value = 283968.55470827285
    scientific_tolerance = 1e-6 + 1e-9 * abs(stage1_value)
    solution = SimpleNamespace(
        stage1=StageCertificate("I", "OPTIMAL", stage1_value, 0.0, metadata=metadata),
        stage2=StageCertificate("II", "OPTIMAL", 38504.21686452, 0.0, metadata=metadata),
        stage3=None,
        replay_summary={"passed": True},
    )
    experiment = SimpleNamespace(
        solver_profile=profile,
        stage1_tolerance=lambda _value: scientific_tolerance,
        stage2_tolerance=lambda _value: 1.0,
    )
    report = SimpleNamespace(
        s_value_mw_h_row=283968.55499339232,
        e_value_mwh=38504.21686452,
        max_budget_violation_h=0.0,
        max_balance_residual_mw=0.0,
        max_x_bound_violation_mw=0.0,
    )
    monkeypatch.setattr("tpwrs.execution.replay_solution", lambda *_a, **_k: report)
    from tpwrs.storage import ReplayFailure
    with pytest.raises(ReplayFailure, match="Stage-I face"):
        validate_policy_solution(object(), experiment, solution, require_stage3=False)


# ============================================================================
# tests.test_engineering_audits_final (consolidated; see SOURCE_MAP.json)
# ============================================================================

import pandas as pd

from tpwrs.engineering import mapped_dc_audit


def test_engineering_audits_final__false_safe_and_false_overload_have_physical_direction() -> None:
    frozen = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "state_id": ["INTACT", "INTACT"],
            "row_id": ["line::0", "line::0"],
            "frozen_flow_mw": [90.0, 110.0],
            "rating_mw": [100.0, 100.0],
        }
    )
    independent = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "state_id": ["INTACT", "INTACT"],
            "row_id": ["line::0", "line::0"],
            "independent_flow_mw": [110.0, 90.0],
            "rating_mw": [100.0, 100.0],
        }
    )
    rows, summary = mapped_dc_audit(frozen, independent)
    assert bool(rows.loc[0, "false_safe"])
    assert not bool(rows.loc[0, "false_overload"])
    assert not bool(rows.loc[1, "false_safe"])
    assert bool(rows.loc[1, "false_overload"])
    assert summary["false_safe_count"] == 1.0
    assert summary["false_overload_count"] == 1.0


@pytest.mark.parametrize("field,value",[("feasibility_tol",float("nan")),("mip_gap",float("nan")),("time_limit_s",float("inf"))])
def test_invalid_nonfinite_solver_settings_rejected(field,value):
    from tpwrs.contracts import SolverProfile
    from tpwrs.storage import ContractError
    with pytest.raises(ContractError):SolverProfile(**{field:value})


def test_frozen_gamma_grid_and_controller_stage_are_aligned():
    import json
    from jobs import gamma
    from tpwrs import orchestrator
    from tpwrs.config import ROOT
    frozen=json.loads((ROOT/"specification/METHODOLOGY_FREEZE_V1.json").read_text(encoding="utf-8"))
    assert tuple(map(float,frozen["m3"]["gamma_grid"])) == gamma.DEFAULT_GAMMA_GRID
    assert gamma.SCREEN_GAMMAS == tuple(map(float,frozen["m3"]["gamma_screen"]))
    assert orchestrator.STAGES["05C"][1] == [("gamma", []), ("bounds", [])]
    assert "GAMMA" not in orchestrator.STAGES and "BOUNDS" not in orchestrator.STAGES


def test_final_evidence_bundle_includes_required_sensitivity_roots():
    from pathlib import Path
    text=(Path(__file__).resolve().parents[1]/"jobs/report.py").read_text(encoding="utf-8")
    assert 'results_m3_sensitivity_r10/gamma' in text
    assert 'results_m3_sensitivity_r10/rate_bounds' in text
    assert 'Required post-freeze sensitivity evidence is incomplete.' in text


def test_frozen_methodology_constants_match_implementation_defaults():
    import json
    from tpwrs.config import ROOT
    from tpwrs.contracts import SolverProfile, ExperimentCase
    from jobs import common, cards, controls, rate_bounds
    frozen=json.loads((ROOT/"specification/METHODOLOGY_FREEZE_V1.json").read_text(encoding="utf-8"))
    m3=frozen["m3"]
    assert cards.FIT_YEARS == tuple(m3["training_years"])
    assert cards.VALIDATION_YEAR == int(m3["validation_year"])
    assert controls.SHUFFLE_SEEDS == tuple(m3["shuffle_seeds"])
    assert common.EVALUATION_YEARS == tuple(m3["held_out_years"]+[m3["confirmation_year"]])
    assert rate_bounds.BOUNDS["primary"] == tuple(m3["primary_rate_bounds"])
    assert rate_bounds.BOUNDS["tight"] == tuple(m3["sensitivity_rate_bounds"][0])
    assert rate_bounds.BOUNDS["wide"] == tuple(m3["sensitivity_rate_bounds"][1])
    sp=SolverProfile()
    tol=frozen["numerical_tolerances"]
    assert sp.feasibility_tol == tol["primal_feasibility"]
    # Constructing a full ExperimentCase is unnecessary here; assert its declared defaults directly.
    assert ExperimentCase.__dataclass_fields__["epsilon_s_abs"].default == tol["lexicographic_absolute"]
    assert ExperimentCase.__dataclass_fields__["epsilon_s_rel"].default == tol["lexicographic_relative"]
    assert frozen["rolling"] == {"implementation_hours":24,"lookahead_hours":48,"terminal_reserve_primary":False}
    assert frozen["m1"]["materiality_absolute_mw"] == 1.0
    assert frozen["m1"]["materiality_capacity_share"] == 0.01


def test_card_selection_lexicographic_tolerance_scales_with_best_eligible_energy():
    from types import SimpleNamespace as NS
    from tpwrs.cards import select_validation_card
    def candidate(name,s,e,rates):
        return NS(year=2028,s_value=s,e_value=e,tie_break_key=name,
            card=NS(rates={'import':rates,'export':rates},model_family='ridge',card_id=name))
    rows=[candidate('large',0.,1e9,(1.,1.,1.)),candidate('best',1e-7,1.,(1.2,1.,.8)),
          candidate('flat',2e-7,1.0001,(1.,1.,1.))]
    assert select_validation_card(rows).card.card_id=='best'
