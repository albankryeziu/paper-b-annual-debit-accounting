from __future__ import annotations


# ============================================================================
# tests.test_annual_lp_gurobi_equivalence_final (consolidated; see SOURCE_MAP.json)
# ============================================================================

import importlib.util
import numpy as np
import pytest

from tpwrs.contracts import nominal_experiment
from tpwrs.contracts import SolverProfile
from tpwrs.model import solve_policy
from tpwrs.contracts import NetworkState, PhysicalCase
from tpwrs.contracts import TimeGrid

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gurobipy") is None,
    reason="Gurobi not installed",
)


@pytest.mark.parametrize("rule", ["M1", "M1_LP", "M2", "M3"])
def test_native_microcases_ignore_production_ram_reserve(simple_case, monkeypatch, rule):
    from types import SimpleNamespace
    from dataclasses import replace
    from tpwrs.execution import validate_policy_solution
    from tpwrs.storage import sha256_array
    import psutil
    memory=psutil.virtual_memory()._replace(available=64*1024**2)
    monkeypatch.setattr("psutil.virtual_memory", lambda: memory)
    monkeypatch.setenv("TPWRS_GUROBI_SOFTMEM_GB", "0.000001")
    exp=nominal_experiment(simple_case,case_id="LOW_RAM",rule=rule,rho=1.,
        solver_profile=SolverProfile(backend="gurobi"),
        **({"card_id":"q", "card_hash":"q"} if rule=="M3" else {}))
    q=np.full(simple_case.requests_mw.shape,0.7) if rule=="M3" else None
    if q is not None: exp=replace(exp,rate_matrix_hash=sha256_array(q))
    solution=solve_policy(simple_case,exp,q=q)
    assert validate_policy_solution(simple_case,exp,solution,require_stage3=False)["passed"]


def _annual_lp_gurobi_equivalence_final__case() -> PhysicalCase:
    grid = TimeGrid.from_index(
        ["2028-01-01T00:00:00Z", "2028-01-01T01:00:00Z"],
        1.0,
    )
    # r1 is initially overloaded and is relieved by p1. r2 is initially secure
    # but can become overloaded after that action, so exact closure is required.
    state = NetworkState(
        "INTACT",
        ("initial", "hidden"),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        np.zeros(2),
        np.array([3.0, 2.0]),
    )
    # Restriction p1 moves 1 MW from bus 1 to bus 2: it relieves r1 while
    # worsening r2. p2 is a second independent action to avoid a degenerate
    # one-column geometry.
    A = np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]])
    return PhysicalCase(
        time_grid=grid,
        participant_ids=("p1", "p2"),
        bus_ids=("b1", "b2", "b3"),
        capacities_mw=np.array([10.0, 10.0]),
        requests_mw=np.array([[6.0, 0.0], [0.0, 0.0]]),
        contract_mask=np.ones((2, 2), dtype=bool),
        eligibility_mask=np.ones((2, 2), dtype=bool),
        # requested_state = pbg - A*request gives [6,0,-6] at t=0.
        p_background_mw=np.array([[0.0, 6.0, -6.0], [0.0, 0.0, 0.0]]),
        A=A,
        network_states=(state,),
        participant_directions=("import", "export"),
    )


def _annual_lp_gurobi_equivalence_final__exp(case: PhysicalCase, rule: str):
    return nominal_experiment(
        case,
        case_id=f"GUR_EQ_{rule}",
        rule=rule,
        rho=1.0,
        solver_profile=SolverProfile(
            name="gurobi_equivalence_micro",
            backend="gurobi",
            presolve=False,
            output_flag=False,
            threads=1,
            lp_method="dual_simplex",
        ),
    )


@pytest.mark.parametrize("rule", ["M1_LP", "M2"])
def test_annual_lp_gurobi_equivalence_final__support_oa_engine_matches_generic_same_model(rule: str):
    case = _annual_lp_gurobi_equivalence_final__case()
    exp = _annual_lp_gurobi_equivalence_final__exp(case, rule)
    optimized = solve_policy(case, exp, use_stage3=False)
    reference = solve_policy(case, exp, use_stage3=False, force_generic=True)

    assert optimized.stage1.status == "OPTIMAL"
    assert optimized.stage2.status == "OPTIMAL"
    assert optimized.replay_summary.get("passed") is True
    assert reference.replay_summary.get("passed") is True
    assert abs(optimized.s_value_mw_h_row - reference.s_value_mw_h_row) <= 2e-6
    assert abs(optimized.e_value_mwh - reference.e_value_mwh) <= 2e-6
    assert optimized.stage1.metadata.get("explicit_security_slack_variables") == 0
    assert optimized.stage1.metadata.get("support_epigraph_variables", 0) <= case.n_time
    assert "SUPPORT_OA" in str(optimized.stage1.metadata.get("engine"))


def test_annual_lp_gurobi_equivalence_final__support_oa_engine_closes_hidden_overload():
    case = _annual_lp_gurobi_equivalence_final__case()
    exp = _annual_lp_gurobi_equivalence_final__exp(case, "M2")
    optimized = solve_policy(case, exp, use_stage3=False)
    assert optimized.replay_summary.get("passed") is True
    assert optimized.closure_rounds >= 1
    assert optimized.replay_summary.get("final_closed_directed_rows", 0) >= 2


def test_annual_lp_gurobi_equivalence_final__M3_native_dual_bound_and_checkpoint(tmp_path):
    from dataclasses import replace
    from tpwrs.annual import load_closed_row_seed, _screen_geometry
    from tpwrs.storage import sha256_array
    case=_annual_lp_gurobi_equivalence_final__case();q=np.full(case.requests_mw.shape,0.7)
    exp=replace(_annual_lp_gurobi_equivalence_final__exp(case,'M2'),rule='M3',card_id='MICRO',card_hash='MICRO',rate_matrix_hash=sha256_array(q))
    optimized=solve_policy(case,exp,q=q,checkpoint_dir=tmp_path)
    reference=solve_policy(case,exp,q=q,force_generic=True)
    assert abs(optimized.stage1.objective_value-reference.stage1.objective_value)<2e-6
    assert abs(optimized.stage2.objective_value-reference.stage2.objective_value)<2e-6
    assert optimized.stage1.best_bound<=optimized.stage1.objective_value+1e-7
    assert optimized.stage2.best_bound<=optimized.stage2.objective_value+1e-7
    geo=_screen_geometry(case,exp)
    hint=load_closed_row_seed(tmp_path,geometry_hash=geo.geometry_hash,n_rows=case.n_flat_rows,n_hours=len(geo.active_hours))
    assert hint is not None
    again=solve_policy(case,exp,q=q,checkpoint_dir=tmp_path)
    assert again.replay_summary['passed']
    assert abs(again.stage1.objective_value-optimized.stage1.objective_value)<2e-6
    (tmp_path/'CLOSED_ROWS.npz').write_bytes(b'corrupt hint')
    assert load_closed_row_seed(tmp_path,geometry_hash=geo.geometry_hash,n_rows=case.n_flat_rows,n_hours=len(geo.active_hours)) is None


def test_annual_lp_gurobi_equivalence_final__native_M1_warm_start_preserves_integer_model():
    from dataclasses import replace
    case=_annual_lp_gurobi_equivalence_final__case();exp=replace(_annual_lp_gurobi_equivalence_final__exp(case,'M2'),rule='M1')
    warm=np.full(case.requests_mw.shape,np.nan);warm[0,0]=3.
    touch=np.full_like(warm,np.nan);touch[0,0]=1.
    sol=solve_policy(case,exp,initial_x=warm,initial_u=touch)
    assert sol.stage1.status=='OPTIMAL' and sol.stage2.status=='OPTIMAL'
    assert sol.replay_summary['passed']
    assert np.max(np.abs(sol.u-np.round(sol.u)))<1e-8


@pytest.mark.parametrize("rule",["M1","M2"])
def test_native_generic_guard_executes_quad_recovery_and_support_cut(simple_case,monkeypatch,rule):
    from dataclasses import replace
    import tpwrs.model as model
    from tpwrs.execution import validate_policy_solution
    exp=nominal_experiment(simple_case,case_id="NATIVE_FACE_RECOVERY",rule=rule,rho=1.,
        solver_profile=SolverProfile(backend="gurobi",presolve=False,output_flag=False))
    original=model.solve_linear_model
    corrupt=2
    returned=[]
    def perturb(matrix,profile,**kwargs):
        nonlocal corrupt
        raw=original(matrix,profile,**kwargs)
        if kwargs["stage"].startswith("STAGE_II") and corrupt:
            corrupt-=1;values=raw.primal.copy();values[0]-=2e-6
            raw=replace(raw,primal=values,objective_value=float(matrix.c@values))
        returned.append(raw)
        return raw
    monkeypatch.setattr(model,"solve_linear_model",perturb)
    sol=model.solve_policy(simple_case,exp,force_generic=True)
    assert corrupt==0
    assert sol.stage2.metadata["physical_support_count"]>=1
    assert sol.stage2.metadata["numeric_focus"]==3
    assert sol.stage2.metadata["quad"]==1
    assert sol.stage2.metadata["effective_feasibility_tol"]==1e-9
    assert validate_policy_solution(simple_case,exp,sol,require_stage3=False)["passed"]
    assert np.array_equal(sol.x_mw,returned[-1].primal[:2].reshape(2,1))


@pytest.mark.parametrize('rule',['M1_LP','M2','M3'])
@pytest.mark.parametrize('seed',[4,17,29])
def test_native_random_support_matches_independent_scipy_formulation(seed,rule):
    from dataclasses import replace
    from test_solver import _release_mathematics__random_case,_release_mathematics__compact_scipy
    from tpwrs.execution import validate_policy_solution
    from tpwrs.storage import sha256_array
    case=_release_mathematics__random_case(seed)
    q=np.tile([.7,1.,1.3],(case.n_time,1)) if rule=='M3' else None
    exp=nominal_experiment(case,case_id=f'NATIVE_RANDOM_{seed}_{rule}',rule=rule,rho=.5,
        card_id='Q' if q is not None else None,card_hash='Q' if q is not None else None,
        solver_profile=SolverProfile(backend='gurobi',feasibility_tol=1e-8))
    if q is not None:exp=replace(exp,rate_matrix_hash=sha256_array(q))
    expected_s,expected_e=_release_mathematics__compact_scipy(case,exp,q)
    actual=solve_policy(case,exp,q=q)
    assert abs(expected_s-actual.stage1.objective_value)<=exp.stage1_tolerance(expected_s)
    assert abs(expected_e-actual.stage2.objective_value)<=exp.stage2_tolerance(expected_e)
    assert validate_policy_solution(case,exp,actual,require_stage3=False)['passed']
