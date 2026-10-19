from __future__ import annotations


# ============================================================================
# tests.test_solver_geometry (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.replay import verify_debit_gap_identity, strict_m1_certificate
from tpwrs.contracts import AllowanceSpec, ExperimentCase, SolverProfile
from tpwrs.cards import matched_allowance, verify_m2_matched_contains_m3, matched_decomposition
from tpwrs.model import solve_policy
from tpwrs.replay import replay_solution


def _solver_geometry__exp(case,rule,H,*,card=False):
    allow=AllowanceSpec.explicit(case.participant_ids,[H])
    return ExperimentCase(f"{rule}_{H}",case.physical_case_id,case.physical_case_hash,case.time_grid.grid_id,rule,allow,card_id="c" if card else None,card_hash="h" if card else None,solver_profile=SolverProfile(output_flag=False))


def test_solver_geometry__stage_i_ii_m2(simple_case):
    sol=solve_policy(simple_case,_solver_geometry__exp(simple_case,"M2",1.0),use_stage3=True)
    assert sol.stage1.objective_value < 1e-6
    assert abs(sol.s_value_mw_h_row) < 1e-5
    assert abs(sol.e_value_mwh-100.0) < 1e-4
    assert sol.stage3 is not None
    assert sol.m2_budget_duals is not None


def test_solver_geometry__m1_is_more_restrictive_on_simple_case(simple_case):
    m1=solve_policy(simple_case,_solver_geometry__exp(simple_case,"M1",1.0),use_stage3=False)
    m2=solve_policy(simple_case,_solver_geometry__exp(simple_case,"M2",1.0),use_stage3=False)
    assert m2.s_value_mw_h_row <= m1.s_value_mw_h_row + 1e-6
    assert m1.s_value_mw_h_row > 49.9


def test_solver_geometry__m1lp_matches_m2_when_xbar_equals_capacity(simple_case):
    lp=solve_policy(simple_case,_solver_geometry__exp(simple_case,"M1_LP",1.0),use_stage3=False)
    m2=solve_policy(simple_case,_solver_geometry__exp(simple_case,"M2",1.0),use_stage3=False)
    assert abs(lp.s_value_mw_h_row-m2.s_value_mw_h_row)<1e-6


def test_solver_geometry__gap_identity(simple_case):
    x=np.array([[20.0],[40.0]])
    assert verify_debit_gap_identity(simple_case,x)


def test_solver_geometry__closure_catches_hidden_rows(simple_case):
    initial=np.zeros((simple_case.n_time,simple_case.n_flat_rows),bool)
    sol=solve_policy(simple_case,_solver_geometry__exp(simple_case,"M2",1.0),initial_active_mask=initial,use_stage3=False)
    assert sol.closure_rounds >= 1
    assert sol.s_value_mw_h_row < 1e-5


def test_solver_geometry__independent_replay(simple_case):
    e=_solver_geometry__exp(simple_case,"M2",1.0); sol=solve_policy(simple_case,e,use_stage3=False); rep=replay_solution(simple_case,e,sol)
    assert rep.passed and rep.max_budget_violation_h < 1e-7


def test_solver_geometry__m2_matched_contains_completed_m3_trajectory(simple_case):
    m3e=_solver_geometry__exp(simple_case,"M3",0.5,card=True)
    q=np.full((2,1),0.5)
    m3=solve_policy(simple_case,m3e,q=q,use_stage3=False)
    matched=matched_allowance(simple_case,m3)
    assert verify_m2_matched_contains_m3(simple_case,m3,matched)
    me=ExperimentCase("matched",simple_case.physical_case_id,simple_case.physical_case_hash,simple_case.time_grid.grid_id,"M2_MATCHED",matched,card_id="c",card_hash="h")
    ms=solve_policy(simple_case,me,use_stage3=False)
    assert ms.stage1.objective_value <= m3.s_value_mw_h_row + 1e-7
    d=matched_decomposition(solve_policy(simple_case,_solver_geometry__exp(simple_case,"M2",0.5),use_stage3=False).s_value_mw_h_row,m3.s_value_mw_h_row,ms.stage1.objective_value)
    assert d["shape_penalty"] >= -1e-8


def test_solver_geometry__strict_certificate_arithmetic():
    cert=strict_m1_certificate(4.0,5.0,0.1)
    assert cert.passed and cert.guaranteed_margin>0


# ============================================================================
# tests.test_annual_support_oa_math (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from tpwrs.annual import _build_budget_matrix, _evaluate_hour_support, _evaluate_support, _initial_zero_support_cuts, _reconstruct_active_x, _screen_geometry, _support_cut_coefficients, build_geometry_audit, load_support_cut_seed, _save_support_cut_pool
from tpwrs.contracts import nominal_experiment
from tpwrs.contracts import SolverProfile
from tpwrs.model import solve_policy
from tpwrs.contracts import NetworkState, PhysicalCase
from tpwrs.replay import physical_objectives
from tpwrs.contracts import TimeGrid


def _annual_support_oa_math__case() -> PhysicalCase:
    grid = TimeGrid.from_index(
        [
            "2028-01-01T00:00:00Z",
            "2028-01-01T01:00:00Z",
            "2028-01-01T02:00:00Z",
        ],
        1.0,
    )
    # r1 starts overloaded and p1 relieves it.  The same action can induce r2,
    # so the zero-action support is insufficient and a second support cut is
    # required for exact closure.  The third row supplies a negative-side case.
    state = NetworkState(
        "INTACT",
        ("initial", "hidden_pos", "hidden_neg"),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.zeros(3),
        np.array([3.0, 2.0, 2.0]),
    )
    A = np.array(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 1.0],
        ]
    )
    return PhysicalCase(
        time_grid=grid,
        participant_ids=("p1", "p2"),
        bus_ids=("b1", "b2", "b3"),
        capacities_mw=np.array([10.0, 10.0]),
        requests_mw=np.array([[6.0, 4.0], [0.0, 0.0], [0.0, 0.0]]),
        contract_mask=np.ones((3, 2), dtype=bool),
        eligibility_mask=np.ones((3, 2), dtype=bool),
        p_background_mw=np.array([[0.0, 6.0, -6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        A=A,
        network_states=(state,),
        participant_directions=("import", "export"),
    )


def _annual_support_oa_math__exp(case: PhysicalCase, rule: str):
    return nominal_experiment(
        case,
        case_id=f"SUPPORT_MATH_{rule}",
        rule=rule,
        rho=1.0,
        card_id="TEST_CARD" if rule == "M3" else None,
        card_hash="TEST_CARD_HASH" if rule == "M3" else None,
        solver_profile=SolverProfile(
            name="support_math_scipy",
            backend="scipy",
            presolve=True,
            output_flag=False,
            threads=1,
        ),
    )


def _annual_support_oa_math__add_cut_rows(case, geo, cuts, theta_index, R, K):
    rows = []
    rhs = []
    for cut in cuts:
        grad, intercept = _support_cut_coefficients(case, geo, cut)
        row = np.zeros(R + K, dtype=float)
        h = cut.local_hour
        start, stop = int(geo.offsets[h]), int(geo.offsets[h + 1])
        row[start:stop] = grad
        row[R + theta_index[h]] = -1.0
        rows.append(row)
        rhs.append(-intercept)
    return rows, rhs


def _annual_support_oa_math__solve_scipy_support_oa(case: PhysicalCase, exp, q=None):
    geo = _screen_geometry(case, exp)
    R = geo.retained_action_coordinates
    variable = geo.variable_local_hours
    K = len(variable)
    theta_index = np.full(len(geo.active_hours), -1, dtype=int)
    theta_index[variable] = np.arange(K)
    B = _build_budget_matrix(case, exp, geo, q)
    budget = np.hstack([B.toarray(), np.zeros((case.n_participants, K))])
    base_rows = [row for row in budget]
    base_rhs = list(exp.allowance.values)
    cuts = list(_initial_zero_support_cuts(case, geo))
    keys = set()

    def key(c):
        return (c.local_hour, tuple(map(int, c.pos_rows)), tuple(map(int, c.neg_rows)))

    for c in cuts:
        keys.add(key(c))

    bounds = [(0.0, float(v)) for v in geo.xbar_coordinate] + [(0.0, None)] * K

    for _ in range(30):
        cut_rows, cut_rhs = _annual_support_oa_math__add_cut_rows(case, geo, cuts, theta_index, R, K)
        A = np.vstack(base_rows + cut_rows)
        b = np.asarray(base_rhs + cut_rhs)
        c = np.concatenate([np.zeros(R), np.ones(K)])
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        assert res.success
        phi, oracle = _evaluate_support(case, geo, res.x[:R])
        gap = phi - res.x[R:]
        if np.max(np.maximum(gap, 0.0), initial=0.0) <= 1e-9:
            s_star = float(geo.constant_s_mw_h_row + phi.sum())
            break
        added = False
        for j, candidate in enumerate(oracle):
            if gap[j] > 1e-9 and key(candidate) not in keys:
                keys.add(key(candidate)); cuts.append(candidate); added = True
        assert added
    else:
        raise AssertionError("Stage-I support OA did not close")

    cap = s_star + exp.stage1_tolerance(s_star)
    for _ in range(30):
        cut_rows, cut_rhs = _annual_support_oa_math__add_cut_rows(case, geo, cuts, theta_index, R, K)
        cap_row = np.concatenate([np.zeros(R), np.ones(K)])
        A = np.vstack(base_rows + cut_rows + [cap_row])
        b = np.asarray(base_rhs + cut_rhs + [cap - geo.constant_s_mw_h_row])
        c = np.concatenate([geo.dt_coordinate, np.zeros(K)])
        res2 = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        assert res2.success
        phi, oracle = _evaluate_support(case, geo, res2.x[:R])
        true_s = float(geo.constant_s_mw_h_row + phi.sum())
        if true_s <= cap + 1e-9:
            return s_star, float(np.dot(geo.dt_coordinate, res2.x[:R])), res2.x[:R], len(cuts)
        gap = phi - res2.x[R:]
        added = False
        for j, candidate in enumerate(oracle):
            if gap[j] > 1e-9 and key(candidate) not in keys:
                keys.add(key(candidate)); cuts.append(candidate); added = True
        assert added
    raise AssertionError("Stage-II support OA did not close")


def test_annual_support_oa_math__support_oracle_equals_complete_physical_objective_on_reduced_box():
    case = _annual_support_oa_math__case(); exp = _annual_support_oa_math__exp(case, "M2"); geo = _screen_geometry(case, exp)
    rng = np.random.default_rng(118)
    for _ in range(40):
        xcoord = rng.random(geo.retained_action_coordinates) * geo.xbar_coordinate
        phi, _ = _evaluate_support(case, geo, xcoord)
        x = _reconstruct_active_x(case, geo, xcoord)
        S, _ = physical_objectives(case, x)
        assert abs(S - (geo.constant_s_mw_h_row + float(phi.sum()))) <= 1e-10


def test_annual_support_oa_math__generated_hourly_support_is_global_underestimator_and_tight_at_oracle_point():
    case = _annual_support_oa_math__case(); exp = _annual_support_oa_math__exp(case, "M2"); geo = _screen_geometry(case, exp)
    h = int(geo.variable_local_hours[0])
    rng = np.random.default_rng(219)
    ub = geo.xbar_coordinate[geo.offsets[h]:geo.offsets[h + 1]]
    x0 = 0.37 * ub
    phi0, cut = _evaluate_hour_support(case, geo, h, x0)
    grad, intercept = _support_cut_coefficients(case, geo, cut)
    assert abs(phi0 - (intercept + float(np.dot(grad, x0)))) <= 1e-10
    for _ in range(100):
        y = rng.random(len(ub)) * ub
        phiy, _ = _evaluate_hour_support(case, geo, h, y)
        assert intercept + float(np.dot(grad, y)) <= phiy + 1e-10


def test_annual_support_oa_math__scipy_support_oa_matches_explicit_generic_lp_stage1_and_stage2():
    case = _annual_support_oa_math__case()
    for rule in ("M1_LP", "M2", "M3", "M2_MATCHED"):
        exp = _annual_support_oa_math__exp(case, rule)
        q = np.array([[0.7, 1.3], [1.0, 1.0], [1.3, 0.7]], dtype=float) if rule == "M3" else None
        s_oa, e_oa, _xcoord, cut_count = _annual_support_oa_math__solve_scipy_support_oa(case, exp, q=q)
        reference = solve_policy(case, exp, q=q, use_stage3=False)
        assert reference.replay_summary["passed"]
        assert abs(s_oa - reference.stage1.objective_value) <= 2e-7
        assert abs(e_oa - reference.stage2.objective_value) <= 2e-7
        # Hidden induced overload requires more than the zero-action support set.
        assert cut_count > len(_screen_geometry(case, exp).variable_local_hours)


def test_annual_support_oa_math__support_model_size_no_longer_scales_with_seed_directed_rows():
    case = _annual_support_oa_math__case(); exp = _annual_support_oa_math__exp(case, "M2"); audit = build_geometry_audit(case, exp)
    assert audit["explicit_security_slack_variables"] == 0
    assert audit["support_epigraph_variables"] == audit["variable_hours"]
    assert audit["support_initial_columns"] == audit["retained_action_coordinates"] + audit["variable_hours"]
    assert audit["legacy_explicit_seed_columns"] == audit["retained_action_coordinates"] + audit["legacy_initial_seed_directed_rows"]


def test_annual_support_oa_math__persisted_support_pool_is_selector_based_and_corruption_is_discarded(tmp_path):
    case = _annual_support_oa_math__case(); exp = _annual_support_oa_math__exp(case, "M2"); geo = _screen_geometry(case, exp)
    cuts = _initial_zero_support_cuts(case, geo)
    _save_support_cut_pool(tmp_path, geo, cuts, stage="TEST", closure_round=0)
    loaded = load_support_cut_seed(tmp_path, geometry=geo)
    assert loaded is not None and len(loaded) == len(cuts)
    (tmp_path / "SUPPORT_CUTS.npz").write_bytes(b"corrupt")
    assert load_support_cut_seed(tmp_path, geometry=geo) is None


def test_annual_support_oa_math__projected_support_dual_reconstruction_is_valid_on_known_one_dimensional_lp():
    from tpwrs.annual import _support_master_dual_bound

    class Con:
        def __init__(self, pi):
            self.Pi = np.asarray(pi, dtype=float)

    # min theta s.t. x<=1, -x-theta<=-2, 0<=x<=3, theta>=0.
    # Exact optimum is x=1, theta=1.  Gurobi's minimization <= convention has
    # Pi=-1 for both constraints, so lambda=zeta=1 is dual optimal.
    B = sp.csr_matrix([[1.0]])
    support_records = [
        (
            np.array([0], dtype=np.int32),
            np.array([2.0]),
            sp.csr_matrix([[-1.0]]),
            Con([-1.0]),
        )
    ]
    cert = _support_master_dual_bound(
        c_x=np.array([0.0]),
        upper_x=np.array([3.0]),
        budget_matrix=B,
        allowance=np.array([1.0]),
        budget_pi=np.array([-1.0]),
        support_records=support_records,
        constant_s=0.0,
        stage1_cap_rhs=None,
        stage1_cap_pi=None,
    )
    assert cert["lower_bound"] <= 1.0 + 1e-12
    assert cert["lower_bound"] >= 1.0 - 1e-10


# ============================================================================
# tests.test_annual_lp_compression_final (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.annual import _screen_geometry, build_geometry_audit
from tpwrs.contracts import nominal_experiment
from tpwrs.contracts import SolverProfile
from tpwrs.contracts import NetworkState, PhysicalCase
from tpwrs.contracts import TimeGrid


def _annual_lp_compression_final__case() -> PhysicalCase:
    grid = TimeGrid.from_index(
        [
            "2028-01-01T00:00:00Z",
            "2028-01-01T01:00:00Z",
            "2028-01-01T02:00:00Z",
        ],
        1.0,
    )
    state = NetworkState(
        "INTACT",
        ("r1", "r2"),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, -1.0]]),
        np.zeros(2),
        np.array([3.0, 4.0]),
    )
    # p1 reduces the positive r1/r2 flow. p2 moves the opposite way and is
    # exactly dominated at the stressed hour when no negative row can bind.
    A = np.array([[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])
    return PhysicalCase(
        time_grid=grid,
        participant_ids=("p1", "p2"),
        bus_ids=("b1", "b2", "b3"),
        capacities_mw=np.array([10.0, 10.0]),
        requests_mw=np.array([[5.0, 5.0], [0.0, 0.0], [0.0, 0.0]]),
        contract_mask=np.ones((3, 2), dtype=bool),
        eligibility_mask=np.ones((3, 2), dtype=bool),
        p_background_mw=np.array([[6.0, 0.0, -6.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        A=A,
        network_states=(state,),
        participant_directions=("import", "export"),
    )


def _annual_lp_compression_final__exp(case: PhysicalCase):
    return nominal_experiment(
        case,
        case_id="GEOMETRY_TEST",
        rule="M2",
        rho=1.0,
        solver_profile=SolverProfile(backend="scipy"),
    )


def test_annual_lp_compression_final__cached_physical_derived_arrays_are_stable_objects():
    case = _annual_lp_compression_final__case()
    assert case.flat_H is case.flat_H
    assert case.flat_G is case.flat_G
    assert case.requested_state_mw is case.requested_state_mw
    assert case.xbar_mw is case.xbar_mw


def test_annual_lp_compression_final__secure_hours_are_removed_exactly_from_annual_geometry():
    case = _annual_lp_compression_final__case()
    geo = _screen_geometry(case, _annual_lp_compression_final__exp(case))
    # Only the first hour has baseline overload.  Secure hours have exact x=0
    # lexicographic optimum and therefore need no annual variables.
    assert geo.active_hours.tolist() == [0]


def test_annual_lp_compression_final__useful_action_dominance_removes_worsening_only_action():
    case = _annual_lp_compression_final__case()
    geo = _screen_geometry(case, _annual_lp_compression_final__exp(case))
    assert geo.candidate_action_coordinates == 2
    assert geo.retained_action_coordinates == 1
    assert geo.participants_by_hour[0].tolist() == [0]


def test_annual_lp_compression_final__directed_outer_box_screen_contains_exact_endpoint_violation():
    case = _annual_lp_compression_final__case()
    geo = _screen_geometry(case, _annual_lp_compression_final__exp(case))
    G = case.flat_G
    t = int(geo.active_hours[0])
    xb = case.xbar_mw[t]
    base = case.requested_state_mw[t] @ case.flat_H.T + case.flat_h
    high = base + np.maximum(G, 0.0) @ xb
    low = base + np.minimum(G, 0.0) @ xb
    exact_pos = set(np.flatnonzero(high >= case.flat_ratings - 1e-9))
    exact_neg = set(np.flatnonzero(-low >= case.flat_ratings - 1e-9))
    assert set(map(int, geo.potential_pos[0])) == exact_pos
    assert set(map(int, geo.potential_neg[0])) == exact_neg


def test_annual_lp_compression_final__geometry_audit_is_deterministic():
    case = _annual_lp_compression_final__case()
    a = build_geometry_audit(case, _annual_lp_compression_final__exp(case))
    b = build_geometry_audit(case, _annual_lp_compression_final__exp(case))
    assert a == b


# ============================================================================
# tests.test_active_coordinate_compression_final (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np
import pytest

from tpwrs.compression import ActiveCoordinateMap, ActiveMVar, add_active_mvar, add_mconstr_active, compression_stats


def test_active_coordinate_compression_final__exact_active_coordinate_map_and_full_reconstruction():
    mapping = ActiveCoordinateMap.from_bounds(
        (2, 4),
        lb=np.zeros((2, 4)),
        ub=np.array([[3.0, 0.0, 2.0, 0.0], [0.0, 7.0, 0.0, 1.0]]),
    )
    assert mapping.full_size == 8
    assert mapping.active_size == 4
    assert mapping.dropped_size == 4
    reduced = mapping.compress_vector(np.arange(8.0))
    assert reduced.tolist() == [0.0, 2.0, 5.0, 7.0]
    expanded = mapping.expand(np.array([11.0, 12.0, 13.0, 14.0]))
    np.testing.assert_array_equal(
        expanded,
        np.array([[11.0, 0.0, 12.0, 0.0], [0.0, 13.0, 0.0, 14.0]]),
    )


def test_active_coordinate_compression_final__dense_and_sparse_column_projection_are_identical():
    scipy_sparse = pytest.importorskip('scipy.sparse')
    mapping = ActiveCoordinateMap.from_bounds(6, np.zeros(6), np.array([1, 0, 2, 0, 3, 0]))
    A = np.arange(18.0).reshape(3, 6)
    expected = A[:, [0, 2, 4]]
    np.testing.assert_array_equal(mapping.compress_columns(A), expected)
    np.testing.assert_array_equal(mapping.compress_columns(scipy_sparse.csr_matrix(A)).toarray(), expected)


def test_active_coordinate_compression_final__no_tolerance_based_scientific_change():
    mapping = ActiveCoordinateMap.from_bounds(3, [0, 0, 0], [0, 1e-15, -0.0])
    # Exactly zero is removed; a tiny nonzero bound remains in the model.
    assert mapping.active_flat.tolist() == [1]


def test_active_coordinate_compression_final__tiny_gurobi_compressed_model_matches_full_model():
    gp = pytest.importorskip('gurobipy')
    try:
        compressed = gp.Model('compressed_smoke')
        compressed.Params.OutputFlag = 0
        ub = np.array([4.0, 0.0, 3.0, 0.0, 2.0])
        x = add_active_mvar(compressed, 5, lb=np.zeros(5), ub=ub, name='x')
        if not isinstance(x, ActiveMVar):
            pytest.fail('fixed-zero coordinates were not compressed')
        A = np.array([
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 1.0],
        ])
        add_mconstr_active(compressed, A, x, '<', np.array([5.0, 9.0, 5.0]), name='cap')
        compressed.setObjective(np.array([3.0, 0.0, 2.0, 0.0, 1.0]) @ x, gp.GRB.MAXIMIZE)
        compressed.optimize()

        full = gp.Model('full_smoke')
        full.Params.OutputFlag = 0
        y = full.addMVar(5, lb=np.zeros(5), ub=ub, name='y')
        full.addMConstr(A, y, '<', np.array([5.0, 9.0, 5.0]), name='cap')
        full.setObjective(np.array([3.0, 0.0, 2.0, 0.0, 1.0]) @ y, gp.GRB.MAXIMIZE)
        full.optimize()
    except gp.GurobiError as exc:
        pytest.skip(f'Gurobi runtime/license unavailable for tiny smoke test: {exc}')

    assert compressed.Status == gp.GRB.OPTIMAL
    assert full.Status == gp.GRB.OPTIMAL
    assert compressed.ObjVal == pytest.approx(full.ObjVal, abs=1e-9)
    np.testing.assert_allclose(x.X, y.X, atol=1e-9)
    stats = compression_stats(compressed)
    assert stats['coordinates_declared'] == 5
    assert stats['coordinates_active'] == 3
    assert stats['coordinates_dropped_fixed_zero'] == 2
    assert stats['matrix_rows_dropped_tautological'] == 1


def test_active_coordinate_compression_final__production_builders_route_through_exact_compression():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    routed=set()
    for path in (root/'tpwrs').glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call)):
            if isinstance(call.func,ast.Attribute) and call.func.attr=='addMVar':
                assert path.name=='compression.py', (path.name,call.lineno)
            if isinstance(call.func,ast.Name) and call.func.id=='add_active_mvar':
                routed.add(path.name)
    assert {'annual.py','linear.py','inputs.py'} <= routed


# ============================================================================
# tests.test_release_mathematics (consolidated; see SOURCE_MAP.json)
# ============================================================================
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye
from tpwrs.annual import _screen_geometry, _build_budget_matrix
from tpwrs.contracts import nominal_experiment
from tpwrs.contracts import SolverProfile
from tpwrs.model import solve_policy
from tpwrs.contracts import PhysicalCase, NetworkState
from tpwrs.contracts import TimeGrid
from tpwrs.replay import box_dual_bound


def _release_mathematics__random_case(seed):
    rng=np.random.default_rng(seed);T=3;N=3;B=4;K=4
    A=rng.normal(size=(B,N));A-=A.mean(axis=0)
    H=rng.normal(size=(K,B));h=rng.normal(size=K)
    background=rng.normal(size=(T,B))*2;background-=background.mean(axis=1)[:,None]
    cap=np.ones(N)*4;req=rng.uniform(0,4,size=(T,N));req[rng.random((T,N))<0.2]=0
    return PhysicalCase(time_grid=TimeGrid.from_index(pd.date_range('2028-01-01',periods=T,freq='h',tz='UTC'),1),
        participant_ids=tuple(f'p{i}' for i in range(N)),bus_ids=tuple(f'b{i}' for i in range(B)),
        capacities_mw=cap,requests_mw=req,contract_mask=np.ones((T,N),bool),eligibility_mask=np.ones((T,N),bool),
        p_background_mw=background,A=A,network_states=(NetworkState('INTACT',tuple(f'r{i}' for i in range(K)),H,h,rng.uniform(1,4,K)),),
        participant_directions=('import','export','import'))


def _release_mathematics__compact_scipy(case,exp,q):
    g=_screen_geometry(case,exp);R=g.retained_action_coordinates
    if not R:return g.constant_s_mw_h_row,0.0
    rows=[];rhs=[];weights=[];blocks=[]
    for hh,t in enumerate(g.active_hours):
        part=g.participants_by_hour[hh]
        if not len(part):continue
        ids=np.r_[g.potential_pos[hh],g.potential_neg[hh]];sign=np.r_[np.ones(len(g.potential_pos[hh])),-np.ones(len(g.potential_neg[hh]))]
        a=sign[:,None]*case.flat_G[np.ix_(ids,part)]
        b=case.flat_ratings[ids]-sign*(case.flat_H[ids]@case.requested_state_mw[t]+case.flat_h[ids])
        aa=np.zeros((len(ids),R));aa[:,g.offsets[hh]:g.offsets[hh+1]]=a
        rows.append(aa);rhs.extend(b);weights.extend(np.full(len(ids),case.time_grid.delta[t]));blocks.append((a,b,np.full(len(ids),case.time_grid.delta[t]),np.arange(g.offsets[hh],g.offsets[hh+1])))
    M=len(rhs);A=csr_matrix(np.vstack(rows));B=_build_budget_matrix(case,exp,g,q)
    U=vstack([hstack([B,csr_matrix((case.n_participants,M))]),hstack([A,-eye(M)])]).tocsr()
    b=np.r_[exp.allowance.values,rhs];c=np.r_[np.zeros(R),weights];bounds=[(0,x) for x in g.xbar_coordinate]+[(0,None)]*M
    opts={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9}
    one=linprog(c,A_ub=U,b_ub=b,bounds=bounds,method='highs',options=opts);assert one.success
    def cert(res,stage,limit=None):
        offset=case.n_participants;thermal=[]
        for a,bb,dt,cols in blocks:
            thermal.append((a,bb,dt,res.ineqlin.marginals[offset:offset+len(bb)],cols));offset+=len(bb)
        return box_dual_bound(np.zeros(R) if stage==1 else g.dt_coordinate,g.xbar_coordinate,B,exp.allowance.values,
            res.ineqlin.marginals[:case.n_participants],thermal,constant_s=g.constant_s_mw_h_row,
            stage1_cap=limit,cap_pi=None if stage==1 else res.ineqlin.marginals[-1])
    C1=cert(one,1);S=one.fun+g.constant_s_mw_h_row
    assert C1['lower_bound']<=S+1e-7
    assert S-C1['lower_bound']<=1e-5
    limit=S+exp.stage1_tolerance(S)
    U2=vstack([U,csr_matrix(c[None,:])]);b2=np.r_[b,limit-g.constant_s_mw_h_row]
    two=linprog(np.r_[g.dt_coordinate,np.zeros(M)],A_ub=U2,b_ub=b2,bounds=bounds,method='highs',options=opts);assert two.success
    C2=cert(two,2,limit)
    assert C2['lower_bound']<=two.fun+1e-7
    assert two.fun-C2['lower_bound']<=1e-5
    return S,two.fun


@pytest.mark.parametrize('rule',['M1_LP','M2','M3'])
@pytest.mark.parametrize('seed',range(20))
def test_release_mathematics__compact_geometry_and_dual_certificate_match_generic(seed,rule):
    case=_release_mathematics__random_case(seed);q=np.tile([0.7,1.0,1.3],(case.n_time,1)) if rule=='M3' else None
    exp=nominal_experiment(case,case_id=f'R{seed}_{rule}',rule=rule,rho=0.5,
        card_id='TEST' if rule=='M3' else None,card_hash='TEST' if rule=='M3' else None,
        solver_profile=SolverProfile(backend='scipy',feasibility_tol=1e-8,presolve=True))
    ref=solve_policy(case,exp,q=q,use_stage3=False,force_generic=True)
    S,E=_release_mathematics__compact_scipy(case,exp,q)
    assert abs(S-ref.stage1.objective_value)<=2e-6
    assert abs(E-ref.stage2.objective_value)<=2e-5


def test_release_mathematics__box_dual_sign_projection_cannot_overstate_optimum():
    # min max(2-x,0), 0<=x<=3, x<=1. Even grossly inaccurate
    # solver multipliers are projected into a mathematically feasible dual.
    for bpi in (-8,0,9):
        for tpi in (-9,-0.5,2):
            v=box_dual_bound([0],[3],csr_matrix([[1]]),[1],[bpi],
                [(np.array([[-1]]),np.array([-2]),np.array([1]),np.array([tpi]),np.array([0]))])
            assert v['lower_bound']<=1+1e-10


@pytest.mark.parametrize('seed', range(20))
def test_release_mathematics__support_projection_replays_complete_random_geometry(seed):
    from tpwrs.annual import _evaluate_support, _reconstruct_active_x
    from tpwrs.replay import physical_objectives
    case = _release_mathematics__random_case(seed)
    exp = nominal_experiment(
        case,
        case_id=f'SUPPORT_RANDOM_{seed}',
        rule='M2',
        rho=0.5,
        solver_profile=SolverProfile(backend='scipy', feasibility_tol=1e-8, presolve=True),
    )
    g = _screen_geometry(case, exp)
    if not g.retained_action_coordinates:
        return
    rng = np.random.default_rng(1000 + seed)
    for _ in range(8):
        xcoord = rng.random(g.retained_action_coordinates) * g.xbar_coordinate
        phi, _ = _evaluate_support(case, g, xcoord)
        full = _reconstruct_active_x(case, g, xcoord)
        S, _ = physical_objectives(case, full)
        assert abs(S - (g.constant_s_mw_h_row + float(phi.sum()))) <= 1e-9

# Physical-face regression tests execute the solver path, not just certificate comparisons.
@pytest.mark.parametrize("rule", ["M1","M1_LP","M2","M3","M2_MATCHED"])
def test_physical_support_is_global_minorant(simple_case,rule):
    from tpwrs.replay import physical_support_cut,physical_objectives
    rng=np.random.default_rng(223)
    for _ in range(12):
        anchor=rng.uniform(0,100,size=(2,1))
        cut=physical_support_cut(simple_case,anchor)
        S,_=physical_objectives(simple_case,anchor)
        assert cut.value(anchor)==pytest.approx(S,abs=1e-11)
        for _ in range(10):
            y=rng.uniform(0,100,size=(2,1))
            assert cut.value(y)<=physical_objectives(simple_case,y)[0]+1e-11


def test_stage1_closes_omitted_rows_before_stage2(simple_case):
    from tpwrs.contracts import nominal_experiment,SolverProfile
    from tpwrs.model import solve_policy
    exp=nominal_experiment(simple_case,case_id="HIDDEN_STAGE_I",rule="M2",rho=.1,
        solver_profile=SolverProfile(backend="scipy"))
    sol=solve_policy(simple_case,exp,force_generic=True,initial_active_mask=np.zeros((2,1),bool))
    assert sol.stage1.objective_value==pytest.approx(100.)
    attempts=sol.stage1.metadata["guard_attempts"]
    assert attempts[0]["action"]=="ADD_PHYSICAL_ROWS"
    assert attempts[0]["stage"]=="STAGE_I_SECURITY_RESTORATION"


@pytest.mark.parametrize("corrupt_count", [1,2])
@pytest.mark.parametrize("stage3", [False,True])
def test_real_recovery_reoptimizes_without_postsolve_projection(simple_case,monkeypatch,corrupt_count,stage3):
    from dataclasses import replace
    import tpwrs.model as model
    from tpwrs.contracts import nominal_experiment,SolverProfile
    from tpwrs.execution import validate_policy_solution
    exp=nominal_experiment(simple_case,case_id="RECOVERY",rule="M2",rho=1.,
        solver_profile=SolverProfile(backend="scipy"))
    reference=model.solve_policy(simple_case,exp,force_generic=True,use_stage3=stage3)
    original=model.solve_linear_model
    remaining=corrupt_count
    outputs=[]
    calls=[]
    def perturbed(matrix,profile,**kwargs):
        nonlocal remaining
        raw=original(matrix,profile,**kwargs)
        calls.append((kwargs["stage"],kwargs.get("numeric_recovery"),matrix.inequality_names))
        target="STAGE_III" if stage3 else "STAGE_II"
        if kwargs["stage"].startswith(target) and remaining:
            remaining-=1
            primal=raw.primal.copy()
            # x are first T*N columns. Change only the synthetic returned test
            # vector, mimicking an auxiliary-to-physical handoff discrepancy.
            primal[0]-=2e-6
            raw=replace(raw,primal=primal,objective_value=float(matrix.c@primal))
        outputs.append(raw.primal.copy())
        return raw
    monkeypatch.setattr(model,"solve_linear_model",perturbed)
    result=model.solve_policy(simple_case,exp,force_generic=True,use_stage3=stage3)
    assert remaining==0
    assert result.stage2.metadata["numerical_recovery_used"]
    assert validate_policy_solution(simple_case,exp,result,require_stage3=stage3)["passed"]
    assert np.array_equal(result.x_mw,outputs[-1][:2].reshape(2,1))
    assert result.e_value_mwh==pytest.approx(reference.e_value_mwh,abs=1e-7)
    assert result.stage2.metadata["physical_support_count"]==(1 if corrupt_count==2 else 0)
    if corrupt_count==2:
        assert any("physical_support::0" in names for _,_,names in calls)


def test_repeated_bad_solver_output_fails_and_saves_epoch(simple_case,monkeypatch,tmp_path):
    from dataclasses import replace
    import tpwrs.model as model
    from tpwrs.contracts import nominal_experiment,SolverProfile
    from tpwrs.storage import SolverFailure
    exp=nominal_experiment(simple_case,case_id="STALLED",rule="M2",rho=1.,solver_profile=SolverProfile(backend="scipy"))
    original=model.solve_linear_model
    def always_bad(matrix,profile,**kwargs):
        raw=original(matrix,profile,**kwargs)
        if kwargs["stage"].startswith("STAGE_II"):
            x=raw.primal.copy();x[0]-=2e-6
            raw=replace(raw,primal=x,objective_value=float(matrix.c@x))
        return raw
    monkeypatch.setattr(model,"solve_linear_model",always_bad)
    with pytest.raises(SolverFailure,match="stalled"):
        model.solve_policy(simple_case,exp,force_generic=True,checkpoint_dir=tmp_path)
    assert (tmp_path/"failure_reproducer/FAILURE.json").is_file()
    assert not (tmp_path/"solution.json").exists()


def test_reported_r9_numbers_still_fail_current_validator(monkeypatch):
    from types import SimpleNamespace
    from tpwrs.contracts import StageCertificate,SolverProfile
    from tpwrs.replay import GENERIC_EXPLICIT_FACE_CONTRACT
    from tpwrs.execution import validate_policy_solution
    from tpwrs.storage import ReplayFailure
    cap=291371.59318655194
    meta={"face_numeric_contract":GENERIC_EXPLICIT_FACE_CONTRACT,"generic_face_certification_tolerance":1e-7}
    sol=SimpleNamespace(stage1=StageCertificate("I","OPTIMAL",cap,0.,metadata=meta),
        stage2=StageCertificate("II","OPTIMAL",1.,0.,metadata=meta),stage3=None,replay_summary={"passed":True})
    exp=SimpleNamespace(solver_profile=SolverProfile(feasibility_tol=1e-8),stage1_tolerance=lambda _:0.,stage2_tolerance=lambda _:0.)
    report=SimpleNamespace(s_value_mw_h_row=291371.59318770148,e_value_mwh=1.)
    monkeypatch.setattr("tpwrs.execution.replay_solution",lambda *a,**k:report)
    with pytest.raises(ReplayFailure,match="Stage-I face"):
        validate_policy_solution(object(),exp,sol,require_stage3=False)


def _independent_dense_two_stage(case,experiment,q):
    """Separate x/slack/u reference: no factorization, screening or production builder."""
    from scipy.optimize import milp,Bounds,LinearConstraint
    from scipy.sparse import csr_matrix
    T,N,R=case.n_time,case.n_participants,case.n_flat_rows
    nx=T*N;ns=T*R;binary=experiment.rule=="M1";size=nx+ns+(nx if binary else 0)
    rows=[];rhs=[]
    G=case.flat_H@case.A
    base=case.requested_state_mw@case.flat_H.T+case.flat_h
    for t in range(T):
        for r in range(R):
            for sign in (-1.,1.):
                row=np.zeros(size);row[t*N:(t+1)*N]=sign*G[r]
                row[nx+t*R+r]=-1.
                rows.append(row);rhs.append(case.flat_ratings[r]-sign*base[t,r])
    for n in range(N):
        row=np.zeros(size)
        for t in range(T):
            j=t*N+n
            if binary:
                row[nx+ns+j]=case.time_grid.delta[t]
                upper=np.zeros(size);upper[j]=1;upper[nx+ns+j]=-case.xbar_mw[t,n]
                lower=np.zeros(size);lower[j]=-1;lower[nx+ns+j]=case.materiality_floor_mw[t,n]
                rows.extend([upper,lower]);rhs.extend([0.,0.])
            elif experiment.rule=="M1_LP":
                row[j]=0 if case.xbar_mw[t,n]==0 else case.time_grid.delta[t]/case.xbar_mw[t,n]
            else:
                row[j]=case.time_grid.delta[t]*(q[t,n] if experiment.rule=="M3" else 1.)/case.capacities_mw[n]
        rows.append(row);rhs.append(experiment.allowance.values[n])
    integrality=np.zeros(size)
    if binary:integrality[nx+ns:]=1
    ub=np.full(size,np.inf);ub[:nx]=case.xbar_mw.ravel()
    if binary:ub[nx+ns:]=1
    S=np.zeros(size);S[nx:nx+ns]=np.repeat(case.time_grid.delta,R)
    E=np.zeros(size);E[:nx]=np.repeat(case.time_grid.delta,N)
    options={"mip_rel_gap":1e-10,"presolve":True,"time_limit":10.}
    def solve(c):
        result=milp(c,integrality=integrality,bounds=Bounds(np.zeros(size),ub),
            constraints=LinearConstraint(csr_matrix(np.asarray(rows)),np.full(len(rhs),-np.inf),np.asarray(rhs)),options=options)
        assert result.status==0,result.message
        return result
    first=solve(S)
    rows.append(S);rhs.append(float(first.fun)+experiment.stage1_tolerance(first.fun))
    second=solve(E)
    return float(first.fun),float(second.fun)


@pytest.mark.parametrize("rule",["M1","M1_LP","M2","M3","M2_MATCHED"])
@pytest.mark.parametrize("seed",[103,271,991])
def test_coupled_multistate_against_independent_dense_formulation(rule,seed):
    from tpwrs.contracts import PhysicalCase,NetworkState,TimeGrid,ExperimentCase,AllowanceSpec,SolverProfile
    from tpwrs.model import solve_policy
    from tpwrs.storage import sha256_array
    rng=np.random.default_rng(seed);T=4;N=3
    grid=TimeGrid.from_index(pd.date_range("2028-01-01",periods=T,freq="h",tz="UTC"),1.)
    A=np.array([[1.,0.,.25],[-1.,1.,.75],[0.,-1.,-1.]])
    states=tuple(NetworkState(str(i),(f"r{i}_0",f"r{i}_1"),rng.normal(size=(2,3)),rng.normal(size=2),np.array([3.,4.])) for i in range(2))
    background=rng.normal(size=(T,3))*2;background-=background.mean(axis=1,keepdims=True)
    case=PhysicalCase(time_grid=grid,participant_ids=("a","b","c"),bus_ids=("1","2","3"),
        capacities_mw=np.array([15.,17.,20.]),requests_mw=rng.uniform(0,10,size=(T,N)),
        contract_mask=np.ones((T,N),bool),eligibility_mask=np.ones((T,N),bool),
        p_background_mw=background,A=A,network_states=states,participant_directions=("import",)*3)
    allowance=AllowanceSpec.explicit(case.participant_ids,[1.,1.,1.])
    q=rng.uniform(.5,1.5,size=(T,N)) if rule=="M3" else None
    exp=ExperimentCase("COUPLED",case.physical_case_id,case.physical_case_hash,grid.grid_id,rule,allowance,
        solver_profile=SolverProfile(backend="scipy",mip_gap=1e-10),
        card_id="synthetic" if q is not None else None,card_hash="synthetic" if q is not None else None,
        rate_matrix_hash=None if q is None else sha256_array(q))
    result=solve_policy(case,exp,q=q,force_generic=True)
    if seed==103 and rule=="M1":
        # Regression for the native-user failure: SciPy/HiGHS returned a tiny
        # nonzero binary residual. The reference path must re-optimize with the
        # intended integer assignment fixed, not accept/edit coordinates.
        assert result.stage2.metadata.get("fixed_integer_polish") is True
        assert result.stage2.metadata.get("raw_integrality_error",0.0) > 1e-7
    s,e=_independent_dense_two_stage(case,exp,q)
    assert result.stage1.objective_value==pytest.approx(s,abs=2e-6,rel=0)
    assert result.stage2.objective_value==pytest.approx(e,abs=2e-6,rel=0)


def test_fixed_integer_native_certificate_uses_declared_relative_mip_gap():
    from tpwrs.contracts import SolverProfile
    from tpwrs.linear import _fixed_integer_bound_certificate

    # Historical production epoch 5856: the fixed-integer LP differed from the
    # certified MIP bound by about 4e-5.  That is inside the declared 1e-8
    # relative MIP gap at this objective scale and must not be rejected by a
    # stricter backend-specific threshold.
    bound=31774.34085031
    polished=31774.34089000
    certificate=_fixed_integer_bound_certificate(
        polished,bound,SolverProfile(backend="gurobi",mip_gap=1e-8))
    assert polished-bound < certificate
    assert certificate == pytest.approx(2e-8*polished)


def test_native_integer_assignment_above_intfeastol_routes_to_certified_polish():
    from tpwrs.linear import _fixed_integer_assignment

    # Historical 2031 rolling failure: an OPTIMAL native assignment can expose
    # a post-read residual above IntFeasTol. It is still unambiguous and must be
    # routed to fixed-integer LP polishing, whose bound certificate is checked
    # by the caller, rather than rejected before continuous re-optimization.
    rounded,error=_fixed_integer_assignment(
        np.array([0.0,1.0000002,4.0]),np.array([False,True,True]))
    assert error>1e-8
    assert rounded.tolist()==[1.0,4.0]

    with pytest.raises(Exception,match="unambiguous"):
        _fixed_integer_assignment(np.array([0.5]),np.array([True]))
