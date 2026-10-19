from __future__ import annotations


# ============================================================================
# tests.test_campaign_lock_cross_host (consolidated; see SOURCE_MAP.json)
# ============================================================================

import json
from pathlib import Path
import socket

import pytest

from tpwrs.execution import CampaignLock
from tpwrs.storage import ContractError


def test_campaign_lock_cross_host__foreign_host_lock_is_never_deleted(tmp_path: Path) -> None:
    lock_path = tmp_path / ".campaign.lock"
    lock_path.write_text(
        json.dumps(
            {
                "schema": "TPWRS_CAMPAIGN_LOCK_V1",
                "campaign_hash": "other",
                "pid": 999999,
                "host": socket.gethostname() + "-OTHER",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError):
        CampaignLock(lock_path, "new").acquire()
    assert lock_path.is_file()


# ============================================================================
# tests.test_io_rolling (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np

from tpwrs.contracts import AllowanceSpec, ExperimentCase
from tpwrs.persistence import save_physical_case, load_physical_case
from tpwrs.rolling import solve_rolling


def test_io_rolling__physical_case_roundtrip(simple_case,tmp_path):
    save_physical_case(simple_case,tmp_path)
    loaded=load_physical_case(tmp_path)
    assert loaded.physical_case_hash==simple_case.physical_case_hash


def test_io_rolling__rolling_coverage_and_balance(simple_case):
    allow=AllowanceSpec.explicit(simple_case.participant_ids,[1.0])
    e=ExperimentCase("roll",simple_case.physical_case_id,simple_case.physical_case_hash,simple_case.time_grid.grid_id,"M2",allow,mode="rolling",lookahead_hours=2,implementation_hours=1)
    result=solve_rolling(simple_case,e,use_stage3=False)
    assert result.complete_coverage and np.min(result.final_balance_hours)>=-1e-8


# ============================================================================
# tests.test_release_io_and_ac (consolidated; see SOURCE_MAP.json)
# ============================================================================
import importlib.util
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from tpwrs.engineering import aligned_matrix, validate_connection_weights, freeze, load
from tpwrs.engineering import reactive_delta
from tpwrs.persistence import save_physical_case
from tpwrs.storage import sha256_json


def test_release_io_and_ac__ac_alignment_and_no_missing_Q(tmp_path):
    t=pd.date_range('2028-01-01',periods=2,freq='h',tz='UTC')
    frame=pd.DataFrame([(str(x),b,i+1) for i,x in enumerate(t) for b in ('b2','b1')],columns=['timestamp_utc','bus_id','q_injection_mvar'])
    path=tmp_path/'q.csv';frame.to_csv(path,index=False)
    np.testing.assert_equal(aligned_matrix(path,t,('b1','b2'),'bus_id','q_injection_mvar'),[[1,1],[2,2]])
    frame.iloc[:-1].to_csv(path,index=False)
    with pytest.raises(ValueError):aligned_matrix(path,t,('b1','b2'),'bus_id','q_injection_mvar')
    pd.concat([frame,frame.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Duplicate'):aligned_matrix(path,t,('b1','b2'),'bus_id','q_injection_mvar')


def test_release_io_and_ac__participant_Q_connections_are_not_balancing_vectors():
    f=pd.DataFrame({'bus_id':['b1','b2'],'participant_id':['p1','p1'],'weight':[0.4,0.6]})
    w=validate_connection_weights(f,('b1','b2'),('p1',))
    assert w.sum()==1
    f['weight']=[1,-1]
    with pytest.raises(ValueError):validate_connection_weights(f,('b1','b2'),('p1',))
    lag=reactive_delta(w,[10],[4],('import',),'lagging_098')
    lead=reactive_delta(w,[10],[4],('import',),'leading_098')
    np.testing.assert_allclose(lag,-lead);assert lag.sum()<0
    np.testing.assert_equal(reactive_delta(w,[10],[4],('import',),'unity'),np.zeros(2))


def test_release_io_and_ac__ac_freeze_hashes_and_missing_source_controls(simple_case,tmp_path):
    case=simple_case;timestamps=case.time_grid.timestamps_utc
    q=pd.DataFrame([(t,b,0.) for t in timestamps for b in case.bus_ids],columns=['timestamp_utc','bus_id','q_injection_mvar'])
    q.to_csv(tmp_path/'q.csv',index=False)
    pd.DataFrame(columns=['timestamp_utc','generator_id','p_mw']).to_csv(tmp_path/'gen.csv',index=False)
    pd.DataFrame({'bus_id':['b1'],'participant_id':['p1'],'weight':[1.]}).to_csv(tmp_path/'w.csv',index=False)
    net=SimpleNamespace(gen=pd.DataFrame(columns=['in_service']),ext_grid=pd.DataFrame({'in_service':[True],'vm_pu':[1.],'min_q_mvar':[-100.],'max_q_mvar':[100.]}))
    network=tmp_path/'native.json';network.write_text('{}')
    cfg={'q_sign_convention':'positive_net_injection_mvar','q_fixed_excludes_voltage_controlled_generators_and_shunts':True,
         'provenance':'Synthetic explicitly supplied unity-Q fixture, not production data.',
         'q_fixed':{'path':'q.csv'},'generator_dispatch':{'path':'gen.csv'},'participant_connection_weights':{'path':'w.csv'}}
    config=tmp_path/'AC_INPUTS.json';config.write_text(json.dumps(cfg))
    meta=freeze(config,case,tmp_path/'frozen',network,net)
    got,arrays=load(tmp_path/'frozen',case,network)
    assert got['identity_hash']==meta['identity_hash'];assert arrays['q_fixed_mvar'].shape==(2,2)
    mf=tmp_path/'frozen/AC_INPUT_FREEZE.json';tampered=json.loads(mf.read_text());tampered['configuration']['provenance']='altered';mf.write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='identity'):load(tmp_path/'frozen',case,network)
    net.ext_grid['min_q_mvar']=np.nan
    with pytest.raises(ValueError,match='finite'):freeze(config,case,tmp_path/'other',network,net)


def test_release_io_and_ac__small_serialized_case_audit(simple_case,tmp_path):
    # Only small serialization/conservation test. Not the Dutch inventory gate.
    from jobs.input_audit import audit_case
    save_physical_case(simple_case,tmp_path/'case')
    _,result=audit_case(tmp_path/'case',strict=False)
    assert result['status']=='PASS'
    assert result['maximum_G_minus_HA']==0


@pytest.mark.skipif(importlib.util.find_spec('pandapower') is None,reason='pandapower is not installed')
def test_release_io_and_ac__native_AC_reconstruction_retains_controls_and_solves():
    import pandapower as pp
    from tpwrs.engineering import prepare, solve
    net=pp.create_empty_network()
    b0=pp.create_bus(net,110);b1=pp.create_bus(net,110)
    pp.create_ext_grid(net,b0,vm_pu=1.0,min_q_mvar=-100,max_q_mvar=100)
    pp.create_line_from_parameters(net,b0,b1,1.,0.05,0.1,0.,1.)
    pp.create_load(net,b1,2.,0.4)
    rebuilt=prepare(net,'INTACT',('0','1'),np.array([2.,-2.]),np.array([0.,-0.4]),(),np.array([]))
    assert not rebuilt.load.in_service.any()
    ok,attempts=solve(rebuilt)
    assert ok,attempts
    assert np.isfinite(rebuilt.res_bus.vm_pu).all()
    assert len(net.sgen)==0  # source not modified


# ============================================================================
# tests.test_release_negative_guards (consolidated; see SOURCE_MAP.json)
# ============================================================================
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from tpwrs.reporting import validate_component_panel
from tpwrs.cards import freeze_preheldout_inventory
from tpwrs.contracts import nominal_experiment
from tpwrs.contracts import SolverProfile
from tpwrs.model import solve_policy
from tpwrs.replay import replay_solution
from tpwrs.storage import ReplayFailure


def test_release_negative_guards__negative_residual_load_is_valid_not_negative_demand():
    f=pd.DataFrame({'timestamp_utc':['2028-01-01T00:00:00Z'],'province':['P'],
        'prov_load_MW':[10.],'prov_solar_MW':[20.],'prov_wind_on_MW':[3.],'prov_wind_off_MW':[0.],'RL_MW':[-13.]})
    assert validate_component_panel(f)['negative_residual_rows']==1
    f.loc[0,'prov_load_MW']=-1
    with pytest.raises(ValueError):validate_component_panel(f)


def test_release_negative_guards__existing_prefreeze_marker_does_not_hide_later_outcomes(tmp_path):
    output=tmp_path/'cards';freeze_preheldout_inventory(tmp_path,output)
    result=tmp_path/'results_m3_final/year_2029/annual/M3';result.mkdir(parents=True)
    (result/'solution.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Held-out'):freeze_preheldout_inventory(tmp_path,output)


def test_release_negative_guards__M1_replay_rejects_integer_two(simple_case):
    e=nominal_experiment(simple_case,case_id='r',rule='M1',rho=1.,solver_profile=SolverProfile(backend='scipy'))
    s=solve_policy(simple_case,e)
    bad=replace(s,u=np.full_like(s.u,2))
    with pytest.raises(ReplayFailure,match=r'\[0,1\]'):replay_solution(simple_case,e,bad)


def test_release_negative_guards__duplicate_state_ids_are_rejected(simple_case):
    from tpwrs.storage import ContractError
    with pytest.raises(ContractError,match='Network-state'):
        replace(simple_case,network_states=(simple_case.network_states[0],simple_case.network_states[0]))


def test_release_negative_guards__qualified_profile_cannot_be_fabricated_by_PASS_only(tmp_path,monkeypatch):
    import json
    import tpwrs.config as cfg
    monkeypatch.setattr(cfg,'ROOT',tmp_path)
    p=tmp_path/'artifacts/performance';p.mkdir(parents=True)
    (p/'SELECTED_SOLVER_PROFILE.json').write_text(json.dumps({'status':'PASS'}))
    q=tmp_path/'results_2028_performance_regression';q.mkdir()
    (q/'STATUS.json').write_text(json.dumps({'status':'PASS'}))
    with pytest.raises(RuntimeError,match='qualification/profile'):cfg.qualified_solver_profile()


def test_release_negative_guards__nonfinite_certificate_cannot_pass():
    from tpwrs.replay import strict_m1_certificate, annual_m1_scalar_bracket
    from tpwrs.storage import ContractError
    import pytest
    for bad in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ContractError): strict_m1_certificate(0.0,bad,1e-6)
        with pytest.raises(ContractError): annual_m1_scalar_bracket(0.0,bad)


# ============================================================================
# tests.test_release_reporting_roundtrip (consolidated; see SOURCE_MAP.json)
# ============================================================================
from dataclasses import replace
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jobs'))
from tpwrs.contracts import nominal_experiment
from tpwrs.contracts import SolverProfile
from tpwrs.storage import sha256_array
from jobs.common import run_annual, run_rolling_case
from jobs.report import annual, rolling


def _release_reporting_roundtrip__experiment(case,rule,mode):
    e=nominal_experiment(case,case_id='IO_'+rule,rule=rule,rho=0.5,mode=mode,
        solver_profile=SolverProfile(backend='scipy',presolve=True),implementation_hash='TEST',result_root_id='SYNTHETIC_ONLY')
    if mode=='rolling':e=replace(e,lookahead_hours=2,implementation_hours=1)
    return e


def test_release_reporting_roundtrip__annual_report_replays_source_arrays(simple_case,tmp_path):
    e=_release_reporting_roundtrip__experiment(simple_case,'M2','annual');root=tmp_path/'annual'
    sol,_=run_annual(simple_case,e,None,root)
    metrics,x,got=annual(root,simple_case)
    assert got.solution_hash==sol.solution_hash
    np.testing.assert_equal(x,sol.x_mw)
    assert metrics['E_restricted_mwh']==pytest.approx(sol.e_value_mwh)
    p=root/'metrics.json';m=json.loads(p.read_text());m['E_restricted_mwh']+=10;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Metric'):annual(root,simple_case)


@pytest.mark.parametrize('rule',['M1','M2','M3'])
def test_release_reporting_roundtrip__full_rolling_report_reconstructs_epoch_identities(simple_case,tmp_path,rule):
    e=_release_reporting_roundtrip__experiment(simple_case,'M2' if rule=='M3' else rule,'rolling')
    q=None
    if rule=='M3':
        q=np.full(simple_case.requests_mw.shape,0.8)
        e=replace(e,rule='M3',card_id='SYNTHETIC',card_hash='SYNTHETIC',rate_matrix_hash=sha256_array(q))
    root=tmp_path/rule
    run_rolling_case(simple_case,e,q,root)
    metrics,x,_=rolling(root,simple_case)
    assert metrics['E_restricted_mwh']==pytest.approx(np.sum(simple_case.time_grid.delta[:,None]*x))
    assert metrics['physical_fde_sum_h']>=0


# ============================================================================
# tests.test_release_workspace_guards (consolidated; see SOURCE_MAP.json)
# ============================================================================

import json
from pathlib import Path

import pytest

import tpwrs.config as rc
from tpwrs.storage import sha256_file
from tpwrs.execution import STAGE_IDENTITY_INPUTS


def test_release_workspace_guards__stage06_binds_ac_input_freeze():
    assert "artifacts/ac_inputs/AC_INPUT_FREEZE.json" in STAGE_IDENTITY_INPUTS["06"]


def test_release_workspace_guards__stage07_binds_evidence_without_manuscript_signoff():
    inputs = set(STAGE_IDENTITY_INPUTS["07"])
    assert "results_m3_controls/M3_CONTROL_STATUS.json" in inputs
    assert "results_engineering_validation/independent_dc/STATUS.json" in inputs
    assert "results_engineering_validation/selected_ac/STATUS.json" in inputs
    assert "artifacts/prices/PRICE_FREEZE.json" in inputs
    assert "local/MANUSCRIPT_SIGNOFF.json" not in inputs


def test_release_workspace_guards__release_manifest_detects_changed_source(tmp_path, monkeypatch):
    source = tmp_path / "tpwrs" / "x.py"
    source.parent.mkdir(parents=True)
    source.write_text("X=1\n", encoding="utf-8")
    manifest = {
        "schema": "TEST",
        "release_id": "TEST_RELEASE",
        "source_files": {"tpwrs/x.py": sha256_file(source)},
    }
    (tmp_path / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    assert rc.verify_release_manifest()["status"] == "PASS"
    source.write_text("X=2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Source release integrity failed"):
        rc.verify_release_manifest()


def test_release_workspace_guards__release_manifest_rejects_unexpected_executable_source(tmp_path, monkeypatch):
    source = tmp_path / "tpwrs" / "x.py"
    source.parent.mkdir(parents=True)
    source.write_text("X=1\n", encoding="utf-8")
    manifest = {
        "schema": "TEST",
        "release_id": "TEST_RELEASE",
        "source_files": {"tpwrs/x.py": sha256_file(source)},
    }
    (tmp_path / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    assert rc.verify_release_manifest()["status"] == "PASS"
    extra = tmp_path / "jobs" / "obsolete_hotfix.py"
    extra.parent.mkdir(parents=True)
    extra.write_text("OLD=True\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected executable source"):
        rc.verify_release_manifest()


def test_release_workspace_guards__release_manifest_rejects_unexpected_methodology_json(tmp_path, monkeypatch):
    source = tmp_path / "tpwrs" / "x.py"
    source.parent.mkdir(parents=True)
    source.write_text("X=1\n", encoding="utf-8")
    manifest = {
        "schema": "TEST",
        "release_id": "TEST_RELEASE",
        "source_files": {"tpwrs/x.py": sha256_file(source)},
    }
    (tmp_path / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    spec = tmp_path / "specification" / "obsolete_method.json"
    spec.parent.mkdir(parents=True)
    spec.write_text('{"old":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected scientific source"):
        rc.verify_release_manifest()


def test_release_workspace_guards__release_manifest_rejects_root_shadow_module(tmp_path, monkeypatch):
    source = tmp_path / "tpwrs" / "x.py"
    source.parent.mkdir(parents=True)
    source.write_text("X=1\n", encoding="utf-8")
    manifest = {
        "schema": "TEST",
        "release_id": "TEST_RELEASE",
        "source_files": {"tpwrs/x.py": sha256_file(source)},
    }
    (tmp_path / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(rc, "ROOT", tmp_path)
    (tmp_path / "json.py").write_text("# obsolete shadow module\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected executable source: json.py"):
        rc.verify_release_manifest()


def test_release_workspace_guards__stage05b_identity_binds_control_card_freeze():
    assert "artifacts/cards/RHO03_CONTROL_CARD_FREEZE.json" in STAGE_IDENTITY_INPUTS["05B"]


def test_release_workspace_guards__numbered_root_runners_move_project_root_to_end_before_import():
    root = Path(__file__).resolve().parents[1]
    runners = ["RUN.py"]
    assert {p.name for p in root.glob("*.py")} == {"RUN.py"}
    for name in runners:
        source = (root / name).read_text(encoding="utf-8")
        reorder = source.index("sys.path[:]")
        production_import = source.index("from tpwrs.orchestrator import main")
        assert reorder < production_import


def test_release_workspace_guards__release_runner_requires_exact_physical_readiness_identity_or_r10_compatibility():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tpwrs" / "orchestrator.py").read_text(encoding="utf-8")
    assert "require_compatible_status" in source
    assert "IMMUTABLE_PHYSICAL_CASE_INPUT" in source


def test_release_workspace_guards__release_runner_rechecks_source_stability_around_child_commands():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tpwrs" / "orchestrator.py").read_text(encoding="utf-8")
    assert "def _assert_release_stable" in source
    # The guard should appear multiple times: before the lock/commands, before
    # each child, after each child, and before PASS.
    assert source.count("_assert_release_stable(") >= 5


def test_release_workspace_guards__release_runner_requires_current_card_and_control_freeze_schemas():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tpwrs" / "orchestrator.py").read_text(encoding="utf-8")
    assert "TPWRS_ALLOWANCE_SPECIFIC_CARD_BUILD_V3" in source
    assert "TPWRS_RHO03_CONTROL_CARD_FREEZE_V2" in source


def test_utf8_relay_survives_cp1252_console():
    import io
    from tpwrs.orchestrator import relay_output
    raw=io.BytesIO()
    console=io.TextIOWrapper(raw,encoding="cp1252",errors="strict")
    text="solver Unicode: \u2713 \ufffd\nfinished\n"
    source=io.StringIO(text);log=io.StringIO();errors=[]
    relay_output(source,log,errors,console)
    assert not errors
    assert log.getvalue()==text
    console.flush()
    assert b"finished" in raw.getvalue()
    assert b"\\ufffd" in raw.getvalue()


def test_relay_log_failure_is_reported_while_output_is_drained():
    import io
    from tpwrs.orchestrator import relay_output
    class BadLog:
        def write(self,_): raise OSError("disk full")
    source=io.StringIO("a\nb\nc\n");errors=[]
    relay_output(source,BadLog(),errors,io.StringIO())
    assert len(errors)==3
    assert source.read()==""


def test_resource_guard_refuses_tiny_floor_budget(monkeypatch):
    from types import SimpleNamespace
    import tpwrs.execution as runtime
    from tpwrs.storage import SolverFailure
    model=SimpleNamespace(Params=SimpleNamespace(SoftMemLimit=0.,TimeLimit=100.),MemUsed=.1)
    monkeypatch.setattr("psutil.virtual_memory",lambda:SimpleNamespace(available=int(.54*1024**3)))
    monkeypatch.delenv("TPWRS_GUROBI_SOFTMEM_GB",raising=False)
    budget=runtime.ExecutionBudget()
    try:
        with pytest.raises(SolverFailure,match="HEADROOM"):
            budget.configure(model)
        assert model.Params.SoftMemLimit==0.
    finally:budget.close()


def test_resource_guard_adds_allocated_gurobi_memory_to_incremental_headroom(monkeypatch):
    from types import SimpleNamespace
    from tpwrs.execution import ExecutionBudget
    model=SimpleNamespace(Params=SimpleNamespace(SoftMemLimit=0.,TimeLimit=100.),MemUsed=.6)
    available=3*1024**3
    monkeypatch.setattr("psutil.virtual_memory",lambda:SimpleNamespace(available=available))
    monkeypatch.delenv("TPWRS_GUROBI_SOFTMEM_GB",raising=False)
    with_reserve=.6+.8*(available-.25*available)/1e9
    budget=ExecutionBudget()
    try:
        budget.configure(model)
        assert model.Params.SoftMemLimit==pytest.approx(with_reserve)
    finally:budget.close()


def test_resource_guard_exempts_bounded_week_qualification(monkeypatch):
    from types import SimpleNamespace
    import tpwrs.execution as runtime
    model=SimpleNamespace(Params=SimpleNamespace(SoftMemLimit=0.,TimeLimit=10000.),MemUsed=0.)
    monkeypatch.setattr("psutil.virtual_memory",lambda:SimpleNamespace(available=64*1024**2))
    monkeypatch.setenv("TPWRS_GUROBI_SOFTMEM_GB","0.000001")
    budget=runtime.ExecutionBudget()
    try:
        budget.configure(model,model_size=(78516,108060,919236))
        assert model.Params.SoftMemLimit==0.
        assert model.Params.TimeLimit==budget.max_stage_s
        assert budget.memory_scope=="BOUNDED_SMALL_MODEL"
    finally:budget.close()


def test_resource_guard_keeps_full_year_coordinate_model_in_production_scope(monkeypatch):
    from types import SimpleNamespace
    import pytest
    import tpwrs.execution as runtime
    from tpwrs.storage import SolverFailure
    model=SimpleNamespace(Params=SimpleNamespace(SoftMemLimit=0.,TimeLimit=10000.),MemUsed=0.)
    monkeypatch.setattr("psutil.virtual_memory",lambda:SimpleNamespace(available=64*1024**2))
    budget=runtime.ExecutionBudget()
    try:
        with pytest.raises(SolverFailure,match="HEADROOM"):
            budget.configure(model,model_size=(221067,0,0))
        assert budget.memory_scope=="PRODUCTION"
    finally:budget.close()


def test_resource_guard_scales_reserve_floor_on_eight_gib_host(monkeypatch):
    from types import SimpleNamespace
    import tpwrs.execution as runtime
    monkeypatch.delenv("TPWRS_GUROBI_SOFTMEM_GB",raising=False)
    model=SimpleNamespace(Params=SimpleNamespace(SoftMemLimit=0.,TimeLimit=10000.),MemUsed=0.)
    monkeypatch.setattr("psutil.virtual_memory",lambda:SimpleNamespace(
        available=int(.72*1024**3),total=8*1024**3))
    budget=runtime.ExecutionBudget()
    try:
        budget.configure(model,model_size=(221067,0,0))
        expected=.8*(int(.72*1024**3)-256*1024**2)/1e9
        assert model.Params.SoftMemLimit==pytest.approx(expected)
        assert budget.memory_scope=="PRODUCTION"
    finally:budget.close()


def test_geometry_cache_is_bounded_across_different_epochs(simple_case):
    from dataclasses import replace
    from tpwrs.annual import _GEOMETRY_CACHE,_screen_geometry,GEOMETRY_CACHE_LIMIT
    from tpwrs.contracts import nominal_experiment,SolverProfile
    _GEOMETRY_CACHE.clear()
    for k in range(8):
        case=replace(simple_case,requests_mw=np.full((2,1),50.+k))
        exp=nominal_experiment(case,case_id=str(k),rule="M2",rho=1.,solver_profile=SolverProfile(backend="scipy"))
        _screen_geometry(case,exp)
        assert len(_GEOMETRY_CACHE)<=GEOMETRY_CACHE_LIMIT
    _GEOMETRY_CACHE.clear()


def test_every_dispatched_job_has_a_cheap_help_interface(monkeypatch):
    import importlib,sys
    from tpwrs.orchestrator import JOBS
    for name,(module,function) in JOBS.items():
        monkeypatch.setattr(sys,"argv",[name,"--help"])
        with pytest.raises(SystemExit) as exc:
            getattr(importlib.import_module(module),function)()
        assert exc.value.code==0


def test_reproduction_is_card_frozen_before_heldout_access():
    from tpwrs.orchestrator import prerequisites
    import inspect
    source=inspect.getsource(prerequisites)
    assert "'REPRO'" in source
    assert "CARD_BUILD_STATUS.json" in source


def test_historical_reproducer_checks_old_ledger_without_rebinding(simple_case,tmp_path):
    from dataclasses import replace,asdict
    from tpwrs.contracts import AllowanceSpec,ExperimentCase,SolverProfile
    from tpwrs.model import solve_policy
    from tpwrs.persistence import save_policy_solution
    from jobs.qualification import _reconstruct_historical_epoch
    allow=AllowanceSpec.explicit(simple_case.participant_ids,[2.])
    parent=ExperimentCase("HIST",simple_case.physical_case_id,simple_case.physical_case_hash,
        simple_case.time_grid.grid_id,"M1",allow,mode="rolling",lookahead_hours=2,
        implementation_hours=1,solver_profile=SolverProfile(backend="scipy"),implementation_hash="old-r9")
    sub= simple_case.slice_time(0,2)
    subexp=replace(parent,case_id="HIST__epoch_000000",physical_case_id=sub.physical_case_id,
        physical_case_hash=sub.physical_case_hash,time_grid_id=sub.time_grid.grid_id,
        allowance=AllowanceSpec.explicit(sub.participant_ids,[2.]),mode="annual",
        lookahead_hours=None,implementation_hours=None,result_root_id="UNSPECIFIED/epochs/epoch_000000")
    solution=solve_policy(sub,subexp,force_generic=True)
    metadata={"face_numeric_contract":"GENERIC_EXPLICIT_FACE_V1","generic_face_certification_tolerance":1e-6}
    old=replace(solution,stage1=replace(solution.stage1,metadata=metadata),stage2=replace(solution.stage2,metadata=metadata))
    folder=tmp_path/"epoch_000000";save_policy_solution(old,folder)
    before={p.name:p.read_bytes() for p in folder.iterdir()}
    target,epoch,previous,lineage=_reconstruct_historical_epoch(simple_case,parent,tmp_path,1)
    assert epoch.allowance.values[0]==pytest.approx(1.)
    assert previous.solution_hash==old.solution_hash
    assert len(lineage)==1 and target.n_time==1
    assert before=={p.name:p.read_bytes() for p in folder.iterdir()}


def test_physical_readiness_timestamp_does_not_reset_scientific_cache(tmp_path):
    from tpwrs.execution import stage_workspace_identity
    from tpwrs.storage import atomic_write_json
    path=tmp_path/"artifacts/FINAL_PHYSICAL_READINESS.json"
    initial={"status":"PASS","scientific_fingerprint":"abc","physical_case":{"physical_case_hash":"physics"},"created_utc":"one","free_disk_gib":10.}
    atomic_write_json(path,initial)
    first=stage_workspace_identity(tmp_path,"03","abc")
    atomic_write_json(path,{**initial,"created_utc":"two","free_disk_gib":5.})
    assert stage_workspace_identity(tmp_path,"03","abc")==first
    atomic_write_json(path,{**initial,"physical_case":{"physical_case_hash":"changed"}})
    assert stage_workspace_identity(tmp_path,"03","abc")!=first


def test_evidence_export_does_not_write_experimental_prose(tmp_path):
    import zipfile
    from jobs.report import export_review_bundle
    from tpwrs.storage import atomic_write_json
    output=tmp_path/"results_final_evidence";output.mkdir()
    atomic_write_json(output/"FINAL_EVIDENCE_MANIFEST.json",{"status":"SYNTHETIC_TEST"})
    (output/"sample.csv").write_text("x\n1\n")
    path=export_review_bundle(tmp_path,output)
    with zipfile.ZipFile(path) as z:
        assert not any(n.endswith((".tex",".md",".pdf")) for n in z.namelist())
        meta=json.loads(z.read("BUNDLE_MANIFEST.json"))
        assert meta["submission_ready"] is False
        assert len(meta["sha256"])==3
        assert json.loads(z.read("results_final_evidence/RAW_OUTPUT_REFERENCES.json"))=={}


def test_orchestrator_local_settings_clear_stale_gurobi_override(monkeypatch,tmp_path):
    from tpwrs.orchestrator import build_child_environment
    monkeypatch.setenv("TPWRS_GUROBI_SOFTMEM_GB","1.5")
    cfg={"max_stage_seconds":3600,"max_case_seconds":14400,"gurobi_softmem_gb":None}
    env=build_child_environment(cfg,tmp_path/"STOP","r")
    assert "TPWRS_GUROBI_SOFTMEM_GB" not in env
    cfg["gurobi_softmem_gb"]=0.75
    env=build_child_environment(cfg,tmp_path/"STOP","r")
    assert env["TPWRS_GUROBI_SOFTMEM_GB"]=="0.75"


def test_stage_workspace_registry_includes_required_sensitivity_stage():
    from tpwrs.execution import STAGE_OUTPUT_ROOTS, STAGE_IDENTITY_INPUTS
    assert STAGE_OUTPUT_ROOTS["03"] == ("artifacts/cards", "artifacts/features")
    assert STAGE_OUTPUT_ROOTS["05C"] == ("results_m3_gamma", "results_m3_rate_bounds")
    assert "results_m3_final/M3_CORE_STATUS.json" in STAGE_IDENTITY_INPUTS["05C"]
    assert "results_m3_gamma/M3_GAMMA_STATUS.json" in STAGE_IDENTITY_INPUTS["07"]
    assert "results_m3_rate_bounds/RATE_BOUND_STATUS.json" in STAGE_IDENTITY_INPUTS["07"]
    assert "GAMMA" not in STAGE_OUTPUT_ROOTS and "BOUNDS" not in STAGE_OUTPUT_ROOTS


def test_gamma_screen_completes_all_years_before_full_path():
    from jobs.gamma import gamma_execution_schedule,SCREEN_GAMMAS,EVALUATION_YEARS,DEFAULT_GAMMA_GRID
    schedule=gamma_execution_schedule()
    prefix=schedule[:len(EVALUATION_YEARS)]
    assert {year for year,points in prefix}==set(EVALUATION_YEARS)
    assert all(points==SCREEN_GAMMAS for year,points in prefix)
    for year in EVALUATION_YEARS:
        assert sorted(g for y,points in schedule if y==year for g in points)==sorted(DEFAULT_GAMMA_GRID)
    with pytest.raises(RuntimeError,match='frozen'):gamma_execution_schedule((.7,1.,1.3))


def test_atomic_json_retries_transient_windows_replace_lock(tmp_path, monkeypatch):
    import tpwrs.storage as storage

    destination = tmp_path / "HEARTBEAT.json"
    real_replace = storage.os.replace
    calls = {"count": 0}

    def intermittently_locked(source, target):
        calls["count"] += 1
        if calls["count"] < 4:
            raise PermissionError(5, "transient sync lock", str(target))
        return real_replace(source, target)

    monkeypatch.setattr(storage.os, "replace", intermittently_locked)
    monkeypatch.setattr(storage.time, "sleep", lambda _seconds: None)
    storage.atomic_write_json(destination, {"status": "PASS"})

    assert calls["count"] == 4
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "PASS"}
