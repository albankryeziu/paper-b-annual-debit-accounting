from __future__ import annotations


# ============================================================================
# tests.test_feature_provenance_final (consolidated; see SOURCE_MAP.json)
# ============================================================================

import pytest

from tpwrs.storage import ProvenanceError
from tpwrs.features import FeatureProvenanceRecord


def test_feature_provenance_final__predeclared_control_input_does_not_fake_issue_timestamp() -> None:
    record = FeatureProvenanceRecord(
        feature="request_fraction",
        source_id="controlled",
        source_hash="abc",
        issue_timestamp=None,
        delivery_period="2025",
        timezone="Europe/Amsterdam",
        forecast_or_realization="predeclared_controlled_input",
        revision_policy="frozen",
        d1_pass=True,
        role="deployable",
        availability_class="PREDECLARED_CONTROLLED_INPUT",
    )
    record.validate()


def test_feature_provenance_final__archived_forecast_still_requires_vintage() -> None:
    record = FeatureProvenanceRecord(
        feature="load_forecast",
        source_id="forecast",
        source_hash="abc",
        issue_timestamp=None,
        delivery_period="2025",
        timezone="Europe/Amsterdam",
        forecast_or_realization="forecast",
        revision_policy="frozen",
        d1_pass=True,
        role="deployable",
        availability_class="ARCHIVED_FORECAST",
    )
    with pytest.raises(ProvenanceError):
        record.validate()


def test_feature_provenance_final__complete_f0_core_accepts_explicit_nonvintaged_classes() -> None:
    from tpwrs.features import F0_CORE, build_feature_schema

    calendar = {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}
    controlled = {"request_fraction"}
    records = []
    for feature in F0_CORE:
        if feature in calendar:
            availability_class = "DETERMINISTIC_CALENDAR"
        elif feature in controlled:
            availability_class = "PREDECLARED_CONTROLLED_INPUT"
        else:
            availability_class = "STATIC_CONTRACT"
        records.append(FeatureProvenanceRecord(
            feature=feature,
            source_id="source",
            source_hash="abc",
            issue_timestamp=None,
            delivery_period="2025-2031",
            timezone="Europe/Amsterdam",
            forecast_or_realization="predeclared_controlled_input_not_archived_forecast",
            revision_policy="immutable_after_physical_freeze",
            d1_pass=True,
            role="deployable",
            availability_class=availability_class,
        ))

    schema = build_feature_schema("F0", records, F0_CORE)
    assert schema.columns == F0_CORE


def test_feature_provenance_final__f0_builder_does_not_reintroduce_model_contract_timestamp_sentinel() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "jobs" / "cards.py").read_text(encoding="utf-8")
    assert "MODEL_CONTRACT::AVAILABLE_AT_OR_BEFORE" not in source
    assert "issue_timestamp=None" in source
    assert "availability_class=availability_class" in source


# ============================================================================
# tests.test_heldout_protocol (consolidated; see SOURCE_MAP.json)
# ============================================================================

import json
from pathlib import Path

import pytest

from tpwrs.cards import ValidationCandidate, freeze_selected_card
from tpwrs.storage import ContractError
from tpwrs.contracts import CardSpec
from tpwrs.cards import HELDOUT_AUDIT_SCHEMA, PROHIBITED_OUTCOME_YEARS, validate_preheldout_audit
from tpwrs.cards import freeze_preheldout_inventory


def _heldout_protocol__candidate() -> ValidationCandidate:
    card = CardSpec(
        card_id="TEST",
        allowance_hash="allowance",
        feature_set_id="F0",
        model_family="ridge",
        thresholds={"import": (0.33, 0.66)},
        rates={"import": (1.2, 1.0, 0.8)},
        q_bounds=(0.5, 1.5),
        heldout_access_audit="PASS",
    )
    return ValidationCandidate(card, 2028, 1.0, 2.0, "F0::ridge")


def test_heldout_protocol__preheldout_producer_and_card_freezer_share_one_contract(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "cards"
    payload = freeze_preheldout_inventory(tmp_path, output)
    audit = output / "PRE_FREEZE_HELDOUT_ACCESS_AUDIT.json"

    assert payload["schema"] == HELDOUT_AUDIT_SCHEMA
    assert tuple(payload["prohibited_outcome_years"]) == PROHIBITED_OUTCOME_YEARS
    assert payload["policy_outcome_files_present"] == []
    assert payload["policy_outcome_files_read"] == []

    validated, audit_hash = validate_preheldout_audit(audit)
    assert validated == payload

    freeze = output / "RHO15_CARD_FREEZE.json"
    freeze_selected_card(
        _heldout_protocol__candidate(),
        freeze,
        heldout_access_audit="PASS",
        heldout_audit_path=audit,
        freeze_metadata={"physical_case_hash": "case"},
    )
    record = json.loads(freeze.read_text(encoding="utf-8"))
    assert record["freeze_metadata"]["heldout_access_audit_sha256"] == audit_hash
    assert record["freeze_metadata"]["heldout_access_audit_schema"] == HELDOUT_AUDIT_SCHEMA


def test_heldout_protocol__preheldout_inventory_rejects_heldout_policy_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "results_m3_gamma_smoke" / "year_2029" / "solution.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Held-out policy outcomes already exist"):
        freeze_preheldout_inventory(tmp_path, tmp_path / "artifacts" / "cards")


def test_heldout_protocol__preheldout_validator_rejects_legacy_or_incomplete_schema(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"schema": "TPWRS_PREFREEZE_INVENTORY_V2", "status": "PASS"}), encoding="utf-8")
    with pytest.raises(ContractError):
        validate_preheldout_audit(audit)


def test_heldout_protocol__production_selected_card_freeze_passes_explicit_audit_path() -> None:
    import ast
    source_path = Path(__file__).resolve().parents[1] / "jobs" / "cards.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "freeze_selected_card"
    ]
    assert calls, "production card-freeze call not found"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "heldout_audit_path" in keywords


# ============================================================================
# tests.test_card_freeze_audit (consolidated; see SOURCE_MAP.json)
# ============================================================================

import json
from pathlib import Path

import pytest

from tpwrs.cards import ValidationCandidate, freeze_selected_card
from tpwrs.storage import ContractError
from tpwrs.contracts import CardSpec


def _card_freeze_audit__candidate() -> ValidationCandidate:
    card = CardSpec(
        card_id="TEST",
        allowance_hash="allowance",
        feature_set_id="F0",
        model_family="ridge",
        thresholds={"import": (0.33, 0.66)},
        rates={"import": (1.2, 1.0, 0.8)},
        q_bounds=(0.5, 1.5),
        heldout_access_audit="PASS",
    )
    return ValidationCandidate(card, 2028, 1.0, 2.0, "F0::ridge")


def test_card_freeze_audit__card_freeze_requires_explicit_audit_artifact(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        freeze_selected_card(
            _card_freeze_audit__candidate(),
            tmp_path / "freeze.json",
            heldout_access_audit="PASS",
        )

    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "TPWRS_PRE_FREEZE_HELDOUT_ACCESS_AUDIT_V3",
                "status": "PASS",
                "fit_years": [2025, 2026, 2027],
                "validation_year": 2028,
                "prohibited_outcome_years": [2029, 2030, 2031],
                "policy_outcome_files_present": [],
                "policy_outcome_files_read": [],
                "loader_scope": "Fitting/selection code reads only 2025-2028 policy/feature outcomes",
            }
        ),
        encoding="utf-8",
    )
    freeze_selected_card(
        _card_freeze_audit__candidate(),
        tmp_path / "freeze.json",
        heldout_access_audit="PASS",
        heldout_audit_path=audit,
    )
    assert (tmp_path / "freeze.json").is_file()


# ============================================================================
# tests.test_frozen_card_audit_binding (consolidated; see SOURCE_MAP.json)
# ============================================================================

from pathlib import Path

from jobs.common import load_frozen_card
from tpwrs.cards import ValidationCandidate, freeze_selected_card
from tpwrs.contracts import CardSpec
from tpwrs.storage import sha256_file
from tpwrs.persistence import save_card
from tpwrs.storage import atomic_write_json
from tpwrs.cards import freeze_preheldout_inventory
from tpwrs.config import scientific_fingerprint


class _frozen_card_audit_binding__Case:
    physical_case_hash = "case-hash"


def test_frozen_card_audit_binding__downstream_frozen_card_loader_revalidates_bound_audit(tmp_path: Path) -> None:
    cards = tmp_path / "cards"
    audit_payload = freeze_preheldout_inventory(tmp_path, cards)
    audit_path = cards / "PRE_FREEZE_HELDOUT_ACCESS_AUDIT.json"

    model_path = cards / "rho_0.15" / "MODEL_ridge.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model-bytes")

    card = CardSpec(
        card_id="RHO15_F0_RIDGE",
        allowance_hash="allowance-contract",
        feature_set_id="F0",
        model_family="ridge",
        thresholds={"import": (0.33, 0.66)},
        rates={"import": (1.2, 1.0, 0.8)},
        source_hashes={"model_bundle": sha256_file(model_path)},
    )
    selected_path = cards / "RHO15_SELECTED_CARD.json"
    save_card(card, selected_path)
    candidate = ValidationCandidate(card, 2028, 1.0, 2.0, "F0::ridge")
    freeze_selected_card(
        candidate,
        cards / "RHO15_CARD_FREEZE.json",
        heldout_access_audit="PASS",
        heldout_audit_path=audit_path,
        freeze_metadata={
            "physical_case_hash": _frozen_card_audit_binding__Case.physical_case_hash,
            "allowance_contract_hash": card.allowance_hash,
        },
    )
    atomic_write_json(
        cards / "CARD_BUILD_STATUS.json",
        {
            "schema": "TPWRS_ALLOWANCE_SPECIFIC_CARD_BUILD_V3",
            "status": "PASS",
            "scientific_fingerprint": scientific_fingerprint(),
            "heldout_access_audit": "PASS",
            "heldout_access_audit_schema": audit_payload["schema"],
        },
    )

    loaded, freeze, loaded_model = load_frozen_card(_frozen_card_audit_binding__Case(), cards, 0.15)
    assert loaded.hash == card.hash
    assert freeze["heldout_access_audit"] == "PASS"
    assert loaded_model == model_path


# ============================================================================
# tests.test_control_freeze_audit_binding (consolidated; see SOURCE_MAP.json)
# ============================================================================

from pathlib import Path

from jobs.controls import CONTROL_KEYS, _load_controls
from tpwrs.contracts import CardSpec
from tpwrs.storage import sha256_file
from tpwrs.persistence import save_card
from tpwrs.storage import atomic_write_json
from tpwrs.cards import freeze_preheldout_inventory
from tpwrs.config import scientific_fingerprint


def _control_freeze_audit_binding__card(card_id: str) -> CardSpec:
    return CardSpec(
        card_id=card_id,
        allowance_hash="allowance",
        feature_set_id="CONTROL",
        model_family="ridge",
        thresholds={"import": (0.33, 0.66)},
        rates={"import": (1.2, 1.0, 0.8)},
    )


def test_control_freeze_audit_binding__control_loader_revalidates_shared_heldout_audit(tmp_path: Path) -> None:
    cards_root = tmp_path / "cards"
    freeze_preheldout_inventory(tmp_path, cards_root)
    audit = cards_root / "PRE_FREEZE_HELDOUT_ACCESS_AUDIT.json"
    selected = _control_freeze_audit_binding__card("SELECTED")

    records = {}
    for key in CONTROL_KEYS:
        card = _control_freeze_audit_binding__card(key)
        path = cards_root / "controls" / f"{key}.json"
        save_card(card, path)
        records[key] = {
            "card_id": card.card_id,
            "card_hash": card.hash,
            "path": str(path.relative_to(cards_root)).replace("\\", "/"),
        }

    from tpwrs.cards import validate_preheldout_audit
    payload, audit_hash = validate_preheldout_audit(audit)
    atomic_write_json(
        cards_root / "RHO03_CONTROL_CARD_FREEZE.json",
        {
            "schema": "TPWRS_RHO03_CONTROL_CARD_FREEZE_V2",
            "status": "PASS",
            "scientific_fingerprint": scientific_fingerprint(),
            "selected_card_hash": selected.hash,
            "heldout_access_audit": "PASS",
            "heldout_access_audit_path": audit.name,
            "heldout_access_audit_sha256": audit_hash,
            "heldout_access_audit_schema": payload["schema"],
            "shuffle_seeds": list(range(118001, 118033)),
            "records": records,
        },
    )

    loaded, freeze = _load_controls(cards_root, selected)
    assert set(loaded) == set(CONTROL_KEYS)
    assert freeze["heldout_access_audit_sha256"] == sha256_file(audit)


# ============================================================================
# tests.test_teacher_cache_identity (consolidated; see SOURCE_MAP.json)
# ============================================================================

import json
from pathlib import Path
import sys

_teacher_cache_identity__ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_teacher_cache_identity__ROOT / "jobs"))

import jobs.cards as ffc
from tpwrs.config import scientific_fingerprint, scientific_source_hashes
from tpwrs.execution import MARKER, prepare_stage_outputs, stage_workspace_identity


def test_teacher_cache_identity__card_build_uses_complete_production_scientific_identity():
    assert ffc.PRODUCTION_IMPLEMENTATION_HASH == scientific_fingerprint()
    hashes = scientific_source_hashes()
    # Regression for the omitted-dependency defect found in the release audit.
    assert "tpwrs/annual.py" in hashes
    assert "tpwrs/cards.py" in hashes
    assert "jobs/cards.py" in hashes
    assert "jobs/primary.py" in hashes
    assert "jobs/m3.py" in hashes


def test_teacher_cache_identity__cross_release_teacher_migration_is_removed():
    assert not hasattr(ffc, "_migrate_legacy_teacher_solution")
    assert not hasattr(ffc, "_experiment_solve_signature")
    source = (_teacher_cache_identity__ROOT / "jobs" / "cards.py").read_text(encoding="utf-8")
    assert "MIGRATED_VERIFIED" not in source
    assert "LEGACY_MIGRATION" not in source


def test_teacher_cache_identity__stage_workspace_archives_legacy_root_and_resumes_exact_identity(tmp_path):
    cards = tmp_path / "artifacts" / "cards"
    cards.mkdir(parents=True)
    (cards / "legacy.txt").write_text("old", encoding="utf-8")

    first = prepare_stage_outputs(tmp_path, "03", "a" * 64)
    assert first["actions"][0]["action"] == "ARCHIVE_AND_RESET"
    archive = tmp_path / first["actions"][0]["archive"]
    assert (archive / "legacy.txt").read_text(encoding="utf-8") == "old"
    marker = json.loads((cards / MARKER).read_text(encoding="utf-8"))
    assert marker["workspace_identity"] == first["workspace_identity"]

    (cards / "current.txt").write_text("keep", encoding="utf-8")
    second = prepare_stage_outputs(tmp_path, "03", "a" * 64)
    assert second["workspace_identity"] == first["workspace_identity"]
    assert second["actions"][0]["action"] == "RESUME"
    assert (cards / "current.txt").read_text(encoding="utf-8") == "keep"


def test_teacher_cache_identity__stage_workspace_changes_on_source_or_prerequisite_identity(tmp_path):
    readiness = tmp_path / "artifacts" / "FINAL_PHYSICAL_READINESS.json"
    readiness.parent.mkdir(parents=True)
    readiness.write_text('{"status":"PASS","v":1}', encoding="utf-8")

    a = stage_workspace_identity(tmp_path, "03", "a" * 64)
    b = stage_workspace_identity(tmp_path, "03", "b" * 64)
    assert a != b

    readiness.write_text('{"status":"PASS","v":2}', encoding="utf-8")
    c = stage_workspace_identity(tmp_path, "03", "a" * 64)
    assert c != a


def test_teacher_cache_identity__stage03_identity_binds_2028_qualification_artifacts(tmp_path):
    # Regression: card teachers use the qualified profile, so changing that
    # qualification must invalidate the card workspace even if source is fixed.
    profile = tmp_path / "artifacts" / "performance" / "SELECTED_SOLVER_PROFILE.json"
    profile.parent.mkdir(parents=True)
    profile.write_text('{"status":"PASS","x":1}', encoding="utf-8")
    gate = tmp_path / "results_2028_performance_regression" / "STATUS.json"
    gate.parent.mkdir(parents=True)
    gate.write_text('{"status":"PASS","x":1}', encoding="utf-8")
    a = stage_workspace_identity(tmp_path, "03", "a" * 64)
    profile.write_text('{"status":"PASS","x":2}', encoding="utf-8")
    b = stage_workspace_identity(tmp_path, "03", "a" * 64)
    assert a != b


# ============================================================================
# tests.test_annual_qualification_identity (consolidated; see SOURCE_MAP.json)
# ============================================================================

from pathlib import Path

import tpwrs.execution as aq


def test_annual_qualification_identity__annual_solver_fingerprint_ignores_unrelated_downstream_file(tmp_path, monkeypatch):
    annual = tmp_path / 'tpwrs' / 'annual.py'
    annual.parent.mkdir(parents=True)
    annual.write_text('ANNUAL=1\n', encoding='utf-8')
    downstream = tmp_path / 'jobs' / 'cards.py'
    downstream.parent.mkdir(parents=True)
    downstream.write_text('F0=1\n', encoding='utf-8')
    monkeypatch.setattr(aq, 'ANNUAL_SOLVER_SOURCE_PATHS', ('tpwrs/annual.py',))
    first = aq.annual_solver_fingerprint(tmp_path)
    downstream.write_text('F0=2\n', encoding='utf-8')
    assert aq.annual_solver_fingerprint(tmp_path) == first


def test_annual_qualification_identity__annual_solver_fingerprint_changes_with_annual_source(tmp_path, monkeypatch):
    annual = tmp_path / 'tpwrs' / 'annual.py'
    annual.parent.mkdir(parents=True)
    annual.write_text('ANNUAL=1\n', encoding='utf-8')
    monkeypatch.setattr(aq, 'ANNUAL_SOLVER_SOURCE_PATHS', ('tpwrs/annual.py',))
    first = aq.annual_solver_fingerprint(tmp_path)
    annual.write_text('ANNUAL=2\n', encoding='utf-8')
    assert aq.annual_solver_fingerprint(tmp_path) != first


def test_annual_qualification_identity__2028_result_root_is_fingerprint_scoped(tmp_path):
    # Import through the script path exactly as the production runner does.
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'jobs' / 'qualification.py'
    spec = importlib.util.spec_from_file_location('run_2028_performance_regression_test', script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    a = module.result_root_for_fingerprint(tmp_path, 'a' * 64)
    b = module.result_root_for_fingerprint(tmp_path, 'b' * 64)
    assert a != b
    assert a.parent == tmp_path / 'solver_fingerprints'
    assert b.parent == tmp_path / 'solver_fingerprints'


def test_fitting_partition_never_opens_full_year_arrays(simple_case,tmp_path,monkeypatch):
    from dataclasses import replace
    from tpwrs.contracts import TimeGrid
    from tpwrs.persistence import save_physical_case,prepare_fitting_inputs,load_fitting_case
    from tpwrs.storage import ContractError
    import pandas as pd
    import numpy as np
    import json
    # Daily intervals keep the full four-year partition small, with exact adjacency.
    idx=pd.date_range('2025-01-01','2029-01-02',freq='D',tz='UTC');n=len(idx)
    case=replace(simple_case,time_grid=TimeGrid.from_index(idx,24.),
        requests_mw=np.full((n,1),100.),contract_mask=np.ones((n,1),bool),
        eligibility_mask=np.ones((n,1),bool),p_background_mw=np.zeros((n,2)))
    save_physical_case(case,tmp_path);record=prepare_fitting_inputs(case,tmp_path)
    original=np.load;opened=[]
    def guarded(path,*args,**kwargs):
        opened.append(str(path))
        assert Path(path).parent.name=='FITTING_2025_2028'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    fitting,bound=load_fitting_case(tmp_path)
    assert set(fitting.time_grid.index.year)=={2025,2026,2027,2028}
    assert opened and bound['physical_case_hash']==case.physical_case_hash
    record['allowances']['0.15']['2029']['values_hours'][0]+=1
    (tmp_path/'FITTING_INPUTS.json').write_text(json.dumps(record))
    with pytest.raises(ContractError,match='record changed'):load_fitting_case(tmp_path)


def test_predictors_are_direction_specific_and_reject_validation_training():
    import numpy as np
    import pandas as pd
    from tpwrs.features import F0_CORE,fit_predictor
    from tpwrs.storage import ProvenanceError
    frame=pd.DataFrame({name:np.ones(12) for name in F0_CORE})
    frame['zone']='Z';frame['technology']='T'
    frame['direction']=['import']*6+['export']*6;frame['year']=2025
    target=np.r_[np.zeros(6),np.ones(6)]
    model=fit_predictor('ridge',frame,F0_CORE,target)
    assert np.allclose(model.predict(frame),target,atol=1e-12)
    frame.loc[0,'year']=2028
    with pytest.raises(ProvenanceError,match='2025'):fit_predictor('ridge',frame,F0_CORE,target)


@pytest.mark.parametrize('delivery,previous',[('2028-03-27 00:30','2028-03-26'),('2028-10-30 00:30','2028-10-29')])
def test_previous_local_calendar_day_across_DST(delivery,previous):
    import pandas as pd
    from tpwrs.features import d1_cutoff
    actual=d1_cutoff(pd.Timestamp(delivery,tz='Europe/Amsterdam'))
    assert str(actual.date())==previous


def test_Q_registry_survives_stage06_preparation(tmp_path):
    from tpwrs.execution import prepare_stage_outputs
    registry=tmp_path/'results_engineering_validation/selected_registry'
    prepare_stage_outputs(tmp_path,'Q','science')
    (registry/'frozen.json').write_text('frozen')
    prepare_stage_outputs(tmp_path,'06','science')
    assert (registry/'frozen.json').read_text()=='frozen'


def test_empirical_reduction_gate_uses_direct_benchmark_quantiles():
    from types import SimpleNamespace
    from tpwrs.network import empirical_embedding_thresholds
    import numpy as np
    rows=[SimpleNamespace(e2=x,einf=2*x,sign_agreement=1-x/10,topk_overlap=1-x/20) for x in (1.,2.,3.,4.)]
    got=empirical_embedding_thresholds(rows)
    expected=(np.quantile([1.,2.,3.,4.],.95),np.quantile([2.,4.,6.,8.],.95),np.quantile([.9,.8,.7,.6],.05),np.quantile([.95,.9,.85,.8],.05))
    assert np.allclose(got,expected,atol=1e-15,rtol=0)
