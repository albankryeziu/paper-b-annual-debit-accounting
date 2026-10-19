from __future__ import annotations


# ============================================================================
# tests.conftest (consolidated; see SOURCE_MAP.json)
# ============================================================================

import numpy as np
import pandas as pd
import pytest

from tpwrs.contracts import NetworkState, PhysicalCase
from tpwrs.contracts import TimeGrid


@pytest.fixture
def simple_case():
    grid=TimeGrid.from_index(pd.date_range("2028-01-01",periods=2,freq="1h",tz="UTC"),1.0)
    H=np.array([[0.5,-0.5]])
    state=NetworkState("INTACT",("line12",),H,np.array([0.0]),np.array([50.0]))
    return PhysicalCase(
        time_grid=grid,
        participant_ids=("p1",),
        bus_ids=("b1","b2"),
        capacities_mw=np.array([100.0]),
        requests_mw=np.array([[100.0],[100.0]]),
        contract_mask=np.ones((2,1),bool),
        eligibility_mask=np.ones((2,1),bool),
        p_background_mw=np.zeros((2,2)),
        A=np.array([[1.0],[-1.0]]),
        network_states=(state,),
        participant_directions=("import",),
        participant_set_id="TEST1",
        contingency_set_id="INTACT_ONLY",
        balancing_key_hash="testkey",
        source_hashes={"test":"abc"},
        declared_case_id="SIMPLE",
    )


def pytest_collection_modifyitems(items):
    """Expose required native dependencies separately from portable/reference tests."""
    native_names={
        'test_active_coordinate_compression_final__tiny_gurobi_compressed_model_matches_full_model',
        'test_release_io_and_ac__native_AC_reconstruction_retains_controls_and_solves',
    }
    for item in items:
        native=item.location[0].replace('\\','/').endswith('test_native.py') or item.originalname in native_names
        item.user_properties.append(('dependency_scope','native_required' if native else 'portable_reference'))
