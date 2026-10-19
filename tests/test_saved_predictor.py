"""Retain the class path and prediction dispatch required by saved flat cards."""

import pickle

import numpy as np
import pandas as pd
import pytest

from tpwrs.features import PredictorBundle
from tpwrs.storage import ProvenanceError


class ColumnModel:
    def predict(self, frame):
        return frame["score"].to_numpy()


def test_saved_predictor_round_trip_and_directional_dispatch():
    bundle = PredictorBundle(
        "retained", ("score",), ("score",), (), {"import": ColumnModel(), "export": ColumnModel()}
    )
    restored = pickle.loads(pickle.dumps(bundle))
    frame = pd.DataFrame({"direction": ["export", "import"], "score": [4.0, 7.0]})
    np.testing.assert_array_equal(restored.predict(frame), [4.0, 7.0])
    with pytest.raises(ProvenanceError, match="direction"):
        restored.predict(frame.drop(columns="direction"))
    with pytest.raises(ProvenanceError, match="Unknown"):
        restored.predict(frame.assign(direction="unknown"))
