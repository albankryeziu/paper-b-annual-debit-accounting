"""Saved predictor compatibility for the original flat-card diagnostic; no fitting.

See docs/CODE_GUIDE.md for the retained function scope.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tpwrs.storage import ProvenanceError


@dataclass
class PredictorBundle:
    family: str
    feature_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    model: object

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if isinstance(self.model, dict):
            if "direction" not in frame:
                raise ProvenanceError("Directional predictor requires direction IDs.")
            result = np.empty(len(frame), float)
            for direction in frame.direction.astype(str).unique():
                if direction not in self.model:
                    raise ProvenanceError("Unknown prediction direction: " + direction)
                mask = frame.direction.astype(str).eq(direction).to_numpy()
                result[mask] = self.model[direction].predict(
                    frame.loc[mask, list(self.feature_columns)]
                )
            return result
        return np.asarray(self.model.predict(frame[list(self.feature_columns)]), dtype=float)
