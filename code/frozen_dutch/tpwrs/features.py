from __future__ import annotations


# ============================================================================
# tpwrs_v3.features_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tpwrs.storage import ProvenanceError
from tpwrs.storage import sha256_json


D1_HOUR_LOCAL = 7
D1_TZ = ZoneInfo("Europe/Amsterdam")

# These classes do not pretend that a deterministic/static control has an
# archived forecast issue timestamp.
NONVINTAGED_DEPLOYABLE_CLASSES = {
    "DETERMINISTIC_CALENDAR",
    "STATIC_CONTRACT",
    "PREDECLARED_CONTROLLED_INPUT",
}


@dataclass(frozen=True)
class FeatureProvenanceRecord:
    feature: str
    source_id: str
    source_hash: str
    issue_timestamp: str | None
    delivery_period: str
    timezone: str
    forecast_or_realization: str
    revision_policy: str
    d1_pass: bool
    role: str = "deployable"
    availability_class: str = "ARCHIVED_FORECAST"

    def validate(self) -> None:
        if not self.feature or not self.source_id or not self.source_hash:
            raise ProvenanceError("Feature provenance requires feature/source/hash.")
        if self.role != "deployable":
            return
        if not self.d1_pass:
            raise ProvenanceError(f"{self.feature}: deployable feature failed D-1 gate.")
        if self.forecast_or_realization.lower() == "realization":
            raise ProvenanceError(f"{self.feature}: realized data cannot be deployed D-1.")

        availability = str(self.availability_class).upper()
        if availability in NONVINTAGED_DEPLOYABLE_CLASSES:
            if self.issue_timestamp not in (None, ""):
                raise ProvenanceError(
                    f"{self.feature}: {availability} must not carry a fictitious "
                    "forecast issue timestamp."
                )
            return

        if availability != "ARCHIVED_FORECAST":
            raise ProvenanceError(
                f"{self.feature}: unsupported availability class {availability!r}."
            )
        if not self.issue_timestamp:
            raise ProvenanceError(
                f"{self.feature}: deployable time-varying forecast lacks issue/vintage time."
            )
        # Parseability and timezone awareness are part of the provenance gate.
        issue = pd.Timestamp(self.issue_timestamp)
        if issue.tzinfo is None:
            raise ProvenanceError(
                f"{self.feature}: archived issue timestamp must be timezone-aware."
            )


def d1_cutoff(delivery_timestamp: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(delivery_timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Europe/Amsterdam")
    else:
        ts = ts.tz_convert("Europe/Amsterdam")
    from datetime import timedelta
    previous_day = ts.date() - timedelta(days=1)
    return pd.Timestamp(
        year=previous_day.year,
        month=previous_day.month,
        day=previous_day.day,
        hour=D1_HOUR_LOCAL,
        tz="Europe/Amsterdam",
    )


def certify_issue_time(
    issue_timestamp: pd.Timestamp,
    delivery_timestamp: pd.Timestamp,
) -> bool:
    issue = pd.Timestamp(issue_timestamp)
    if issue.tzinfo is None:
        raise ProvenanceError("Issue timestamp must be timezone-aware.")
    return bool(issue.tz_convert("Europe/Amsterdam") <= d1_cutoff(delivery_timestamp))


@dataclass(frozen=True)
class FeatureSchema:
    feature_set_id: str
    columns: tuple[str, ...]
    provenance_hash: str

    @property
    def hash(self) -> str:
        return sha256_json(asdict(self))


F0_BASE = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "capacity_mw",
    "request_fraction",
    "national_load_forecast_mw",
    "renewable_availability_forecast",
    "outage_availability",
    "zone",
    "technology",
)
F0_CORE = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "capacity_mw",
    "request_fraction",
    "zone",
    "technology",
)
F0_OPTIONAL_FORECAST = (
    "national_load_forecast_mw",
    "renewable_availability_forecast",
    "outage_availability",
)
F1_EXTRA = (
    "directional_provincial_residual_load_forecast_mw",
    "directional_national_residual_load_forecast_mw",
)
F2_EXTRA = ("directional_provincial_minus_national_normalized",)


def build_feature_schema(
    feature_set_id: str,
    provenance: Iterable[FeatureProvenanceRecord],
    available_columns: Iterable[str],
) -> FeatureSchema:
    records = {record.feature: record for record in provenance}
    columns = set(map(str, available_columns))
    f0 = F0_CORE
    if feature_set_id == "F0":
        requested = f0
    elif feature_set_id == "F1":
        requested = f0 + F1_EXTRA
    elif feature_set_id == "F2":
        requested = f0 + F1_EXTRA + F2_EXTRA
    else:
        raise ProvenanceError(f"Unknown feature set {feature_set_id!r}.")

    missing = [feature for feature in requested if feature not in columns]
    if missing:
        raise ProvenanceError(
            f"{feature_set_id}: required columns are unavailable: {missing}"
        )
    for feature in requested:
        if feature not in records:
            raise ProvenanceError(f"{feature_set_id}: {feature} lacks provenance.")
        records[feature].validate()

    return FeatureSchema(
        feature_set_id,
        tuple(requested),
        sha256_json([asdict(records[feature]) for feature in requested]),
    )


def provenance_from_csv(path: str | Path) -> list[FeatureProvenanceRecord]:
    frame = pd.read_csv(Path(path))
    required = [
        "feature",
        "source_id",
        "source_hash",
        "issue_timestamp",
        "delivery_period",
        "timezone",
        "forecast_or_realization",
        "revision_policy",
        "d1_pass",
        "role",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ProvenanceError(f"Provenance file missing columns: {missing}")

    if "availability_class" not in frame.columns:
        frame["availability_class"] = "ARCHIVED_FORECAST"

    records: list[FeatureProvenanceRecord] = []
    for row in frame.to_dict("records"):
        row["d1_pass"] = str(row["d1_pass"]).strip().lower() in {
            "1",
            "true",
            "yes",
            "pass",
        }
        if pd.isna(row["issue_timestamp"]):
            row["issue_timestamp"] = None
        records.append(
            FeatureProvenanceRecord(
                **{
                    key: row[key]
                    for key in required + ["availability_class"]
                }
            )
        )
    return records


def directional_residual_features(
    panel: pd.DataFrame,
    participant_map: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "timestamp",
        "province",
        "demand_mw",
        "pv_mw",
        "wind_onshore_mw",
        "wind_offshore_mw",
    }
    if not required.issubset(panel.columns):
        raise ProvenanceError(
            f"Residual-load panel missing {sorted(required - set(panel.columns))}"
        )

    work = panel.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["residual_load_mw"] = (
        work["demand_mw"]
        - work["pv_mw"]
        - work["wind_onshore_mw"]
        - work["wind_offshore_mw"]
    )
    national = (
        work.groupby("timestamp", as_index=False)["residual_load_mw"]
        .sum()
        .rename(columns={"residual_load_mw": "national_residual_load_mw"})
    )
    work = work.merge(national, on="timestamp", validate="many_to_one")

    mapping = participant_map[
        ["participant_id", "province", "direction"]
    ].copy()
    out = work.merge(mapping, on="province", how="inner")
    is_import = out["direction"].astype(str).str.lower().eq("import")
    out["directional_provincial_residual_load_forecast_mw"] = np.where(
        is_import,
        np.maximum(out["residual_load_mw"], 0.0),
        np.maximum(-out["residual_load_mw"], 0.0),
    )
    out["directional_national_residual_load_forecast_mw"] = np.where(
        is_import,
        np.maximum(out["national_residual_load_mw"], 0.0),
        np.maximum(-out["national_residual_load_mw"], 0.0),
    )
    denominator = np.maximum(np.abs(out["national_residual_load_mw"]), 1.0)
    out["directional_provincial_minus_national_normalized"] = (
        out["directional_provincial_residual_load_forecast_mw"]
        - out["directional_national_residual_load_forecast_mw"]
    ) / denominator
    return out


# ============================================================================
# tpwrs_v3.predictors_v3 (consolidated; see SOURCE_MAP.json)
# ============================================================================

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass
class PredictorBundle:
    family: str
    feature_columns: tuple[str,...]
    numeric_columns: tuple[str,...]
    categorical_columns: tuple[str,...]
    model: object

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if isinstance(self.model,dict):
            if 'direction' not in frame:
                raise ProvenanceError('Directional predictor requires direction IDs.')
            result=np.empty(len(frame),float)
            for direction in frame.direction.astype(str).unique():
                if direction not in self.model:
                    raise ProvenanceError('Unknown prediction direction: '+direction)
                mask=frame.direction.astype(str).eq(direction).to_numpy()
                result[mask]=self.model[direction].predict(frame.loc[mask,list(self.feature_columns)])
            return result
        return np.asarray(self.model.predict(frame[list(self.feature_columns)]),dtype=float)


def _split(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[list[str],list[str]]:
    numeric=[]; categorical=[]
    for c in columns:
        (numeric if pd.api.types.is_numeric_dtype(frame[c]) else categorical).append(c)
    return numeric,categorical


def fit_predictor(family: str, frame: pd.DataFrame, feature_columns: Sequence[str], target: np.ndarray) -> PredictorBundle:
    columns=tuple(feature_columns); numeric,categorical=_split(frame,columns)
    if set(columns)!=set(F0_CORE):
        raise ProvenanceError('Production predictor must use the frozen F0 feature family exactly.')
    if 'year' in frame and not frame.year.isin((2025,2026,2027)).all():
        raise ProvenanceError('Predictor fitting may use only 2025--2027.')
    if 'direction' not in frame:
        raise ProvenanceError('Predictor fitting requires direction classes.')
    if family=="ridge":
        pre=ColumnTransformer([
            ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
            ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical),
        ],remainder="drop")
        model=Pipeline([("pre",pre),("reg",Ridge(alpha=2.0))])
    elif family=="histgb":
        pre=ColumnTransformer([
            ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
            ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ord",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),categorical),
        ],remainder="drop")
        model=Pipeline([("pre",pre),("reg",HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=15,max_depth=3,min_samples_leaf=40,l2_regularization=1.0,learning_rate=0.08,random_state=118))])
    else: raise ValueError(f"Unknown predictor family {family!r}.")
    from sklearn.base import clone
    models={}
    for direction in sorted(frame.direction.astype(str).unique()):
        mask=frame.direction.astype(str).eq(direction).to_numpy()
        fitted=clone(model)
        fitted.fit(frame.loc[mask,list(columns)],np.asarray(target,float)[mask])
        models[direction]=fitted
    return PredictorBundle(family,columns,tuple(numeric),tuple(categorical),models)
