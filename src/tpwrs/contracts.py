"""Validated time grids, physical cases, allowances, policies and certificates.

See docs/CODE_GUIDE.md for the manuscript mapping and execution context.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from tpwrs.storage import ContractError
from tpwrs.storage import sha256_json


@dataclass(frozen=True)
class TimeGrid:
    timestamps_utc: tuple[str, ...]
    delta_t_hours: tuple[float, ...]
    timezone: str = "UTC"
    expected_full_year: bool = False

    def __post_init__(self) -> None:
        if not self.timestamps_utc:
            raise ContractError("TimeGrid cannot be empty.")
        if len(self.timestamps_utc) != len(self.delta_t_hours):
            raise ContractError("timestamps and delta_t_hours must have identical length.")
        dt = np.asarray(self.delta_t_hours, dtype=float)
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
            raise ContractError("Every interval duration must be finite and positive.")
        idx = self.index
        if idx.has_duplicates or not idx.is_monotonic_increasing:
            raise ContractError("TimeGrid timestamps must be unique and increasing.")
        if len(idx) > 1:
            gaps = np.diff(idx.asi8) / 3.6e12
            if np.any(gaps <= 0):
                raise ContractError("TimeGrid chronology is invalid.")
            # The timestamp marks interval start.  Consecutive gaps must equal the
            # duration of all but the final interval.
            if not np.allclose(gaps, dt[:-1], rtol=0.0, atol=1e-9):
                raise ContractError(
                    "Timestamp adjacency does not match explicit interval durations."
                )
        if self.expected_full_year:
            year = int(idx[0].year)
            if np.any(idx.year != year):
                raise ContractError("A full-year TimeGrid cannot cross calendar years.")
            expected = 8784.0 if pd.Timestamp(f"{year}-12-31").is_leap_year else 8760.0
            if abs(float(dt.sum()) - expected) > 1e-7:
                raise ContractError(f"Expected full year {expected:g} h, got {dt.sum():g} h.")

    @classmethod
    def from_index(
        cls,
        timestamps: Iterable[object],
        delta_t_hours: Iterable[float] | float,
        *,
        timezone: str = "UTC",
        expected_full_year: bool = False,
    ) -> "TimeGrid":
        idx = pd.DatetimeIndex(pd.to_datetime(list(timestamps), utc=True))
        if np.isscalar(delta_t_hours):
            dt = np.full(len(idx), float(delta_t_hours), dtype=float)
        else:
            dt = np.asarray(list(delta_t_hours), dtype=float)
        return cls(
            tuple(ts.isoformat() for ts in idx),
            tuple(float(v) for v in dt),
            timezone,
            expected_full_year,
        )

    @classmethod
    def uniform_hourly(cls, year: int, *, expected_full_year: bool = True) -> "TimeGrid":
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        idx = pd.date_range(start, end, inclusive="left", freq="1h")
        return cls.from_index(idx, 1.0, expected_full_year=expected_full_year)

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(self.timestamps_utc, utc=True))

    @property
    def delta(self) -> np.ndarray:
        return np.asarray(self.delta_t_hours, dtype=float)

    @property
    def n_intervals(self) -> int:
        return len(self.timestamps_utc)

    @property
    def uniform_delta(self) -> float | None:
        dt = self.delta
        return float(dt[0]) if np.allclose(dt, dt[0], rtol=0.0, atol=1e-12) else None

    @property
    def total_hours(self) -> float:
        return float(self.delta.sum())

    @property
    def grid_id(self) -> str:
        return (
            "time_"
            + sha256_json({"timestamps": self.timestamps_utc, "delta": self.delta_t_hours})[:16]
        )

    @property
    def hash(self) -> str:
        return sha256_json(
            {
                "timestamps": self.timestamps_utc,
                "delta": self.delta_t_hours,
                "timezone": self.timezone,
            }
        )

    def slice(self, start: int, stop: int) -> "TimeGrid":
        return TimeGrid(
            self.timestamps_utc[start:stop], self.delta_t_hours[start:stop], self.timezone, False
        )


from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np

from tpwrs.storage import ContractError
from tpwrs.storage import sha256_json

Rule = Literal["M1", "M1_LP", "M2", "M3", "M2_MATCHED"]
Mode = Literal["annual", "rolling"]
AllowanceSource = Literal["nominal_contract_share", "m3_realized_physical_fde", "explicit"]


@dataclass(frozen=True)
class SolverProfile:
    name: str = "scientific_default"
    backend: str = "auto"
    time_limit_s: float | None = None
    mip_gap: float = 1e-8
    feasibility_tol: float = 1e-8
    presolve: bool = True
    output_flag: bool = False
    threads: int = 1
    seed: int = 0
    lp_method: str = "auto"

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "gurobi", "scipy"}:
            raise ContractError("Solver backend must be auto, gurobi, or scipy.")
        if self.time_limit_s is not None and (
            not np.isfinite(self.time_limit_s) or self.time_limit_s <= 0
        ):
            raise ContractError("time_limit_s must be positive.")
        if (
            not np.isfinite(self.mip_gap)
            or not np.isfinite(self.feasibility_tol)
            or self.mip_gap < 0
            or self.feasibility_tol <= 0
        ):
            raise ContractError("Invalid solver tolerance.")
        if self.threads < 1 or self.seed < 0:
            raise ContractError("Solver threads must be positive and seed nonnegative.")
        if self.lp_method not in {"auto", "primal_simplex", "dual_simplex", "barrier"}:
            raise ContractError("Unsupported LP method.")


@dataclass(frozen=True)
class AllowanceSpec:
    participant_ids: tuple[str, ...]
    values_hours: tuple[float, ...]
    source: AllowanceSource
    nominal_rho: float | None = None
    source_solution_hash: str | None = None
    contract_mask_hash: str | None = None

    def __post_init__(self) -> None:
        if len(self.participant_ids) != len(self.values_hours) or not self.participant_ids:
            raise ContractError("Allowance participant IDs and values must align.")
        values = np.asarray(self.values_hours, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < -1e-12):
            raise ContractError("Allowance values must be finite and nonnegative.")
        if len(set(self.participant_ids)) != len(self.participant_ids):
            raise ContractError("Allowance participant IDs must be unique.")
        if self.source == "nominal_contract_share" and self.nominal_rho is None:
            raise ContractError("Nominal allowance requires nominal_rho.")
        if self.nominal_rho is not None and not (0 < self.nominal_rho <= 1):
            raise ContractError("nominal_rho must lie in (0,1].")
        if self.source == "m3_realized_physical_fde" and not self.source_solution_hash:
            raise ContractError("M2-MATCHED allowance must record its M3 source solution hash.")

    @classmethod
    def from_nominal(
        cls,
        participant_ids: Sequence[str],
        contract_mask: np.ndarray,
        time_grid: TimeGrid,
        rho: float,
    ) -> "AllowanceSpec":
        mask = np.asarray(contract_mask, dtype=bool)
        if mask.shape != (time_grid.n_intervals, len(participant_ids)):
            raise ContractError("contract_mask shape does not match time/participants.")
        if not (0 < rho <= 1):
            raise ContractError("rho must lie in (0,1].")
        dt = time_grid.delta
        uniform = time_grid.uniform_delta
        if uniform is not None:
            counts = mask.sum(axis=0).astype(int)
            k = np.floor(rho * counts + 1e-12).astype(int)
            values = k.astype(float) * uniform
        else:
            # General duration-consistent analogue.  The final Dutch case is hourly,
            # so theorem-comparison cases should use the uniform branch above.
            values = rho * np.sum(dt[:, None] * mask, axis=0)
        import hashlib

        h = hashlib.sha256(np.ascontiguousarray(mask.astype(np.uint8)).tobytes()).hexdigest()
        return cls(
            tuple(map(str, participant_ids)),
            tuple(map(float, values)),
            "nominal_contract_share",
            float(rho),
            None,
            h,
        )

    @classmethod
    def explicit(
        cls,
        participant_ids: Sequence[str],
        values_hours: Sequence[float],
        *,
        source: AllowanceSource = "explicit",
        nominal_rho: float | None = None,
        source_solution_hash: str | None = None,
    ) -> "AllowanceSpec":
        return cls(
            tuple(map(str, participant_ids)),
            tuple(map(float, values_hours)),
            source,
            nominal_rho,
            source_solution_hash,
        )

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.values_hours, dtype=float)

    @property
    def hash(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class CardSpec:
    card_id: str
    allowance_hash: str
    feature_set_id: str
    model_family: str
    thresholds: Mapping[str, tuple[float, float]]
    rates: Mapping[str, tuple[float, float, float]]
    q_bounds: tuple[float, float] = (0.5, 1.5)
    lambda_q: float = 1e-3
    direction_fallbacks: Mapping[str, str] = field(default_factory=dict)
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    fit_years: tuple[int, ...] = (2025, 2026, 2027)
    validation_year: int = 2028
    heldout_access_audit: str = "PASS"

    def __post_init__(self) -> None:
        qlo, qhi = self.q_bounds
        if not (0 < qlo <= 1 <= qhi):
            raise ContractError("Card q bounds must be positive and contain one.")
        if self.lambda_q < 0:
            raise ContractError("lambda_q must be nonnegative.")
        if self.validation_year in self.fit_years:
            raise ContractError("Validation year cannot be a fitting year.")
        if self.heldout_access_audit != "PASS":
            raise ContractError(
                "A card cannot be deployed without a passing held-out access audit."
            )
        for direction, q in self.rates.items():
            if len(q) != 3 or not (q[0] + 1e-10 >= q[1] >= q[2] - 1e-10):
                raise ContractError(f"{direction}: rates must be ordered q1>=q2>=q3.")
            if min(q) < qlo - 1e-9 or max(q) > qhi + 1e-9:
                raise ContractError(f"{direction}: rate outside frozen bounds.")

    @property
    def hash(self) -> str:
        return sha256_json(
            {
                "card_id": self.card_id,
                "allowance_hash": self.allowance_hash,
                "feature_set_id": self.feature_set_id,
                "model_family": self.model_family,
                "thresholds": dict(self.thresholds),
                "rates": dict(self.rates),
                "q_bounds": self.q_bounds,
                "lambda_q": self.lambda_q,
                "fallbacks": dict(self.direction_fallbacks),
                "sources": dict(self.source_hashes),
                "fit_years": self.fit_years,
                "validation_year": self.validation_year,
            }
        )


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    physical_case_id: str
    physical_case_hash: str
    time_grid_id: str
    rule: Rule
    allowance: AllowanceSpec
    mode: Mode = "annual"
    card_id: str | None = None
    card_hash: str | None = None
    feature_set_id: str | None = None
    information_level: str = "FROZEN_INPUT"
    lookahead_hours: int | None = None
    implementation_hours: int | None = None
    solver_profile: SolverProfile = field(default_factory=SolverProfile)
    epsilon_s_abs: float = 1e-6
    epsilon_s_rel: float = 1e-9
    epsilon_e_abs: float = 1e-6
    epsilon_e_rel: float = 1e-9
    source_root_id: str = "FINAL_DUTCH_CASE"
    result_root_id: str = "UNSPECIFIED"
    rate_matrix_hash: str | None = None
    implementation_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id or not self.physical_case_id or not self.physical_case_hash:
            raise ContractError("ExperimentCase requires case and physical-case identity.")
        if self.rule not in {"M1", "M1_LP", "M2", "M3", "M2_MATCHED"}:
            raise ContractError("Unknown accounting rule.")
        if self.mode not in {"annual", "rolling"}:
            raise ContractError("Unknown operation mode.")
        if self.mode == "rolling":
            if not self.lookahead_hours or not self.implementation_hours:
                raise ContractError("Rolling cases require lookahead and implementation durations.")
            if self.implementation_hours > self.lookahead_hours:
                raise ContractError("Rolling implementation block cannot exceed look-ahead.")
        if self.rule == "M3" and (not self.card_id or not self.card_hash):
            raise ContractError("M3 case requires an immutable card ID/hash.")
        if self.rule != "M3" and self.card_id is not None and self.rule != "M2_MATCHED":
            raise ContractError("Only M3/M2-MATCHED cases may carry a card reference.")
        if (
            not np.isfinite(
                [self.epsilon_s_abs, self.epsilon_s_rel, self.epsilon_e_abs, self.epsilon_e_rel]
            ).all()
            or min(self.epsilon_s_abs, self.epsilon_s_rel, self.epsilon_e_abs, self.epsilon_e_rel)
            < 0
        ):
            raise ContractError("Lexicographic tolerances cannot be negative.")
        if not self.source_root_id or not self.result_root_id:
            raise ContractError("ExperimentCase requires logical source/result root identifiers.")

    @property
    def hash(self) -> str:
        payload = asdict(self)
        payload["allowance"] = asdict(self.allowance)
        payload["solver_profile"] = asdict(self.solver_profile)
        return sha256_json(payload)

    def stage1_tolerance(self, value: float) -> float:
        return self.epsilon_s_abs + self.epsilon_s_rel * max(1.0, abs(float(value)))

    def stage2_tolerance(self, value: float) -> float:
        return self.epsilon_e_abs + self.epsilon_e_rel * max(1.0, abs(float(value)))


from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from tpwrs.storage import sha256_array, sha256_json


def readonly(array: np.ndarray, dtype=float) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class StageCertificate:
    stage: str
    status: str
    objective_value: float
    runtime_s: float
    best_bound: float | None = None
    relative_gap: float | None = None
    solver_message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySolution:
    case_id: str
    experiment_hash: str
    physical_case_id: str
    physical_case_hash: str
    rule: str
    allowance_hash: str
    x_mw: np.ndarray
    u: np.ndarray | None
    q: np.ndarray | None
    stage1: StageCertificate
    stage2: StageCertificate
    stage3: StageCertificate | None
    s_value_mw_h_row: float
    e_value_mwh: float
    m2_budget_duals: Mapping[str, float] = field(default_factory=dict)
    closure_rounds: int = 0
    active_row_count: int = 0
    replay_summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_mw", readonly(self.x_mw))
        if self.u is not None:
            object.__setattr__(self, "u", readonly(self.u))
        if self.q is not None:
            object.__setattr__(self, "q", readonly(self.q))

    @property
    def solution_hash(self) -> str:
        return sha256_json(
            {
                "case": self.case_id,
                "experiment": self.experiment_hash,
                "physical": self.physical_case_hash,
                "rule": self.rule,
                "allowance": self.allowance_hash,
                "x": sha256_array(self.x_mw),
                "u": None if self.u is None else sha256_array(self.u),
                "q": None if self.q is None else sha256_array(self.q),
                "S": self.s_value_mw_h_row,
                "E": self.e_value_mwh,
            }
        )


@dataclass(frozen=True)
class ReplayReport:
    passed: bool
    s_value_mw_h_row: float
    e_value_mwh: float
    max_row_violation_mw: float
    max_balance_residual_mw: float
    max_x_bound_violation_mw: float
    max_budget_violation_h: float
    physical_fde_hours: tuple[float, ...]
    weighted_debit_hours: tuple[float, ...] | None
    touched_duration_hours: tuple[float, ...] | None
    violated_mask: np.ndarray
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "violated_mask", readonly(self.violated_mask, dtype=bool))


from dataclasses import dataclass, field
from functools import cached_property
from typing import Mapping, Sequence

import numpy as np

from tpwrs.storage import ContractError
from tpwrs.storage import combined_hash, sha256_array, sha256_json


@dataclass(frozen=True)
class NetworkState:
    state_id: str
    row_ids: tuple[str, ...]
    H: np.ndarray
    h_mw: np.ndarray
    ratings_mw: np.ndarray
    state_kind: str = "INTACT"
    outaged_element_id: str | None = None
    source_hash: str | None = None

    def __post_init__(self) -> None:
        H = readonly(self.H)
        h = readonly(self.h_mw)
        ratings = readonly(self.ratings_mw)
        if H.ndim != 2 or h.shape != (H.shape[0],) or ratings.shape != (H.shape[0],):
            raise ContractError(f"{self.state_id}: H/h/rating dimensions do not align.")
        if len(self.row_ids) != H.shape[0] or len(set(self.row_ids)) != len(self.row_ids):
            raise ContractError(f"{self.state_id}: invalid row IDs.")
        if (
            np.any(~np.isfinite(H))
            or np.any(~np.isfinite(h))
            or np.any(~np.isfinite(ratings))
            or np.any(ratings <= 0)
        ):
            raise ContractError(
                f"{self.state_id}: network arrays must be finite with positive ratings."
            )
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "h_mw", h)
        object.__setattr__(self, "ratings_mw", ratings)

    @property
    def hash(self) -> str:
        return sha256_json(
            {
                "state_id": self.state_id,
                "row_ids": self.row_ids,
                "H": sha256_array(self.H),
                "h": sha256_array(self.h_mw),
                "ratings": sha256_array(self.ratings_mw),
                "kind": self.state_kind,
                "outage": self.outaged_element_id,
            }
        )


@dataclass(frozen=True)
class PhysicalCase:
    time_grid: TimeGrid
    participant_ids: tuple[str, ...]
    bus_ids: tuple[str, ...]
    capacities_mw: np.ndarray
    requests_mw: np.ndarray
    contract_mask: np.ndarray
    eligibility_mask: np.ndarray
    p_background_mw: np.ndarray
    A: np.ndarray
    network_states: tuple[NetworkState, ...]
    participant_directions: tuple[str, ...] | None = None
    participant_set_id: str = "UNSPECIFIED"
    contingency_set_id: str = "UNSPECIFIED"
    balancing_key_hash: str = "UNSPECIFIED"
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    declared_case_id: str | None = None
    balance_tolerance_mw: float = 1e-7

    def __post_init__(self) -> None:
        T = self.time_grid.n_intervals
        N = len(self.participant_ids)
        B = len(self.bus_ids)
        cap = readonly(self.capacities_mw)
        req = readonly(self.requests_mw)
        ctr = readonly(self.contract_mask, dtype=bool)
        elig = readonly(self.eligibility_mask, dtype=bool)
        pbg = readonly(self.p_background_mw)
        A = readonly(self.A)
        if len(set(self.participant_ids)) != N or len(set(self.bus_ids)) != B:
            raise ContractError("Participant and bus IDs must be unique.")
        if cap.shape != (N,) or np.any(~np.isfinite(cap)) or np.any(cap <= 0):
            raise ContractError("capacities_mw must be positive and participant aligned.")
        if req.shape != (T, N) or np.any(~np.isfinite(req)) or np.any(req < -1e-10):
            raise ContractError(
                "requests_mw must have shape (time, participant) and be nonnegative."
            )
        if ctr.shape != (T, N) or elig.shape != (T, N):
            raise ContractError("contract and eligibility masks must align with time/participants.")
        if np.any(elig & ~ctr):
            raise ContractError(
                "Eligibility mask must be a subset of the independent contract-duration mask."
            )
        if pbg.shape != (T, B) or A.shape != (B, N):
            raise ContractError(
                "Background/A dimensions do not align with time/buses/participants."
            )
        if np.any(~np.isfinite(pbg)) or np.any(~np.isfinite(A)):
            raise ContractError("Background and participant-action arrays must be finite.")
        if not self.network_states:
            raise ContractError("At least one network state is required.")
        if len({state.state_id for state in self.network_states}) != len(self.network_states):
            raise ContractError("Network-state IDs must be unique.")
        for state in self.network_states:
            if state.H.shape[1] != B:
                raise ContractError(f"{state.state_id}: H columns must match bus IDs.")
        if np.max(np.abs(pbg.sum(axis=1))) > self.balance_tolerance_mw:
            raise ContractError("Background injections are not balanced for every interval.")
        if np.max(np.abs(A.sum(axis=0))) > self.balance_tolerance_mw:
            raise ContractError("Participant transaction columns are not balanced.")
        if self.participant_directions is not None:
            if len(self.participant_directions) != N or any(
                d not in {"import", "export"} for d in self.participant_directions
            ):
                raise ContractError(
                    "participant_directions must be import/export and participant aligned."
                )
        object.__setattr__(self, "capacities_mw", cap)
        object.__setattr__(self, "requests_mw", req)
        object.__setattr__(self, "contract_mask", ctr)
        object.__setattr__(self, "eligibility_mask", elig)
        object.__setattr__(self, "p_background_mw", pbg)
        object.__setattr__(self, "A", A)

    @property
    def n_time(self) -> int:
        return self.time_grid.n_intervals

    @property
    def n_participants(self) -> int:
        return len(self.participant_ids)

    @property
    def n_buses(self) -> int:
        return len(self.bus_ids)

    @cached_property
    def xbar_mw(self) -> np.ndarray:
        return readonly(
            self.eligibility_mask.astype(float)
            * np.minimum(self.capacities_mw[None, :], self.requests_mw)
        )

    @cached_property
    def materiality_floor_mw(self) -> np.ndarray:
        theta = np.maximum(1.0, 0.01 * self.capacities_mw)
        return readonly(np.minimum(self.xbar_mw, theta[None, :]))

    @cached_property
    def flat_H(self) -> np.ndarray:
        return readonly(np.vstack([state.H for state in self.network_states]))

    @cached_property
    def flat_h(self) -> np.ndarray:
        return readonly(np.concatenate([state.h_mw for state in self.network_states]))

    @cached_property
    def flat_ratings(self) -> np.ndarray:
        return readonly(np.concatenate([state.ratings_mw for state in self.network_states]))

    @cached_property
    def flat_G(self) -> np.ndarray:
        """Cached participant flow-effect map G = H A for all registered rows."""
        value = np.asarray(self.flat_H @ self.A, dtype=float)
        value.flags.writeable = False
        return value

    @cached_property
    def flat_row_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{state.state_id}::{rid}" for state in self.network_states for rid in state.row_ids
        )

    @cached_property
    def flat_state_ids(self) -> tuple[str, ...]:
        return tuple(state.state_id for state in self.network_states for _ in state.row_ids)

    @cached_property
    def n_flat_rows(self) -> int:
        return len(self.flat_row_ids)

    @cached_property
    def requested_state_mw(self) -> np.ndarray:
        return readonly(self.p_background_mw - self.requests_mw @ self.A.T)

    def post_state_mw(self, x_mw: np.ndarray) -> np.ndarray:
        x = np.asarray(x_mw, dtype=float)
        if x.shape != self.requests_mw.shape:
            raise ContractError("x_mw shape does not match the physical case.")
        return self.requested_state_mw + x @ self.A.T

    def flows_mw(self, x_mw: np.ndarray) -> np.ndarray:
        p = self.post_state_mw(x_mw)
        return p @ self.flat_H.T + self.flat_h[None, :]

    @cached_property
    def physical_case_hash(self) -> str:
        parts = {
            "time": self.time_grid.hash,
            "cap": sha256_array(self.capacities_mw),
            "req": sha256_array(self.requests_mw),
            "ctr": sha256_array(self.contract_mask.astype(np.uint8)),
            "elig": sha256_array(self.eligibility_mask.astype(np.uint8)),
            "pbg": sha256_array(self.p_background_mw),
            "A": sha256_array(self.A),
            "rows": sha256_json(self.flat_row_ids),
            "network": combined_hash({s.state_id: s.hash for s in self.network_states}),
            "balancing": self.balancing_key_hash,
            "participant_set": self.participant_set_id,
            "contingency_set": self.contingency_set_id,
            "sources": sha256_json(dict(sorted(self.source_hashes.items()))),
        }
        return combined_hash(parts)

    @cached_property
    def physical_case_id(self) -> str:
        return self.declared_case_id or ("physical_" + self.physical_case_hash[:16])

    def verify_experiment_identity(
        self, physical_case_id: str, physical_case_hash: str, time_grid_id: str
    ) -> None:
        if (
            physical_case_id != self.physical_case_id
            or physical_case_hash != self.physical_case_hash
        ):
            raise ContractError("Experiment physical-case ID/hash does not match loaded case.")
        if time_grid_id != self.time_grid.grid_id:
            raise ContractError("Experiment time-grid ID does not match loaded case.")

    def slice_time(self, start: int, stop: int) -> "PhysicalCase":
        return PhysicalCase(
            time_grid=self.time_grid.slice(start, stop),
            participant_ids=self.participant_ids,
            bus_ids=self.bus_ids,
            capacities_mw=self.capacities_mw,
            requests_mw=self.requests_mw[start:stop],
            contract_mask=self.contract_mask[start:stop],
            eligibility_mask=self.eligibility_mask[start:stop],
            p_background_mw=self.p_background_mw[start:stop],
            A=self.A,
            network_states=self.network_states,
            participant_directions=self.participant_directions,
            participant_set_id=self.participant_set_id,
            contingency_set_id=self.contingency_set_id,
            balancing_key_hash=self.balancing_key_hash,
            source_hashes=self.source_hashes,
            declared_case_id=None,
            balance_tolerance_mw=self.balance_tolerance_mw,
        )


from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CampaignCaseSpec:
    case_id: str
    rho: float
    rule: str
    mode: str
    matched: bool = False


PRIMARY_MATRIX = (
    CampaignCaseSpec("RHO15_M1LP", 0.15, "M1_LP", "annual"),
    CampaignCaseSpec("RHO15_M2", 0.15, "M2", "annual"),
    CampaignCaseSpec("RHO15_M3", 0.15, "M3", "annual"),
    CampaignCaseSpec("RHO15_ROLL_M1", 0.15, "M1", "rolling"),
    CampaignCaseSpec("RHO15_ROLL_M2", 0.15, "M2", "rolling"),
    CampaignCaseSpec("RHO15_ROLL_M3", 0.15, "M3", "rolling"),
    CampaignCaseSpec("RHO03_M1LP", 0.03, "M1_LP", "annual"),
    CampaignCaseSpec("RHO03_M2", 0.03, "M2", "annual"),
    CampaignCaseSpec("RHO03_M3", 0.03, "M3", "annual"),
    CampaignCaseSpec("RHO03_M2MATCHED", 0.03, "M2_MATCHED", "annual", True),
    CampaignCaseSpec("RHO03_ROLL_M1", 0.03, "M1", "rolling"),
    CampaignCaseSpec("RHO03_ROLL_M2", 0.03, "M2", "rolling"),
    CampaignCaseSpec("RHO03_ROLL_M3", 0.03, "M3", "rolling"),
)


def nominal_experiment(
    case: PhysicalCase,
    *,
    case_id: str,
    rule: str,
    rho: float,
    mode: str = "annual",
    card_id: str | None = None,
    card_hash: str | None = None,
    feature_set_id: str | None = None,
    solver_profile: SolverProfile | None = None,
    source_root_id: str = "FINAL_DUTCH_CASE",
    result_root_id: str = "UNSPECIFIED",
    implementation_hash: str | None = None,
) -> ExperimentCase:
    allowance = AllowanceSpec.from_nominal(
        case.participant_ids, case.contract_mask, case.time_grid, rho
    )
    return ExperimentCase(
        case_id,
        case.physical_case_id,
        case.physical_case_hash,
        case.time_grid.grid_id,
        rule,
        allowance,
        mode,
        card_id,
        card_hash,
        feature_set_id,
        "FROZEN_D1" if rule == "M3" else "FROZEN_INPUT",
        48 if mode == "rolling" else None,
        24 if mode == "rolling" else None,
        solver_profile or SolverProfile(),
        source_root_id=source_root_id,
        result_root_id=result_root_id,
        implementation_hash=implementation_hash,
    )
