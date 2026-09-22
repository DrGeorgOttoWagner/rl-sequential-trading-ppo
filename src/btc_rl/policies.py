"""
policies.py — deterministic baseline policies (docs/methodology.md).

A policy exposes ``reset(seed)`` and ``act(obs, info) -> int``.  The
``info`` dict is the one returned by the environment for the current decision
date; policies must not touch the environment's price arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .data import REPO_ROOT
from .env import FLAT, LONG
from .splits import SplitWindow, usable_decision_index_range


class Policy(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int: ...


class AlwaysFlat:
    name = "always_flat"

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        return FLAT


class AlwaysLong:
    """Buy-and-hold: enters LONG at the first fill and never leaves."""

    name = "buy_and_hold"

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        return LONG


class RandomPolicy:
    """Bernoulli(p_long) action each step from an explicit, self-owned RNG."""

    def __init__(self, seed: int, p_long: float = 0.5) -> None:
        self.seed = int(seed)
        self.p_long = float(p_long)
        self.name = f"random_seed{self.seed}"
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
            self.name = f"random_seed{self.seed}"
        self._rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        return LONG if self._rng.random() < self.p_long else FLAT


# ── Replay of a committed position series (F7: strict) ────────────────────────

class ReplayError(RuntimeError):
    """A replay position series is incomplete, non-unique or non-binary."""


def validate_position_series(positions: pd.Series) -> pd.Series:
    """Unique, sorted, datetime-indexed, binary {0, 1} position series."""
    s = positions.copy()
    try:
        s.index = pd.to_datetime(s.index)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"position index is not datetime-like: {exc}") from exc
    if s.index.has_duplicates:
        raise ReplayError(f"duplicate dates in position series: {s.index[s.index.duplicated()][:5].tolist()}")
    vals = s.to_numpy()
    if len(vals) == 0:
        raise ReplayError("empty position series")
    if not np.all(np.isfinite(vals.astype(np.float64))):
        raise ReplayError("non-finite positions")
    if not np.all(np.isin(vals.astype(np.float64), (0.0, 1.0))):
        raise ReplayError("positions must be binary {0, 1} (LONG/FLAT only)")
    return s.astype(int).sort_index()


def replay_fill_dates(frame: pd.DataFrame, window: SplitWindow) -> pd.DatetimeIndex:
    """Held-bar (fill) dates the environment will ask a replay policy for on ``window``."""
    i0, last = usable_decision_index_range(frame, window)
    return frame.index[i0 + 1 : last + 2]


def validate_replay_coverage(positions: pd.Series, fill_dates: pd.DatetimeIndex) -> None:
    missing = fill_dates.difference(positions.index)
    if len(missing):
        raise ReplayError(
            f"position series lacks {len(missing)} required fill date(s); first {missing[0].date()}"
        )


class FixedPositionPolicy:
    """
    Replay a committed position series.

    ``positions`` is indexed by the HELD-BAR date (the bar during which the
    position is active, i.e. the fill date).  At decision date t the policy
    returns positions[t + 1 day].  A missing date raises ``ReplayError``;
    it never silently becomes FLAT.
    """

    def __init__(self, positions: pd.Series, name: str = "fixed_positions") -> None:
        self._positions = validate_position_series(positions)
        self.name = name

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        held_date = pd.Timestamp(info["state_instant_date"])  # fill instant open[t+1]
        try:
            return int(self._positions.loc[held_date])
        except KeyError as exc:
            raise ReplayError(f"no replay position for fill date {held_date.date()}") from exc


V5B_EQUITY_CURVE_RELATIVE = (
    "data/reference/long_cash_equity_curve_v5b.csv"
)
V5B_VARIANT_A_LABEL = "vA_baseline_ohlcv_23_consv_cost"


def v5b_artifact_path(path: Path | None = None) -> Path:
    return Path(path) if path is not None else REPO_ROOT / V5B_EQUITY_CURVE_RELATIVE


def load_v5b_positions(run_label: str = V5B_VARIANT_A_LABEL, path: Path | None = None) -> pd.Series:
    """
    Load and validate the committed v5B LONG/CASH position series for one run label.

    Provenance: ``reports/backtesting/long_cash_equity_curve_v5b.csv`` of the ML
    project; column ``position`` is the position held during the bar ``date``
    (ML convention: signal at close[t-1], fill at open[t]).  Only the POSITION
    series is reused: the replay is re-accounted inside ``BtcUsdtTradingEnv``
    (open-to-open, purge rule).  The historical v5B equity and metrics use
    intrabar open-to-close accounting and an end-of-test force-close, so they
    are historical context only and never directly comparable RL baselines.
    """
    p = v5b_artifact_path(path)
    if not p.exists():
        raise FileNotFoundError(f"mandatory ML replay artifact not found: {p}")
    df = pd.read_csv(p, parse_dates=["date"])
    for col in ("run_label", "date", "position"):
        if col not in df.columns:
            raise ReplayError(f"replay artifact missing column {col}")
    sub = df.loc[df["run_label"] == run_label]
    if sub.empty:
        raise ReplayError(f"run_label {run_label!r} not found in {p}")
    return validate_position_series(sub.set_index("date")["position"])


def make_variant_a_replay(
    frame: pd.DataFrame, window: SplitWindow, path: Path | None = None
) -> FixedPositionPolicy:
    """Variant A replay policy for ``window`` with complete-coverage validation."""
    positions = load_v5b_positions(V5B_VARIANT_A_LABEL, path)
    validate_replay_coverage(positions, replay_fill_dates(frame, window))
    return FixedPositionPolicy(positions, name="ml_variant_a_replay")
