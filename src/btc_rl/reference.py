"""
reference.py — independent, vectorised accounting used to cross-check the
environment in tests.  Written deliberately without reusing ``env.py`` code.

Conventions (identical to the frozen study design (docs/methodology.md)): decision at row t, fill at
open[t+1], mark at open[t+2]; equity is multiplied by (1 - c) per leg at the
fill and then by (1 + p * (open[t+2] / open[t+1] - 1)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import CostConfig


def open_to_open_simple_returns(opens: np.ndarray, t_start: int, n_steps: int) -> np.ndarray:
    """r_k = open[t_start + k + 2] / open[t_start + k + 1] - 1 for k in 0..n_steps-1."""
    o = np.asarray(opens, dtype=np.float64)
    fills = o[t_start + 1 : t_start + 1 + n_steps]
    marks = o[t_start + 2 : t_start + 2 + n_steps]
    return marks / fills - 1.0


def position_path_equity(
    opens: np.ndarray,
    t_start: int,
    positions: np.ndarray,
    cost: CostConfig,
    initial_equity: float = 1.0,
) -> np.ndarray:
    """
    Equity after each step for a sequence of target positions chosen at
    decisions t_start, t_start+1, ...  (positions[k] = p_{t_start+k+1}).
    """
    p = np.asarray(positions, dtype=np.float64)
    n = p.shape[0]
    r = open_to_open_simple_returns(opens, t_start, n)
    prev = np.concatenate([[0.0], p[:-1]])
    legs = np.abs(p - prev)
    step_factor = (cost.leg_factor ** legs) * (1.0 + p * r)
    return initial_equity * np.cumprod(step_factor)


def buy_and_hold_equity(
    opens: np.ndarray, t_start: int, n_steps: int, cost: CostConfig, initial_equity: float = 1.0
) -> np.ndarray:
    """Analytic path: one entry leg, then open[t_start+k+2] / open[t_start+1]."""
    o = np.asarray(opens, dtype=np.float64)
    entry = o[t_start + 1]
    marks = o[t_start + 2 : t_start + 2 + n_steps]
    return initial_equity * cost.leg_factor * marks / entry


def max_drawdown(equity: np.ndarray, initial_equity: float = 1.0) -> float:
    """Minimum of equity / running peak - 1, with the initial equity as the first peak."""
    e = np.concatenate([[initial_equity], np.asarray(equity, dtype=np.float64)])
    peak = np.maximum.accumulate(e)
    return float(np.min(e / peak - 1.0))


def as_frame(dates: pd.DatetimeIndex, equity: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"equity": equity}, index=dates[: len(equity)])
