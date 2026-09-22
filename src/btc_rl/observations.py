"""
observations.py — O1 minimal observation (docs/methodology.md).

O1 = [r_{t-L+1}, ..., r_t, position_t]  with r_k = log(close_k / close_{k-1}).

Causality by construction
-------------------------
``build_o1_observation`` receives *only* the close history up to and including
the decision bar (``closes_up_to_t``), so it cannot read any later row.  The
environment enforces this structurally by passing ``closes[: t + 1]``.

``position_t`` is the position held during bar t, i.e. the one chosen at
decision t-1 and filled at open[t].  It is known at close[t].

Bounds and scaling (F4)
-----------------------
``validate_ohlcv`` guarantees |log close return| <= MAX_ABS_LOG_RETURN on
frozen data.  The declared Box bound for the return components is therefore
MAX_ABS_LOG_RETURN / return_scale, so every observation built from validated
data satisfies ``observation_space.contains``.  Scaled returns are
additionally clipped to the declared bound as a defensive measure for
unvalidated (synthetic) frames.  A scale, if used, must be fitted on the
training split only (``fit_return_scale``).  Default 1.0 (raw log returns).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

MAX_ABS_LOG_RETURN = 1.0  # data-level bound enforced by data.validate_ohlcv (observed max on frozen data: 0.503)


@dataclass(frozen=True)
class ObservationConfig:
    lookback: int = 10
    return_scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.lookback, int) or isinstance(self.lookback, bool) or self.lookback < 1:
            raise ValueError("lookback must be a positive integer")
        if not isinstance(self.return_scale, (int, float)) or isinstance(self.return_scale, bool):
            raise ValueError("return_scale must be a number")
        if not math.isfinite(self.return_scale) or self.return_scale <= 0:
            raise ValueError("return_scale must be finite and positive")
        with np.errstate(over="ignore", under="ignore"):
            b32 = np.float32(MAX_ABS_LOG_RETURN / self.return_scale)
        # bound must be a finite, NORMAL float32 (denormals such as 1e-40 are degenerate)
        if not np.isfinite(b32) or b32 < np.finfo(np.float32).tiny:
            raise ValueError(
                f"return_scale={self.return_scale!r} yields a float32 observation bound of {b32!r}; "
                "bounds must be finite normal positive float32 values"
            )

    @property
    def size(self) -> int:
        return self.lookback + 1

    @property
    def return_bound(self) -> float:
        """Declared |bound| of each scaled return component."""
        return MAX_ABS_LOG_RETURN / self.return_scale


def observation_space(cfg: ObservationConfig) -> spaces.Box:
    b = np.float32(cfg.return_bound)
    if not np.isfinite(b) or b < np.finfo(np.float32).tiny:
        raise ValueError("observation bound is not a finite normal positive float32")
    low = np.concatenate([np.full(cfg.lookback, -b, dtype=np.float32), np.zeros(1, dtype=np.float32)])
    high = np.concatenate([np.full(cfg.lookback, b, dtype=np.float32), np.ones(1, dtype=np.float32)])
    return spaces.Box(low=low, high=high, dtype=np.float32)


def build_o1_observation(
    closes_up_to_t: np.ndarray, position: int, cfg: ObservationConfig
) -> np.ndarray:
    """
    Build O1 from the close history ending at the decision bar.

    ``closes_up_to_t`` must contain at least ``lookback + 1`` finite positive
    closes; only the last ``lookback + 1`` are used.
    """
    c = np.asarray(closes_up_to_t, dtype=np.float64)
    if c.ndim != 1 or c.shape[0] < cfg.lookback + 1:
        raise ValueError(f"need a 1-d array of at least {cfg.lookback + 1} closes")
    window = c[-(cfg.lookback + 1):]
    if not np.all(np.isfinite(window)) or np.any(window <= 0):
        raise ValueError("closes must be finite and positive")
    if position not in (0, 1):
        raise ValueError("position must be 0 or 1")
    rets = np.diff(np.log(window)) / cfg.return_scale
    b = cfg.return_bound
    rets = np.clip(rets, -b, b)
    obs = np.empty(cfg.size, dtype=np.float32)
    obs[: cfg.lookback] = rets
    obs[cfg.lookback] = float(position)
    # float32 rounding can exceed the float32 bound by 1 ulp; clip once more in float32
    np.clip(obs[: cfg.lookback], np.float32(-b), np.float32(b), out=obs[: cfg.lookback])
    if not np.all(np.isfinite(obs)):
        raise ValueError("non-finite observation produced")
    return obs


def fit_return_scale(train_closes: np.ndarray) -> float:
    """Population std of daily log returns on the training closes (train-only statistic)."""
    c = np.asarray(train_closes, dtype=np.float64)
    if c.ndim != 1 or c.shape[0] < 3 or not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("training closes must be a 1-d array of >= 3 finite positive prices")
    r = np.diff(np.log(c))
    s = float(np.std(r))
    if not math.isfinite(s) or s <= 0:
        raise ValueError("degenerate training returns")
    return s
