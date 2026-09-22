"""
observations_o2.py — O2 engineered observation (docs/methodology.md).

O2 = [z(return_1d), z(return_7d), z(return_30d), z(volatility_14), z(volatility_30),
      z(ma_50_ratio), z(ma_200_ratio), z(rsi_14), position_t]            (size 9)

where z(x) = (x - mean_TRAIN) / std_TRAIN is a per-feature standardisation whose
statistics are fitted on TRAIN closes only (``fit_o2_scaler``), and
``position_t`` is the portfolio state held during bar t (0 = CASH, 1 = LONG).

Feature definitions (decision bar t, closes c_k, daily log return r_k = ln(c_k / c_{k-1}))
-----------------------------------------------------------------------------------------
    return_1d      = ln(c_t / c_{t-1})
    return_7d      = ln(c_t / c_{t-7})
    return_30d     = ln(c_t / c_{t-30})
    volatility_14  = sample std (ddof=1) of r_{t-13}, ..., r_t      (14 returns, closes c_{t-14..t})
    volatility_30  = sample std (ddof=1) of r_{t-29}, ..., r_t      (30 returns, closes c_{t-30..t})
    ma_50_ratio    = c_t / mean(c_{t-49}, ..., c_t)                 (SMA50 includes c_t)
    ma_200_ratio   = c_t / mean(c_{t-199}, ..., c_t)                (SMA200 includes c_t)
    rsi_14         = 100 - 100 / (1 + RS),  RS = mean(gain_{t-13..t}) / mean(loss_{t-13..t})
                     gain_k = max(c_k - c_{k-1}, 0), loss_k = max(c_{k-1} - c_k, 0)
                     SIMPLE ROLLING-MEAN RSI (Cutler's RSI).  This is the ML project's
                     ``features.compute_rsi`` exactly: it is NOT Wilder's exponentially
                     smoothed RSI although the ML docstring calls it "Wilder".  The
                     definition is kept unchanged on purpose.
                     Degenerate windows (never present on the frozen dataset, see tests):
                     mean loss == 0 and mean gain > 0 -> 100; both zero -> 50.  The ML
                     implementation yields NaN there and drops the row; an RL observation
                     cannot be dropped, so the conventional limits are used instead.

Every feature is a function of the closes c_{t-199}, ..., c_t only (200 closes):
``O2_WINDOW`` = 200, warm-up ``O2_WARMUP_ROWS`` = 199 rows before the first
decision.  ``build_o2_observation`` receives only ``closes[: t + 1]`` from the
environment and uses the last 200 of them, so future rows are unreachable by
construction (same structural causality as O1).  Opens, highs, lows and
volume are not used.

Scaling (TRAIN only)
--------------------
``fit_o2_scaler`` computes the per-feature population mean and std (ddof=0,
the same statistic type as the E1 return scale) over the feature rows whose
COMPLETE 200-close window lies inside the TRAIN fit closes
[TRAIN start, last usable TRAIN decision] = [2018-03-04, 2020-12-29].  These are
the decision rows 2018-09-19 .. 2020-12-29 (833 rows).  Consequently the fitted
statistics are a function of exactly the 1032 TRAIN closes that also define the
E1 scale; no 2017 warm-up close, no settlement close, no VALIDATION and no TEST
close can enter them (proved by perturbation tests).  Observation warm-up is a
different matter: the first TRAIN observation (2018-03-04) still uses the 199
causal closes before TRAIN, as permitted; those rows are simply not part of the
fit population.

Bounds (finite Box, no active clipping)
---------------------------------------
``validate_ohlcv`` guarantees |r_k| <= MAX_ABS_LOG_RETURN (=1) on frozen data.
Every raw feature therefore has an analytic bound (``raw_feature_bounds``):
returns +-1, +-7, +-30; volatilities [0, sqrt(n/(n-1))]; MA ratios
(0, n / sum_{j<n} e^{-j}]; RSI [0, 100].  The declared Box is the standardised
image of those bounds, so every observation built from validated data lies
inside it without clipping.  As in O1, values are additionally clipped to the
declared bound as a defence for unvalidated (synthetic) frames only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from gymnasium import spaces

from .observations import MAX_ABS_LOG_RETURN
from .splits import SplitWindow, usable_decision_index_range

O2_FEATURES: tuple[str, ...] = (
    "return_1d", "return_7d", "return_30d", "volatility_14", "volatility_30",
    "ma_50_ratio", "ma_200_ratio", "rsi_14",
)
O2_N_FEATURES = len(O2_FEATURES)      # 8 engineered market features
O2_SIZE = O2_N_FEATURES + 1           # + current portfolio state
O2_WINDOW = 200                       # closes c_{t-199..t} needed at decision t
O2_WARMUP_ROWS = O2_WINDOW - 1        # rows required before the first decision row
RSI_PERIOD = 14
RSI_NO_LOSS = 100.0                   # mean loss == 0, mean gain > 0
RSI_FLAT = 50.0                       # mean loss == 0 and mean gain == 0
SCALER_STATISTIC = "per-feature population mean/std (ddof=0) over TRAIN-only feature rows"
RSI_DEFINITION = ("simple rolling-mean RSI (Cutler): 100 - 100/(1 + mean(gain,14)/mean(loss,14)); "
                  "identical to the ML project's compute_rsi (not Wilder smoothing)")

FEATURE_DEFINITIONS: dict[str, str] = {
    "return_1d": "ln(close[t] / close[t-1])",
    "return_7d": "ln(close[t] / close[t-7])",
    "return_30d": "ln(close[t] / close[t-30])",
    "volatility_14": "sample std (ddof=1) of the 14 daily log close returns ending at t (closes t-14..t)",
    "volatility_30": "sample std (ddof=1) of the 30 daily log close returns ending at t (closes t-30..t)",
    "ma_50_ratio": "close[t] / mean(close[t-49..t])  (SMA50 includes close[t])",
    "ma_200_ratio": "close[t] / mean(close[t-199..t])  (SMA200 includes close[t])",
    "rsi_14": RSI_DEFINITION + f"; degenerate windows: no loss -> {RSI_NO_LOSS:.0f}, flat -> {RSI_FLAT:.0f}",
    "position": "portfolio state held during bar t: 0 = CASH, 1 = LONG",
}


def _ma_ratio_upper_bound(n: int, b: float = MAX_ABS_LOG_RETURN) -> float:
    """Tight bound of close[t] / SMA_n[t] when every daily |log return| <= b: n / sum_{j=0}^{n-1} e^{-jb}."""
    return n / float(np.sum(np.exp(-b * np.arange(n))))


def raw_feature_bounds(b: float = MAX_ABS_LOG_RETURN) -> dict[str, tuple[float, float]]:
    """Analytic [low, high] of every raw feature under the data-level bound |log return| <= b."""
    return {
        "return_1d": (-b, b),
        "return_7d": (-7.0 * b, 7.0 * b),
        "return_30d": (-30.0 * b, 30.0 * b),
        "volatility_14": (0.0, b * math.sqrt(14.0 / 13.0)),
        "volatility_30": (0.0, b * math.sqrt(30.0 / 29.0)),
        "ma_50_ratio": (0.0, _ma_ratio_upper_bound(50, b)),
        "ma_200_ratio": (0.0, _ma_ratio_upper_bound(200, b)),
        "rsi_14": (0.0, 100.0),
    }


RAW_FEATURE_BOUNDS: dict[str, tuple[float, float]] = raw_feature_bounds()


# ───────────────────────────────────────────── raw features (one decision row)
def o2_raw_features(closes_up_to_t: np.ndarray) -> np.ndarray:
    """
    The 8 raw O2 features at the decision bar t = last element of
    ``closes_up_to_t``.  Only the last ``O2_WINDOW`` closes are used.
    """
    c = np.asarray(closes_up_to_t, dtype=np.float64)
    if c.ndim != 1 or c.shape[0] < O2_WINDOW:
        raise ValueError(f"need a 1-d array of at least {O2_WINDOW} closes")
    w = c[-O2_WINDOW:]
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("closes must be finite and positive")
    r = np.diff(np.log(w))                     # r_{t-198} .. r_t (199 returns)
    d = np.diff(w[-(RSI_PERIOD + 1):])         # price changes of the last 14 bars
    gain = float(np.mean(np.clip(d, 0.0, None)))
    loss = float(np.mean(np.clip(-d, 0.0, None)))
    if loss > 0.0:
        rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    else:
        rsi = RSI_NO_LOSS if gain > 0.0 else RSI_FLAT
    out = np.array([
        math.log(w[-1] / w[-2]),
        math.log(w[-1] / w[-8]),
        math.log(w[-1] / w[-31]),
        float(np.std(r[-14:], ddof=1)),
        float(np.std(r[-30:], ddof=1)),
        w[-1] / float(np.mean(w[-50:])),
        w[-1] / float(np.mean(w[-200:])),
        rsi,
    ], dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite raw O2 feature")
    return out


def o2_feature_table(frame: pd.DataFrame, first_index: int, last_index: int) -> pd.DataFrame:
    """Raw O2 features for every decision row ``first_index .. last_index`` (inclusive), one builder call per row."""
    closes = frame["close"].to_numpy(dtype=np.float64)
    if first_index < O2_WARMUP_ROWS:
        raise ValueError(f"row {first_index} has fewer than {O2_WARMUP_ROWS} warm-up rows before it")
    if last_index < first_index or last_index >= len(closes):
        raise ValueError("invalid row range")
    rows = [o2_raw_features(closes[: t + 1]) for t in range(first_index, last_index + 1)]
    return pd.DataFrame(np.vstack(rows), index=frame.index[first_index : last_index + 1], columns=list(O2_FEATURES))


# ───────────────────────────────────────────── scaler (TRAIN only)
@dataclass(frozen=True)
class O2ScalerFit:
    means: tuple[float, ...]
    stds: tuple[float, ...]
    fit_close_start: pd.Timestamp    # first close that may enter the statistic (TRAIN start)
    fit_close_end: pd.Timestamp      # last close that may enter it (last usable TRAIN decision)
    fit_row_start: pd.Timestamp      # first feature row in the fit population (window fully inside the fit closes)
    fit_row_end: pd.Timestamp        # last feature row in the fit population
    n_closes: int
    n_rows: int
    window_name: str
    statistic: str = SCALER_STATISTIC
    features: tuple[str, ...] = O2_FEATURES

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("fit_close_start", "fit_close_end", "fit_row_start", "fit_row_end"):
            d[k] = str(getattr(self, k).date())
        d["means"] = list(self.means)
        d["stds"] = list(self.stds)
        d["features"] = list(self.features)
        d["per_feature"] = {
            name: {"mean": self.means[i], "std": self.stds[i],
                   "raw_low": RAW_FEATURE_BOUNDS[name][0], "raw_high": RAW_FEATURE_BOUNDS[name][1],
                   "clipping": "none active (declared bound = standardised analytic bound)"}
            for i, name in enumerate(self.features)
        }
        return d


def fit_o2_scaler(frame: pd.DataFrame, train_window: SplitWindow) -> O2ScalerFit:
    """
    Per-feature mean/std over the O2 feature rows whose whole 200-close window
    lies inside the TRAIN fit closes [window start, last usable decision].
    """
    i0, last = usable_decision_index_range(frame, train_window)
    if i0 < O2_WARMUP_ROWS:
        raise RuntimeError(f"TRAIN start row {i0} lacks the {O2_WARMUP_ROWS} O2 warm-up rows")
    row0 = i0 + O2_WARMUP_ROWS            # first row whose window starts at the TRAIN start close
    if row0 > last:
        raise RuntimeError("TRAIN window too short to fit the O2 scaler on TRAIN-only feature rows")
    table = o2_feature_table(frame, row0, last)
    x = table.to_numpy(dtype=np.float64)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0, ddof=0)
    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(stds)) and np.all(stds > 0)):
        raise RuntimeError("degenerate O2 scaler statistics (non-finite or zero std)")
    fit = O2ScalerFit(
        means=tuple(float(v) for v in means),
        stds=tuple(float(v) for v in stds),
        fit_close_start=pd.Timestamp(frame.index[i0]),
        fit_close_end=pd.Timestamp(frame.index[last]),
        fit_row_start=pd.Timestamp(frame.index[row0]),
        fit_row_end=pd.Timestamp(frame.index[last]),
        n_closes=int(last - i0 + 1),
        n_rows=int(last - row0 + 1),
        window_name=train_window.name,
    )
    if fit.fit_close_start != train_window.start:
        raise RuntimeError("O2 scaler fit closes do not start at the TRAIN window start")
    if fit.fit_close_end >= train_window.end:
        raise RuntimeError("O2 scaler fit closes reach the TRAIN settlement rows")
    return fit


# ───────────────────────────────────────────── observation config / space / builder
@dataclass(frozen=True)
class O2ObservationConfig:
    means: tuple[float, ...]
    stds: tuple[float, ...]

    def __post_init__(self) -> None:
        means = tuple(float(v) for v in self.means)
        stds = tuple(float(v) for v in self.stds)
        if len(means) != O2_N_FEATURES or len(stds) != O2_N_FEATURES:
            raise ValueError(f"O2 needs exactly {O2_N_FEATURES} means and stds")
        if not all(math.isfinite(v) for v in means) or not all(math.isfinite(v) and v > 0 for v in stds):
            raise ValueError("O2 scaler statistics must be finite with positive stds")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "stds", stds)
        low, high = self.standardised_bounds()
        for b in np.concatenate([low, high]):
            with np.errstate(over="ignore", under="ignore"):
                b32 = np.float32(b)
            if not np.isfinite(b32) or (b32 != 0 and abs(b32) < np.finfo(np.float32).tiny):
                raise ValueError(f"standardised bound {b!r} is not a finite normal float32")
        if not np.all(high > low):
            raise ValueError("standardised bounds are not ordered")

    @property
    def size(self) -> int:
        return O2_SIZE

    @property
    def features(self) -> tuple[str, ...]:
        return O2_FEATURES

    def standardised_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Declared [low, high] of every standardised feature (float64)."""
        lows = np.array([RAW_FEATURE_BOUNDS[f][0] for f in O2_FEATURES], dtype=np.float64)
        highs = np.array([RAW_FEATURE_BOUNDS[f][1] for f in O2_FEATURES], dtype=np.float64)
        m = np.asarray(self.means, dtype=np.float64)
        s = np.asarray(self.stds, dtype=np.float64)
        return (lows - m) / s, (highs - m) / s

    @classmethod
    def from_fit(cls, fit: O2ScalerFit) -> "O2ObservationConfig":
        return cls(means=fit.means, stds=fit.stds)


def o2_observation_space(cfg: O2ObservationConfig) -> spaces.Box:
    low, high = cfg.standardised_bounds()
    lo = np.concatenate([low.astype(np.float32), np.zeros(1, dtype=np.float32)])
    hi = np.concatenate([high.astype(np.float32), np.ones(1, dtype=np.float32)])
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("O2 observation bounds are not finite")
    return spaces.Box(low=lo, high=hi, dtype=np.float32)


def build_o2_observation(closes_up_to_t: np.ndarray, position: int, cfg: O2ObservationConfig) -> np.ndarray:
    """O2 observation at the decision bar = last element of ``closes_up_to_t`` (only the last 200 closes are used)."""
    if position not in (0, 1):
        raise ValueError("position must be 0 or 1")
    raw = o2_raw_features(closes_up_to_t)
    z = (raw - np.asarray(cfg.means)) / np.asarray(cfg.stds)
    low, high = cfg.standardised_bounds()
    z = np.clip(z, low, high)            # inactive on validated data (analytic bounds); defence for synthetic frames
    obs = np.empty(O2_SIZE, dtype=np.float32)
    obs[:O2_N_FEATURES] = z
    obs[O2_N_FEATURES] = float(position)
    # float32 rounding can exceed the float32 bound by 1 ulp; clip once more in float32
    np.clip(obs[:O2_N_FEATURES], low.astype(np.float32), high.astype(np.float32), out=obs[:O2_N_FEATURES])
    if not np.all(np.isfinite(obs)):
        raise ValueError("non-finite O2 observation produced")
    return obs


def verify_o2_warmup(frame: pd.DataFrame, window: SplitWindow) -> dict:
    """The first decision of ``window`` must have >= 199 rows before it (all O2 features defined there)."""
    i0, last = usable_decision_index_range(frame, window)
    if i0 < O2_WARMUP_ROWS:
        raise RuntimeError(
            f"O2 cannot produce valid values at the first {window.name} decision {frame.index[i0].date()}: "
            f"{i0} warm-up rows available, {O2_WARMUP_ROWS} required"
        )
    first = o2_raw_features(frame["close"].to_numpy(dtype=np.float64)[: i0 + 1])
    return {
        "window": window.name,
        "first_decision": str(frame.index[i0].date()),
        "warmup_rows_required": O2_WARMUP_ROWS,
        "warmup_rows_available": int(i0),
        "warmup_slack_rows": int(i0 - O2_WARMUP_ROWS),
        "first_window_close": str(frame.index[i0 - O2_WARMUP_ROWS].date()),
        "first_decision_raw_features": {k: float(v) for k, v in zip(O2_FEATURES, first)},
    }


__all__ = [
    "O2_FEATURES", "O2_N_FEATURES", "O2_SIZE", "O2_WINDOW", "O2_WARMUP_ROWS", "FEATURE_DEFINITIONS",
    "RAW_FEATURE_BOUNDS", "RSI_DEFINITION", "SCALER_STATISTIC", "O2ScalerFit", "O2ObservationConfig",
    "o2_raw_features", "o2_feature_table", "fit_o2_scaler", "o2_observation_space", "build_o2_observation",
    "verify_o2_warmup", "raw_feature_bounds",
]
