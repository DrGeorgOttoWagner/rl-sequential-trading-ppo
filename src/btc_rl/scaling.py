"""
scaling.py — TRAIN-only fit of the O1 return scale.

Preprocessing statistics must be fitted on the declared TRAIN interval and
nothing else (frozen study design (docs/methodology.md)).  The fit population is the set of daily
closes a TRAIN observation can actually contain:

    fit_start = TRAIN window start            (2018-03-04)
    fit_end   = last usable TRAIN decision    (2020-12-29 under the purge rule)

The statistic is the population standard deviation of the daily log close
returns log(close_k / close_{k-1}) for consecutive closes inside
[fit_start, fit_end].  The first return therefore starts at the second fit
close; no close before ``fit_start`` (2017 warm-up history) and no close
after ``fit_end`` (settlement rows, validation, test) enters the fit.

Observation warm-up is a different matter: the environment may still read
earlier causal closes to form the first observations of a window.  Only the
*statistic* is restricted here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .observations import ObservationConfig, fit_return_scale
from .splits import SplitWindow, usable_decision_index_range

SCALE_STATISTIC = "population std of daily log close returns"


@dataclass(frozen=True)
class ScaleFit:
    scale: float
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    n_closes: int
    n_returns: int
    window_name: str
    statistic: str = SCALE_STATISTIC

    def to_json(self) -> dict:
        d = asdict(self)
        d["fit_start"] = str(self.fit_start.date())
        d["fit_end"] = str(self.fit_end.date())
        return d


def fit_train_return_scale(frame: pd.DataFrame, train_window: SplitWindow) -> ScaleFit:
    """
    Fit the O1 return scale on closes strictly inside ``train_window`` from
    its first decision date through its last usable decision date.
    """
    i0, last = usable_decision_index_range(frame, train_window)
    closes = frame["close"].to_numpy(dtype=np.float64)[i0 : last + 1]
    scale = fit_return_scale(closes)
    fit = ScaleFit(
        scale=float(scale),
        fit_start=pd.Timestamp(frame.index[i0]),
        fit_end=pd.Timestamp(frame.index[last]),
        n_closes=int(last - i0 + 1),
        n_returns=int(last - i0),
        window_name=train_window.name,
    )
    if fit.fit_start != train_window.start:
        raise RuntimeError("scale fit does not start at the TRAIN window start")
    if fit.fit_end >= train_window.end:
        raise RuntimeError("scale fit reaches the TRAIN settlement rows")
    return fit


def scaled_observation_config(base: ObservationConfig, fit: ScaleFit) -> ObservationConfig:
    return ObservationConfig(lookback=base.lookback, return_scale=fit.scale)
