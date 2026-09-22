"""
splits.py — frozen train / validation / test windows (docs/methodology.md).

Interpretation
--------------
A split window lists DECISION dates: the dates t on which an observation is
formed after close[t] and an action is chosen.  Consequences:

* Warm-up: the observation at the first decision date needs ``lookback``
  earlier closes.  These come from rows before the window (past data only).
* Settlement / purge rule: a decision at t consumes open[t+1] and open[t+2].
  A transition belongs to a split only if its observation AND its whole
  execution/reward interval lie inside the split.  Hence the last USABLE
  decision date is ``end - SETTLEMENT_BARS`` rows; the final two dates of
  every window are purged from decisions and serve only as settlement prices.
  No train reward touches validation prices, no validation reward touches
  test prices, and test never consumes prices after its frozen end date.
* Observation history may reach into the previous split (warm-up) because it
  is strictly causal.

The windows are contiguous and non-overlapping; ``check_split_layout``
verifies this together with warm-up availability and the purge rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SETTLEMENT_BARS = 2  # open[t+1] (fill) and open[t+2] (mark) after the last decision


@dataclass(frozen=True)
class SplitWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        object.__setattr__(self, "end", pd.Timestamp(self.end))
        if self.end < self.start:
            raise ValueError(f"split {self.name}: end before start")


TRAIN = SplitWindow("train", "2018-03-04", "2020-12-31")
VALIDATION = SplitWindow("validation", "2021-01-01", "2021-08-29")
TEST = SplitWindow("test", "2021-08-30", "2026-02-04")

FROZEN_SPLITS: dict[str, SplitWindow] = {
    "train": TRAIN,
    "validation": VALIDATION,
    "test": TEST,
}


def decision_index_range(frame: pd.DataFrame, window: SplitWindow) -> tuple[int, int]:
    """Integer positions (inclusive) of the first and last decision date of ``window``."""
    idx = frame.index
    if window.start not in idx or window.end not in idx:
        raise KeyError(f"split {window.name} boundaries not present in frame")
    i0 = int(idx.get_loc(window.start))
    i1 = int(idx.get_loc(window.end))
    return i0, i1


def decision_dates(frame: pd.DataFrame, window: SplitWindow) -> pd.DatetimeIndex:
    i0, i1 = decision_index_range(frame, window)
    return frame.index[i0 : i1 + 1]


def usable_decision_index_range(
    frame: pd.DataFrame, window: SplitWindow, settlement_bars: int = SETTLEMENT_BARS
) -> tuple[int, int]:
    """Purge rule: last usable decision index = window end index - settlement_bars."""
    i0, i1 = decision_index_range(frame, window)
    last = i1 - settlement_bars
    if last < i0:
        raise ValueError(f"split {window.name} too short for {settlement_bars} settlement bars")
    return i0, last


def usable_decision_dates(frame: pd.DataFrame, window: SplitWindow) -> pd.DatetimeIndex:
    i0, last = usable_decision_index_range(frame, window)
    return frame.index[i0 : last + 1]


def check_split_layout(
    frame: pd.DataFrame,
    splits: dict[str, SplitWindow] | None = None,
    lookback: int = 10,
    settlement_bars: int = SETTLEMENT_BARS,
) -> dict[str, dict]:
    """
    Verify that the splits are contiguous, ordered, non-overlapping, that
    warm-up rows exist in ``frame``, and report the purged decision range so
    that settlement never crosses a split end.  Returns per-split info.
    """
    splits = splits or FROZEN_SPLITS
    ordered = [splits[k] for k in ("train", "validation", "test") if k in splits]
    info: dict[str, dict] = {}
    n = len(frame)
    for w in ordered:
        i0, i1 = decision_index_range(frame, w)
        if i0 - lookback < 0:
            raise ValueError(f"split {w.name}: not enough warm-up rows for lookback {lookback}")
        if i1 > n - 1:
            raise ValueError(f"split {w.name}: end {w.end.date()} beyond available data")
        _, last_usable = usable_decision_index_range(frame, w, settlement_bars)
        info[w.name] = {
            "start": w.start,
            "end": w.end,
            "n_dates": i1 - i0 + 1,
            "first_index": i0,
            "last_index": i1,
            "last_usable_decision": frame.index[last_usable],
            "n_usable_decisions": last_usable - i0 + 1,
            "warmup_first_date": frame.index[i0 - lookback],
            "last_mark_date": frame.index[last_usable + settlement_bars],
        }
    for a, b in zip(ordered, ordered[1:]):
        if b.start != a.end + pd.Timedelta(days=1):
            raise ValueError(f"splits {a.name} and {b.name} are not contiguous")
    return info
