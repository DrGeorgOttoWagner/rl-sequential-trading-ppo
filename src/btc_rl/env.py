"""
env.py — Gymnasium environment for BTC/USDT daily INVESTED/CASH (internal labels LONG/FLAT) trading.

Exact timeline for one step (decision index t, frame row t)
-----------------------------------------------------------

    close[t]            open[t+1]                       open[t+2]
      |                    |                                |
  last price in       state s_t EXISTS here;            state s_{t+1} exists
  s_t (closes only    action a_t chosen and FILLED       here (closes through
  through close[t])   here; one cost leg per unit of     close[t+1] plus
                      |p_{t+1} - p_t|                    p_{t+1}); r_t settles
                                                         here on open[t+2]

* State s_t exists at the instant open[t+1] and contains only features
  computed through close[t] (the environment passes ``closes[: t + 1]`` to
  the observation builder) plus the position p_t held during bar t.
* Action a_t is chosen and executed at open[t+1].  No same-close fill.
* Reward r_t is determined by exactly three quantities: open[t+1],
  open[t+2] and the number of cost legs |p_{t+1} - p_t|:

      r_t = |Δp| * log(1 - c) + p_{t+1} * log(open[t+2] / open[t+1])

  which equals log(equity_{t+1} / equity_t).  Overnight gaps are inside the
  held return because both ends are opens.
* State s_{t+1} exists at open[t+2].  Every component of r_t is known at
  that instant and none is known earlier; ``info['state_instant_date']`` of
  the returned state equals ``info['mark_date']`` of the settled step.
  Equity, drawdown and peak in ``info`` are marked at the same instant.
* ``price_end`` (optional) is a hard boundary: the environment keeps only
* Observation at t uses closes[: t + 1] only (structural causality) plus the
  position held during bar t.
* The action chosen at t is the target position for bar t+1, executed at
  open[t+1].  There is no same-close fill.
* Reward for step t is the portfolio log return between open[t+1] (after
  costs) and open[t+2]:

      reward_t = |Δp| * log(1 - c) + p_{t+1} * log(open[t+2] / open[t+1])

  which is exactly log(equity_{t+1} / equity_t).  Open-to-open returns keep
  overnight gaps inside the held-position return (frozen study design (docs/methodology.md)).
* Equity is marked at execution points (opens).  FLAT bars earn zero.
* ``price_end`` (optional) is a hard boundary: the environment keeps only
  rows up to and including that date, so no price after it can be read for
  any purpose.  ``make_split_env`` sets it to the split end and purges the
  last two dates from the decision range (see ``splits.py``).
* No leverage, no short, no forced liquidation at episode end.  A position
  still LONG at the end of an episode is reported via ``info['position']``
  and the metrics; no exit leg is charged.

Termination semantics
---------------------
``terminated`` is always False in v1: there is no absorbing state (equity
cannot reach zero with LONG/FLAT on spot and no leverage).  ``truncated`` is
True when the episode reaches its last decision index (a data-window time
limit).  Calling ``step`` after truncation raises.

Provenance
----------
Cost profile from the ML project (see ``costs.py``).  Drawdown definition
(equity / running peak - 1) follows
the author's earlier supervised-learning study (not part of this repository).
The ML open-to-close (intrabar) accounting and its end-of-test force-close
are intentionally NOT reused.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .costs import CONSERVATIVE_COST, CostConfig
from .observations import ObservationConfig, build_o1_observation, observation_space
from .splits import (
    SETTLEMENT_BARS,
    FROZEN_SPLITS,
    SplitWindow,
    decision_index_range,
    usable_decision_index_range,
)

FLAT = 0
LONG = 1


class BtcUsdtTradingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        frame: pd.DataFrame,
        decision_start: pd.Timestamp | str,
        decision_end: pd.Timestamp | str,
        cost_config: CostConfig = CONSERVATIVE_COST,
        obs_config: ObservationConfig | None = None,
        episode_length: int | None = None,
        random_start: bool = False,
        initial_equity: float = 1.0,
        render_mode: str | None = None,
        price_end: pd.Timestamp | str | None = None,
    ) -> None:
        super().__init__()
        obs_config = obs_config or ObservationConfig()
        if price_end is not None:
            # Structural boundary: rows after price_end do not exist inside the env.
            frame = frame.loc[: pd.Timestamp(price_end)]
            if len(frame) == 0 or frame.index[-1] != pd.Timestamp(price_end):
                raise ValueError("price_end must be a date present in frame")
        self.price_end = None if price_end is None else pd.Timestamp(price_end)
        if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing:
            raise ValueError("frame must be sorted with a DatetimeIndex")
        if frame.index.has_duplicates:
            raise ValueError("frame index has duplicates")
        for col in ("open", "close"):
            if col not in frame.columns:
                raise ValueError(f"frame missing column {col}")
        if isinstance(initial_equity, bool) or not isinstance(initial_equity, (int, float)):
            raise ValueError("initial_equity must be a number")
        if not math.isfinite(initial_equity) or initial_equity <= 0:
            raise ValueError("initial_equity must be finite and positive")
        if not isinstance(cost_config, CostConfig):
            raise TypeError("cost_config must be a CostConfig")

        self._dates = frame.index
        self._opens = frame["open"].to_numpy(dtype=np.float64)
        self._closes = frame["close"].to_numpy(dtype=np.float64)
        if not (np.all(np.isfinite(self._opens)) and np.all(np.isfinite(self._closes))):
            raise ValueError("prices must be finite")
        if (self._opens <= 0).any() or (self._closes <= 0).any():
            raise ValueError("prices must be positive")
        n = len(frame)

        self.cost_config = cost_config
        self.obs_config = obs_config
        self.initial_equity = float(initial_equity)
        self.render_mode = render_mode

        window = SplitWindow("episode", pd.Timestamp(decision_start), pd.Timestamp(decision_end))
        self._i0, self._i1 = decision_index_range(frame, window)
        if self._i0 - obs_config.lookback < 0:
            raise ValueError("not enough warm-up rows before decision_start")
        if self._i1 + SETTLEMENT_BARS > n - 1:
            raise ValueError(
                "not enough settlement rows after decision_end "
                "(a decision needs open[t+1] and open[t+2] within the allowed price range)"
            )
        self.n_decisions_available = self._i1 - self._i0 + 1

        if episode_length is not None:
            if episode_length < 1 or episode_length > self.n_decisions_available:
                raise ValueError("episode_length outside available decision range")
        self.episode_length = episode_length
        self.random_start = random_start

        self.action_space = spaces.Discrete(2)
        self.observation_space = observation_space(obs_config)

        # episode state (set in reset)
        self._t = -1
        self._t_start = -1
        self._t_end = -1
        self._position = FLAT
        self._equity = self.initial_equity
        self._peak = self.initial_equity
        self._max_drawdown = 0.0
        self._legs_paid = 0
        self._n_steps = 0
        self._done = True

    # ------------------------------------------------------------------ helpers
    @property
    def decision_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self._dates[self._i0], self._dates[self._i1]

    def _observe(self) -> np.ndarray:
        # Structural causality: only closes up to and including bar t are visible.
        return build_o1_observation(self._closes[: self._t + 1], self._position, self.obs_config)

    def _state_info(self, **extra: Any) -> dict[str, Any]:
        """State as seen at the current decision index (what a policy may use)."""
        info = {
            "step": self._n_steps,
            "decision_index": int(self._t),
            "decision_date": self._dates[self._t],
            # the state exists at the fill instant open[t+1]; its features end at close[t]
            "state_instant_date": self._dates[self._t + 1],
            "last_feature_date": self._dates[self._t],
            "position": int(self._position),
            "equity": float(self._equity),
            "peak_equity": float(self._peak),
            "drawdown": float(self._equity / self._peak - 1.0),
            "max_drawdown": float(self._max_drawdown),
            "legs_paid": int(self._legs_paid),
            "episode_start_index": int(self._t_start),
            "episode_end_index": int(self._t_end),
        }
        info.update(extra)
        return info

    def _choose_start(self, options: dict[str, Any] | None) -> int:
        options = options or {}
        if "start_index" in options:
            start = int(options["start_index"])
        elif "start_date" in options:
            start = int(self._dates.get_loc(pd.Timestamp(options["start_date"])))
        elif self.random_start:
            length = self.episode_length or self.n_decisions_available
            hi = self._i1 - length + 1  # inclusive upper bound for start
            start = int(self.np_random.integers(self._i0, hi + 1))
        else:
            start = self._i0
        if start < self._i0 or start > self._i1:
            raise ValueError(f"start index {start} outside decision window")
        return start

    # ---------------------------------------------------------- gymnasium API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._t_start = self._choose_start(options)
        length = self.episode_length or (self._i1 - self._t_start + 1)
        self._t_end = min(self._t_start + length - 1, self._i1)
        self._t = self._t_start
        self._position = FLAT
        self._equity = self.initial_equity
        self._peak = self.initial_equity
        self._max_drawdown = 0.0
        self._legs_paid = 0
        self._n_steps = 0
        self._done = False
        return self._observe(), self._state_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        action = _validate_action(action)

        t = self._t
        prev_position = self._position
        new_position = action
        legs = abs(new_position - prev_position)

        open_fill = self._opens[t + 1]
        open_mark = self._opens[t + 2]
        equity_before = self._equity

        # 1) fill at open[t+1]: one cost leg per unit of position change
        cost_factor = self.cost_config.factor_for_legs(legs)
        equity_after_cost = equity_before * cost_factor
        # 2) hold p_{t+1} from open[t+1] to open[t+2] (overnight gap included)
        gross_simple_return = open_mark / open_fill - 1.0
        equity_after = equity_after_cost * (1.0 + new_position * gross_simple_return)

        reward = float(np.log(equity_after / equity_before))
        if not (math.isfinite(equity_after) and equity_after > 0 and math.isfinite(reward)):
            raise RuntimeError(f"non-finite accounting at decision index {t}")

        self._equity = equity_after
        self._position = new_position
        self._legs_paid += legs
        self._n_steps += 1
        if self._equity > self._peak:
            self._peak = self._equity
        drawdown = self._equity / self._peak - 1.0
        if drawdown < self._max_drawdown:
            self._max_drawdown = drawdown

        truncated = t >= self._t_end
        terminated = False
        self._done = truncated
        self._t = t + 1  # s_{t+1}: features through close[t+1], exists at open[t+2]

        # info describes the state at the NEXT decision (row t+1: 'decision_date',
        # 'position', 'equity', ...) plus the step just settled under 'settled_*',
        # 'fill_date', 'mark_date', 'action', 'legs', ... keys.
        info = self._state_info(
            settled_decision_index=int(t),
            settled_decision_date=self._dates[t],
            action=action,
            prev_position=int(prev_position),
            legs=int(legs),
            cost_fraction=float(1.0 - cost_factor),
            fill_date=self._dates[t + 1],
            mark_date=self._dates[t + 2],
            open_fill=float(open_fill),
            open_mark=float(open_mark),
            gross_simple_return=float(gross_simple_return),
            position_return=float(new_position * gross_simple_return),
            reward_log_cost=float(legs * self.cost_config.log_leg_cost),
            reward_log_gross=float(np.log1p(new_position * gross_simple_return)),
        )
        return self._observe(), reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return (
            f"t={self._dates[self._t].date()} pos={self._position} "
            f"equity={self._equity:.6f} dd={self._equity / self._peak - 1.0:+.4f}"
        )


def _validate_action(action: Any) -> int:
    """Accept only integer-typed actions (python int, numpy integer, 0-d integer array) in {0, 1}."""
    if isinstance(action, (bool, np.bool_)):
        raise TypeError("action must be an integer 0 or 1, not a bool")
    if isinstance(action, np.ndarray):
        if action.shape != () or not np.issubdtype(action.dtype, np.integer):
            raise TypeError("array actions must be 0-d with an integer dtype")
        value = int(action)
    elif isinstance(action, (int, np.integer)):
        value = int(action)
    else:
        raise TypeError(f"action must be an integer, got {type(action).__name__}")
    if value not in (FLAT, LONG):
        raise ValueError(f"invalid action {value}; expected 0 (FLAT) or 1 (LONG)")
    return value


def make_split_env(
    frame: pd.DataFrame,
    split: str | SplitWindow,
    cost_config: CostConfig = CONSERVATIVE_COST,
    obs_config: ObservationConfig | None = None,
    splits: Mapping[str, SplitWindow] | None = None,
    **kwargs: Any,
) -> BtcUsdtTradingEnv:
    """
    Build an environment for one split under the purge rule: decisions run
    from the split start to ``end - 2`` rows, and ``price_end`` is the split
    end, so every fill and mark stays inside the split.

    ``splits`` is the configured window mapping (``FoundationConfig.splits``);
    it defaults to the frozen constants, which ``load_foundation_config``
    guarantees to be identical (F8).  A ``SplitWindow`` may be passed directly.
    """
    table = FROZEN_SPLITS if splits is None else splits
    window = table[split] if isinstance(split, str) else split
    if not isinstance(window, SplitWindow):
        raise TypeError("split must be a split name or a SplitWindow")
    i0, last_usable = usable_decision_index_range(frame, window)
    return BtcUsdtTradingEnv(
        frame,
        decision_start=frame.index[i0],
        decision_end=frame.index[last_usable],
        cost_config=cost_config,
        obs_config=obs_config,
        price_end=window.end,
        **kwargs,
    )
