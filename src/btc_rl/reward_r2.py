"""
reward_r2.py — the prospectively frozen risk-aware reward R2 (docs/methodology.md).

R2 is R1 minus a turnover penalty minus an incremental-drawdown penalty.  It is
a TRAINING INCENTIVE ONLY: it never touches the financial accounting.  Equity,
costs, positions, drawdown and every evaluation metric are the accepted E1
accounting (``env.BtcUsdtTradingEnv.step``), which this module calls unchanged
and does not re-implement.

Definition (per decision step t; all quantities marked at opens)
-----------------------------------------------------------------
    financial_r1_t          = log(E_{t+1} / E_t)                       accepted E1 reward after modelled 0.10% fee + 0.05% slippage allowance per trade
    legs_t                  = |p_{t+1} - p_t|  in {0, 1}               exactly the accounting's cost legs
    running_peak_t          = max(E_0, ..., E_t)                        peak of realised equity BEFORE the transition
    D_t                     = 1 - E_t / running_peak_t                 positive drawdown magnitude before the transition
    running_peak_{t+1}      = max(running_peak_t, E_{t+1})              peak updated with the settled equity
    D_{t+1}                 = 1 - E_{t+1} / running_peak_{t+1}          drawdown magnitude after the transition
    incremental_drawdown_t  = max(0, D_{t+1} - D_t)                    only DEEPER drawdown; recovery / unchanged = 0

    R2_t = financial_r1_t - TURNOVER_PENALTY_LAMBDA * legs_t - DRAWDOWN_PENALTY_LAMBDA * incremental_drawdown_t

    TURNOVER_PENALTY_LAMBDA = 0.0005
    DRAWDOWN_PENALTY_LAMBDA = 0.10

The coefficients are fixed prospectively.  They are never tuned,
never changed after VALIDATION is seen, and must be reused unchanged by
E4 = O2 + R2.

Timing / causality
------------------
r_t settles at open[t+2] (the E1 reward instant).  The drawdown term depends
on the realised transition open[t+1] -> open[t+2] (E_{t+1}) and on the running
peak of equity realised up to and including that transition; nothing later is
read.  Episode reset: E_0 = running_peak_0 = 1 (initial equity), D_0 = 0.

Implementation
--------------
``compute_r2`` is a pure function of (financial_r1, legs, E_t, peak_t, E_{t+1},
peak_{t+1}) and returns the complete decomposition.  ``R2RewardMixin`` wraps
any certified environment class: its ``step`` records E_t / peak_t, calls the
parent ``step`` (the E1 accounting, untouched), reads E_{t+1} / peak_{t+1}
from the parent's own running-peak state, and returns R2 instead of R1 while
every ``info`` field of the parent is preserved.  ``BtcUsdtTradingEnvR2`` is
the O1 environment with R2 (E3).  E4 can combine the same mixin with the O2
environment without changing this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .costs import CONSERVATIVE_COST, CostConfig
from .env import BtcUsdtTradingEnv
from .observations import ObservationConfig

REWARD_DEFINITION = "R2"

# ───────────────────────────────────────────── prospectively frozen coefficients
TURNOVER_PENALTY_LAMBDA: float = 0.0005
DRAWDOWN_PENALTY_LAMBDA: float = 0.10

R2_FORMULA = ("R2_t = financial_r1_t - TURNOVER_PENALTY_LAMBDA * legs_t "
              "- DRAWDOWN_PENALTY_LAMBDA * max(0, D_{t+1} - D_t)")
R2_TERMS: dict[str, str] = {
    "financial_r1_t": "log(E_{t+1} / E_t): the accepted E1 portfolio log return after real transaction costs "
                      "(legs_t * log(1 - c) + p_{t+1} * log(open[t+2] / open[t+1]))",
    "legs_t": "|p_{t+1} - p_t| in {0, 1}: exactly the cost legs charged by the accounting",
    "D_t": "1 - E_t / running_peak_t, running_peak_t = max realised equity before the transition (initial equity counts)",
    "D_{t+1}": "1 - E_{t+1} / running_peak_{t+1}, running_peak_{t+1} = max(running_peak_t, E_{t+1})",
    "incremental_drawdown_t": "max(0, D_{t+1} - D_t): only deeper drawdown; new highs, recovery and unchanged drawdown give 0",
}
R2_TIMING = ("r_t settles at open[t+2] exactly as R1; the drawdown term uses the realised transition "
             "open[t+1] -> open[t+2] and the running peak of realised equity up to that transition; "
             "no later price or equity is read")
R2_RESET = "initial equity 1, running peak 1, drawdown 0 at every episode reset"
R2_ACCOUNTING_SEPARATION = ("reward shaping only: equity, costs (0.10% fee + 0.05% slippage allowance per trade), positions, drawdown and every evaluation "
                            "metric are computed by the unchanged E1 accounting from the financial equity path")


@dataclass(frozen=True)
class R2Coefficients:
    """The frozen penalty coefficients.  Non-default values are legal only for unit tests of the formula."""

    turnover_penalty_lambda: float = TURNOVER_PENALTY_LAMBDA
    drawdown_penalty_lambda: float = DRAWDOWN_PENALTY_LAMBDA

    def __post_init__(self) -> None:
        for name in ("turnover_penalty_lambda", "drawdown_penalty_lambda"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(v))

    @property
    def is_frozen(self) -> bool:
        return (self.turnover_penalty_lambda == TURNOVER_PENALTY_LAMBDA
                and self.drawdown_penalty_lambda == DRAWDOWN_PENALTY_LAMBDA)

    def to_json(self) -> dict[str, float]:
        return asdict(self)


FROZEN_R2 = R2Coefficients()
assert FROZEN_R2.is_frozen and FROZEN_R2.turnover_penalty_lambda == 0.0005 and FROZEN_R2.drawdown_penalty_lambda == 0.10


# ───────────────────────────────────────────── pure reward mathematics
def drawdown_magnitude(equity: float, running_peak: float) -> float:
    """D = 1 - equity / running_peak >= 0 (the environment's ``drawdown`` is the negative of this)."""
    if not (math.isfinite(equity) and math.isfinite(running_peak)) or equity <= 0 or running_peak <= 0:
        raise ValueError("equity and running peak must be finite and positive")
    if equity > running_peak * (1.0 + 1e-12):
        raise ValueError("equity exceeds the running peak: the peak must already include this equity")
    return max(0.0, 1.0 - equity / running_peak)


def incremental_drawdown(drawdown_before: float, drawdown_after: float) -> float:
    """max(0, D_{t+1} - D_t): deeper drawdown only; recovery and unchanged drawdown contribute zero."""
    if not (math.isfinite(drawdown_before) and math.isfinite(drawdown_after)):
        raise ValueError("drawdowns must be finite")
    if drawdown_before < 0 or drawdown_after < 0:
        raise ValueError("drawdown magnitudes must be non-negative")
    return max(0.0, drawdown_after - drawdown_before)


@dataclass(frozen=True)
class R2Decomposition:
    """Every term of one R2 step, recorded so the reward can be audited term by term."""

    financial_reward_r1: float
    legs: int
    equity_before: float
    running_peak_before: float
    equity_after: float
    running_peak_after: float
    drawdown_before: float
    drawdown_after: float
    incremental_drawdown: float
    turnover_penalty: float
    drawdown_penalty: float
    reward_r2: float

    def as_info(self) -> dict[str, Any]:
        d = asdict(self)
        d["reward_definition"] = REWARD_DEFINITION
        return d


def compute_r2(financial_reward_r1: float, legs: int, equity_before: float, running_peak_before: float,
               equity_after: float, running_peak_after: float,
               coefficients: R2Coefficients = FROZEN_R2) -> R2Decomposition:
    """
    R2 for one settled transition from quantities the accounting already produced.

    ``running_peak_before`` is the peak of realised equity up to E_t (initial
    equity included); ``running_peak_after`` must equal
    ``max(running_peak_before, equity_after)``.  Nothing else enters.
    """
    if isinstance(legs, bool) or not isinstance(legs, (int, np.integer)) or int(legs) not in (0, 1):
        raise ValueError("legs must be the integer 0 or 1 (|p_{t+1} - p_t| for CASH/LONG)")
    if not math.isfinite(financial_reward_r1):
        raise ValueError("financial_reward_r1 must be finite")
    if not isinstance(coefficients, R2Coefficients):
        raise TypeError("coefficients must be an R2Coefficients")
    expected_peak = max(running_peak_before, equity_after)
    if not math.isclose(running_peak_after, expected_peak, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("running_peak_after must equal max(running_peak_before, equity_after)")
    d_before = drawdown_magnitude(equity_before, running_peak_before)
    d_after = drawdown_magnitude(equity_after, running_peak_after)
    inc = incremental_drawdown(d_before, d_after)
    turnover_penalty = coefficients.turnover_penalty_lambda * int(legs)
    drawdown_penalty = coefficients.drawdown_penalty_lambda * inc
    reward_r2 = float(financial_reward_r1) - turnover_penalty - drawdown_penalty
    if not math.isfinite(reward_r2):
        raise RuntimeError("non-finite R2 reward")
    return R2Decomposition(
        financial_reward_r1=float(financial_reward_r1), legs=int(legs),
        equity_before=float(equity_before), running_peak_before=float(running_peak_before),
        equity_after=float(equity_after), running_peak_after=float(running_peak_after),
        drawdown_before=float(d_before), drawdown_after=float(d_after), incremental_drawdown=float(inc),
        turnover_penalty=float(turnover_penalty), drawdown_penalty=float(drawdown_penalty), reward_r2=float(reward_r2),
    )


R2_INFO_KEYS: tuple[str, ...] = tuple(R2Decomposition.__dataclass_fields__) + ("reward_definition",)


# ───────────────────────────────────────────── environment: certified accounting + R2 reward
class R2RewardMixin:
    """
    Replace the RETURNED reward of a certified environment by R2 and add the
    decomposition to ``info``.  ``reset``, observation, costs, equity, running
    peak, drawdown and every other ``info`` field are the parent's (E1) code;
    the parent ``step`` is called exactly once per step and its accounting
    state is read, never written.
    """

    r2_coefficients: R2Coefficients = FROZEN_R2

    def _init_r2(self, coefficients: R2Coefficients) -> None:
        if not isinstance(coefficients, R2Coefficients):
            raise TypeError("coefficients must be an R2Coefficients")
        self.r2_coefficients = coefficients
        self.reward_definition = REWARD_DEFINITION
        self._r2_last: R2Decomposition | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):  # type: ignore[override]
        self._r2_last = None
        return super().reset(seed=seed, options=options)  # type: ignore[misc]  # E1 reset: equity = peak = initial, D = 0

    def step(self, action: int):  # type: ignore[override]
        if getattr(self, "_done", True):
            raise RuntimeError("step() called on a finished episode; call reset()")
        # E_t and running_peak_t: the parent's own accounting state BEFORE the transition
        equity_before = float(self._equity)  # type: ignore[attr-defined]
        peak_before = float(self._peak)      # type: ignore[attr-defined]
        obs, financial_r1, terminated, truncated, info = super().step(action)  # type: ignore[misc]  # E1 accounting
        # E_{t+1} and running_peak_{t+1}: the parent updated them with the settled equity; nothing later exists
        equity_after = float(self._equity)   # type: ignore[attr-defined]
        peak_after = float(self._peak)       # type: ignore[attr-defined]
        dec = compute_r2(float(financial_r1), int(info["legs"]), equity_before, peak_before, equity_after, peak_after,
                         self.r2_coefficients)
        # consistency with the parent's reported state (marked at the same instant open[t+2])
        if not (math.isclose(dec.equity_after, float(info["equity"]), rel_tol=0.0, abs_tol=0.0)
                and math.isclose(-dec.drawdown_after, float(info["drawdown"]), rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(dec.running_peak_after, float(info["peak_equity"]), rel_tol=0.0, abs_tol=0.0)):
            raise RuntimeError("R2 decomposition disagrees with the environment's accounting state")
        self._r2_last = dec
        info.update(dec.as_info())
        return obs, dec.reward_r2, terminated, truncated, info


class BtcUsdtTradingEnvR2(R2RewardMixin, BtcUsdtTradingEnv):
    """The certified O1 INVESTED/CASH (internal labels LONG/FLAT) environment with the returned reward replaced by R2 (E3 = O1 + R2)."""

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
        r2_coefficients: R2Coefficients = FROZEN_R2,
    ) -> None:
        BtcUsdtTradingEnv.__init__(self, frame, decision_start, decision_end, cost_config=cost_config,
                                   obs_config=obs_config, episode_length=episode_length, random_start=random_start,
                                   initial_equity=initial_equity, render_mode=render_mode, price_end=price_end)
        self._init_r2(r2_coefficients)


# the accounting, reset and observation code of the R2 environment is literally the certified E1 code
assert BtcUsdtTradingEnvR2._observe is BtcUsdtTradingEnv._observe
assert BtcUsdtTradingEnvR2._state_info is BtcUsdtTradingEnv._state_info
assert BtcUsdtTradingEnvR2._choose_start is BtcUsdtTradingEnv._choose_start
