"""
evaluation.py — episode runner and metrics.

Metrics follow the ML project's ``trade_metrics.py`` in spirit (total return,
annualised return, annualised volatility, Sharpe, max drawdown, exposure,
trade count) but are recomputed here from the environment's open-to-open
equity path with 365 periods per year (frozen study design (docs/methodology.md)).  Classification
metrics and the ML trade log are not reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .env import LONG, BtcUsdtTradingEnv
from .policies import Policy

PERIODS_PER_YEAR = 365


@dataclass
class EpisodeResult:
    policy: str
    curve: pd.DataFrame  # one row per step, indexed by decision date
    metrics: dict[str, Any]
    start_index: int
    end_index: int


def run_episode(
    env: BtcUsdtTradingEnv,
    policy: Policy,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> EpisodeResult:
    """Run one full episode; the policy sees only (obs, info)."""
    policy.reset(seed)
    obs, info = env.reset(seed=seed, options=options)
    rows: list[dict[str, Any]] = []
    done = False
    while not done:
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(
            {
                "decision_date": info["settled_decision_date"],
                "fill_date": info["fill_date"],
                "mark_date": info["mark_date"],
                "action": info["action"],
                "position": info["position"],
                "legs": info["legs"],
                "cost_fraction": info["cost_fraction"],
                "gross_simple_return": info["gross_simple_return"],
                "position_return": info["position_return"],
                "reward": reward,
                "equity": info["equity"],
                "drawdown": info["drawdown"],
            }
        )
        done = terminated or truncated
    curve = pd.DataFrame(rows).set_index("decision_date")
    metrics = compute_metrics(curve, env.initial_equity, periods_per_year)
    metrics["policy"] = policy.name
    return EpisodeResult(
        policy=policy.name,
        curve=curve,
        metrics=metrics,
        start_index=info["episode_start_index"],
        end_index=info["episode_end_index"],
    )


def compute_metrics(
    curve: pd.DataFrame, initial_equity: float = 1.0, periods_per_year: int = PERIODS_PER_YEAR
) -> dict[str, Any]:
    equity = curve["equity"].to_numpy(dtype=np.float64)
    n = len(equity)
    if n == 0:
        raise ValueError("empty curve")
    prev = np.concatenate([[initial_equity], equity[:-1]])
    step_ret = equity / prev - 1.0
    final = float(equity[-1])
    total_return = final / initial_equity - 1.0
    ann_return = (final / initial_equity) ** (periods_per_year / n) - 1.0
    std = float(np.std(step_ret, ddof=1)) if n > 1 else 0.0
    ann_vol = std * np.sqrt(periods_per_year)
    sharpe = float(np.mean(step_ret) / std * np.sqrt(periods_per_year)) if std > 0 else None
    peak = np.maximum.accumulate(np.concatenate([[initial_equity], equity]))
    dd = np.concatenate([[initial_equity], equity]) / peak - 1.0
    legs = int(curve["legs"].sum())
    positions = curve["position"].to_numpy()
    entries = int(((positions == LONG) & (np.concatenate([[0], positions[:-1]]) == 0)).sum())
    return {
        "n_steps": n,
        "first_decision_date": curve.index[0],
        "last_decision_date": curve.index[-1],
        "final_equity": final,
        "total_return": float(total_return),
        "annualised_return": float(ann_return),
        "annualised_volatility": float(ann_vol),
        "sharpe_ratio": sharpe,
        "max_drawdown": float(dd.min()),
        "exposure": float(np.mean(positions == LONG)),
        "n_legs": legs,
        "n_entries": entries,
        "total_cost_fraction": float(1.0 - np.prod(1.0 - curve["cost_fraction"].to_numpy())),
        "final_position": int(positions[-1]),
        "sum_reward": float(curve["reward"].sum()),
        "periods_per_year": periods_per_year,
    }
