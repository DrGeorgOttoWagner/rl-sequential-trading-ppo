"""
baselines.py — deterministic baseline sanity run (no files written).

Usage (from the package directory, with the venv active):

    python -m btc_rl.baselines --split test --seeds 0 1 2 3 4 [--no-replay]

Runs always-FLAT, buy-and-hold, the seeded random policy and (by default) the
ML Variant A replay through the environment under the zero and conservative
cost profiles and prints a metrics table.  Nothing is saved.  Running this on
the test split is mechanical verification only (see README, test-window
policy); it is not model selection.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import FoundationConfig, load_foundation_config
from .data import load_dataset
from .env import make_split_env
from .evaluation import run_episode
from .policies import AlwaysFlat, AlwaysLong, RandomPolicy, make_variant_a_replay
from .splits import check_split_layout

COLUMNS = [
    "policy", "cost", "n_steps", "final_equity", "total_return", "annualised_return",
    "annualised_volatility", "sharpe_ratio", "max_drawdown", "exposure", "n_legs",
    "total_cost_fraction", "final_position",
]


def run_baselines(
    split: str = "test",
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    include_replay: bool = True,
    cfg: FoundationConfig | None = None,
    frame: pd.DataFrame | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    cfg = cfg or load_foundation_config()
    if frame is None:
        frame, report = load_dataset(cfg.dataset)
        if verbose:
            print(f"dataset: {report['rows']} rows, {report['first_date'].date()} .. "
                  f"{report['last_date'].date()}, sha256={report['sha256'][:12]}, identity verified")
    layout = check_split_layout(frame, cfg.splits, lookback=cfg.observation.lookback)
    if verbose:
        for name, li in layout.items():
            print(f"split {name:<10} {li['start'].date()} .. {li['end'].date()}  "
                  f"usable decisions {li['start'].date()} .. {li['last_usable_decision'].date()} "
                  f"(n={li['n_usable_decisions']}, last mark {li['last_mark_date'].date()})")

    window = cfg.splits[split]
    policies = [AlwaysFlat(), AlwaysLong()]
    if include_replay:
        policies.append(make_variant_a_replay(frame, window))

    rows = []
    for cost_name, cost in cfg.costs.items():
        env = make_split_env(frame, window, cost_config=cost, obs_config=cfg.observation,
                             splits=cfg.splits, initial_equity=cfg.initial_equity)
        for policy in policies:
            m = run_episode(env, policy, periods_per_year=cfg.periods_per_year).metrics
            rows.append({**m, "cost": cost_name})
        rand = []
        for s in seeds:
            m = run_episode(env, RandomPolicy(s), periods_per_year=cfg.periods_per_year).metrics
            rand.append(m)
            rows.append({**m, "cost": cost_name})
        agg = {k: float(np.mean([r[k] for r in rand])) for k in
               ("final_equity", "total_return", "annualised_return", "annualised_volatility",
                "max_drawdown", "exposure", "n_legs", "total_cost_fraction")}
        agg["sharpe_ratio"] = float(np.mean([r["sharpe_ratio"] for r in rand if r["sharpe_ratio"] is not None]))
        rows.append({"policy": f"random_mean({len(seeds)} seeds)", "cost": cost_name,
                     "n_steps": rand[0]["n_steps"], "final_position": np.nan, **agg})
    return pd.DataFrame(rows)[COLUMNS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-replay", action="store_true", help="skip the ML Variant A replay row")
    args = ap.parse_args()
    table = run_baselines(args.split, tuple(args.seeds), include_replay=not args.no_replay)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(f"\nbaselines on split={args.split}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
