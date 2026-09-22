"""
config.py — load ``configs/foundation.toml`` and reject any mismatch with the
frozen constants hard-coded in the modules (F8).

There is exactly one set of frozen parameters.  The TOML file is the written
record; the module constants are the runtime defaults.  Loading fails if the
two disagree, so reporting and execution can never use different windows,
costs, cutoff or dataset identity.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .costs import COST_PROFILES, CostConfig
from .data import (
    EXPECTED_FIRST_DATE,
    EXPECTED_LAST_DATE,
    EXPECTED_ROWS_AFTER_CUTOFF,
    EXPECTED_SHA256,
    HARD_CUTOFF,
    REPO_ROOT,
    DatasetConfig,
    resolve_raw_csv_path,
)
from .observations import ObservationConfig
from .splits import FROZEN_SPLITS, SplitWindow

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "foundation.toml"


class ConfigMismatchError(RuntimeError):
    """The TOML config disagrees with the frozen module constants."""


@dataclass(frozen=True)
class FoundationConfig:
    dataset: DatasetConfig
    splits: dict[str, SplitWindow]
    costs: dict[str, CostConfig]
    observation: ObservationConfig
    periods_per_year: int
    initial_equity: float


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigMismatchError(msg)


def load_foundation_config(path: Path | None = None) -> FoundationConfig:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    d = raw["dataset"]
    dataset = DatasetConfig(
        raw_csv=resolve_raw_csv_path(d["raw_csv"]),
        hard_cutoff=pd.Timestamp(d["hard_cutoff"]),
        expected_sha256=d["expected_sha256"],
        expected_first_date=pd.Timestamp(d["expected_first_date"]),
        expected_last_date=pd.Timestamp(d["expected_last_date"]),
        expected_rows_after_cutoff=int(d["expected_rows_after_cutoff"]),
        verify_identity=True,
    )
    _check(dataset.hard_cutoff == HARD_CUTOFF, f"hard_cutoff {d['hard_cutoff']} != frozen {HARD_CUTOFF.date()}")
    _check(dataset.expected_sha256 == EXPECTED_SHA256, "expected_sha256 differs from frozen constant")
    _check(dataset.expected_first_date == EXPECTED_FIRST_DATE, "expected_first_date differs from frozen constant")
    _check(dataset.expected_last_date == EXPECTED_LAST_DATE, "expected_last_date differs from frozen constant")
    _check(dataset.expected_rows_after_cutoff == EXPECTED_ROWS_AFTER_CUTOFF,
           "expected_rows_after_cutoff differs from frozen constant")

    splits = {k: SplitWindow(k, v[0], v[1]) for k, v in raw["splits"].items()}
    _check(set(splits) == set(FROZEN_SPLITS), f"split names {sorted(splits)} != {sorted(FROZEN_SPLITS)}")
    for k, w in FROZEN_SPLITS.items():
        _check(splits[k] == w, f"split {k} {splits[k].start.date()}..{splits[k].end.date()} != frozen "
                               f"{w.start.date()}..{w.end.date()}")

    costs = {k: CostConfig(**v) for k, v in raw["costs"].items()}
    _check(set(costs) == set(COST_PROFILES), f"cost profiles {sorted(costs)} != {sorted(COST_PROFILES)}")
    for k, c in COST_PROFILES.items():
        _check(costs[k] == c, f"cost profile {k} differs from frozen constant")

    obs = ObservationConfig(**raw["observation"])
    _check(obs.lookback == ObservationConfig().lookback, "observation.lookback differs from module default")

    acc = raw["accounting"]
    periods = int(acc["periods_per_year"])
    _check(periods == 365, "periods_per_year must be 365")
    return FoundationConfig(
        dataset=dataset,
        splits=splits,
        costs=costs,
        observation=obs,
        periods_per_year=periods,
        initial_equity=float(acc["initial_equity"]),
    )


__all__ = ["ConfigMismatchError", "FoundationConfig", "load_foundation_config", "DEFAULT_CONFIG_PATH", "REPO_ROOT"]
