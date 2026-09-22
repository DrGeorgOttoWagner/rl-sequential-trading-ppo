"""configs/foundation.toml must agree with the module-level frozen constants."""
import pandas as pd

from btc_rl.costs import CONSERVATIVE_COST, ZERO_COST
from btc_rl.data import HARD_CUTOFF
from btc_rl.splits import FROZEN_SPLITS


def test_toml_matches_frozen_constants(cfg):
    assert cfg.dataset.hard_cutoff == HARD_CUTOFF == pd.Timestamp("2026-04-29")
    for name, w in FROZEN_SPLITS.items():
        assert cfg.splits[name].start == w.start and cfg.splits[name].end == w.end
    assert cfg.costs["conservative"] == CONSERVATIVE_COST
    assert cfg.costs["zero"] == ZERO_COST
    assert abs(CONSERVATIVE_COST.one_leg_fraction - 0.0015) < 1e-15
    assert abs(CONSERVATIVE_COST.round_trip_fraction - 0.0030) < 1e-15
    assert cfg.observation.lookback == 10
    assert cfg.periods_per_year == 365
    assert cfg.initial_equity == 1.0
