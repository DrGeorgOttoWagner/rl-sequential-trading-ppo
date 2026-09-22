"""
btc_rl — deterministic data/accounting foundation for the BTC/USDT RL research study.

Scope: data loading, integrity validation, frozen
splits, minimal causal observation (O1), Gymnasium INVESTED/CASH (internal labels LONG/FLAT) environment with
next-bar-open execution and open-to-open accounting, deterministic baselines.

Out of scope here: PPO training, reward shaping (R2), engineered observations
(O2), deep-learning forecasting.

Research use only.  Not financial advice.
"""

from .costs import CONSERVATIVE_COST, ZERO_COST, CostConfig
from .data import DataIntegrityError, DatasetConfig, load_dataset
from .env import FLAT, LONG, BtcUsdtTradingEnv, make_split_env
from .observations import ObservationConfig
from .splits import FROZEN_SPLITS, SplitWindow

__all__ = [
    "CONSERVATIVE_COST",
    "ZERO_COST",
    "CostConfig",
    "DataIntegrityError",
    "DatasetConfig",
    "load_dataset",
    "FLAT",
    "LONG",
    "BtcUsdtTradingEnv",
    "make_split_env",
    "ObservationConfig",
    "FROZEN_SPLITS",
    "SplitWindow",
]
