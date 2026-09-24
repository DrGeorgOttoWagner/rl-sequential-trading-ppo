import numpy as np
import pandas as pd
import pytest

from btc_rl.config import load_foundation_config
from btc_rl.data import load_dataset


@pytest.fixture(scope="session")
def cfg():
    return load_foundation_config()


@pytest.fixture(scope="session")
def frame(cfg):
    # Real-data fixture: the dataset ships with the repository; skip only if the checkout is incomplete.
    if not cfg.dataset.raw_csv.exists():
        pytest.skip("dataset file is missing; restore the complete repository checkout")
    df, _ = load_dataset(cfg.dataset)
    return df


def make_synthetic_frame(n: int = 80, seed: int = 7, start: str = "2020-01-01") -> pd.DataFrame:
    """Deterministic daily OHLCV with genuine overnight gaps (close[t] != open[t+1])."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    opens = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, n)))
    closes = opens * np.exp(rng.normal(0.0, 0.02, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol = rng.uniform(100, 200, n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def make_constant_frame(n: int = 40, price: float = 100.0, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="D")
    p = np.full(n, price)
    return pd.DataFrame(
        {"open": p, "high": p, "low": p, "close": p, "volume": np.ones(n)},
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def synthetic_frame():
    return make_synthetic_frame()


@pytest.fixture
def constant_frame():
    return make_constant_frame()
