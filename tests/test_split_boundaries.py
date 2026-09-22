"""
Purge rule: a transition belongs to a split only when its observation and its
whole execution/reward interval (open[t+1], open[t+2]) lie inside the split.
"""
import numpy as np
import pandas as pd
import pytest

from btc_rl.env import BtcUsdtTradingEnv, make_split_env
from btc_rl.evaluation import run_episode
from btc_rl.observations import build_o1_observation
from btc_rl.policies import AlwaysLong, RandomPolicy
from btc_rl.splits import FROZEN_SPLITS, usable_decision_dates

TRAIN_END, VAL_START = pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01")
VAL_END, TEST_START = pd.Timestamp("2021-08-29"), pd.Timestamp("2021-08-30")
TEST_END = pd.Timestamp("2026-02-04")


def _tamper(frame, from_date=None, to_date=None, factor=3.0):
    df = frame.copy()
    mask = pd.Series(True, index=df.index)
    if from_date is not None:
        mask &= df.index >= pd.Timestamp(from_date)
    if to_date is not None:
        mask &= df.index <= pd.Timestamp(to_date)
    df.loc[mask, ["open", "high", "low", "close"]] *= factor
    return df


def _curve(frame, split, policy, seed=0):
    return run_episode(make_split_env(frame, split), policy, seed=seed).curve


def test_usable_decision_ranges(frame):
    expect = {
        "train": ("2018-03-04", "2020-12-29", 1032),
        "validation": ("2021-01-01", "2021-08-27", 239),
        "test": ("2021-08-30", "2026-02-02", 1618),
    }
    for name, (s, e, n) in expect.items():
        u = usable_decision_dates(frame, FROZEN_SPLITS[name])
        assert (u[0], u[-1], len(u)) == (pd.Timestamp(s), pd.Timestamp(e), n)


@pytest.mark.parametrize("split,last_price", [
    ("train", TRAIN_END), ("validation", VAL_END), ("test", TEST_END)])
def test_every_fill_and_mark_stays_inside_split(frame, split, last_price):
    c = _curve(frame, split, RandomPolicy(3), seed=3)
    assert c["fill_date"].max() <= last_price
    assert c["mark_date"].max() == last_price
    assert c.index.max() == last_price - pd.Timedelta(days=2)
    assert c.index.min() == FROZEN_SPLITS[split].start


def test_train_rewards_never_touch_validation_prices(frame):
    tampered = _tamper(frame, from_date=VAL_START)  # every validation and test price
    a = _curve(frame, "train", RandomPolicy(1), seed=1)
    b = _curve(tampered, "train", RandomPolicy(1), seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_validation_rewards_never_touch_test_prices(frame):
    tampered = _tamper(frame, from_date=TEST_START)
    a = _curve(frame, "validation", RandomPolicy(2), seed=2)
    b = _curve(tampered, "validation", RandomPolicy(2), seed=2)
    pd.testing.assert_frame_equal(a, b)


def test_test_never_consumes_prices_after_frozen_end(frame):
    tampered = _tamper(frame, from_date=TEST_END + pd.Timedelta(days=1))
    a = _curve(frame, "test", AlwaysLong())
    b = _curve(tampered, "test", AlwaysLong())
    pd.testing.assert_frame_equal(a, b)
    env = make_split_env(frame, "test")
    assert env.price_end == TEST_END
    assert env._dates[-1] == TEST_END  # rows after the split end do not exist inside the env


def test_observation_history_may_use_earlier_split(frame):
    """Causal warm-up from the previous split is allowed and actually used."""
    env = make_split_env(frame, "validation")
    obs, info = env.reset()
    assert info["decision_date"] == VAL_START
    closes = frame.loc["2020-12-22":"2021-01-01", "close"].to_numpy()  # 10 train closes + 1
    expected = build_o1_observation(closes, 0, env.obs_config)
    np.testing.assert_array_equal(obs, expected)
    # perturbing train closes changes the first validation observation ...
    tampered = _tamper(frame, to_date=TRAIN_END)
    obs_t, _ = make_split_env(tampered, "validation").reset()
    assert not np.array_equal(obs, obs_t)
    # ... but never a validation reward or equity value
    a = _curve(frame, "validation", RandomPolicy(5), seed=5)
    b = _curve(tampered, "validation", RandomPolicy(5), seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_price_end_boundary_is_enforced_by_env(synthetic_frame):
    df = synthetic_frame
    end = df.index[40]
    # decision at index 38 marks at open[40] == price_end: allowed
    BtcUsdtTradingEnv(df, df.index[15], df.index[38], price_end=end)
    # decision at index 39 would need open[41] > price_end: refused
    with pytest.raises(ValueError, match="settlement"):
        BtcUsdtTradingEnv(df, df.index[15], df.index[39], price_end=end)
    with pytest.raises(ValueError):
        BtcUsdtTradingEnv(df, df.index[15], df.index[30], price_end="2019-06-01")
