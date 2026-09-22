"""Open-to-open accounting cross-checked against independent reference paths."""
import numpy as np
import pandas as pd
import pytest

from btc_rl import reference
from btc_rl.costs import CONSERVATIVE_COST, ZERO_COST
from btc_rl.env import FLAT, LONG, BtcUsdtTradingEnv, make_split_env
from btc_rl.evaluation import run_episode
from btc_rl.policies import AlwaysFlat, AlwaysLong, RandomPolicy

from conftest import make_synthetic_frame


def _equities(env, policy, seed=None):
    res = run_episode(env, policy, seed=seed)
    return res.curve["equity"].to_numpy(), res


@pytest.mark.parametrize("cost", [ZERO_COST, CONSERVATIVE_COST], ids=["zero", "conservative"])
def test_buy_and_hold_matches_analytic_open_to_open_synthetic(synthetic_frame, cost):
    df = synthetic_frame
    t0, t1 = 15, 70
    env = BtcUsdtTradingEnv(df, df.index[t0], df.index[t1], cost_config=cost)
    eq, res = _equities(env, AlwaysLong())
    expected = reference.buy_and_hold_equity(df["open"].to_numpy(), t0, t1 - t0 + 1, cost)
    np.testing.assert_allclose(eq, expected, rtol=1e-12)
    assert res.metrics["n_legs"] == 1
    # closed-form final equity: (1-c) * open[t1+2] / open[t0+1]
    assert eq[-1] == pytest.approx(cost.leg_factor * df["open"].iloc[t1 + 2] / df["open"].iloc[t0 + 1])


@pytest.mark.parametrize("cost", [ZERO_COST, CONSERVATIVE_COST], ids=["zero", "conservative"])
def test_buy_and_hold_matches_analytic_on_real_test_split(frame, cost):
    env = make_split_env(frame, "test", cost_config=cost)
    eq, res = _equities(env, AlwaysLong())
    i0 = frame.index.get_loc(pd.Timestamp("2021-08-30"))
    expected = reference.buy_and_hold_equity(frame["open"].to_numpy(), i0, 1618, cost)
    np.testing.assert_allclose(eq, expected, rtol=1e-12)
    assert res.curve.index[0] == pd.Timestamp("2021-08-30")
    assert res.curve.index[-1] == pd.Timestamp("2026-02-02")
    assert res.curve["fill_date"].iloc[0] == pd.Timestamp("2021-08-31")
    assert res.curve["mark_date"].iloc[-1] == pd.Timestamp("2026-02-04")
    entry = frame.loc["2021-08-31", "open"]
    final_mark = frame.loc["2026-02-04", "open"]
    assert eq[-1] == pytest.approx(cost.leg_factor * final_mark / entry, rel=1e-12)


@pytest.mark.parametrize("cost", [ZERO_COST, CONSERVATIVE_COST], ids=["zero", "conservative"])
def test_always_flat_stays_at_initial_equity(synthetic_frame, cost):
    df = synthetic_frame
    env = BtcUsdtTradingEnv(df, df.index[15], df.index[70], cost_config=cost, initial_equity=1000.0)
    eq, res = _equities(env, AlwaysFlat())
    assert (eq == 1000.0).all()
    assert res.metrics["n_legs"] == 0
    assert res.metrics["max_drawdown"] == 0.0
    assert res.curve["reward"].abs().max() == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_position_path_matches_vectorised_reference(synthetic_frame, seed):
    df = synthetic_frame
    t0, t1 = 15, 70
    env = BtcUsdtTradingEnv(df, df.index[t0], df.index[t1], cost_config=CONSERVATIVE_COST)
    eq, res = _equities(env, RandomPolicy(seed))
    positions = res.curve["position"].to_numpy()
    assert positions.min() == 0 and positions.max() == 1  # both actions occurred
    expected = reference.position_path_equity(df["open"].to_numpy(), t0, positions, CONSERVATIVE_COST)
    np.testing.assert_allclose(eq, expected, rtol=1e-12)
    # rewards are log equity ratios
    logs = np.diff(np.log(np.concatenate([[1.0], eq])))
    np.testing.assert_allclose(res.curve["reward"].to_numpy(), logs, atol=1e-12)
    assert res.metrics["max_drawdown"] == pytest.approx(reference.max_drawdown(eq))


def test_overnight_gap_is_earned_not_close_to_open(synthetic_frame):
    """Position return must be open[t+2]/open[t+1]-1, not close[t+1]/open[t+1]-1."""
    df = synthetic_frame
    t = 20
    env = BtcUsdtTradingEnv(df, df.index[t], df.index[t + 5])
    env.reset()
    _, _, _, _, info = env.step(LONG)
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    assert info["gross_simple_return"] == pytest.approx(o[t + 2] / o[t + 1] - 1.0)
    assert info["gross_simple_return"] != pytest.approx(c[t + 1] / o[t + 1] - 1.0)
    assert info["fill_date"] == df.index[t + 1] and info["mark_date"] == df.index[t + 2]


def test_no_same_close_fill(synthetic_frame):
    """Reward for the decision at t must not depend on any price at or before close[t]."""
    df = synthetic_frame
    t = 20
    env_a = BtcUsdtTradingEnv(df, df.index[t], df.index[t + 3])
    tampered = df.copy()
    tampered.iloc[: t + 1, :] *= 2.0  # all prices up to and including bar t
    env_b = BtcUsdtTradingEnv(tampered, df.index[t], df.index[t + 3])
    env_a.reset(); env_b.reset()
    _, ra, _, _, ia = env_a.step(LONG)
    _, rb, _, _, ib = env_b.step(LONG)
    assert ra == rb and ia["equity"] == ib["equity"]


def test_drawdown_tracking(synthetic_frame):
    df = synthetic_frame
    env = BtcUsdtTradingEnv(df, df.index[15], df.index[70], cost_config=ZERO_COST)
    env.reset()
    peak, worst = 1.0, 0.0
    done = False
    while not done:
        _, _, _, done, info = env.step(LONG)
        peak = max(peak, info["equity"])
        worst = min(worst, info["equity"] / peak - 1.0)
        assert info["peak_equity"] == pytest.approx(peak)
        assert info["drawdown"] == pytest.approx(info["equity"] / peak - 1.0)
        assert info["max_drawdown"] == pytest.approx(worst)
    assert worst < 0.0  # the synthetic path is not monotone


def test_no_forced_liquidation_at_episode_end(synthetic_frame):
    df = synthetic_frame
    env = BtcUsdtTradingEnv(df, df.index[15], df.index[25], cost_config=CONSERVATIVE_COST)
    _, res = _equities(env, AlwaysLong())
    assert res.metrics["final_position"] == LONG and res.metrics["n_legs"] == 1
