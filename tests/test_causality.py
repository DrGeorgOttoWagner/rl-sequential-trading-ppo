"""Observation at decision t may depend only on rows <= t."""
import numpy as np
import pandas as pd

from btc_rl.env import LONG, BtcUsdtTradingEnv
from btc_rl.observations import ObservationConfig, build_o1_observation

from conftest import make_synthetic_frame


def _run(env, actions):
    obs, _ = env.reset()
    out = [obs.copy()]
    for a in actions:
        obs, *_ = env.step(a)
        out.append(obs.copy())
    return np.stack(out)


def test_future_rows_do_not_change_observations():
    base = make_synthetic_frame(80)
    tampered = base.copy()
    t_cut = 40  # rows strictly after this index are perturbed
    tampered.iloc[t_cut + 1 :, :] *= 3.0  # every column, every future row
    cfg = ObservationConfig(lookback=10)
    kw = dict(decision_start=base.index[15], decision_end=base.index[t_cut], obs_config=cfg)
    actions = [1, 0, 1, 1, 0, 0, 1] * 4
    actions = actions[: t_cut - 15 + 1]
    o1 = _run(BtcUsdtTradingEnv(base, **kw), actions)
    o2 = _run(BtcUsdtTradingEnv(tampered, **kw), actions)
    # The final observation is formed at close[t_cut + 1], which is a perturbed row,
    # so compare all observations formed at or before t_cut.
    np.testing.assert_array_equal(o1[:-1], o2[:-1])
    assert not np.array_equal(o1[-1], o2[-1])  # sanity: the perturbation was real


def test_observation_matches_hand_computed_log_returns():
    df = make_synthetic_frame(60)
    cfg = ObservationConfig(lookback=10)
    env = BtcUsdtTradingEnv(df, df.index[20], df.index[30], obs_config=cfg)
    obs, _ = env.reset()
    closes = df["close"].to_numpy()
    expected = np.diff(np.log(closes[10:21]))  # r_11 .. r_20, last one uses close[20]/close[19]
    np.testing.assert_allclose(obs[:10], expected.astype(np.float32), rtol=1e-6)
    assert obs[10] == 0.0
    obs, *_ = env.step(LONG)
    expected = np.diff(np.log(closes[11:22]))
    np.testing.assert_allclose(obs[:10], expected.astype(np.float32), rtol=1e-6)
    assert obs[10] == 1.0  # position held during bar 21 is the one just chosen


def test_build_o1_uses_only_prefix():
    closes = np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 50)))
    cfg = ObservationConfig(lookback=10)
    t = 30
    full_prefix = build_o1_observation(closes[: t + 1], 0, cfg)
    exact_window = build_o1_observation(closes[t - 10 : t + 1], 0, cfg)
    np.testing.assert_array_equal(full_prefix, exact_window)


def test_env_never_reads_open_of_decision_bar_or_close_after_t(frame):
    """On the real data: obs[t] is identical when opens are randomised (O1 uses closes only)."""
    df = frame.copy()
    rng = np.random.default_rng(0)
    df["open"] = df["open"] * rng.uniform(0.5, 1.5, len(df))
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    e1 = BtcUsdtTradingEnv(frame, "2021-01-01", "2021-01-20")
    e2 = BtcUsdtTradingEnv(df, "2021-01-01", "2021-01-20")
    a = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0] * 2
    np.testing.assert_array_equal(_run(e1, a), _run(e2, a))
