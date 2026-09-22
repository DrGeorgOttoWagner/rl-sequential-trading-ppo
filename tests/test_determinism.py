"""Same seed and config -> identical episode start and results."""
import numpy as np
import pandas as pd

from btc_rl.env import BtcUsdtTradingEnv, make_split_env
from btc_rl.evaluation import run_episode
from btc_rl.policies import RandomPolicy


def test_random_start_and_random_policy_are_seed_deterministic(synthetic_frame):
    df = synthetic_frame
    def one(seed):
        env = BtcUsdtTradingEnv(df, df.index[15], df.index[70], episode_length=20, random_start=True)
        return run_episode(env, RandomPolicy(seed), seed=seed)
    a, b = one(123), one(123)
    assert a.start_index == b.start_index and a.end_index == b.end_index
    assert a.end_index - a.start_index + 1 == 20
    pd.testing.assert_frame_equal(a.curve, b.curve)
    c = one(124)
    assert c.start_index != a.start_index or not c.curve["action"].equals(a.curve["action"])


def test_random_start_stays_inside_window(synthetic_frame):
    df = synthetic_frame
    env = BtcUsdtTradingEnv(df, df.index[15], df.index[70], episode_length=20, random_start=True)
    starts = set()
    for s in range(200):
        _, info = env.reset(seed=s)
        starts.add(info["episode_start_index"])
        assert 15 <= info["episode_start_index"] <= 70 - 20 + 1
        assert info["episode_end_index"] == info["episode_start_index"] + 19
    assert len(starts) > 10


def test_explicit_start_options(synthetic_frame):
    df = synthetic_frame
    env = BtcUsdtTradingEnv(df, df.index[15], df.index[70], episode_length=5)
    _, info = env.reset(options={"start_index": 40})
    assert info["decision_date"] == df.index[40] and info["episode_end_index"] == 44
    _, info = env.reset(options={"start_date": df.index[68]})
    assert info["episode_start_index"] == 68 and info["episode_end_index"] == 70  # clipped to window
    _, info = env.reset()
    assert info["episode_start_index"] == 15


def test_real_data_full_episode_is_reproducible(frame):
    r1 = run_episode(make_split_env(frame, "validation"), RandomPolicy(7), seed=7)
    r2 = run_episode(make_split_env(frame, "validation"), RandomPolicy(7), seed=7)
    pd.testing.assert_frame_equal(r1.curve, r2.curve)
    assert r1.metrics == r2.metrics
    assert r1.metrics["n_steps"] == 239
