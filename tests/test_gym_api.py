"""Gymnasium API conformance and termination semantics."""
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from btc_rl.env import FLAT, LONG, BtcUsdtTradingEnv, make_split_env
from btc_rl.observations import ObservationConfig


@pytest.mark.filterwarnings("ignore:.*render modes.*:UserWarning")
def test_check_env_synthetic(synthetic_frame):
    df = synthetic_frame
    check_env(BtcUsdtTradingEnv(df, df.index[15], df.index[70]))
    check_env(BtcUsdtTradingEnv(df, df.index[15], df.index[70], episode_length=10, random_start=True))


@pytest.mark.filterwarnings("ignore:.*render modes.*:UserWarning")
def test_check_env_real_splits(frame):
    for split in ("train", "validation", "test"):
        check_env(make_split_env(frame, split))


def test_spaces_and_signatures(synthetic_frame):
    df = synthetic_frame
    env = BtcUsdtTradingEnv(df, df.index[15], df.index[70], obs_config=ObservationConfig(lookback=10))
    assert env.action_space == spaces.Discrete(2)
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (11,)
    assert env.observation_space.dtype == np.float32
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (11,) and env.observation_space.contains(obs)
    assert isinstance(info, dict)
    out = env.step(LONG)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float) and isinstance(terminated, bool) and isinstance(truncated, bool)
    assert env.observation_space.contains(obs)


def test_truncation_at_window_end_and_terminated_never(synthetic_frame):
    df = synthetic_frame
    env = BtcUsdtTradingEnv(df, df.index[15], df.index[20])  # 6 decisions
    env.reset()
    flags = []
    for _ in range(6):
        _, _, terminated, truncated, info = env.step(LONG)
        flags.append((terminated, truncated))
    assert all(t is False for t, _ in flags)
    assert [tr for _, tr in flags] == [False] * 5 + [True]
    assert info["step"] == 6
    with pytest.raises(RuntimeError):
        env.step(FLAT)


def test_invalid_action_and_window_errors(synthetic_frame):
    df = synthetic_frame
    env = BtcUsdtTradingEnv(df, df.index[15], df.index[20])
    env.reset()
    with pytest.raises(ValueError):
        env.step(2)
    with pytest.raises(ValueError):  # not enough warm-up closes
        BtcUsdtTradingEnv(df, df.index[5], df.index[20])
    with pytest.raises(ValueError):  # not enough settlement opens after the last decision
        BtcUsdtTradingEnv(df, df.index[15], df.index[len(df) - 2])
    with pytest.raises(ValueError):
        BtcUsdtTradingEnv(df, df.index[15], df.index[20], episode_length=7)
