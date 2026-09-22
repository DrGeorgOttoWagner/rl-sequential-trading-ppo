"""
Temporal ordering proof by perturbation.

Claimed timeline (decision index t):
  s_t exists at open[t+1], features through close[t] and position p_t;
  a_t fills at open[t+1];
  r_t is a function of open[t+1], open[t+2] and the cost legs only;
  s_{t+1} exists at open[t+2], features through close[t+1] and position p_{t+1};
  every component of r_t is known at the instant s_{t+1} exists, none earlier.
"""
import math

import numpy as np
import pandas as pd
import pytest

from btc_rl.costs import CONSERVATIVE_COST
from btc_rl.env import FLAT, LONG, BtcUsdtTradingEnv

from conftest import make_synthetic_frame

T0, T1 = 15, 60
ACTIONS = [LONG, LONG, FLAT, LONG, FLAT, FLAT, LONG, LONG, LONG, FLAT] * 5


def _run(frame, n_steps):
    env = BtcUsdtTradingEnv(frame, frame.index[T0], frame.index[T1], cost_config=CONSERVATIVE_COST)
    obs, info = env.reset()
    out = [(obs.copy(), None, info)]
    for a in ACTIONS[:n_steps]:
        obs, r, _, _, info = env.step(a)
        out.append((obs.copy(), r, info))
    return out


def _perturbed(frame, rows, cols=("open", "high", "low", "close", "volume"), factor=1.7):
    df = frame.copy()
    df.iloc[rows, [df.columns.get_loc(c) for c in cols]] *= factor
    return df


@pytest.fixture
def base():
    return make_synthetic_frame(80)


def test_state_instants_and_feature_dates(base):
    env = BtcUsdtTradingEnv(base, base.index[T0], base.index[T1])
    _, info = env.reset()
    assert info["last_feature_date"] == base.index[T0]          # closes through close[t]
    assert info["state_instant_date"] == base.index[T0 + 1]     # s_t exists at open[t+1]
    _, _, _, _, info = env.step(LONG)
    assert info["fill_date"] == base.index[T0 + 1]              # a_t fills at open[t+1]
    assert info["mark_date"] == base.index[T0 + 2]              # r_t settles at open[t+2]
    assert info["last_feature_date"] == base.index[T0 + 1]      # s_{t+1} features through close[t+1]
    assert info["state_instant_date"] == base.index[T0 + 2]     # s_{t+1} exists at open[t+2]
    assert info["state_instant_date"] == info["mark_date"]      # r_t fully known when s_{t+1} exists


def test_reward_depends_only_on_open_fill_open_mark_and_legs(base):
    k = 3  # examine step k, decision index t = T0 + k
    t = T0 + k
    ref = _run(base, k + 1)
    r_ref, info = ref[-1][1], ref[-1][2]
    legs = info["legs"]
    p_new = info["position"]
    o = base["open"].to_numpy()
    expected = legs * math.log(CONSERVATIVE_COST.leg_factor) + p_new * math.log(o[t + 2] / o[t + 1])
    assert r_ref == pytest.approx(expected, abs=1e-12)

    # perturbing open[t+1] or open[t+2] changes r_t ...
    for row in (t + 1, t + 2):
        r_p = _run(_perturbed(base, [row], cols=("open",)), k + 1)[-1][1]
        assert r_p != pytest.approx(r_ref, abs=1e-12)
    # ... while every close (all rows), every other open, and all rows after t+2 leave r_t unchanged
    n = len(base)
    r_all_closes = _run(_perturbed(base, list(range(n)), cols=("close", "high", "low", "volume")), k + 1)[-1][1]
    assert r_all_closes == pytest.approx(r_ref, abs=1e-15)
    other_opens = [i for i in range(n) if i not in (t + 1, t + 2)]
    r_other = _run(_perturbed(base, other_opens, cols=("open",)), k + 1)[-1][1]
    assert r_other == pytest.approx(r_ref, abs=1e-15)
    r_future = _run(_perturbed(base, list(range(t + 3, n))), k + 1)[-1][1]
    assert r_future == pytest.approx(r_ref, abs=1e-15)


def test_state_after_step_uses_close_t_plus_1_but_nothing_after_open_t_plus_2(base):
    k = 3
    t = T0 + k
    ref = _run(base, k + 1)
    obs_ref, r_ref, info_ref = ref[-1]

    # close[t+1] is inside s_{t+1}: perturbing it changes the observation but not the reward
    p = _run(_perturbed(base, [t + 1], cols=("close",)), k + 1)[-1]
    assert not np.array_equal(p[0], obs_ref) and p[1] == pytest.approx(r_ref, abs=1e-15)

    # everything at row t+2 except open (close, high, low, volume) is after the state instant open[t+2]
    p = _run(_perturbed(base, [t + 2], cols=("close", "high", "low", "volume")), k + 1)[-1]
    np.testing.assert_array_equal(p[0], obs_ref)
    assert p[1] == pytest.approx(r_ref, abs=1e-15)
    assert p[2]["equity"] == info_ref["equity"] and p[2]["drawdown"] == info_ref["drawdown"]

    # all rows strictly after t+2: nothing returned so far may change
    p = _run(_perturbed(base, list(range(t + 3, len(base)))), k + 1)
    for (o1, r1, i1), (o2, r2, i2) in zip(ref, p):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2
        assert i1["equity"] == i2["equity"] and i1["max_drawdown"] == i2["max_drawdown"]


def test_information_frontier_is_open_t_plus_2_at_every_step(base):
    """For every step k the whole history returned so far is invariant to rows > t_k + 2."""
    n_steps = 20
    ref = _run(base, n_steps)
    for k in range(n_steps + 1):
        t = T0 + k
        # s_k was returned at instant open[t+1] (k-th state); rows > t+1 must not affect it,
        # and rows > t+2 must not affect anything returned up to and including r_{t}.
        p_state = _run(_perturbed(base, list(range(t + 2, len(base)))), k)
        np.testing.assert_array_equal(p_state[-1][0], ref[k][0])
        assert p_state[-1][2]["equity"] == ref[k][2]["equity"]
        if k < n_steps:
            p_step = _run(_perturbed(base, list(range(t + 3, len(base)))), k + 1)
            assert p_step[-1][1] == ref[k + 1][1]
            np.testing.assert_array_equal(p_step[-1][0], ref[k + 1][0])


def test_position_component_is_the_position_held_during_the_feature_bar(base):
    env = BtcUsdtTradingEnv(base, base.index[T0], base.index[T1])
    obs, _ = env.reset()
    assert obs[-1] == FLAT                       # nothing held during bar T0
    obs, *_ = env.step(LONG)
    assert obs[-1] == LONG                       # p_{t+1}: held from open[t+1], reported in s_{t+1}
    obs, *_ = env.step(FLAT)
    assert obs[-1] == FLAT
