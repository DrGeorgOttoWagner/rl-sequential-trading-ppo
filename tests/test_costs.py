"""Exactly one cost leg per position change; none otherwise."""
import math

import numpy as np
import pytest

from btc_rl.costs import CONSERVATIVE_COST, ZERO_COST, CostConfig
from btc_rl.env import FLAT, LONG, BtcUsdtTradingEnv


@pytest.fixture
def env(constant_frame):
    # constant prices: equity changes only through costs
    return BtcUsdtTradingEnv(constant_frame, constant_frame.index[12], constant_frame.index[30],
                             cost_config=CONSERVATIVE_COST)


LEG = CONSERVATIVE_COST.leg_factor  # 1 - 0.0015


def test_flat_to_flat_charges_nothing(env):
    env.reset()
    _, r, _, _, info = env.step(FLAT)
    assert info["legs"] == 0 and info["cost_fraction"] == 0.0
    assert info["equity"] == 1.0 and r == 0.0


def test_flat_to_long_charges_exactly_one_entry_leg(env):
    env.reset()
    _, r, _, _, info = env.step(LONG)
    assert info["legs"] == 1
    assert info["equity"] == pytest.approx(LEG, rel=1e-15)
    assert r == pytest.approx(math.log(LEG), rel=1e-12)


def test_long_to_long_charges_nothing(env):
    env.reset()
    env.step(LONG)
    _, r, _, _, info = env.step(LONG)
    assert info["legs"] == 0
    assert info["equity"] == pytest.approx(LEG, rel=1e-15)
    assert r == 0.0


def test_long_to_flat_charges_exactly_one_exit_leg(env):
    env.reset()
    env.step(LONG)
    env.step(LONG)
    _, r, _, _, info = env.step(FLAT)
    assert info["legs"] == 1
    assert info["equity"] == pytest.approx(LEG ** 2, rel=1e-15)
    assert r == pytest.approx(math.log(LEG), rel=1e-12)
    _, r, _, _, info = env.step(FLAT)
    assert info["legs"] == 0 and r == 0.0 and info["legs_paid"] == 2


def test_round_trip_total(env):
    env.reset()
    for a in (LONG, LONG, FLAT, FLAT, LONG, FLAT):
        _, _, _, _, info = env.step(a)
    assert info["legs_paid"] == 4
    assert info["equity"] == pytest.approx(LEG ** 4, rel=1e-14)
    assert 1.0 - info["equity"] == pytest.approx(1 - (1 - CONSERVATIVE_COST.one_leg_fraction) ** 4)


def test_zero_cost_profile_never_charges(constant_frame):
    env = BtcUsdtTradingEnv(constant_frame, constant_frame.index[12], constant_frame.index[30],
                            cost_config=ZERO_COST)
    env.reset()
    for a in (LONG, FLAT, LONG, FLAT):
        _, r, _, _, info = env.step(a)
        assert r == 0.0 and info["equity"] == 1.0
    assert info["legs_paid"] == 4


def test_cost_config_values():
    c = CostConfig(fee_bps=10, slippage_bps=5)
    assert c.one_leg_fraction == pytest.approx(0.0015)
    assert c.round_trip_fraction == pytest.approx(0.0030)
    assert c.factor_for_legs(2) == pytest.approx((1 - 0.0015) ** 2)
    with pytest.raises(ValueError):
        CostConfig(fee_bps=-1)
