"""
ppo_e1.py — PPO baseline experiment E1 (O1 / R1 / 0.10% fee + 0.05% slippage allowance per trade).

    python -m btc_rl.ppo_e1 --seed 42                       # full E1 run (200k timesteps)
    python -m btc_rl.ppo_e1 --seed 42 --smoke               # 16,384-step integration run
    python -m btc_rl.ppo_e1 --seed 42 --skip-preflight      # only for already-certified sessions

E1 is the control configuration of docs/methodology.md: minimal
observation O1, reward R1 (per-step portfolio log return after costs),
conservative costs (0.10% fee + 0.05% slippage allowance per trade), Stable-Baselines3 PPO with a small MLP.
It establishes a baseline; it does not optimise performance.

Portfolio action semantics (target portfolio state, not an exchange order)
--------------------------------------------------------------------------
    0 = CASH   remain in / return to USDT
    1 = LONG   hold BTC with full capital

    CASH -> CASH   no trade, no cost leg
    CASH -> LONG   buy BTC, exactly one cost leg
    LONG -> LONG   keep the position, no new buy, no cost leg
    LONG -> CASH   sell BTC, exactly one cost leg

The environment's internal constant ``FLAT`` is the same integer as ``CASH``
(0); it is kept unchanged to avoid touching certified code.  No SHORT state
exists, no leverage, full capital.

Split policy (enforced at every runtime boundary)
---------------------------------------------------------------
Every split specification that reaches E1 — a name, a ``SplitWindow``, a
substituted split mapping inside a ``FoundationConfig``, or a prebuilt
environment handed to a diagnostic — is resolved and compared with the
frozen windows before anything is built or run:

* role ``train``:      only the frozen TRAIN window is legal;
* role ``evaluation``: only the frozen VALIDATION window is legal;
* role ``diagnostic``: TRAIN or VALIDATION.

Anything else (the test split under any name or type, a window with
different dates, a mapping whose windows differ from the frozen table, an
environment whose decision range or ``price_end`` differs from the legal
window) raises ``ForbiddenSplitError``.  E1 never constructs a test
environment and never runs a learned policy on one.

Prebuilt environments are verified for semantics as well as
dates: certified ``BtcUsdtTradingEnv``, ``Discrete(2)`` CASH/LONG actions,
O1 observation config (and the exact config expected for the run when it is
known), conservative 0.10% fee + 0.05% slippage allowance per trade, initial equity 1.0, deterministic
full-window episodes (no ``random_start``, no ``episode_length``), and a
wrapper chain consisting only of the supported instrumentation wrapper
``stable_baselines3.common.monitor.Monitor``; any reward/observation/action
transform, ``TimeLimit`` or other wrapper is rejected.  The deterministic
policy adapter ``SB3DeterministicPolicy`` must be bound to a verified
environment and role at construction and checks every decision date against
the bound window before the first ``model.predict`` call, so composing it
with the generic ``evaluation.run_episode`` on an illegal environment fails
before prediction.

Supported guarantee boundary
----------------------------
The isolation guarantees hold for the project's E1 APIs: ``make_e1_env``,
``make_train_env``, ``build_model``, ``evaluate_e1``,
``check_finite_policy_outputs``, ``SB3DeterministicPolicy`` (bound),
``run_e1`` and the canonical report.  ``load_e1_model`` returns an ordinary
Stable-Baselines3 model; calling raw SB3 APIs such as ``model.predict`` on
arbitrary observations, or monkeypatching Python objects, is outside the
project's guarantees.  This is a research-integrity boundary, not a sandbox.

Training episodes and SB3 boundary behaviour
--------------------------------------------
Training episodes are the full TRAIN window (1032 decisions), deterministic
start in CASH, no random start.  200,704 timesteps therefore traverse the
same TRAIN history about 194 times.  At the window end the environment
returns ``truncated=True`` (a time limit, not a terminal state); SB3's
``DummyVecEnv`` flags it as ``TimeLimit.truncated`` and PPO bootstraps the
last rollout reward with ``gamma * V(terminal_observation)``.  The terminal
observation contains close 2020-12-30 (the last usable TRAIN decision close
is 2020-12-29).  Bootstrapping does not change the financial R1 / equity
accounting for a fixed policy (the evaluator adds no such term), but it DOES
change the learning targets and can therefore affect which policy is learned.

Frozen timeline
-------------------------------------------
state s_t exists at open[t+1] with information through close[t]; action a_t
selects CASH or LONG at open[t+1]; reward r_t settles open[t+1] -> open[t+2]
minus the cost of the state transition; state s_{t+1} exists at open[t+2].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, FoundationConfig, load_foundation_config
from .costs import CONSERVATIVE_COST
from .data import load_dataset
from .env import FLAT, LONG, BtcUsdtTradingEnv, make_split_env
from .evaluation import EpisodeResult, run_episode
from .observations import ObservationConfig
from .preflight import LOCK_PATH, PACKAGE_ROOT
from .scaling import ScaleFit, fit_train_return_scale, scaled_observation_config
from .splits import FROZEN_SPLITS, SETTLEMENT_BARS, SplitWindow, usable_decision_index_range

# ───────────────────────────────────────────── portfolio-state semantics
CASH = FLAT  # 0: in USDT.  Same integer as the certified env constant FLAT.
assert CASH == 0 and LONG == 1

ACTION_SEMANTICS: dict[int, str] = {CASH: "CASH", LONG: "LONG"}
TRANSITION_SEMANTICS: dict[str, dict[str, Any]] = {
    "CASH->CASH": {"trade": "none", "legs": 0},
    "CASH->LONG": {"trade": "buy BTC (enter long)", "legs": 1},
    "LONG->LONG": {"trade": "hold existing BTC (no re-buy)", "legs": 0},
    "LONG->CASH": {"trade": "sell BTC (exit long)", "legs": 1},
}

# ───────────────────────────────────────────── split policy
EXPERIMENT = "E1"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "validation"
FORBIDDEN_SPLITS: tuple[str, ...] = ("test",)
LEGAL_WINDOWS: dict[str, tuple[str, ...]] = {
    "train": (TRAIN_SPLIT,),
    "evaluation": (EVAL_SPLIT,),
    "diagnostic": (TRAIN_SPLIT, EVAL_SPLIT),
}
COST_PROFILE = "conservative"
OBSERVATION_DEFINITION = "O1"
REWARD_DEFINITION = "R1"
DEVELOPMENT_SEEDS: tuple[int, ...] = (42, 123, 2026)
FULL_BUDGET_TIMESTEPS = 200_000
SMOKE_BUDGET_TIMESTEPS = 16_384  # 8 rollouts of n_steps=2048, inside the 10k-20k smoke band

DEFAULT_ARTIFACTS_ROOT = PACKAGE_ROOT / "artifacts" / "ppo"


class ForbiddenSplitError(RuntimeError):
    """An E1 boundary received a split, window, mapping or environment that is not legal for its role."""


def _same_window(a: SplitWindow, b: SplitWindow) -> bool:
    return a.name == b.name and a.start == b.start and a.end == b.end


def verify_split_table(splits: Mapping[str, SplitWindow]) -> None:
    """The split mapping in use must be exactly the frozen table (names and dates)."""
    if set(splits) != set(FROZEN_SPLITS):
        raise ForbiddenSplitError(f"split table names {sorted(splits)} != frozen {sorted(FROZEN_SPLITS)}")
    for name, frozen in FROZEN_SPLITS.items():
        w = splits[name]
        if not isinstance(w, SplitWindow) or not _same_window(w, frozen):
            raise ForbiddenSplitError(f"split table entry {name!r} differs from the frozen window")


def resolve_e1_window(spec: Any, cfg: FoundationConfig, role: str) -> SplitWindow:
    """
    Resolve ``spec`` (split name or SplitWindow) to a frozen window legal for
    ``role`` (``train`` / ``evaluation`` / ``diagnostic``).  Fails closed:
    unknown types, unknown names, the test split under any name or type, any
    window whose dates differ from the legal frozen window, and any config
    whose split table differs from the frozen table raise ForbiddenSplitError.
    """
    if role not in LEGAL_WINDOWS:
        raise ValueError(f"unknown E1 role {role!r}")
    verify_split_table(cfg.splits)
    if isinstance(spec, str):
        if spec in FORBIDDEN_SPLITS or spec not in cfg.splits:
            raise ForbiddenSplitError(f"E1 {role}: split name {spec!r} is not legal (legal: {LEGAL_WINDOWS[role]})")
        window = cfg.splits[spec]
    elif isinstance(spec, SplitWindow):
        window = spec
    else:
        raise ForbiddenSplitError(f"E1 {role}: unsupported split specification of type {type(spec).__name__}")
    for legal_name in LEGAL_WINDOWS[role]:
        if _same_window(window, FROZEN_SPLITS[legal_name]):
            return FROZEN_SPLITS[legal_name]  # always return the frozen object, never the caller's
    raise ForbiddenSplitError(
        f"E1 {role}: window {window.name} {window.start.date()}..{window.end.date()} is not a legal "
        f"{'/'.join(LEGAL_WINDOWS[role]).upper()} window; the test split is off-limits for PPO training, "
        "evaluation and every performance-driven decision"
    )


def guard_split(spec: Any, role: str, cfg: FoundationConfig | None = None) -> SplitWindow:
    """Backward-compatible name for ``resolve_e1_window`` (loads the frozen config when none is given)."""
    return resolve_e1_window(spec, cfg or load_foundation_config(), role)


E1_INITIAL_EQUITY = 1.0
E1_LOOKBACK = 10


def _supported_wrapper_types() -> tuple[type, ...]:
    from stable_baselines3.common.monitor import Monitor

    return (Monitor,)


def _wrapper_chain(env: Any) -> tuple[list[type], Any]:
    """Wrapper types from the outside in, and the innermost (unwrapped) environment."""
    import gymnasium as gym

    chain: list[type] = []
    e = env
    seen = 0
    while isinstance(e, gym.Wrapper):
        chain.append(type(e))
        e = e.env
        seen += 1
        if seen > 32:
            raise ForbiddenSplitError("E1: wrapper chain too deep")
    return chain, e


def verify_e1_env(env: Any, role: str, obs_cfg: ObservationConfig | None = None) -> SplitWindow:
    """
    A prebuilt environment is legal for ``role`` only if ALL of the following
    hold:

    * the wrapper chain contains only supported instrumentation wrappers
      (exactly ``stable_baselines3.common.monitor.Monitor``); reward /
      observation / action transforms, ``TimeLimit`` and any other wrapper
      are rejected;
    * the unwrapped environment is the certified ``BtcUsdtTradingEnv``;
    * ``action_space == Discrete(2)`` (0 = CASH, 1 = LONG);
    * the observation config is O1 (lookback 10) and equals ``obs_cfg`` when
      one is given; the declared observation space matches that config;
    * the cost profile is the frozen conservative profile (0.10% fee + 0.05% slippage allowance per trade);
    * initial equity is 1.0;
    * deterministic full-window episodes: ``random_start`` False and no
      ``episode_length``;
    * decision range, last mark and ``price_end`` equal a legal frozen
      window for ``role``, and no later row exists inside the environment.

    Returns the matching frozen window.
    """
    from gymnasium import spaces

    from .observations import observation_space as _obs_space

    if role not in LEGAL_WINDOWS:
        raise ValueError(f"unknown E1 role {role!r}")
    chain, base = _wrapper_chain(env)
    supported = _supported_wrapper_types()
    for w in chain:
        if w not in supported:
            raise ForbiddenSplitError(
                f"E1 {role}: unsupported wrapper {w.__module__}.{w.__name__}; only "
                f"{', '.join(t.__module__ + '.' + t.__name__ for t in supported)} may wrap an E1 environment"
            )
    if type(base) is not BtcUsdtTradingEnv:
        raise ForbiddenSplitError(f"E1 {role}: environment of type {type(base).__name__} is not the certified env")
    for attr in ("_dates", "_i1", "price_end", "obs_config", "cost_config", "random_start", "episode_length",
                 "initial_equity", "action_space", "observation_space"):
        if not hasattr(base, attr):
            raise ForbiddenSplitError(f"E1: environment lacks the certified attribute {attr!r}")
    if base.action_space != spaces.Discrete(2):
        raise ForbiddenSplitError(f"E1 {role}: action space {base.action_space} is not Discrete(2) CASH/LONG")
    oc = base.obs_config
    if not isinstance(oc, ObservationConfig) or oc.lookback != E1_LOOKBACK:
        raise ForbiddenSplitError(f"E1 {role}: observation config {oc!r} is not O1 (lookback {E1_LOOKBACK})")
    if obs_cfg is not None and oc != obs_cfg:
        raise ForbiddenSplitError(f"E1 {role}: observation config {oc!r} differs from the expected {obs_cfg!r}")
    if base.observation_space != _obs_space(oc):
        raise ForbiddenSplitError(f"E1 {role}: observation space does not match the O1 config")
    cost = base.cost_config
    if cost != CONSERVATIVE_COST or cost.one_leg_fraction != 0.0015:
        raise ForbiddenSplitError(f"E1 {role}: cost profile {cost!r} is not the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    if base.initial_equity != E1_INITIAL_EQUITY:
        raise ForbiddenSplitError(f"E1 {role}: initial equity {base.initial_equity} != {E1_INITIAL_EQUITY}")
    if base.random_start is not False or base.episode_length is not None:
        raise ForbiddenSplitError(
            f"E1 {role}: episode options random_start={base.random_start} episode_length={base.episode_length}; "
            "E1 uses deterministic full-window episodes only"
        )
    dates: pd.DatetimeIndex = base._dates
    first, last = base.decision_window
    last_mark = dates[base._i1 + SETTLEMENT_BARS]
    for legal_name in LEGAL_WINDOWS[role]:
        w = FROZEN_SPLITS[legal_name]
        if (first == w.start and last_mark == w.end and base.price_end == w.end and dates[-1] == w.end
                and last == dates[base._i1]):
            return w
    raise ForbiddenSplitError(
        f"E1 {role}: environment decisions {first.date()}..{last.date()}, price_end "
        f"{None if base.price_end is None else base.price_end.date()} do not match a legal frozen "
        f"{'/'.join(LEGAL_WINDOWS[role]).upper()} window"
    )


# ───────────────────────────────────────────── PPO configuration
@dataclass(frozen=True)
class PPOHyperparameters:
    """
    One fixed, non-tuned configuration.  Every field is the Stable-Baselines3
    2.9 PPO default unless the rationale says otherwise; the values are listed
    explicitly so that the metadata is complete even if SB3 defaults change.
    """

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    net_arch: tuple[int, ...] = (64, 64)
    activation: str = "Tanh"

    def sb3_kwargs(self) -> dict[str, Any]:
        import torch.nn as nn

        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "normalize_advantage": self.normalize_advantage,
            "policy_kwargs": {
                "net_arch": {"pi": list(self.net_arch), "vf": list(self.net_arch)},
                "activation_fn": getattr(nn, self.activation),
            },
        }

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["net_arch"] = list(self.net_arch)
        d["policy"] = "MlpPolicy"
        return d


HYPERPARAMETER_RATIONALE: dict[str, str] = {
    "policy": "MlpPolicy: O1 is an 11-dimensional flat vector; no recurrence, no CNN.",
    "net_arch": "[64, 64] separate pi/vf heads: the small MLP the task asks for (equals the SB3 default).",
    "activation": "Tanh: SB3 default for MlpPolicy; bounded activations suit small scaled inputs.",
    "learning_rate": "3e-4: SB3 default, not tuned.",
    "n_steps": "2048: SB3 default; ~2 full training episodes (1032 steps each) per rollout.",
    "batch_size": "64: SB3 default, not tuned.",
    "n_epochs": "10: SB3 default, not tuned.",
    "gamma": "0.99: SB3 default; daily steps, ~100-day effective horizon.",
    "gae_lambda": "0.95: SB3 default, not tuned.",
    "clip_range": "0.2: SB3 default, not tuned.",
    "ent_coef": "0.0: SB3 default; no entropy bonus so behaviour is not artificially randomised.",
    "vf_coef": "0.5: SB3 default, not tuned.",
    "max_grad_norm": "0.5: SB3 default, not tuned.",
    "normalize_advantage": "True: SB3 default.",
    "seed": "Passed to PPO (torch/numpy/random) and to the environment reset; three development seeds.",
    "device": "cpu: a [64, 64] MLP on an 11-d input gains nothing from a GPU; CPU keeps runs reproducible.",
    "torch_threads": "1: single-threaded CPU kernels so that same-seed runs are bit-for-bit repeatable.",
    "total_timesteps": "200,000 fixed budget per seed (SB3 rounds up to a multiple of n_steps: 200,704).",
    "episodes": "Deterministic full-window training episodes (start CASH at the first usable train decision); "
                "~194 passes over the same TRAIN history; time-limit truncation bootstrapped by SB3.",
    "return_scale": "O1 log returns divided by the population std of daily log close returns over TRAIN closes "
                    "only (fit_start = TRAIN start, fit_end = last usable TRAIN decision; frozen study design (docs/methodology.md)); "
                    "content of O1 unchanged.",
}


@dataclass(frozen=True)
class E1Config:
    seed: int
    total_timesteps: int = FULL_BUDGET_TIMESTEPS
    hyperparameters: PPOHyperparameters = field(default_factory=PPOHyperparameters)
    scale_returns_on_train: bool = True
    torch_threads: int = 1
    device: str = "cpu"
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT
    run_group: str = "e1"          # artifacts/ppo/<run_group>/<seed>/
    verbose: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.device != "cpu":
            raise ValueError("E1 runs on CPU only (task requirement); pass device='cpu'")
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root))

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_root / self.run_group / str(self.seed)


# ───────────────────────────────────────────── frozen canonical E1 specification
# The canonical E1 cohort must satisfy this specification exactly (checked by ppo_e1_report on both the
# metadata and the loaded checkpoints).  Smoke, tiny and every alternative run are NON-CANONICAL.
CANONICAL_E1_SPEC: dict[str, Any] = {
    "experiment": EXPERIMENT,
    "seeds": DEVELOPMENT_SEEDS,
    "total_timesteps_requested": FULL_BUDGET_TIMESTEPS,
    "total_timesteps_trained": 200_704,
    "ppo_hyperparameters": PPOHyperparameters().to_json(),
    "target_kl": None,
    "clip_range_vf": None,
    "use_sde": False,
    "vec_envs": 1,
    "device": "cpu",
    "torch_threads": 1,
    "observation": {"definition": OBSERVATION_DEFINITION, "lookback": E1_LOOKBACK, "size": E1_LOOKBACK + 1},
    "scale_fit": {"fit_start": "2018-03-04", "fit_end": "2020-12-29", "n_closes": 1032, "n_returns": 1031,
                  "window_name": TRAIN_SPLIT},  # the VALUE is recomputed from the frozen dataset by the report
    "reward": REWARD_DEFINITION,
    "costs": {"profile": COST_PROFILE, "bps_per_leg": 15.0, "fee_bps": 10.0, "slippage_bps": 5.0, "spread_bps": 0.0,
              "round_trip_bps": 30.0},
    "action_semantics": {"0": "CASH", "1": "LONG"},
    "dataset": {"sha256": "7ff14ebd8f2236eb733073cbea7bfd36d2e1c6977aff89e00bdfde74f3af1305",
                "rows_after_cutoff": 3178, "first_date": "2017-08-17", "last_date": "2026-04-29",
                "hard_cutoff": "2026-04-29"},
    "train_split": {"name": "train", "window_start": "2018-03-04", "window_end": "2020-12-31",
                    "first_usable_decision": "2018-03-04", "last_usable_decision": "2020-12-29",
                    "n_usable_decisions": 1032, "last_mark_open": "2020-12-31"},
    "validation_split": {"name": "validation", "window_start": "2021-01-01", "window_end": "2021-08-29",
                         "first_usable_decision": "2021-01-01", "last_usable_decision": "2021-08-27",
                         "n_usable_decisions": 239, "last_mark_open": "2021-08-29"},
    "training_episodes": {"split": TRAIN_SPLIT, "episode_length": 1032, "random_start": False},
    "random_baseline_seeds": (0, 1, 2, 3, 4),
    "python_version": "3.12.14",
}


# ───────────────────────────────────────────── observation scaling (TRAIN only)
def resolve_observation_config(frame: pd.DataFrame, cfg: FoundationConfig, scale_on_train: bool
                               ) -> tuple[ObservationConfig, ScaleFit | None]:
    """O1, optionally with the return scale fitted on TRAIN closes only (``scaling.fit_train_return_scale``)."""
    train = resolve_e1_window(TRAIN_SPLIT, cfg, "train")
    if not scale_on_train:
        return cfg.observation, None
    fit = fit_train_return_scale(frame, train)
    return scaled_observation_config(cfg.observation, fit), fit


# ───────────────────────────────────────────── environments
def make_e1_env(frame: pd.DataFrame, split: Any, cfg: FoundationConfig, obs_cfg: ObservationConfig,
                role: str = "diagnostic") -> BtcUsdtTradingEnv:
    """Certified split environment under the conservative cost profile for a window legal for ``role``."""
    window = resolve_e1_window(split, cfg, role)
    cost = cfg.costs[COST_PROFILE]
    if cost != CONSERVATIVE_COST or abs(cost.one_leg_fraction - 0.0015) > 1e-15:
        raise RuntimeError("E1 requires the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    env = make_split_env(frame, window, cost_config=cost, obs_config=obs_cfg, splits=FROZEN_SPLITS,
                         initial_equity=cfg.initial_equity)
    verify_e1_env(env, role)
    return env


def make_train_env(frame: pd.DataFrame, cfg: FoundationConfig, obs_cfg: ObservationConfig):
    """Monitor-wrapped TRAIN environment (full-window deterministic episodes)."""
    from stable_baselines3.common.monitor import Monitor

    return Monitor(make_e1_env(frame, TRAIN_SPLIT, cfg, obs_cfg, role="train"))


# ───────────────────────────────────────────── model
def build_model(env, config: E1Config, log_dir: Path | None = None, obs_cfg: ObservationConfig | None = None):
    """PPO on ``env``, which must be (a Monitor-wrapped) frozen TRAIN environment with E1 semantics."""
    import torch
    from stable_baselines3 import PPO

    verify_e1_env(env, "train", obs_cfg)
    torch.set_num_threads(int(config.torch_threads))
    model = PPO("MlpPolicy", env, seed=int(config.seed), device=config.device, verbose=config.verbose,
                **config.hyperparameters.sb3_kwargs())
    if log_dir is not None:
        from stable_baselines3.common.logger import configure

        log_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(log_dir), ["csv"]))
    return model


def load_e1_model(path: Path):
    from stable_baselines3 import PPO

    return PPO.load(str(path), device="cpu")


class SB3DeterministicPolicy:
    """
    Adapter: a trained SB3 model as a ``policies.Policy`` with deterministic
    (argmax) actions, BOUND to a verified E1 environment and role.

    Binding happens at construction (``verify_e1_env``; the model's spaces
    must equal the environment's).  Every ``act`` call checks, before the
    first ``model.predict``, that the decision described by ``info`` lies
    inside the bound frozen window and that the observation is inside the
    model's observation space.  Composing this adapter with the generic
    ``evaluation.run_episode`` on a TEST or otherwise illegal environment
    therefore fails before prediction.  ``role`` is ``evaluation``
    (VALIDATION only) or ``diagnostic`` (TRAIN or VALIDATION).
    """

    def __init__(self, model, env: Any, role: str = "evaluation", name: str = "ppo_e1",
                 obs_cfg: ObservationConfig | None = None) -> None:
        if role not in ("evaluation", "diagnostic"):
            raise ForbiddenSplitError(f"E1 adapter: role {role!r} is not a learned-policy role (evaluation/diagnostic)")
        self.window = verify_e1_env(env, role, obs_cfg)
        base = getattr(env, "unwrapped", env)
        if model.observation_space != base.observation_space or model.action_space != base.action_space:
            raise ForbiddenSplitError("E1 adapter: model spaces differ from the bound environment's spaces")
        self.model = model
        self.role = role
        self.name = name
        self._obs_space = base.observation_space
        self._first_decision, self._last_decision = base.decision_window
        self._predictions = 0

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        try:
            decision = pd.Timestamp(info["decision_date"])
            instant = pd.Timestamp(info["state_instant_date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ForbiddenSplitError("E1 adapter: info lacks the certified decision/state dates") from exc
        if not (self._first_decision <= decision <= self._last_decision) or instant > self.window.end:
            raise ForbiddenSplitError(
                f"E1 adapter bound to {self.window.name} refused decision {decision.date()} "
                f"(legal {self._first_decision.date()}..{self._last_decision.date()}) before prediction"
            )
        if not self._obs_space.contains(np.asarray(obs, dtype=np.float32)):
            raise ForbiddenSplitError("E1 adapter: observation outside the bound observation space")
        action, _ = self.model.predict(obs, deterministic=True)
        self._predictions += 1
        return int(np.asarray(action).item())


# ───────────────────────────────────────────── evaluation (validation only)
def evaluate_e1(model, frame: pd.DataFrame, cfg: FoundationConfig, obs_cfg: ObservationConfig,
                split: Any = EVAL_SPLIT, name: str = "ppo_e1") -> EpisodeResult:
    """Deterministic evaluation of a learned policy; only the frozen VALIDATION window is legal."""
    env = make_e1_env(frame, split, cfg, obs_cfg, role="evaluation")
    policy = SB3DeterministicPolicy(model, env, role="evaluation", name=name, obs_cfg=obs_cfg)
    result = run_episode(env, policy, periods_per_year=cfg.periods_per_year)
    _assert_finite_curve(result)
    result.metrics.update(state_distribution(result))
    return result


def state_distribution(result: EpisodeResult) -> dict[str, float]:
    pos = result.curve["position"].to_numpy()
    return {"fraction_LONG": float(np.mean(pos == LONG)), "fraction_CASH": float(np.mean(pos == CASH))}


def _assert_finite_curve(result: EpisodeResult) -> None:
    num = result.curve.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(num)):
        raise RuntimeError("non-finite value in evaluation curve")


def check_finite_policy_outputs(model, env: BtcUsdtTradingEnv, obs_cfg: ObservationConfig | None = None) -> dict[str, Any]:
    """
    Walk one deterministic episode with the model and assert that every
    observation, reward, action logit and value estimate is finite.  ``env``
    must be a verified TRAIN or VALIDATION environment with E1 semantics
    (``verify_e1_env``) whose spaces equal the model's.
    """
    import torch

    window = verify_e1_env(env, "diagnostic", obs_cfg)
    base = getattr(env, "unwrapped", env)
    if model.observation_space != base.observation_space or model.action_space != base.action_space:
        raise ForbiddenSplitError("E1 diagnostic: model spaces differ from the environment's spaces")
    obs, _ = env.reset()
    n = 0
    done = False
    with torch.no_grad():
        while True:
            if not (np.all(np.isfinite(obs)) and env.observation_space.contains(obs)):
                raise RuntimeError(f"non-finite or out-of-space observation at step {n}")
            obs_t, _ = model.policy.obs_to_tensor(obs)
            logits = model.policy.get_distribution(obs_t).distribution.logits
            value = model.policy.predict_values(obs_t)
            if not (torch.isfinite(logits).all() and torch.isfinite(value).all()):
                raise RuntimeError(f"non-finite policy output at step {n}")
            if done:
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(np.asarray(action).item()))
            if not np.isfinite(reward):
                raise RuntimeError(f"non-finite reward at step {n}")
            done = bool(terminated or truncated)
            n += 1
    return {"steps_checked": n, "window": window.name}


# ───────────────────────────────────────────── provenance
SOURCE_FINGERPRINT_GLOBS: tuple[str, ...] = (
    "pyproject.toml", "uv.lock", ".python-version", "configs/*.toml", "src/btc_rl/*.py", "tests/*.py",
)
KEY_SOURCE_FILES: dict[str, str] = {
    "pyproject_toml": "pyproject.toml",
    "uv_lock": "uv.lock",
    "foundation_config": "configs/foundation.toml",
    "ppo_source": "src/btc_rl/ppo_e1.py",
    "report_source": "src/btc_rl/ppo_e1_report.py",
    "scaler_source": "src/btc_rl/scaling.py",
    "observation_source": "src/btc_rl/observations.py",
    "env_source": "src/btc_rl/env.py",
    "splits_source": "src/btc_rl/splits.py",
    "costs_source": "src/btc_rl/costs.py",
    "evaluation_source": "src/btc_rl/evaluation.py",
    "data_source": "src/btc_rl/data.py",
    "preflight_source": "src/btc_rl/preflight.py",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_files(root: Path = PACKAGE_ROOT, globs: Iterable[str] = SOURCE_FINGERPRINT_GLOBS) -> dict[str, str]:
    """relative path -> sha256 of every source/config file matching the fingerprint globs (git-independent)."""
    root = Path(root)
    out: dict[str, str] = {}
    for pattern in globs:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = sha256_of(p)
    return out


def source_fingerprint(files: Mapping[str, str]) -> str:
    """Single digest over (path, sha256) pairs; identifies the code that ran regardless of commit state."""
    return sha256_text("\n".join(f"{k}:{v}" for k, v in sorted(files.items())))


def _git(*args: str, root: Path = PACKAGE_ROOT) -> str:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.rstrip("\n")


def git_identity(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """HEAD, branch, tracked-diff hash (package scope) and the untracked, non-ignored files in the package."""
    rel = "."
    tracked_diff = _git("diff", "HEAD", "--", rel, root=root)
    untracked = [ln for ln in _git("ls-files", "--others", "--exclude-standard", "--", rel, root=root).splitlines() if ln]
    return {
        "commit": _git("rev-parse", "HEAD", root=root),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", root=root),
        "tracked_changes": bool(tracked_diff),
        "tracked_diff_sha256": sha256_text(tracked_diff),
        "untracked_files": untracked,
        "dirty": bool(tracked_diff) or bool(untracked),
        "scope": "package directory (standalone-public-rl-study)",
    }


def provenance(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    files = source_files(root)
    key = {}
    for label, rel in KEY_SOURCE_FILES.items():
        if rel not in files:
            raise RuntimeError(f"provenance: key source file missing: {rel}")
        key[label] = files[rel]
    return {
        "git": git_identity(root),
        "source_fingerprint": source_fingerprint(files),
        "source_fingerprint_globs": list(SOURCE_FINGERPRINT_GLOBS),
        "source_files": files,
        "key_hashes": key,
    }


# ───────────────────────────────────────────── metadata
def _split_record(frame: pd.DataFrame, window: SplitWindow) -> dict[str, Any]:
    i0, last = usable_decision_index_range(frame, window)
    return {
        "name": window.name,
        "window_start": str(window.start.date()),
        "window_end": str(window.end.date()),
        "first_usable_decision": str(frame.index[i0].date()),
        "last_usable_decision": str(frame.index[last].date()),
        "n_usable_decisions": int(last - i0 + 1),
        "last_mark_open": str(frame.index[last + SETTLEMENT_BARS].date()),
    }


REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "experiment", "git", "source", "python_version", "uv_lock_sha256", "dataset", "seed", "ppo_hyperparameters",
    "hyperparameter_rationale", "total_timesteps_requested", "total_timesteps_trained", "train_split",
    "validation_split", "observation", "reward", "costs", "action_semantics", "transition_semantics",
    "timestamp_utc", "packages", "device", "torch_threads", "training_episodes", "artifacts",
)
# Fields that every run of one canonical experiment must share exactly (checked by the report).
IDENTITY_KEYS: tuple[str, ...] = (
    "experiment", "ppo_hyperparameters", "total_timesteps_requested", "total_timesteps_trained", "train_split",
    "validation_split", "observation", "reward", "costs", "action_semantics", "transition_semantics", "device",
    "torch_threads", "training_episodes",
)


def collect_metadata(config: E1Config, model, frame: pd.DataFrame, dataset_report: dict[str, Any],
                     cfg: FoundationConfig, obs_cfg: ObservationConfig, scale_fit: ScaleFit | None,
                     train_seconds: float, preflight: dict[str, Any] | None = None,
                     source_before: dict[str, Any] | None = None) -> dict[str, Any]:
    import gymnasium
    import stable_baselines3
    import torch

    cost = cfg.costs[COST_PROFILE]
    train = resolve_e1_window(TRAIN_SPLIT, cfg, "train")
    validation = resolve_e1_window(EVAL_SPLIT, cfg, "evaluation")
    train_rec = _split_record(frame, train)
    prov = provenance()
    if scale_fit is not None and scale_fit.scale != obs_cfg.return_scale:
        raise RuntimeError("observation scale differs from the recorded fit")
    source = {k: prov[k] for k in ("source_fingerprint", "source_fingerprint_globs", "key_hashes", "source_files")}
    if source_before is not None:
        # the source snapshot captured BEFORE learning must equal the one captured after it
        source["fingerprint_before_training"] = source_before["source_fingerprint"]
        source["source_stable_during_run"] = source_before["source_fingerprint"] == prov["source_fingerprint"]
        if not source["source_stable_during_run"]:
            raise RuntimeError("source files changed while the run was executing; the run is not attributable")
    meta = {
        "experiment": EXPERIMENT,
        "study_stage": "ppo-e1",
        "run_group": config.run_group,
        "git": prov["git"],
        "source": source,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pyproject_sha256": prov["key_hashes"]["pyproject_toml"],
        "uv_lock_sha256": sha256_of(LOCK_PATH),
        "foundation_config_sha256": sha256_of(DEFAULT_CONFIG_PATH),
        "dataset": {
            "path": str(cfg.dataset.raw_csv),
            "sha256": dataset_report["sha256"],
            "rows_after_cutoff": int(dataset_report["rows"]),
            "first_date": str(pd.Timestamp(dataset_report["first_date"]).date()),
            "last_date": str(pd.Timestamp(dataset_report["last_date"]).date()),
            "hard_cutoff": str(cfg.dataset.hard_cutoff.date()),
        },
        "seed": int(config.seed),
        "ppo_hyperparameters": config.hyperparameters.to_json(),
        "hyperparameter_rationale": HYPERPARAMETER_RATIONALE,
        "total_timesteps_requested": int(config.total_timesteps),
        "total_timesteps_trained": int(model.num_timesteps),
        "train_split": train_rec,
        "validation_split": _split_record(frame, validation),
        "forbidden_splits": list(FORBIDDEN_SPLITS),
        "observation": {
            "definition": OBSERVATION_DEFINITION,
            "content": "10 most recent daily log close returns (scaled) + current portfolio state (0=CASH, 1=LONG)",
            "lookback": int(obs_cfg.lookback),
            "return_scale": float(obs_cfg.return_scale),
            "scale_fit": (scale_fit.to_json() if scale_fit is not None
                          else {"scale": 1.0, "fit_start": None, "fit_end": None, "statistic": "not scaled"}),
            "size": int(obs_cfg.size),
        },
        "reward": {
            "definition": REWARD_DEFINITION,
            "formula": "r_t = legs_t * log(1 - c) + p_{t+1} * log(open[t+2] / open[t+1])",
            "shaping": "none (no drawdown, turnover or Sharpe terms)",
            "sb3_training_target_note": ("at the TRAIN window end (truncated=True) SB3 bootstraps the last rollout "
                                         "reward with gamma * V(terminal_observation); environment reward unchanged"),
        },
        "costs": {
            "profile": COST_PROFILE,
            "bps_per_leg": float(cost.one_leg_fraction * 1e4),
            "fee_bps": cost.fee_bps, "slippage_bps": cost.slippage_bps, "spread_bps": cost.spread_bps,
            "round_trip_bps": float(cost.round_trip_fraction * 1e4),
            "cost_drag_definition": ("total_cost_fraction = 1 - prod_t (1 - cost_fraction_t): proportional "
                                     "terminal-equity reduction from transaction costs for the same action path"),
        },
        "action_semantics": {str(k): v for k, v in ACTION_SEMANTICS.items()},
        "transition_semantics": TRANSITION_SEMANTICS,
        "position_types": "spot only: no SHORT, no leverage, full capital",
        "timeline": ("s_t at open[t+1] (info through close[t]); a_t at open[t+1]; r_t settles "
                     "open[t+1]->open[t+2] minus transition cost; s_{t+1} at open[t+2]"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": float(train_seconds),
        "packages": {
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "device": config.device,
        "torch_threads": int(config.torch_threads),
        "training_episodes": {
            "split": TRAIN_SPLIT,
            "start": "deterministic, first usable train decision, initial state CASH",
            "episode_length": int(train_rec["n_usable_decisions"]),
            "random_start": False,
            "passes_over_train_history": float(model.num_timesteps / train_rec["n_usable_decisions"]),
            "vec_envs": 1,
            "wrapper": "stable_baselines3.common.monitor.Monitor + DummyVecEnv (SB3 default)",
            "time_limit": "truncated=True at the window end; SB3 TimeLimit.truncated bootstrapping applies",
        },
        "artifacts": {
            "dir": str(config.artifacts_dir),
            "model": "model.zip",
            "metadata": "metadata.json",
            "validation_metrics": "validation_metrics.json",
            "validation_curve": "validation_curve.csv",
            "train_log": "train_log/progress.csv",
        },
        "preflight": preflight or {"verdict": "not run inside this process"},
    }
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if missing:
        raise RuntimeError(f"metadata incomplete: {missing}")
    return meta


def validate_metadata(meta: dict[str, Any]) -> list[str]:
    """Return the list of missing or empty required keys (empty list = complete)."""
    problems = []
    for k in REQUIRED_METADATA_KEYS:
        if k not in meta:
            problems.append(f"missing {k}")
        elif meta[k] in (None, "", {}, []):
            problems.append(f"empty {k}")
    if "source" in meta and not meta["source"].get("source_fingerprint"):
        problems.append("empty source.source_fingerprint")
    obs = meta.get("observation", {})
    if isinstance(obs, dict) and obs.get("scale_fit", {}).get("statistic") != "not scaled":
        for k in ("fit_start", "fit_end", "scale"):
            if not obs.get("scale_fit", {}).get(k):
                problems.append(f"missing observation.scale_fit.{k}")
    return problems


# ───────────────────────────────────────────── workflow
@dataclass
class E1RunResult:
    config: E1Config
    model: Any
    obs_config: ObservationConfig
    scale_fit: ScaleFit | None
    metadata: dict[str, Any]
    validation: EpisodeResult
    artifacts_dir: Path


def _json_default(o: Any) -> Any:
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=_json_default, sort_keys=True), encoding="utf-8")


def run_e1(config: E1Config, frame: pd.DataFrame | None = None, cfg: FoundationConfig | None = None,
           dataset_report: dict[str, Any] | None = None, preflight: dict[str, Any] | None = None,
           write_artifacts: bool = True) -> E1RunResult:
    """
    Complete E1 workflow for one seed: TRAIN-only training, checkpoint save,
    reload, finite-output check, deterministic VALIDATION evaluation, metadata.
    Every window is resolved against the frozen table; the test split is
    never built.
    """
    cfg = cfg or load_foundation_config()
    verify_split_table(cfg.splits)
    if frame is None:
        frame, dataset_report = load_dataset(cfg.dataset)
    if dataset_report is None:
        raise ValueError("dataset_report is required when frame is supplied")
    obs_cfg, scale_fit = resolve_observation_config(frame, cfg, config.scale_returns_on_train)
    out = config.artifacts_dir
    if write_artifacts:
        out.mkdir(parents=True, exist_ok=True)

    source_before = provenance()  # source identity captured BEFORE learning (verified stable afterwards)
    train_env = make_train_env(frame, cfg, obs_cfg)
    model = build_model(train_env, config, log_dir=(out / "train_log") if write_artifacts else None, obs_cfg=obs_cfg)
    t0 = time.time()
    model.learn(total_timesteps=int(config.total_timesteps), progress_bar=False)
    train_seconds = time.time() - t0

    if write_artifacts:
        model.save(str(out / "model.zip"))
        model = load_e1_model(out / "model.zip")  # evaluate the checkpoint, not the in-memory object

    finite = check_finite_policy_outputs(model, make_e1_env(frame, EVAL_SPLIT, cfg, obs_cfg, role="diagnostic"), obs_cfg)
    validation = evaluate_e1(model, frame, cfg, obs_cfg, EVAL_SPLIT, name=f"ppo_e1_seed{config.seed}")
    meta = collect_metadata(config, model, frame, dataset_report, cfg, obs_cfg, scale_fit, train_seconds, preflight,
                            source_before=source_before)
    meta["finite_output_check"] = finite
    if write_artifacts:
        write_json(out / "metadata.json", meta)
        write_json(out / "validation_metrics.json", validation.metrics)
        validation.curve.to_csv(out / "validation_curve.csv")
    return E1RunResult(config, model, obs_cfg, scale_fit, meta, validation, out)


def _preflight_or_die() -> dict[str, Any]:
    from .preflight import format_report, run_preflight

    res = run_preflight(require_replay=True, run_tests=True)
    print(format_report(res))
    if not res.certified:
        print("preflight not CERTIFIED: refusing to train PPO", file=sys.stderr)
        sys.exit(res.exit_code or 1)
    return {"verdict": res.verdict, "exit_code": res.exit_code,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checks": [{"name": c.name, "ok": c.ok} for c in res.checks]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1: PPO baseline (O1, R1; modelled costs: 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale), train-only, validation-only evaluation.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=FULL_BUDGET_TIMESTEPS)
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_BUDGET_TIMESTEPS}-step integration run into artifacts/ppo/e1_smoke/")
    ap.add_argument("--skip-preflight", action="store_true", help="do not re-run the certification gate in-process")
    ap.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args(argv)

    config = E1Config(seed=args.seed, total_timesteps=SMOKE_BUDGET_TIMESTEPS if args.smoke else args.timesteps,
                      artifacts_root=args.artifacts_root, run_group="e1_smoke" if args.smoke else "e1",
                      verbose=args.verbose)
    preflight = None if args.skip_preflight else _preflight_or_die()
    res = run_e1(config, preflight=preflight)
    m = res.validation.metrics
    fit = res.metadata["observation"]["scale_fit"]
    print(f"\nE1 seed={config.seed} timesteps={res.metadata['total_timesteps_trained']} "
          f"train {res.metadata['train_seconds']:.0f}s -> {res.artifacts_dir}")
    print(f"scale {fit['scale']:.10f} fitted on TRAIN closes {fit['fit_start']}..{fit['fit_end']} "
          f"| source fingerprint {res.metadata['source']['source_fingerprint'][:12]}")
    print(f"validation ({m['first_decision_date'].date()}..{m['last_decision_date'].date()}, {m['n_steps']} steps): "
          f"total_return={m['total_return']:+.4f} final_equity={m['final_equity']:.4f} "
          f"sharpe={m['sharpe_ratio'] if m['sharpe_ratio'] is None else round(m['sharpe_ratio'], 3)} "
          f"max_dd={m['max_drawdown']:+.4f} legs={m['n_legs']} entries={m['n_entries']} "
          f"exposure={m['exposure']:.3f} cost_drag={m['total_cost_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
