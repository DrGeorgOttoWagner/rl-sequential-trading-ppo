"""
ppo_e4.py — PPO experiment E4 (O2 engineered observation / R2 risk-aware reward / 0.10% fee + 0.05% slippage allowance per trade).

    python -m btc_rl.ppo_e4 --seed 42                       # full E4 run (200k timesteps)
    python -m btc_rl.ppo_e4 --seed 42 --smoke               # 16,384-step integration run
    python -m btc_rl.ppo_e4 --seed 42 --skip-preflight      # only for already-certified sessions

E4 completes the frozen 2x2 matrix of docs/methodology.md
(E1 = O1+R1, E2 = O2+R1, E3 = O1+R2, E4 = O2+R2).  It is a COMPOSITION of two
accepted components and defines nothing new:

* observation: the accepted E2 O2 environment ``ppo_e2.BtcUsdtTradingEnvO2``
  (8 engineered close-derived features + position, TRAIN-only per-feature
  standardisation, ``observations_o2.py``), reused unchanged;
* reward: the accepted E3 ``reward_r2.R2RewardMixin`` with the frozen
  coefficients (turnover 0.0005 per leg, drawdown 0.10 per unit of incremental
  drawdown), reused unchanged.

``BtcUsdtTradingEnvO2R2 = R2RewardMixin + BtcUsdtTradingEnvO2``: the mixin's
``step`` calls the O2 environment's inherited certified E1 ``step`` for every
financial quantity and replaces only the returned reward; ``_observe`` is the
O2 override.  Everything else (dataset, frozen splits, timeline, CASH/LONG
actions, 0.10% fee + 0.05% slippage allowance per trade, SB3 PPO and every hyperparameter, [64, 64] Tanh MLP, CPU,
one torch thread, one environment, seeds 42/123/2026, 200,000-step budget,
deterministic full-TRAIN-window episodes, validation accounting, baselines) is
the E1 configuration.  ``CANONICAL_E4_SPEC`` is derived from
``CANONICAL_E2_SPEC`` by replacing only the reward block with E3's frozen R2
block; tests assert the spec equals E2's outside the reward keys and E3's
outside the observation/scaler keys.

Controls: E2 (same observation, reward R1) and E3 (same reward, observation
O1).  Artifacts go to ``artifacts/ppo/e4/<seed>/``; ``E4Config`` refuses run
groups / resolved paths that alias the accepted E1, E2 or E3 directories
(guard extended to E3).  The accepted E1/E2/E3 run files and the
training-core sources are anchored by SHA-256 and verified before and after
every comparison.
"""

from __future__ import annotations

import argparse
import copy
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, FoundationConfig, load_foundation_config
from .costs import CONSERVATIVE_COST, CostConfig
from .data import load_dataset
from .env import FLAT, LONG
from .evaluation import EpisodeResult
from .observations_o2 import (
    FEATURE_DEFINITIONS,
    O2_FEATURES,
    O2_N_FEATURES,
    O2_SIZE,
    O2_WARMUP_ROWS,
    O2_WINDOW,
    RAW_FEATURE_BOUNDS,
    RSI_DEFINITION,
    O2ObservationConfig,
    O2ScalerFit,
    o2_observation_space,
    verify_o2_warmup,
)
from .ppo_e1 import (
    ACTION_SEMANTICS,
    CASH,
    COST_PROFILE,
    DEFAULT_ARTIFACTS_ROOT,
    DEVELOPMENT_SEEDS,
    E1_INITIAL_EQUITY,
    EVAL_SPLIT,
    FORBIDDEN_SPLITS,
    FULL_BUDGET_TIMESTEPS,
    KEY_SOURCE_FILES,
    LEGAL_WINDOWS,
    REQUIRED_METADATA_KEYS,
    SMOKE_BUDGET_TIMESTEPS,
    SOURCE_FINGERPRINT_GLOBS,
    TRAIN_SPLIT,
    TRANSITION_SEMANTICS,
    ForbiddenSplitError,
    PPOHyperparameters,
    _assert_finite_curve,
    _preflight_or_die,
    _same_window,
    _split_record,
    _supported_wrapper_types,
    _wrapper_chain,
    git_identity,
    load_e1_model,
    sha256_of,
    source_files,
    source_fingerprint,
    state_distribution,
    verify_split_table,
    write_json,
)
from .ppo_e2 import (
    CANONICAL_E2_SPEC,
    E1_FROZEN_ARTIFACTS,
    E1_RUN_GROUP,
    HYPERPARAMETER_RATIONALE_E2,
    BtcUsdtTradingEnvO2,
    resolve_o2_config,
)
from .ppo_e3 import (
    E2_FROZEN_ARTIFACTS,
    E2_FROZEN_CORE_SOURCE_HASHES,
    E2_RUN_GROUP,
    E3_RUN_GROUP,
    R2_CURVE_COLUMNS,
    R2_SPEC,
    REWARD_DIAGNOSTIC_KEYS,
    RUN_GROUP_PATTERN,
    artifact_status,
    reward_diagnostics,
    reward_metadata,
    run_episode_e3,
    verify_e3_output_dir,
)
from .preflight import LOCK_PATH, PACKAGE_ROOT
from .reward_r2 import (
    DRAWDOWN_PENALTY_LAMBDA,
    FROZEN_R2,
    REWARD_DEFINITION,
    TURNOVER_PENALTY_LAMBDA,
    R2Coefficients,
    R2RewardMixin,
)
from .splits import FROZEN_SPLITS, SETTLEMENT_BARS, SplitWindow, usable_decision_index_range

assert CASH == FLAT == 0 and LONG == 1

# ───────────────────────────────────────────── experiment identity
EXPERIMENT = "E4"
CONTROL_EXPERIMENTS: tuple[str, ...] = ("E2", "E3")
STUDY_STAGE = "ppo-e4"
OBSERVATION_DEFINITION = "O2"
E4_RUN_GROUP = "e4"
E4_SMOKE_RUN_GROUP = "e4_smoke"
FORBIDDEN_RUN_GROUP_PREFIXES: tuple[str, ...] = ("e1", "e2", "e3")   # E4 never writes into an E1/E2/E3 run group
PROTECTED_ARTIFACT_DIRS: tuple[Path, ...] = tuple(
    (DEFAULT_ARTIFACTS_ROOT / g).resolve() for g in (E1_RUN_GROUP, E2_RUN_GROUP, E3_RUN_GROUP))

# provenance: E1's key sources plus the O2, R2 and E4 sources
KEY_SOURCE_FILES_E4: dict[str, str] = {
    **KEY_SOURCE_FILES,
    "o2_observation_source": "src/btc_rl/observations_o2.py",
    "reward_r2_source": "src/btc_rl/reward_r2.py",
    "ppo_e4_source": "src/btc_rl/ppo_e4.py",
    "e4_report_source": "src/btc_rl/ppo_e4_report.py",
}

# SHA-256 anchors of the accepted E3 artifacts (closure, commit eab52b80; checkpoint hashes equal those in
# the independent review record).  Taken before any E4 work.
E3_FROZEN_ARTIFACTS: dict[int, dict[str, str]] = {
    42: {
        "model.zip": "fba80e8a06706563d09635b22ccbf67a27069e45f79e77b24c93d4188836f763",
        "metadata.json": "d661d75b3c61d325aadacb31b19a31cd53aa99d91bc59ba370204e0ca38347a4",
        "validation_curve.csv": "6db14d1ecf04a0c7476299fe7ea757631d69fe8ac789850d16a557214e871f37",
        "validation_metrics.json": "0f71b4607c7c08bd23bb3ede1337a46d65beef22d2d4693d86d7de492aa6eedf",
    },
    123: {
        "model.zip": "c308962a9186a3aa429af67a8baafac9f6a5c1356f62a0aa8d4d6f0fda09832b",
        "metadata.json": "327af857fee4e4d96e6941ef0a54046917dce931df362571ac512a8d10cef195",
        "validation_curve.csv": "4c74e22ff4061bfe40cd990ea404b9a9f0b3e7d8bd415756a0c6671995987895",
        "validation_metrics.json": "a0538bacffe25f3af55cc3d4905734b240b2d2f601b855173caaa456d3dcb70f",
    },
    2026: {
        "model.zip": "9fbc160cddb65abf955b5ec5deb093f978ed45fbc7491978b3d696c51c0cacaf",
        "metadata.json": "c238e61c19fefe432d17d4c059ba996e78a725690c9b68f51814b33aa12c6c2d",
        "validation_curve.csv": "b4b40033d55ec19a1c6bfa2323bf2d4a78c1cccc99c4be80c2f833ba309e2b0d",
        "validation_metrics.json": "ab2c1e483cbc6a3d23c12d436d9172b85a6d71b89542084286b93076bab7cf07",
    },
}
# E1 core + O2 source (as recorded by the accepted E2 runs) + R2 source (as recorded by the accepted E3 runs)
E3_FROZEN_CORE_SOURCE_HASHES: dict[str, str] = {
    **E2_FROZEN_CORE_SOURCE_HASHES,
    "reward_r2_source": "5d2a673cbb93cfe9bd857c92ed501c3a350a1647923f7bceae0fc8df7c1a42ec",
}
ACCEPTED_ANCHORS: tuple[tuple[str, dict[int, dict[str, str]], str], ...] = (
    ("E1", E1_FROZEN_ARTIFACTS, E1_RUN_GROUP), ("E2", E2_FROZEN_ARTIFACTS, E2_RUN_GROUP), ("E3", E3_FROZEN_ARTIFACTS, E3_RUN_GROUP),
)


def verify_accepted_core_sources_unchanged(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Every E1, E2 and E3 training-core source of the current tree must hash to the accepted values."""
    hashes = {label: sha256_of(Path(root) / KEY_SOURCE_FILES_E4[label]) for label in E3_FROZEN_CORE_SOURCE_HASHES}
    changed = sorted(k for k, v in hashes.items() if v != E3_FROZEN_CORE_SOURCE_HASHES[k])
    if changed:
        raise RuntimeError(f"E1/E2/E3 training-core sources changed since the accepted runs: {changed}")
    return {"unchanged": True, "hashes": hashes}


def verify_accepted_artifacts_unchanged(artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, Any]:
    """The accepted E1, E2 AND E3 run files must be present and byte-identical to their anchors."""
    out = {}
    for label, anchors, group in ACCEPTED_ANCHORS:
        status = artifact_status(anchors, artifacts_root, group)
        if not status["all_present"]:
            raise RuntimeError(f"{label} canonical artifacts missing: {status['missing']}")
        if not status["unchanged"]:
            raise RuntimeError(f"{label} canonical artifacts differ from the frozen anchors: {status['changed']}")
        out[label] = status
    return out


# ───────────────────────────────────────────── output-path guard (E3 added to the controls)
def validate_run_group(run_group: Any) -> str:
    if not isinstance(run_group, str) or not run_group:
        raise ValueError("run_group must be a non-empty string")
    if not RUN_GROUP_PATTERN.fullmatch(run_group):
        raise ValueError(
            f"run_group {run_group!r} is not a single plain path component (letters, digits, '_' and '-' only; "
            "no '/', '\\\\', '.', '..' or absolute paths); E4 never writes outside artifacts/<run_group>/<seed>"
        )
    if any(run_group == p or run_group.startswith(p + "_") for p in FORBIDDEN_RUN_GROUP_PREFIXES):
        raise ValueError(f"run_group {run_group!r} collides with the frozen E1/E2/E3 artifacts; E4 never writes there")
    return run_group


def verify_e4_output_dir(path: Path, protected: tuple[Path, ...] | None = None) -> Path:
    """Resolved output directory must not equal, descend into or contain the accepted E1/E2/E3 directories."""
    return verify_e3_output_dir(path, PROTECTED_ARTIFACT_DIRS if protected is None else protected)


# ───────────────────────────────────────────── split policy (mirrors E1, E4-labelled)
def resolve_e4_window(spec: Any, cfg: FoundationConfig, role: str) -> SplitWindow:
    if role not in LEGAL_WINDOWS:
        raise ValueError(f"unknown E4 role {role!r}")
    verify_split_table(cfg.splits)
    if isinstance(spec, str):
        if spec in FORBIDDEN_SPLITS or spec not in cfg.splits:
            raise ForbiddenSplitError(f"E4 {role}: split name {spec!r} is not legal (legal: {LEGAL_WINDOWS[role]})")
        window = cfg.splits[spec]
    elif isinstance(spec, SplitWindow):
        window = spec
    else:
        raise ForbiddenSplitError(f"E4 {role}: unsupported split specification of type {type(spec).__name__}")
    for legal_name in LEGAL_WINDOWS[role]:
        if _same_window(window, FROZEN_SPLITS[legal_name]):
            return FROZEN_SPLITS[legal_name]
    raise ForbiddenSplitError(
        f"E4 {role}: window {window.name} {window.start.date()}..{window.end.date()} is not a legal "
        f"{'/'.join(LEGAL_WINDOWS[role]).upper()} window; the test split is off-limits for PPO training, "
        "evaluation and every performance-driven decision"
    )


# ───────────────────────────────────────────── environment: accepted O2 env + accepted R2 mixin
class BtcUsdtTradingEnvO2R2(R2RewardMixin, BtcUsdtTradingEnvO2):
    """
    The accepted O2 environment with the returned reward replaced by the
    accepted R2 (E4 = O2 + R2).  Nothing is overridden here: ``_observe`` is
    ``BtcUsdtTradingEnvO2._observe``, ``step`` is ``R2RewardMixin.step`` (which
    calls the inherited certified E1 accounting once), ``reset`` is the mixin's
    (clears the decomposition, then E1 reset).
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        decision_start: pd.Timestamp | str,
        decision_end: pd.Timestamp | str,
        o2_config: O2ObservationConfig,
        cost_config: CostConfig = CONSERVATIVE_COST,
        episode_length: int | None = None,
        random_start: bool = False,
        initial_equity: float = 1.0,
        render_mode: str | None = None,
        price_end: pd.Timestamp | str | None = None,
        r2_coefficients: R2Coefficients = FROZEN_R2,
    ) -> None:
        BtcUsdtTradingEnvO2.__init__(self, frame, decision_start, decision_end, o2_config, cost_config=cost_config,
                                     episode_length=episode_length, random_start=random_start,
                                     initial_equity=initial_equity, render_mode=render_mode, price_end=price_end)
        self._init_r2(r2_coefficients)


assert BtcUsdtTradingEnvO2R2._observe is BtcUsdtTradingEnvO2._observe
assert BtcUsdtTradingEnvO2R2.step is R2RewardMixin.step
assert BtcUsdtTradingEnvO2R2.reset is R2RewardMixin.reset


def make_o2r2_split_env(frame: pd.DataFrame, window: SplitWindow, o2_config: O2ObservationConfig,
                        cost_config: CostConfig = CONSERVATIVE_COST, **kwargs: Any) -> BtcUsdtTradingEnvO2R2:
    if not isinstance(window, SplitWindow):
        raise TypeError("window must be a SplitWindow")
    i0, last_usable = usable_decision_index_range(frame, window)
    return BtcUsdtTradingEnvO2R2(frame, decision_start=frame.index[i0], decision_end=frame.index[last_usable],
                                 o2_config=o2_config, cost_config=cost_config, price_end=window.end, **kwargs)


def verify_e4_env(env: Any, role: str, obs_cfg: O2ObservationConfig | None = None) -> SplitWindow:
    """Mirror of ``verify_e2_env`` (O2 checks) plus ``verify_e3_env``'s frozen-R2 check, type exactly ``BtcUsdtTradingEnvO2R2``."""
    from gymnasium import spaces

    if role not in LEGAL_WINDOWS:
        raise ValueError(f"unknown E4 role {role!r}")
    chain, base = _wrapper_chain(env)
    supported = _supported_wrapper_types()
    for w in chain:
        if w not in supported:
            raise ForbiddenSplitError(
                f"E4 {role}: unsupported wrapper {w.__module__}.{w.__name__}; only "
                f"{', '.join(t.__module__ + '.' + t.__name__ for t in supported)} may wrap an E4 environment"
            )
    if type(base) is not BtcUsdtTradingEnvO2R2:
        raise ForbiddenSplitError(f"E4 {role}: environment of type {type(base).__name__} is not the certified O2+R2 env")
    for attr in ("_dates", "_i1", "price_end", "obs_config", "cost_config", "random_start", "episode_length",
                 "initial_equity", "action_space", "observation_space", "r2_coefficients"):
        if not hasattr(base, attr):
            raise ForbiddenSplitError(f"E4: environment lacks the certified attribute {attr!r}")
    if base.action_space != spaces.Discrete(2):
        raise ForbiddenSplitError(f"E4 {role}: action space {base.action_space} is not Discrete(2) CASH/LONG")
    oc = base.obs_config
    if not isinstance(oc, O2ObservationConfig):
        raise ForbiddenSplitError(f"E4 {role}: observation config {oc!r} is not O2")
    if obs_cfg is not None and oc != obs_cfg:
        raise ForbiddenSplitError(f"E4 {role}: observation config differs from the expected O2 scaler")
    if base.observation_space != o2_observation_space(oc) or base.observation_space.shape != (O2_SIZE,):
        raise ForbiddenSplitError(f"E4 {role}: observation space does not match the O2 config")
    if not isinstance(base.r2_coefficients, R2Coefficients) or not base.r2_coefficients.is_frozen:
        raise ForbiddenSplitError(
            f"E4 {role}: R2 coefficients {base.r2_coefficients!r} are not the frozen values "
            f"(turnover {TURNOVER_PENALTY_LAMBDA}, drawdown {DRAWDOWN_PENALTY_LAMBDA})"
        )
    cost = base.cost_config
    if cost != CONSERVATIVE_COST or cost.one_leg_fraction != 0.0015:
        raise ForbiddenSplitError(f"E4 {role}: cost profile {cost!r} is not the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    if base.initial_equity != E1_INITIAL_EQUITY:
        raise ForbiddenSplitError(f"E4 {role}: initial equity {base.initial_equity} != {E1_INITIAL_EQUITY}")
    if base.random_start is not False or base.episode_length is not None:
        raise ForbiddenSplitError(
            f"E4 {role}: episode options random_start={base.random_start} episode_length={base.episode_length}; "
            "E4 uses deterministic full-window episodes only (as E1)"
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
        f"E4 {role}: environment decisions {first.date()}..{last.date()}, price_end "
        f"{None if base.price_end is None else base.price_end.date()} do not match a legal frozen "
        f"{'/'.join(LEGAL_WINDOWS[role]).upper()} window"
    )


# ───────────────────────────────────────────── configuration (identical PPO settings to E1/E2/E3)
@dataclass(frozen=True)
class E4Config:
    seed: int
    total_timesteps: int = FULL_BUDGET_TIMESTEPS
    hyperparameters: PPOHyperparameters = field(default_factory=PPOHyperparameters)
    torch_threads: int = 1
    device: str = "cpu"
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT
    run_group: str = E4_RUN_GROUP
    verbose: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.device != "cpu":
            raise ValueError("E4 runs on CPU only (identical to E1); pass device='cpu'")
        validate_run_group(self.run_group)
        if not isinstance(self.artifacts_root, (str, Path)):
            raise ValueError("artifacts_root must be a path")
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root))
        verify_e4_output_dir(self.artifacts_dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_root / self.run_group / str(self.seed)


HYPERPARAMETER_RATIONALE_E4: dict[str, str] = {
    **HYPERPARAMETER_RATIONALE_E2,
    "reward": (f"R2 = R1 - {TURNOVER_PENALTY_LAMBDA} * legs - {DRAWDOWN_PENALTY_LAMBDA} * max(0, D_(t+1) - D_t): "
               "the accepted E3 reward, coefficients frozen prospectively, reused unchanged; "
               "training incentive only, financial accounting is the E1 accounting."),
    "experimental_control": "composition of accepted components: observation = accepted E2 O2 (unchanged), reward = accepted "
                            "E3 R2 (unchanged); every PPO/env/cost/seed/budget setting equals E1/E2/E3 by construction "
                            "(CANONICAL_E4_SPEC is derived from CANONICAL_E2_SPEC with E3's reward block).",
}

# ───────────────────────────────────────────── frozen canonical E4 specification (derived from E2 + E3)
E4_VS_E2_DIFFERENCE_KEYS: tuple[str, ...] = ("experiment", "reward", "reward_r2")
E4_VS_E3_DIFFERENCE_KEYS: tuple[str, ...] = ("experiment", "observation", "scale_fit", "scaler_fit")


def _derive_canonical_e4_spec() -> dict[str, Any]:
    spec = copy.deepcopy(CANONICAL_E2_SPEC)      # O2 observation + scaler blocks of the accepted E2
    spec["experiment"] = EXPERIMENT
    spec["reward"] = REWARD_DEFINITION           # R2 block of the accepted E3
    spec["reward_r2"] = copy.deepcopy(R2_SPEC)
    return spec


CANONICAL_E4_SPEC: dict[str, Any] = _derive_canonical_e4_spec()


# ───────────────────────────────────────────── environments
def make_e4_env(frame: pd.DataFrame, split: Any, cfg: FoundationConfig, obs_cfg: O2ObservationConfig,
                role: str = "diagnostic") -> BtcUsdtTradingEnvO2R2:
    window = resolve_e4_window(split, cfg, role)
    cost = cfg.costs[COST_PROFILE]
    if cost != CONSERVATIVE_COST or abs(cost.one_leg_fraction - 0.0015) > 1e-15:
        raise RuntimeError("E4 requires the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    env = make_o2r2_split_env(frame, window, obs_cfg, cost_config=cost, initial_equity=cfg.initial_equity,
                              r2_coefficients=FROZEN_R2)
    verify_e4_env(env, role)
    return env


def make_train_env(frame: pd.DataFrame, cfg: FoundationConfig, obs_cfg: O2ObservationConfig):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(make_e4_env(frame, TRAIN_SPLIT, cfg, obs_cfg, role="train"))


# ───────────────────────────────────────────── model
def build_model(env, config: E4Config, log_dir: Path | None = None, obs_cfg: O2ObservationConfig | None = None):
    import torch
    from stable_baselines3 import PPO

    verify_e4_env(env, "train", obs_cfg)
    torch.set_num_threads(int(config.torch_threads))
    model = PPO("MlpPolicy", env, seed=int(config.seed), device=config.device, verbose=config.verbose,
                **config.hyperparameters.sb3_kwargs())
    if log_dir is not None:
        from stable_baselines3.common.logger import configure

        log_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(log_dir), ["csv"]))
    return model


load_e4_model = load_e1_model


class SB3DeterministicPolicyE4:
    """Deterministic (argmax) adapter bound to a verified E4 environment and role (mirror of the E1 adapter)."""

    def __init__(self, model, env: Any, role: str = "evaluation", name: str = "ppo_e4",
                 obs_cfg: O2ObservationConfig | None = None) -> None:
        if role not in ("evaluation", "diagnostic"):
            raise ForbiddenSplitError(f"E4 adapter: role {role!r} is not a learned-policy role (evaluation/diagnostic)")
        self.window = verify_e4_env(env, role, obs_cfg)
        base = getattr(env, "unwrapped", env)
        if model.observation_space != base.observation_space or model.action_space != base.action_space:
            raise ForbiddenSplitError("E4 adapter: model spaces differ from the bound environment's spaces")
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
            raise ForbiddenSplitError("E4 adapter: info lacks the certified decision/state dates") from exc
        if not (self._first_decision <= decision <= self._last_decision) or instant > self.window.end:
            raise ForbiddenSplitError(
                f"E4 adapter bound to {self.window.name} refused decision {decision.date()} "
                f"(legal {self._first_decision.date()}..{self._last_decision.date()}) before prediction"
            )
        if not self._obs_space.contains(np.asarray(obs, dtype=np.float32)):
            raise ForbiddenSplitError("E4 adapter: observation outside the bound observation space")
        action, _ = self.model.predict(obs, deterministic=True)
        self._predictions += 1
        return int(np.asarray(action).item())


# ───────────────────────────────────────────── evaluation (validation only)
run_episode_e4 = run_episode_e3   # the E3 runner: financial metrics from the equity path + R2 decomposition/diagnostics


def evaluate_e4(model, frame: pd.DataFrame, cfg: FoundationConfig, obs_cfg: O2ObservationConfig,
                split: Any = EVAL_SPLIT, name: str = "ppo_e4") -> EpisodeResult:
    env = make_e4_env(frame, split, cfg, obs_cfg, role="evaluation")
    policy = SB3DeterministicPolicyE4(model, env, role="evaluation", name=name, obs_cfg=obs_cfg)
    result = run_episode_e4(env, policy, periods_per_year=cfg.periods_per_year)
    _assert_finite_curve(result)
    result.metrics.update(state_distribution(result))
    return result


def check_finite_policy_outputs(model, env: Any, obs_cfg: O2ObservationConfig | None = None) -> dict[str, Any]:
    import torch

    window = verify_e4_env(env, "diagnostic", obs_cfg)
    base = getattr(env, "unwrapped", env)
    if model.observation_space != base.observation_space or model.action_space != base.action_space:
        raise ForbiddenSplitError("E4 diagnostic: model spaces differ from the environment's spaces")
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
            obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
            if not np.isfinite(reward):
                raise RuntimeError(f"non-finite reward at step {n}")
            for k in R2_CURVE_COLUMNS:
                if not np.isfinite(info[k]):
                    raise RuntimeError(f"non-finite R2 term {k} at step {n}")
            done = bool(terminated or truncated)
            n += 1
    return {"steps_checked": n, "window": window.name, "observation": OBSERVATION_DEFINITION, "reward": REWARD_DEFINITION}


# ───────────────────────────────────────────── provenance / metadata
def provenance_e4(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    files = source_files(root)
    key = {}
    for label, rel in KEY_SOURCE_FILES_E4.items():
        if rel not in files:
            raise RuntimeError(f"provenance: key source file missing: {rel}")
        key[label] = files[rel]
    return {"git": git_identity(root), "source_fingerprint": source_fingerprint(files),
            "source_fingerprint_globs": list(SOURCE_FINGERPRINT_GLOBS), "source_files": files, "key_hashes": key}


def collect_metadata(config: E4Config, model, frame: pd.DataFrame, dataset_report: dict[str, Any],
                     cfg: FoundationConfig, obs_cfg: O2ObservationConfig, scaler_fit: O2ScalerFit,
                     train_seconds: float, preflight: dict[str, Any] | None = None,
                     source_before: dict[str, Any] | None = None, model_sha256: str | None = None) -> dict[str, Any]:
    import gymnasium
    import stable_baselines3
    import torch

    cost = cfg.costs[COST_PROFILE]
    train = resolve_e4_window(TRAIN_SPLIT, cfg, "train")
    validation = resolve_e4_window(EVAL_SPLIT, cfg, "evaluation")
    train_rec = _split_record(frame, train)
    prov = provenance_e4()
    if obs_cfg != O2ObservationConfig.from_fit(scaler_fit):
        raise RuntimeError("O2 observation config differs from the recorded scaler fit")
    source = {k: prov[k] for k in ("source_fingerprint", "source_fingerprint_globs", "key_hashes", "source_files")}
    if source_before is not None:
        source["fingerprint_before_training"] = source_before["source_fingerprint"]
        source["source_stable_during_run"] = source_before["source_fingerprint"] == prov["source_fingerprint"]
        if not source["source_stable_during_run"]:
            raise RuntimeError("source files changed while the run was executing; the run is not attributable")
    low, high = obs_cfg.standardised_bounds()
    meta = {
        "experiment": EXPERIMENT,
        "control_experiments": list(CONTROL_EXPERIMENTS),
        "differs_from_controls_in": {"E2": ["reward"], "E3": ["observation"]},
        "composition": {"observation": "accepted E2 O2 (ppo_e2.BtcUsdtTradingEnvO2, observations_o2.py) unchanged",
                        "reward": "accepted E3 R2 (reward_r2.R2RewardMixin, FROZEN_R2) unchanged"},
        "study_stage": STUDY_STAGE,
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
        "hyperparameter_rationale": HYPERPARAMETER_RATIONALE_E4,
        "total_timesteps_requested": int(config.total_timesteps),
        "total_timesteps_trained": int(model.num_timesteps),
        "train_split": train_rec,
        "validation_split": _split_record(frame, validation),
        "forbidden_splits": list(FORBIDDEN_SPLITS),
        "observation": {   # identical to the accepted E2 record
            "definition": OBSERVATION_DEFINITION,
            "content": "8 standardised engineered close-derived features + current portfolio state (0=CASH, 1=LONG)",
            "features": list(O2_FEATURES),
            "feature_definitions": FEATURE_DEFINITIONS,
            "rsi_definition": RSI_DEFINITION,
            "inputs": "closes only (no open/high/low/volume); feature[t] uses closes t-199..t",
            "n_features": O2_N_FEATURES,
            "size": O2_SIZE,
            "window": O2_WINDOW,
            "warmup_rows": O2_WARMUP_ROWS,
            "warmup": verify_o2_warmup(frame, train),
            "scaler_fit": scaler_fit.to_json(),
            "scaling": "z = (x - mean) / std per feature; TRAIN-only statistics (see scaler_fit)",
            "raw_feature_bounds": {k: list(v) for k, v in RAW_FEATURE_BOUNDS.items()},
            "declared_bounds_low": [float(v) for v in low] + [0.0],
            "declared_bounds_high": [float(v) for v in high] + [1.0],
            "clipping": "none active: declared bounds are the standardised analytic bounds implied by "
                        "|log return| <= 1 (data-level invariant); defensive clip to the declared bound only",
        },
        "reward": reward_metadata(),   # identical to the accepted E3 record
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
                     "open[t+1]->open[t+2] minus transition cost; s_{t+1} at open[t+2]; R2 penalties use only "
                     "legs_t and the realised equity / running peak through open[t+2]"),
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
            "model_sha256": model_sha256,
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
    """Missing/empty required keys of an E4 record: E2's O2 schema AND E3's R2 schema (empty list = complete)."""
    from .ppo_e2 import validate_metadata as _v2
    from .ppo_e3 import REQUIRED_REWARD_KEYS

    problems = _v2(meta)   # required keys, fingerprint, O2 definition and scaler record
    rew = meta.get("reward", {})
    if not isinstance(rew, dict) or rew.get("definition") != REWARD_DEFINITION:
        problems.append("reward.definition is not R2")
    else:
        for k in REQUIRED_REWARD_KEYS:
            if k not in rew or rew[k] in (None, "", [], {}):
                problems.append(f"missing reward.{k}")
    return problems


# ───────────────────────────────────────────── workflow
@dataclass
class E4RunResult:
    config: E4Config
    model: Any
    obs_config: O2ObservationConfig
    scaler_fit: O2ScalerFit
    metadata: dict[str, Any]
    validation: EpisodeResult
    artifacts_dir: Path


def run_e4(config: E4Config, frame: pd.DataFrame | None = None, cfg: FoundationConfig | None = None,
           dataset_report: dict[str, Any] | None = None, preflight: dict[str, Any] | None = None,
           write_artifacts: bool = True) -> E4RunResult:
    """Complete E4 workflow for one seed (mirror of ``run_e2`` / ``run_e3`` with the O2 + R2 environment)."""
    validate_run_group(config.run_group)                 # destination guard first: before any loading, mkdir or write
    out = verify_e4_output_dir(config.artifacts_dir)
    cfg = cfg or load_foundation_config()
    verify_split_table(cfg.splits)
    if frame is None:
        frame, dataset_report = load_dataset(cfg.dataset)
    if dataset_report is None:
        raise ValueError("dataset_report is required when frame is supplied")
    obs_cfg, scaler_fit = resolve_o2_config(frame, cfg)       # the accepted E2 scaler, unchanged
    verify_o2_warmup(frame, resolve_e4_window(TRAIN_SPLIT, cfg, "train"))
    if write_artifacts:
        out.mkdir(parents=True, exist_ok=True)

    source_before = provenance_e4()
    train_env = make_train_env(frame, cfg, obs_cfg)
    model = build_model(train_env, config, log_dir=(out / "train_log") if write_artifacts else None, obs_cfg=obs_cfg)
    t0 = time.time()
    model.learn(total_timesteps=int(config.total_timesteps), progress_bar=False)
    train_seconds = time.time() - t0

    model_sha256 = None
    if write_artifacts:
        model.save(str(out / "model.zip"))
        model_sha256 = sha256_of(out / "model.zip")
        model = load_e4_model(out / "model.zip")

    finite = check_finite_policy_outputs(model, make_e4_env(frame, EVAL_SPLIT, cfg, obs_cfg, role="diagnostic"), obs_cfg)
    validation = evaluate_e4(model, frame, cfg, obs_cfg, EVAL_SPLIT, name=f"ppo_e4_seed{config.seed}")
    meta = collect_metadata(config, model, frame, dataset_report, cfg, obs_cfg, scaler_fit, train_seconds, preflight,
                            source_before=source_before, model_sha256=model_sha256)
    meta["finite_output_check"] = finite
    if write_artifacts:
        write_json(out / "metadata.json", meta)
        write_json(out / "validation_metrics.json", validation.metrics)
        validation.curve.to_csv(out / "validation_curve.csv")
    return E4RunResult(config, model, obs_cfg, scaler_fit, meta, validation, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E4: PPO with the O2 engineered observation and the R2 risk-aware "
                                             "reward (modelled costs: 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale), train-only, validation-only evaluation.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=FULL_BUDGET_TIMESTEPS)
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_BUDGET_TIMESTEPS}-step integration run into artifacts/ppo/e4_smoke/")
    ap.add_argument("--skip-preflight", action="store_true", help="do not re-run the certification gate in-process")
    ap.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args(argv)

    config = E4Config(seed=args.seed, total_timesteps=SMOKE_BUDGET_TIMESTEPS if args.smoke else args.timesteps,
                      artifacts_root=args.artifacts_root, run_group=E4_SMOKE_RUN_GROUP if args.smoke else E4_RUN_GROUP,
                      verbose=args.verbose)
    preflight = None if args.skip_preflight else _preflight_or_die()
    res = run_e4(config, preflight=preflight)
    m = res.validation.metrics
    fit = res.metadata["observation"]["scaler_fit"]
    print(f"\nE4 seed={config.seed} timesteps={res.metadata['total_timesteps_trained']} "
          f"train {res.metadata['train_seconds']:.0f}s -> {res.artifacts_dir}")
    print(f"O2 scaler fitted on TRAIN feature rows {fit['fit_row_start']}..{fit['fit_row_end']} ({fit['n_rows']} rows) "
          f"| R2 lambdas turnover={TURNOVER_PENALTY_LAMBDA} drawdown={DRAWDOWN_PENALTY_LAMBDA} "
          f"| source fingerprint {res.metadata['source']['source_fingerprint'][:12]}")
    print(f"validation ({m['first_decision_date'].date()}..{m['last_decision_date'].date()}, {m['n_steps']} steps): "
          f"total_return={m['total_return']:+.4f} final_equity={m['final_equity']:.4f} "
          f"sharpe={m['sharpe_ratio'] if m['sharpe_ratio'] is None else round(m['sharpe_ratio'], 3)} "
          f"max_dd={m['max_drawdown']:+.4f} legs={m['n_legs']} entries={m['n_entries']} "
          f"exposure={m['exposure']:.3f} cost_drag={m['total_cost_fraction']:.4f}")
    print(f"reward diagnostics (training reward, NOT return): cum R1={m['cumulative_financial_r1']:+.4f} "
          f"cum turnover pen={m['cumulative_turnover_penalty']:.4f} cum drawdown pen={m['cumulative_drawdown_penalty']:.4f} "
          f"cum R2={m['cumulative_reward_r2']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
