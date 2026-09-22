"""
ppo_e2.py — PPO experiment E2 (O2 engineered observation / R1 / 0.10% fee + 0.05% slippage allowance per trade).

    python -m btc_rl.ppo_e2 --seed 42                       # full E2 run (200k timesteps)
    python -m btc_rl.ppo_e2 --seed 42 --smoke               # 16,384-step integration run
    python -m btc_rl.ppo_e2 --seed 42 --skip-preflight      # only for already-certified sessions

E2 is the RQ2 treatment of docs/methodology.md: it differs from the
control E1 in exactly ONE respect, the observation representation
(O2 engineered, ``observations_o2.py``) instead of O1 minimal.  Everything
else is inherited unchanged from E1 (``ppo_e1.py``): dataset, frozen splits,
market timeline, CASH/LONG target-state actions, 0.10% fee + 0.05% slippage allowance per trade,
reward R1, the SB3 PPO implementation and every hyperparameter, the [64, 64]
Tanh MLP, CPU with one torch thread, one environment, the seeds 42 / 123 /
2026, the 200,000-step budget (200,704 trained), deterministic full-window
TRAIN episodes, validation accounting and the baseline definitions.
``CANONICAL_E2_SPEC`` is DERIVED from ``CANONICAL_E1_SPEC`` by replacing only
the observation and scaler blocks; a test asserts the two specifications are
otherwise identical.

Environment
-----------
``BtcUsdtTradingEnvO2`` is a subclass of the certified ``BtcUsdtTradingEnv``
that overrides only ``_observe`` (and the declared observation space).  The
accounting, timing, cost and episode code is literally the E1 code; the
observation is built from ``closes[: t + 1]`` exactly as O1 is, so structural
causality is unchanged.  E1 boundaries reject this subclass (``type(base) is
not BtcUsdtTradingEnv``) and E2 boundaries require it, so E1 and E2 artifacts
cannot be mixed at the environment level either.

Split policy, guarantee boundary, artifacts
-------------------------------------------
Identical to E1 (fail-closed window resolution, prebuilt-environment
verification, bound deterministic adapter; TRAIN for training, VALIDATION for
evaluation, TEST never built by this module).  Artifacts go to
``artifacts/ppo/e2/<seed>/`` (``e2_smoke`` for the smoke run); ``E2Config``
refuses run groups that could collide with E1's.  E1 artifacts are never
written: the module records the SHA-256 anchors of the accepted E1 checkpoints
and files (``E1_FROZEN_ARTIFACTS``) and can verify they are unchanged.
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
from .env import FLAT, LONG, BtcUsdtTradingEnv
from .evaluation import EpisodeResult, run_episode
from .observations_o2 import (
    FEATURE_DEFINITIONS,
    O2_FEATURES,
    O2_N_FEATURES,
    O2_SIZE,
    O2_WARMUP_ROWS,
    O2_WINDOW,
    RAW_FEATURE_BOUNDS,
    RSI_DEFINITION,
    SCALER_STATISTIC,
    O2ObservationConfig,
    O2ScalerFit,
    build_o2_observation,
    fit_o2_scaler,
    o2_observation_space,
    verify_o2_warmup,
)
from .ppo_e1 import (
    ACTION_SEMANTICS,
    CANONICAL_E1_SPEC,
    CASH,
    COST_PROFILE,
    DEFAULT_ARTIFACTS_ROOT,
    DEVELOPMENT_SEEDS,
    E1_INITIAL_EQUITY,
    EVAL_SPLIT,
    FORBIDDEN_SPLITS,
    FULL_BUDGET_TIMESTEPS,
    HYPERPARAMETER_RATIONALE,
    IDENTITY_KEYS,
    KEY_SOURCE_FILES,
    LEGAL_WINDOWS,
    REQUIRED_METADATA_KEYS,
    REWARD_DEFINITION,
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
from .preflight import LOCK_PATH, PACKAGE_ROOT
from .splits import FROZEN_SPLITS, SETTLEMENT_BARS, SplitWindow, usable_decision_index_range

assert CASH == FLAT == 0 and LONG == 1

# ───────────────────────────────────────────── experiment identity
EXPERIMENT = "E2"
CONTROL_EXPERIMENT = "E1"
STUDY_STAGE = "ppo-e2"
OBSERVATION_DEFINITION = "O2"
E2_RUN_GROUP = "e2"
E2_SMOKE_RUN_GROUP = "e2_smoke"
FORBIDDEN_RUN_GROUP_PREFIXES: tuple[str, ...] = ("e1",)   # E2 must never write into an E1 run group

# provenance: E1's key sources plus the E2/O2 sources
KEY_SOURCE_FILES_E2: dict[str, str] = {
    **KEY_SOURCE_FILES,
    "o2_observation_source": "src/btc_rl/observations_o2.py",
    "ppo_e2_source": "src/btc_rl/ppo_e2.py",
    "e2_report_source": "src/btc_rl/ppo_e2_report.py",
}

# SHA-256 anchors of the accepted E1 artifacts (closure, commit 7ffb9ed2; checkpoint hashes
# equal those recorded in the independent review record).  E2 never writes these files;
# the E1-vs-E2 comparison verifies them before reading E1 results.
E1_FROZEN_ARTIFACTS: dict[int, dict[str, str]] = {
    42: {
        "model.zip": "c0c8a75722fdcf3a98689d8a082d83dfbd5c6bbcc4158b08c84717a25c6310f3",
        "metadata.json": "8977ed156c61010b17bcef6c3d07ed6da7c1b706b50e6b08ef7751eae7fbbc6b",
        "validation_curve.csv": "e4d2e6e4bf230ae53526e4cdc7f28ffcd77c5d0c9746b61dc8c0b0a72fca1bc0",
        "validation_metrics.json": "b232f90ca73327dc190ec6677de6cffb6cf733b716750a44ece83bed9ea5756b",
    },
    123: {
        "model.zip": "06fb7976f0e5767ca8fef5cd0a812765a57fe6f995d7bda4c0586108bd144c48",
        "metadata.json": "d96d8d381aa02183696f7b95505cf6de42453ec8cf840f4d6770e3de62b1bbcd",
        "validation_curve.csv": "15f7644f19da890e0ebd84245584a4ee28c816ca087ee098ec00e9725663981a",
        "validation_metrics.json": "d69dcc1681da5124bd408bc24d8970a6d352daf9e48891dc96fa1ee175643c6a",
    },
    2026: {
        "model.zip": "c9fc3879ab614e39b3e83d8f914db9793c32419d562866cfd4abe656c609e9d7",
        "metadata.json": "ffb02ab46e2a1a8017c68e8b3b1e8a329a60bde10cb4bf2474c3f679b1149af9",
        "validation_curve.csv": "c12ddad368b1777849ca348c98c2f3de9faf9d3464c95908db527fedb1af096b",
        "validation_metrics.json": "1e4efbe28d4759c672f2458f44698e5329525011d9537c407a8472a3fa35361d",
    },
}
E1_RUN_GROUP = "e1"
# SHA-256 of the E1 training-relevant core sources as recorded in the accepted E1 runs' metadata (key_hashes).
# E2 must not change any of them; the canonical E1 report would reject the tree otherwise.
E1_FROZEN_CORE_SOURCE_HASHES: dict[str, str] = {
    "env_source": "e2b23c8412502630f78cac6688a99b16c8825be216319f1094d2e96e3314df68",
    "splits_source": "1052457e104f5b2a5ff5d4ff7669457ed1a603075164fa9682e948b749ba4aff",
    "observation_source": "801d339806a4e614288b73bc8de5ac8b32177fba21333c83aacf268595d1d6e3",
    "scaler_source": "53afbbb59086648e0ae3cf039ddc58e220e6712ded44dfd24f56dc1342bb6122",
    "costs_source": "810a5badb116fc64f6e1d33e0a1631332a19e3e92aa04bcf5547ae313d268a7e",
    "evaluation_source": "5821d877a04a40f438475fca39e1090f05a58ba87e26ddaa17f5ae5bc10931ed",
    "data_source": "f7fa51b63ff025a97775311691b0d64f8c476cfb0987fd3112e353c574cf938e",
    "foundation_config": "6437ce3d08fc67fd9e9fc123f485f7556320f323a70470aa4135af1efe8bbae3",
    "pyproject_toml": "573ac68062ddcf22745002b7e315981da03be3ac283e49c83f8010514ee045f1",
    "uv_lock": "4c1ae8fde806ef6261daec00f1af7087aec665717a83227bd00aa9ee2d47f56f",
}


def verify_e1_core_sources_unchanged(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Every E1 training-core source of the current tree must hash to the value recorded by the accepted E1 runs."""
    hashes = {label: sha256_of(Path(root) / KEY_SOURCE_FILES[label]) for label in E1_FROZEN_CORE_SOURCE_HASHES}
    changed = sorted(k for k, v in hashes.items() if v != E1_FROZEN_CORE_SOURCE_HASHES[k])
    if changed:
        raise RuntimeError(f"E1 training-core sources changed since the accepted E1 runs: {changed}")
    return {"unchanged": True, "hashes": hashes}


def e1_artifact_status(artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, Any]:
    """Compare the E1 run files on disk with the frozen anchors (missing files are reported, not hashed)."""
    root = Path(artifacts_root) / E1_RUN_GROUP
    status: dict[str, Any] = {"present": {}, "changed": [], "missing": []}
    for seed, files in E1_FROZEN_ARTIFACTS.items():
        for name, digest in files.items():
            p = root / str(seed) / name
            if not p.exists():
                status["missing"].append(f"{seed}/{name}")
                continue
            now = sha256_of(p)
            status["present"][f"{seed}/{name}"] = now
            if now != digest:
                status["changed"].append(f"{seed}/{name}")
    status["all_present"] = not status["missing"]
    status["unchanged"] = not status["changed"]
    return status


def verify_e1_artifacts_unchanged(artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, Any]:
    status = e1_artifact_status(artifacts_root)
    if not status["all_present"]:
        raise RuntimeError(f"E1 canonical artifacts missing: {status['missing']}")
    if not status["unchanged"]:
        raise RuntimeError(f"E1 canonical artifacts differ from the frozen anchors: {status['changed']}")
    return status


# ───────────────────────────────────────────── split policy (mirrors E1, E2-labelled)
def resolve_e2_window(spec: Any, cfg: FoundationConfig, role: str) -> SplitWindow:
    """Same fail-closed resolution as ``ppo_e1.resolve_e1_window`` (TRAIN for training, VALIDATION for evaluation)."""
    if role not in LEGAL_WINDOWS:
        raise ValueError(f"unknown E2 role {role!r}")
    verify_split_table(cfg.splits)
    if isinstance(spec, str):
        if spec in FORBIDDEN_SPLITS or spec not in cfg.splits:
            raise ForbiddenSplitError(f"E2 {role}: split name {spec!r} is not legal (legal: {LEGAL_WINDOWS[role]})")
        window = cfg.splits[spec]
    elif isinstance(spec, SplitWindow):
        window = spec
    else:
        raise ForbiddenSplitError(f"E2 {role}: unsupported split specification of type {type(spec).__name__}")
    for legal_name in LEGAL_WINDOWS[role]:
        if _same_window(window, FROZEN_SPLITS[legal_name]):
            return FROZEN_SPLITS[legal_name]
    raise ForbiddenSplitError(
        f"E2 {role}: window {window.name} {window.start.date()}..{window.end.date()} is not a legal "
        f"{'/'.join(LEGAL_WINDOWS[role]).upper()} window; the test split is off-limits for PPO training, "
        "evaluation and every performance-driven decision"
    )


# ───────────────────────────────────────────── environment: certified env with the O2 observation
class BtcUsdtTradingEnvO2(BtcUsdtTradingEnv):
    """
    The certified INVESTED/CASH (internal labels LONG/FLAT) environment with ONLY the observation replaced by O2.

    ``reset``/``step``/accounting/costs/episode logic are inherited unchanged.
    ``_observe`` builds O2 from ``closes[: t + 1]`` (structural causality) and
    the position held during bar t.  Requires ``O2_WARMUP_ROWS`` rows before the
    first decision (parent requires only the O1 lookback).
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
    ) -> None:
        if not isinstance(o2_config, O2ObservationConfig):
            raise TypeError("o2_config must be an O2ObservationConfig")
        super().__init__(frame, decision_start, decision_end, cost_config=cost_config, obs_config=None,
                         episode_length=episode_length, random_start=random_start, initial_equity=initial_equity,
                         render_mode=render_mode, price_end=price_end)
        if self._i0 < O2_WARMUP_ROWS:
            raise ValueError(f"not enough O2 warm-up rows before decision_start ({self._i0} < {O2_WARMUP_ROWS})")
        self.obs_config = o2_config
        self.observation_space = o2_observation_space(o2_config)

    def _observe(self) -> np.ndarray:
        # Structural causality: only closes up to and including bar t are visible.
        return build_o2_observation(self._closes[: self._t + 1], self._position, self.obs_config)


def make_o2_split_env(frame: pd.DataFrame, window: SplitWindow, o2_config: O2ObservationConfig,
                      cost_config: CostConfig = CONSERVATIVE_COST, **kwargs: Any) -> BtcUsdtTradingEnvO2:
    """O2 environment for one frozen window under the purge rule (same construction as ``env.make_split_env``)."""
    if not isinstance(window, SplitWindow):
        raise TypeError("window must be a SplitWindow")
    i0, last_usable = usable_decision_index_range(frame, window)
    return BtcUsdtTradingEnvO2(frame, decision_start=frame.index[i0], decision_end=frame.index[last_usable],
                               o2_config=o2_config, cost_config=cost_config, price_end=window.end, **kwargs)


def verify_e2_env(env: Any, role: str, obs_cfg: O2ObservationConfig | None = None) -> SplitWindow:
    """
    A prebuilt environment is legal for ``role`` only if (mirror of ``verify_e1_env``):
    only ``Monitor`` wrappers; unwrapped type exactly ``BtcUsdtTradingEnvO2``;
    ``Discrete(2)``; O2 config (equal to ``obs_cfg`` when given) and matching
    9-d space; frozen conservative costs; initial equity 1.0; deterministic
    full-window episodes; decision range / last mark / ``price_end`` equal a
    legal frozen window for ``role``.
    """
    from gymnasium import spaces

    if role not in LEGAL_WINDOWS:
        raise ValueError(f"unknown E2 role {role!r}")
    chain, base = _wrapper_chain(env)
    supported = _supported_wrapper_types()
    for w in chain:
        if w not in supported:
            raise ForbiddenSplitError(
                f"E2 {role}: unsupported wrapper {w.__module__}.{w.__name__}; only "
                f"{', '.join(t.__module__ + '.' + t.__name__ for t in supported)} may wrap an E2 environment"
            )
    if type(base) is not BtcUsdtTradingEnvO2:
        raise ForbiddenSplitError(f"E2 {role}: environment of type {type(base).__name__} is not the certified O2 env")
    for attr in ("_dates", "_i1", "price_end", "obs_config", "cost_config", "random_start", "episode_length",
                 "initial_equity", "action_space", "observation_space"):
        if not hasattr(base, attr):
            raise ForbiddenSplitError(f"E2: environment lacks the certified attribute {attr!r}")
    if base.action_space != spaces.Discrete(2):
        raise ForbiddenSplitError(f"E2 {role}: action space {base.action_space} is not Discrete(2) CASH/LONG")
    oc = base.obs_config
    if not isinstance(oc, O2ObservationConfig):
        raise ForbiddenSplitError(f"E2 {role}: observation config {oc!r} is not O2")
    if obs_cfg is not None and oc != obs_cfg:
        raise ForbiddenSplitError(f"E2 {role}: observation config differs from the expected O2 scaler")
    if base.observation_space != o2_observation_space(oc) or base.observation_space.shape != (O2_SIZE,):
        raise ForbiddenSplitError(f"E2 {role}: observation space does not match the O2 config")
    cost = base.cost_config
    if cost != CONSERVATIVE_COST or cost.one_leg_fraction != 0.0015:
        raise ForbiddenSplitError(f"E2 {role}: cost profile {cost!r} is not the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    if base.initial_equity != E1_INITIAL_EQUITY:
        raise ForbiddenSplitError(f"E2 {role}: initial equity {base.initial_equity} != {E1_INITIAL_EQUITY}")
    if base.random_start is not False or base.episode_length is not None:
        raise ForbiddenSplitError(
            f"E2 {role}: episode options random_start={base.random_start} episode_length={base.episode_length}; "
            "E2 uses deterministic full-window episodes only (as E1)"
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
        f"E2 {role}: environment decisions {first.date()}..{last.date()}, price_end "
        f"{None if base.price_end is None else base.price_end.date()} do not match a legal frozen "
        f"{'/'.join(LEGAL_WINDOWS[role]).upper()} window"
    )


# ───────────────────────────────────────────── configuration (identical PPO settings to E1)
@dataclass(frozen=True)
class E2Config:
    seed: int
    total_timesteps: int = FULL_BUDGET_TIMESTEPS
    hyperparameters: PPOHyperparameters = field(default_factory=PPOHyperparameters)
    torch_threads: int = 1
    device: str = "cpu"
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT
    run_group: str = E2_RUN_GROUP          # artifacts/ppo/<run_group>/<seed>/
    verbose: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.device != "cpu":
            raise ValueError("E2 runs on CPU only (identical to E1); pass device='cpu'")
        if not isinstance(self.run_group, str) or not self.run_group:
            raise ValueError("run_group must be a non-empty string")
        if any(self.run_group == p or self.run_group.startswith(p + "_") for p in FORBIDDEN_RUN_GROUP_PREFIXES):
            raise ValueError(f"run_group {self.run_group!r} collides with the frozen E1 artifacts; E2 never writes there")
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root))

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_root / self.run_group / str(self.seed)


HYPERPARAMETER_RATIONALE_E2: dict[str, str] = {
    **HYPERPARAMETER_RATIONALE,
    "policy": "MlpPolicy: O2 is a 9-dimensional flat vector (8 standardised engineered features + position); "
              "no recurrence, no CNN.  Same policy class and architecture as E1.",
    "device": "cpu: a [64, 64] MLP on a 9-d input gains nothing from a GPU; CPU keeps runs reproducible (as E1).",
    "return_scale": "not applicable: O2 does not contain the O1 return history.",
    "observation_scaling": ("per-feature standardisation z = (x - mean)/std with population statistics fitted on "
                            "TRAIN-only feature rows (window fully inside TRAIN closes 2018-03-04 .. 2020-12-29); "
                            "no VALIDATION/TEST/warm-up close enters the statistic; no active clipping."),
    "experimental_control": "every PPO/env/cost/reward/seed/budget setting equals E1 by construction (CANONICAL_E2_SPEC "
                            "is derived from CANONICAL_E1_SPEC); only the observation differs (RQ2).",
}

# ───────────────────────────────────────────── frozen canonical E2 specification (derived from E1)
E2_SPEC_DIFFERENCE_KEYS: tuple[str, ...] = ("experiment", "observation", "scale_fit", "scaler_fit")


def _derive_canonical_e2_spec() -> dict[str, Any]:
    spec = copy.deepcopy(CANONICAL_E1_SPEC)
    spec["experiment"] = EXPERIMENT
    spec["observation"] = {"definition": OBSERVATION_DEFINITION, "features": list(O2_FEATURES),
                           "n_features": O2_N_FEATURES, "size": O2_SIZE, "window": O2_WINDOW,
                           "warmup_rows": O2_WARMUP_ROWS}
    del spec["scale_fit"]
    spec["scaler_fit"] = {"fit_close_start": "2018-03-04", "fit_close_end": "2020-12-29",
                          "fit_row_start": "2018-09-19", "fit_row_end": "2020-12-29",
                          "n_closes": 1032, "n_rows": 833, "window_name": TRAIN_SPLIT,
                          "statistic": SCALER_STATISTIC, "features": list(O2_FEATURES)}
    # the VALUES (means/stds) are recomputed from the frozen dataset by the report
    return spec


CANONICAL_E2_SPEC: dict[str, Any] = _derive_canonical_e2_spec()


# ───────────────────────────────────────────── O2 scaler resolution (TRAIN only)
def resolve_o2_config(frame: pd.DataFrame, cfg: FoundationConfig) -> tuple[O2ObservationConfig, O2ScalerFit]:
    """O2 with the per-feature scaler fitted on TRAIN-only feature rows (``observations_o2.fit_o2_scaler``)."""
    train = resolve_e2_window(TRAIN_SPLIT, cfg, "train")
    fit = fit_o2_scaler(frame, train)
    return O2ObservationConfig.from_fit(fit), fit


# ───────────────────────────────────────────── environments
def make_e2_env(frame: pd.DataFrame, split: Any, cfg: FoundationConfig, obs_cfg: O2ObservationConfig,
                role: str = "diagnostic") -> BtcUsdtTradingEnvO2:
    window = resolve_e2_window(split, cfg, role)
    cost = cfg.costs[COST_PROFILE]
    if cost != CONSERVATIVE_COST or abs(cost.one_leg_fraction - 0.0015) > 1e-15:
        raise RuntimeError("E2 requires the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    env = make_o2_split_env(frame, window, obs_cfg, cost_config=cost, initial_equity=cfg.initial_equity)
    verify_e2_env(env, role)
    return env


def make_train_env(frame: pd.DataFrame, cfg: FoundationConfig, obs_cfg: O2ObservationConfig):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(make_e2_env(frame, TRAIN_SPLIT, cfg, obs_cfg, role="train"))


# ───────────────────────────────────────────── model
def build_model(env, config: E2Config, log_dir: Path | None = None, obs_cfg: O2ObservationConfig | None = None):
    """PPO on ``env`` (a Monitor-wrapped frozen TRAIN O2 environment); same SB3 call as E1."""
    import torch
    from stable_baselines3 import PPO

    verify_e2_env(env, "train", obs_cfg)
    torch.set_num_threads(int(config.torch_threads))
    model = PPO("MlpPolicy", env, seed=int(config.seed), device=config.device, verbose=config.verbose,
                **config.hyperparameters.sb3_kwargs())
    if log_dir is not None:
        from stable_baselines3.common.logger import configure

        log_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(log_dir), ["csv"]))
    return model


load_e2_model = load_e1_model  # plain PPO.load on CPU; the E2 boundaries verify spaces/config afterwards


class SB3DeterministicPolicyE2:
    """Deterministic (argmax) adapter bound to a verified E2 environment and role (mirror of the E1 adapter)."""

    def __init__(self, model, env: Any, role: str = "evaluation", name: str = "ppo_e2",
                 obs_cfg: O2ObservationConfig | None = None) -> None:
        if role not in ("evaluation", "diagnostic"):
            raise ForbiddenSplitError(f"E2 adapter: role {role!r} is not a learned-policy role (evaluation/diagnostic)")
        self.window = verify_e2_env(env, role, obs_cfg)
        base = getattr(env, "unwrapped", env)
        if model.observation_space != base.observation_space or model.action_space != base.action_space:
            raise ForbiddenSplitError("E2 adapter: model spaces differ from the bound environment's spaces")
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
            raise ForbiddenSplitError("E2 adapter: info lacks the certified decision/state dates") from exc
        if not (self._first_decision <= decision <= self._last_decision) or instant > self.window.end:
            raise ForbiddenSplitError(
                f"E2 adapter bound to {self.window.name} refused decision {decision.date()} "
                f"(legal {self._first_decision.date()}..{self._last_decision.date()}) before prediction"
            )
        if not self._obs_space.contains(np.asarray(obs, dtype=np.float32)):
            raise ForbiddenSplitError("E2 adapter: observation outside the bound observation space")
        action, _ = self.model.predict(obs, deterministic=True)
        self._predictions += 1
        return int(np.asarray(action).item())


# ───────────────────────────────────────────── evaluation (validation only)
def evaluate_e2(model, frame: pd.DataFrame, cfg: FoundationConfig, obs_cfg: O2ObservationConfig,
                split: Any = EVAL_SPLIT, name: str = "ppo_e2") -> EpisodeResult:
    env = make_e2_env(frame, split, cfg, obs_cfg, role="evaluation")
    policy = SB3DeterministicPolicyE2(model, env, role="evaluation", name=name, obs_cfg=obs_cfg)
    result = run_episode(env, policy, periods_per_year=cfg.periods_per_year)
    _assert_finite_curve(result)
    result.metrics.update(state_distribution(result))
    return result


def check_finite_policy_outputs(model, env: Any, obs_cfg: O2ObservationConfig | None = None) -> dict[str, Any]:
    import torch

    window = verify_e2_env(env, "diagnostic", obs_cfg)
    base = getattr(env, "unwrapped", env)
    if model.observation_space != base.observation_space or model.action_space != base.action_space:
        raise ForbiddenSplitError("E2 diagnostic: model spaces differ from the environment's spaces")
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


# ───────────────────────────────────────────── provenance / metadata
def provenance_e2(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    files = source_files(root)
    key = {}
    for label, rel in KEY_SOURCE_FILES_E2.items():
        if rel not in files:
            raise RuntimeError(f"provenance: key source file missing: {rel}")
        key[label] = files[rel]
    return {"git": git_identity(root), "source_fingerprint": source_fingerprint(files),
            "source_fingerprint_globs": list(SOURCE_FINGERPRINT_GLOBS), "source_files": files, "key_hashes": key}


def collect_metadata(config: E2Config, model, frame: pd.DataFrame, dataset_report: dict[str, Any],
                     cfg: FoundationConfig, obs_cfg: O2ObservationConfig, scaler_fit: O2ScalerFit,
                     train_seconds: float, preflight: dict[str, Any] | None = None,
                     source_before: dict[str, Any] | None = None) -> dict[str, Any]:
    import gymnasium
    import stable_baselines3
    import torch

    cost = cfg.costs[COST_PROFILE]
    train = resolve_e2_window(TRAIN_SPLIT, cfg, "train")
    validation = resolve_e2_window(EVAL_SPLIT, cfg, "evaluation")
    train_rec = _split_record(frame, train)
    prov = provenance_e2()
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
        "control_experiment": CONTROL_EXPERIMENT,
        "differs_from_control_in": ["observation"],
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
        "hyperparameter_rationale": HYPERPARAMETER_RATIONALE_E2,
        "total_timesteps_requested": int(config.total_timesteps),
        "total_timesteps_trained": int(model.num_timesteps),
        "train_split": train_rec,
        "validation_split": _split_record(frame, validation),
        "forbidden_splits": list(FORBIDDEN_SPLITS),
        "observation": {
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


REQUIRED_SCALER_KEYS: tuple[str, ...] = ("means", "stds", "fit_close_start", "fit_close_end", "fit_row_start",
                                         "fit_row_end", "n_closes", "n_rows", "window_name", "statistic", "features")


def validate_metadata(meta: dict[str, Any]) -> list[str]:
    """Missing/empty required keys of an E2 record (empty list = complete)."""
    problems = []
    for k in REQUIRED_METADATA_KEYS:
        if k not in meta:
            problems.append(f"missing {k}")
        elif meta[k] in (None, "", {}, []):
            problems.append(f"empty {k}")
    if "source" in meta and not meta["source"].get("source_fingerprint"):
        problems.append("empty source.source_fingerprint")
    obs = meta.get("observation", {})
    if not isinstance(obs, dict) or obs.get("definition") != OBSERVATION_DEFINITION:
        problems.append("observation.definition is not O2")
    fit = obs.get("scaler_fit", {}) if isinstance(obs, dict) else {}
    for k in REQUIRED_SCALER_KEYS:
        if k not in fit or fit[k] in (None, "", [], {}):
            problems.append(f"missing observation.scaler_fit.{k}")
    return problems


# ───────────────────────────────────────────── workflow
@dataclass
class E2RunResult:
    config: E2Config
    model: Any
    obs_config: O2ObservationConfig
    scaler_fit: O2ScalerFit
    metadata: dict[str, Any]
    validation: EpisodeResult
    artifacts_dir: Path


def run_e2(config: E2Config, frame: pd.DataFrame | None = None, cfg: FoundationConfig | None = None,
           dataset_report: dict[str, Any] | None = None, preflight: dict[str, Any] | None = None,
           write_artifacts: bool = True) -> E2RunResult:
    """Complete E2 workflow for one seed (mirror of ``ppo_e1.run_e1`` with the O2 observation)."""
    cfg = cfg or load_foundation_config()
    verify_split_table(cfg.splits)
    if frame is None:
        frame, dataset_report = load_dataset(cfg.dataset)
    if dataset_report is None:
        raise ValueError("dataset_report is required when frame is supplied")
    obs_cfg, scaler_fit = resolve_o2_config(frame, cfg)
    verify_o2_warmup(frame, resolve_e2_window(TRAIN_SPLIT, cfg, "train"))
    out = config.artifacts_dir
    if write_artifacts:
        out.mkdir(parents=True, exist_ok=True)

    source_before = provenance_e2()
    train_env = make_train_env(frame, cfg, obs_cfg)
    model = build_model(train_env, config, log_dir=(out / "train_log") if write_artifacts else None, obs_cfg=obs_cfg)
    t0 = time.time()
    model.learn(total_timesteps=int(config.total_timesteps), progress_bar=False)
    train_seconds = time.time() - t0

    if write_artifacts:
        model.save(str(out / "model.zip"))
        model = load_e2_model(out / "model.zip")

    finite = check_finite_policy_outputs(model, make_e2_env(frame, EVAL_SPLIT, cfg, obs_cfg, role="diagnostic"), obs_cfg)
    validation = evaluate_e2(model, frame, cfg, obs_cfg, EVAL_SPLIT, name=f"ppo_e2_seed{config.seed}")
    meta = collect_metadata(config, model, frame, dataset_report, cfg, obs_cfg, scaler_fit, train_seconds, preflight,
                            source_before=source_before)
    meta["finite_output_check"] = finite
    if write_artifacts:
        write_json(out / "metadata.json", meta)
        write_json(out / "validation_metrics.json", validation.metrics)
        validation.curve.to_csv(out / "validation_curve.csv")
    return E2RunResult(config, model, obs_cfg, scaler_fit, meta, validation, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E2: PPO with the O2 engineered observation (R1; modelled costs: 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale), "
                                             "train-only, validation-only evaluation.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=FULL_BUDGET_TIMESTEPS)
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_BUDGET_TIMESTEPS}-step integration run into artifacts/ppo/e2_smoke/")
    ap.add_argument("--skip-preflight", action="store_true", help="do not re-run the certification gate in-process")
    ap.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args(argv)

    config = E2Config(seed=args.seed, total_timesteps=SMOKE_BUDGET_TIMESTEPS if args.smoke else args.timesteps,
                      artifacts_root=args.artifacts_root, run_group=E2_SMOKE_RUN_GROUP if args.smoke else E2_RUN_GROUP,
                      verbose=args.verbose)
    preflight = None if args.skip_preflight else _preflight_or_die()
    res = run_e2(config, preflight=preflight)
    m = res.validation.metrics
    fit = res.metadata["observation"]["scaler_fit"]
    print(f"\nE2 seed={config.seed} timesteps={res.metadata['total_timesteps_trained']} "
          f"train {res.metadata['train_seconds']:.0f}s -> {res.artifacts_dir}")
    print(f"O2 scaler fitted on TRAIN feature rows {fit['fit_row_start']}..{fit['fit_row_end']} "
          f"(closes {fit['fit_close_start']}..{fit['fit_close_end']}, {fit['n_rows']} rows) "
          f"| source fingerprint {res.metadata['source']['source_fingerprint'][:12]}")
    print(f"validation ({m['first_decision_date'].date()}..{m['last_decision_date'].date()}, {m['n_steps']} steps): "
          f"total_return={m['total_return']:+.4f} final_equity={m['final_equity']:.4f} "
          f"sharpe={m['sharpe_ratio'] if m['sharpe_ratio'] is None else round(m['sharpe_ratio'], 3)} "
          f"max_dd={m['max_drawdown']:+.4f} legs={m['n_legs']} entries={m['n_entries']} "
          f"exposure={m['exposure']:.3f} cost_drag={m['total_cost_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
