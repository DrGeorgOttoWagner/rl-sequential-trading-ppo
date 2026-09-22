"""
ppo_e3.py — PPO experiment E3 (O1 minimal observation / R2 risk-aware reward / 0.10% fee + 0.05% slippage allowance per trade).

    python -m btc_rl.ppo_e3 --seed 42                       # full E3 run (200k timesteps)
    python -m btc_rl.ppo_e3 --seed 42 --smoke               # 16,384-step integration run
    python -m btc_rl.ppo_e3 --seed 42 --skip-preflight      # only for already-certified sessions

E3 is the RQ3 treatment of docs/methodology.md: it differs from the
control E1 in exactly ONE respect, the training reward (R2, ``reward_r2.py``)
instead of R1.  Everything else is inherited unchanged from E1 (``ppo_e1.py``):
dataset, frozen splits, market timeline, O1 observation (10 scaled log returns
+ position, TRAIN-only scale), CASH/LONG target-state actions, 0.10% fee +
0.05% slippage allowance per trade, the SB3 PPO implementation and every hyperparameter, the
[64, 64] Tanh MLP, CPU with one torch thread, one environment, the seeds
42 / 123 / 2026, the 200,000-step budget (200,704 trained), deterministic
full-window TRAIN episodes, validation accounting and the baseline definitions.
``CANONICAL_E3_SPEC`` is DERIVED from ``CANONICAL_E1_SPEC`` by replacing only
the reward block; a test asserts the two specifications are otherwise identical.

Reward R2 (prospectively frozen, see ``reward_r2.py``)
-----------------------------------------------------
    R2_t = financial_r1_t - 0.0005 * legs_t - 0.10 * max(0, D_{t+1} - D_t)

R2 is a training incentive only.  ``BtcUsdtTradingEnvR2`` calls the certified
E1 ``step`` for every financial quantity and replaces only the returned reward,
so equity, costs, positions, drawdown and every evaluation metric are the
accepted E1 accounting of the realised equity path.  The per-step decomposition
(financial_reward_r1, turnover_penalty, drawdown_penalty, reward_r2, ...) is
recorded in ``info``, in the saved validation curve and as cumulative reward
diagnostics next to (never instead of) the financial metrics.

Split policy, guarantee boundary, artifacts
-------------------------------------------
Identical to E1/E2 (fail-closed window resolution, prebuilt-environment
verification incl. the frozen R2 coefficients, bound deterministic adapter;
TRAIN for training, VALIDATION for evaluation, TEST never built by this
module).  Artifacts go to ``artifacts/ppo/e3/<seed>/`` (``e3_smoke`` for the
smoke run); ``E3Config`` refuses run groups that could collide with E1's or
E2's.  The accepted E1 and E2 artifacts are never written: their SHA-256
anchors (``E1_FROZEN_ARTIFACTS`` from ``ppo_e2``, ``E2_FROZEN_ARTIFACTS`` here)
and the training-core source hashes recorded by the accepted runs are verified
before and after every comparison.
"""

from __future__ import annotations

import argparse
import copy
import platform
import re
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
from .evaluation import EpisodeResult, compute_metrics
from .observations import ObservationConfig, observation_space
from .policies import Policy
from .ppo_e1 import (
    ACTION_SEMANTICS,
    CANONICAL_E1_SPEC,
    CASH,
    COST_PROFILE,
    DEFAULT_ARTIFACTS_ROOT,
    DEVELOPMENT_SEEDS,
    E1_INITIAL_EQUITY,
    E1_LOOKBACK,
    EVAL_SPLIT,
    FORBIDDEN_SPLITS,
    FULL_BUDGET_TIMESTEPS,
    HYPERPARAMETER_RATIONALE,
    IDENTITY_KEYS,
    KEY_SOURCE_FILES,
    LEGAL_WINDOWS,
    OBSERVATION_DEFINITION,
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
    resolve_observation_config,
    sha256_of,
    source_files,
    source_fingerprint,
    state_distribution,
    verify_split_table,
    write_json,
)
from .ppo_e2 import E1_FROZEN_ARTIFACTS, E1_FROZEN_CORE_SOURCE_HASHES, E1_RUN_GROUP
from .preflight import LOCK_PATH, PACKAGE_ROOT
from .reward_r2 import (
    DRAWDOWN_PENALTY_LAMBDA,
    FROZEN_R2,
    R2_ACCOUNTING_SEPARATION,
    R2_FORMULA,
    R2_INFO_KEYS,
    R2_RESET,
    R2_TERMS,
    R2_TIMING,
    REWARD_DEFINITION,
    TURNOVER_PENALTY_LAMBDA,
    BtcUsdtTradingEnvR2,
    R2Coefficients,
)
from .scaling import ScaleFit
from .splits import FROZEN_SPLITS, SETTLEMENT_BARS, SplitWindow, usable_decision_index_range

assert CASH == FLAT == 0 and LONG == 1

# ───────────────────────────────────────────── experiment identity
EXPERIMENT = "E3"
CONTROL_EXPERIMENT = "E1"
STUDY_STAGE = "ppo-e3"
E3_RUN_GROUP = "e3"
E3_SMOKE_RUN_GROUP = "e3_smoke"
FORBIDDEN_RUN_GROUP_PREFIXES: tuple[str, ...] = ("e1", "e2")   # E3 must never write into an E1 or E2 run group
E2_RUN_GROUP = "e2"
# the accepted E1/E2 artifact directories, resolved; no E3 output path may equal or descend
# into them (or contain them), whatever run group / artifacts root spelling is used.
PROTECTED_ARTIFACT_DIRS: tuple[Path, ...] = tuple((DEFAULT_ARTIFACTS_ROOT / g).resolve() for g in (E1_RUN_GROUP, E2_RUN_GROUP))
# a run group is exactly ONE plain path component: letters, digits, '_' and '-' only (no separators, no '.', not empty)
RUN_GROUP_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_run_group(run_group: Any) -> str:
    """
    A legal E3 run group is a single plain path component that does not
    collide with the E1/E2 groups.  Everything else — separators (``e1/``,
    ``e3/../e2``), dot components (``./e1``, ``../e1``), absolute paths, empty
    or non-string values — is rejected BEFORE any path is formed.
    """
    if not isinstance(run_group, str) or not run_group:
        raise ValueError("run_group must be a non-empty string")
    if not RUN_GROUP_PATTERN.fullmatch(run_group):
        raise ValueError(
            f"run_group {run_group!r} is not a single plain path component (letters, digits, '_' and '-' only; "
            "no '/', '\\', '.', '..' or absolute paths); E3 never writes outside artifacts/<run_group>/<seed>"
        )
    if any(run_group == p or run_group.startswith(p + "_") for p in FORBIDDEN_RUN_GROUP_PREFIXES):
        raise ValueError(f"run_group {run_group!r} collides with the frozen E1/E2 artifacts; E3 never writes there")
    return run_group


def verify_e3_output_dir(path: Path, protected: tuple[Path, ...] | None = None) -> Path:
    """
    Resolve ``path`` (symlinks and '..' included) and refuse it if it equals,
    descends into, or contains an accepted E1/E2 artifact directory.  Returns
    the resolved path.  Pure: creates nothing.  ``protected`` defaults to the
    module constant read at call time.
    """
    if protected is None:
        protected = PROTECTED_ARTIFACT_DIRS
    resolved = Path(path).resolve()
    for prot in protected:
        if resolved == prot or resolved.is_relative_to(prot) or prot.is_relative_to(resolved):
            raise ValueError(
                f"E3 output directory {resolved} would alias the protected accepted artifacts {prot}; "
                "E3 never writes into or over E1/E2 artifacts"
            )
    return resolved

# provenance: E1's key sources plus the R2/E3 sources
KEY_SOURCE_FILES_E3: dict[str, str] = {
    **KEY_SOURCE_FILES,
    "reward_r2_source": "src/btc_rl/reward_r2.py",
    "ppo_e3_source": "src/btc_rl/ppo_e3.py",
    "e3_report_source": "src/btc_rl/ppo_e3_report.py",
}

# SHA-256 anchors of the accepted E2 artifacts (closure, commit 5c9f4a5c; checkpoint hashes equal
# those recorded in the independent review record).  Taken before any E3 work.  E3 never writes
# these files; the E1-vs-E3 comparison verifies E1 and E2 anchors before and after reading E1 results.
E2_FROZEN_ARTIFACTS: dict[int, dict[str, str]] = {
    42: {
        "model.zip": "fda38d95ab6e813c544df2d31ce83e412d610f3551310bbfa93a956de2cf0e81",
        "metadata.json": "f55e147100846754543a3435e72c11c9434f474f6e2d26ddae6e6e40d815567c",
        "validation_curve.csv": "1e44ea35ec1350cfa1907c4091945c9ef270eaca61d397dd5f47b25ed010f0f2",
        "validation_metrics.json": "fd78856eb878f40b4bb7578b73d1c23d4256c2184561bb0328e171c2b042a429",
    },
    123: {
        "model.zip": "b13599661ff5cf8d8c5582be0bb375b0ec6e7c677e097331cce457e9860280cf",
        "metadata.json": "1cb2df3c087a75b02bdc5b51be638801475adf0db9cf7df92b5cb5ce7daa8823",
        "validation_curve.csv": "cc9495e1dcc951fc9493cb1a3c836027098eb2a8d2db9855d61626d79c8124d3",
        "validation_metrics.json": "501508b8b6d9b66e1d485cff04344ef63ea8d64aa868ecd0e11fd749db59f1c3",
    },
    2026: {
        "model.zip": "1423ac351b4cf38cc58d75ddf7582c059e424b5b7d98b479063a86db57d4a1aa",
        "metadata.json": "67cc854a99f45479700ec9a2573120c5c0f3ce3f7c28df526a4e3da5464aaee7",
        "validation_curve.csv": "9536758f7103924650f519b8e266671c53021caf5c425af4103a3b21411c1c5b",
        "validation_metrics.json": "ee81012dd3d8d1c7d0b9c037f61e1010cc441564176342d3c80ef3387c312515",
    },
}
# The E2 training-core source (O2 observation) as recorded by the accepted E2 runs; E3 does not touch it.
E2_FROZEN_CORE_SOURCE_HASHES: dict[str, str] = {
    **E1_FROZEN_CORE_SOURCE_HASHES,
    "o2_observation_source": "7ad68a27df2de01ce8e7f13937da82f11282c74a6de1d34013b9dd6ddf84f4cc",
}
_E2_KEY_SOURCE_FILES: dict[str, str] = {**KEY_SOURCE_FILES, "o2_observation_source": "src/btc_rl/observations_o2.py"}


def verify_accepted_core_sources_unchanged(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Every E1 and E2 training-core source of the current tree must hash to the accepted values."""
    hashes = {label: sha256_of(Path(root) / _E2_KEY_SOURCE_FILES[label]) for label in E2_FROZEN_CORE_SOURCE_HASHES}
    changed = sorted(k for k, v in hashes.items() if v != E2_FROZEN_CORE_SOURCE_HASHES[k])
    if changed:
        raise RuntimeError(f"E1/E2 training-core sources changed since the accepted runs: {changed}")
    return {"unchanged": True, "hashes": hashes}


def artifact_status(anchors: dict[int, dict[str, str]], artifacts_root: Path, run_group: str) -> dict[str, Any]:
    """Compare the run files on disk with frozen anchors (missing files are reported, not hashed)."""
    root = Path(artifacts_root) / run_group
    status: dict[str, Any] = {"run_group": run_group, "present": {}, "changed": [], "missing": []}
    for seed, files in anchors.items():
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


def verify_accepted_artifacts_unchanged(artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, Any]:
    """The accepted E1 AND E2 run files must be present and byte-identical to their anchors."""
    out = {}
    for label, anchors, group in (("E1", E1_FROZEN_ARTIFACTS, E1_RUN_GROUP), ("E2", E2_FROZEN_ARTIFACTS, E2_RUN_GROUP)):
        status = artifact_status(anchors, artifacts_root, group)
        if not status["all_present"]:
            raise RuntimeError(f"{label} canonical artifacts missing: {status['missing']}")
        if not status["unchanged"]:
            raise RuntimeError(f"{label} canonical artifacts differ from the frozen anchors: {status['changed']}")
        out[label] = status
    return out


# ───────────────────────────────────────────── split policy (mirrors E1, E3-labelled)
def resolve_e3_window(spec: Any, cfg: FoundationConfig, role: str) -> SplitWindow:
    """Same fail-closed resolution as ``ppo_e1.resolve_e1_window`` (TRAIN for training, VALIDATION for evaluation)."""
    if role not in LEGAL_WINDOWS:
        raise ValueError(f"unknown E3 role {role!r}")
    verify_split_table(cfg.splits)
    if isinstance(spec, str):
        if spec in FORBIDDEN_SPLITS or spec not in cfg.splits:
            raise ForbiddenSplitError(f"E3 {role}: split name {spec!r} is not legal (legal: {LEGAL_WINDOWS[role]})")
        window = cfg.splits[spec]
    elif isinstance(spec, SplitWindow):
        window = spec
    else:
        raise ForbiddenSplitError(f"E3 {role}: unsupported split specification of type {type(spec).__name__}")
    for legal_name in LEGAL_WINDOWS[role]:
        if _same_window(window, FROZEN_SPLITS[legal_name]):
            return FROZEN_SPLITS[legal_name]
    raise ForbiddenSplitError(
        f"E3 {role}: window {window.name} {window.start.date()}..{window.end.date()} is not a legal "
        f"{'/'.join(LEGAL_WINDOWS[role]).upper()} window; the test split is off-limits for PPO training, "
        "evaluation and every performance-driven decision"
    )


# ───────────────────────────────────────────── environment: certified O1 env with the R2 reward
def make_r2_split_env(frame: pd.DataFrame, window: SplitWindow, obs_cfg: ObservationConfig,
                      cost_config: CostConfig = CONSERVATIVE_COST, **kwargs: Any) -> BtcUsdtTradingEnvR2:
    """R2 environment for one frozen window under the purge rule (same construction as ``env.make_split_env``)."""
    if not isinstance(window, SplitWindow):
        raise TypeError("window must be a SplitWindow")
    i0, last_usable = usable_decision_index_range(frame, window)
    return BtcUsdtTradingEnvR2(frame, decision_start=frame.index[i0], decision_end=frame.index[last_usable],
                               cost_config=cost_config, obs_config=obs_cfg, price_end=window.end, **kwargs)


def verify_e3_env(env: Any, role: str, obs_cfg: ObservationConfig | None = None) -> SplitWindow:
    """
    A prebuilt environment is legal for ``role`` only if (mirror of ``verify_e1_env``):
    only ``Monitor`` wrappers; unwrapped type exactly ``BtcUsdtTradingEnvR2``;
    ``Discrete(2)``; O1 config (lookback 10, equal to ``obs_cfg`` when given) and
    matching 11-d space; the FROZEN R2 coefficients; frozen conservative costs;
    initial equity 1.0; deterministic full-window episodes; decision range /
    last mark / ``price_end`` equal a legal frozen window for ``role``.
    """
    from gymnasium import spaces

    if role not in LEGAL_WINDOWS:
        raise ValueError(f"unknown E3 role {role!r}")
    chain, base = _wrapper_chain(env)
    supported = _supported_wrapper_types()
    for w in chain:
        if w not in supported:
            raise ForbiddenSplitError(
                f"E3 {role}: unsupported wrapper {w.__module__}.{w.__name__}; only "
                f"{', '.join(t.__module__ + '.' + t.__name__ for t in supported)} may wrap an E3 environment"
            )
    if type(base) is not BtcUsdtTradingEnvR2:
        raise ForbiddenSplitError(f"E3 {role}: environment of type {type(base).__name__} is not the certified R2 env")
    for attr in ("_dates", "_i1", "price_end", "obs_config", "cost_config", "random_start", "episode_length",
                 "initial_equity", "action_space", "observation_space", "r2_coefficients"):
        if not hasattr(base, attr):
            raise ForbiddenSplitError(f"E3: environment lacks the certified attribute {attr!r}")
    if base.action_space != spaces.Discrete(2):
        raise ForbiddenSplitError(f"E3 {role}: action space {base.action_space} is not Discrete(2) CASH/LONG")
    oc = base.obs_config
    if not isinstance(oc, ObservationConfig) or oc.lookback != E1_LOOKBACK:
        raise ForbiddenSplitError(f"E3 {role}: observation config {oc!r} is not O1 (lookback {E1_LOOKBACK})")
    if obs_cfg is not None and oc != obs_cfg:
        raise ForbiddenSplitError(f"E3 {role}: observation config {oc!r} differs from the expected {obs_cfg!r}")
    if base.observation_space != observation_space(oc) or base.observation_space.shape != (E1_LOOKBACK + 1,):
        raise ForbiddenSplitError(f"E3 {role}: observation space does not match the O1 config")
    if not isinstance(base.r2_coefficients, R2Coefficients) or not base.r2_coefficients.is_frozen:
        raise ForbiddenSplitError(
            f"E3 {role}: R2 coefficients {base.r2_coefficients!r} are not the frozen values "
            f"(turnover {TURNOVER_PENALTY_LAMBDA}, drawdown {DRAWDOWN_PENALTY_LAMBDA})"
        )
    cost = base.cost_config
    if cost != CONSERVATIVE_COST or cost.one_leg_fraction != 0.0015:
        raise ForbiddenSplitError(f"E3 {role}: cost profile {cost!r} is not the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    if base.initial_equity != E1_INITIAL_EQUITY:
        raise ForbiddenSplitError(f"E3 {role}: initial equity {base.initial_equity} != {E1_INITIAL_EQUITY}")
    if base.random_start is not False or base.episode_length is not None:
        raise ForbiddenSplitError(
            f"E3 {role}: episode options random_start={base.random_start} episode_length={base.episode_length}; "
            "E3 uses deterministic full-window episodes only (as E1)"
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
        f"E3 {role}: environment decisions {first.date()}..{last.date()}, price_end "
        f"{None if base.price_end is None else base.price_end.date()} do not match a legal frozen "
        f"{'/'.join(LEGAL_WINDOWS[role]).upper()} window"
    )


# ───────────────────────────────────────────── configuration (identical PPO settings to E1)
@dataclass(frozen=True)
class E3Config:
    seed: int
    total_timesteps: int = FULL_BUDGET_TIMESTEPS
    hyperparameters: PPOHyperparameters = field(default_factory=PPOHyperparameters)
    scale_returns_on_train: bool = True
    torch_threads: int = 1
    device: str = "cpu"
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT
    run_group: str = E3_RUN_GROUP          # artifacts/ppo/<run_group>/<seed>/
    verbose: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        if self.device != "cpu":
            raise ValueError("E3 runs on CPU only (identical to E1); pass device='cpu'")
        validate_run_group(self.run_group)
        if not isinstance(self.artifacts_root, (str, Path)):
            raise ValueError("artifacts_root must be a path")
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root))
        verify_e3_output_dir(self.artifacts_dir)   # resolved destination must not alias the accepted E1/E2 artifacts

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_root / self.run_group / str(self.seed)


HYPERPARAMETER_RATIONALE_E3: dict[str, str] = {
    **HYPERPARAMETER_RATIONALE,
    "reward": (f"R2 = R1 - {TURNOVER_PENALTY_LAMBDA} * legs - {DRAWDOWN_PENALTY_LAMBDA} * max(0, D_(t+1) - D_t): "
               "coefficients fixed prospectively, never tuned, unchanged after VALIDATION; "
               "training incentive only, financial accounting is the E1 accounting."),
    "experimental_control": "every PPO/env/observation/cost/seed/budget setting equals E1 by construction (CANONICAL_E3_SPEC "
                            "is derived from CANONICAL_E1_SPEC); only the training reward differs (RQ3).",
}

# ───────────────────────────────────────────── frozen canonical E3 specification (derived from E1)
E3_SPEC_DIFFERENCE_KEYS: tuple[str, ...] = ("experiment", "reward", "reward_r2")
R2_SPEC: dict[str, Any] = {
    "definition": REWARD_DEFINITION,
    "formula": R2_FORMULA,
    "turnover_penalty_lambda": TURNOVER_PENALTY_LAMBDA,
    "drawdown_penalty_lambda": DRAWDOWN_PENALTY_LAMBDA,
    "financial_reward": "R1 (E1 accounting after modelled 0.10% fee + 0.05% slippage allowance per trade)",
    "incremental_drawdown": "max(0, D_{t+1} - D_t), D = 1 - equity / running peak (initial equity is the first peak)",
}


def _derive_canonical_e3_spec() -> dict[str, Any]:
    spec = copy.deepcopy(CANONICAL_E1_SPEC)
    spec["experiment"] = EXPERIMENT
    spec["reward"] = REWARD_DEFINITION
    spec["reward_r2"] = copy.deepcopy(R2_SPEC)
    return spec


CANONICAL_E3_SPEC: dict[str, Any] = _derive_canonical_e3_spec()


# ───────────────────────────────────────────── environments
def make_e3_env(frame: pd.DataFrame, split: Any, cfg: FoundationConfig, obs_cfg: ObservationConfig,
                role: str = "diagnostic") -> BtcUsdtTradingEnvR2:
    window = resolve_e3_window(split, cfg, role)
    cost = cfg.costs[COST_PROFILE]
    if cost != CONSERVATIVE_COST or abs(cost.one_leg_fraction - 0.0015) > 1e-15:
        raise RuntimeError("E3 requires the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    env = make_r2_split_env(frame, window, obs_cfg, cost_config=cost, initial_equity=cfg.initial_equity,
                            r2_coefficients=FROZEN_R2)
    verify_e3_env(env, role)
    return env


def make_train_env(frame: pd.DataFrame, cfg: FoundationConfig, obs_cfg: ObservationConfig):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(make_e3_env(frame, TRAIN_SPLIT, cfg, obs_cfg, role="train"))


# ───────────────────────────────────────────── model
def build_model(env, config: E3Config, log_dir: Path | None = None, obs_cfg: ObservationConfig | None = None):
    """PPO on ``env`` (a Monitor-wrapped frozen TRAIN R2 environment); same SB3 call as E1."""
    import torch
    from stable_baselines3 import PPO

    verify_e3_env(env, "train", obs_cfg)
    torch.set_num_threads(int(config.torch_threads))
    model = PPO("MlpPolicy", env, seed=int(config.seed), device=config.device, verbose=config.verbose,
                **config.hyperparameters.sb3_kwargs())
    if log_dir is not None:
        from stable_baselines3.common.logger import configure

        log_dir.mkdir(parents=True, exist_ok=True)
        model.set_logger(configure(str(log_dir), ["csv"]))
    return model


load_e3_model = load_e1_model  # plain PPO.load on CPU; the E3 boundaries verify spaces/config afterwards


class SB3DeterministicPolicyE3:
    """Deterministic (argmax) adapter bound to a verified E3 environment and role (mirror of the E1 adapter)."""

    def __init__(self, model, env: Any, role: str = "evaluation", name: str = "ppo_e3",
                 obs_cfg: ObservationConfig | None = None) -> None:
        if role not in ("evaluation", "diagnostic"):
            raise ForbiddenSplitError(f"E3 adapter: role {role!r} is not a learned-policy role (evaluation/diagnostic)")
        self.window = verify_e3_env(env, role, obs_cfg)
        base = getattr(env, "unwrapped", env)
        if model.observation_space != base.observation_space or model.action_space != base.action_space:
            raise ForbiddenSplitError("E3 adapter: model spaces differ from the bound environment's spaces")
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
            raise ForbiddenSplitError("E3 adapter: info lacks the certified decision/state dates") from exc
        if not (self._first_decision <= decision <= self._last_decision) or instant > self.window.end:
            raise ForbiddenSplitError(
                f"E3 adapter bound to {self.window.name} refused decision {decision.date()} "
                f"(legal {self._first_decision.date()}..{self._last_decision.date()}) before prediction"
            )
        if not self._obs_space.contains(np.asarray(obs, dtype=np.float32)):
            raise ForbiddenSplitError("E3 adapter: observation outside the bound observation space")
        action, _ = self.model.predict(obs, deterministic=True)
        self._predictions += 1
        return int(np.asarray(action).item())


# ───────────────────────────────────────────── episode runner with the reward decomposition
R2_CURVE_COLUMNS: tuple[str, ...] = ("financial_reward_r1", "turnover_penalty", "drawdown_penalty", "reward_r2",
                                     "incremental_drawdown", "drawdown_before", "drawdown_after",
                                     "running_peak_before", "running_peak_after")
REWARD_DIAGNOSTIC_KEYS: tuple[str, ...] = ("cumulative_financial_r1", "cumulative_turnover_penalty",
                                           "cumulative_drawdown_penalty", "cumulative_reward_r2",
                                           "n_turnover_penalised_steps", "n_drawdown_penalised_steps",
                                           "max_step_drawdown_penalty", "log_final_equity")


def reward_diagnostics(curve: pd.DataFrame, initial_equity: float = 1.0) -> dict[str, Any]:
    """
    Cumulative reward terms of one episode.  These are TRAINING-REWARD
    diagnostics, never financial returns: cumulative_financial_r1 equals
    log(final_equity / initial_equity) by construction (a check, not a metric),
    and cumulative_reward_r2 is what the agent was trained to maximise.
    """
    r1 = curve["financial_reward_r1"].to_numpy(dtype=np.float64)
    tp = curve["turnover_penalty"].to_numpy(dtype=np.float64)
    dp = curve["drawdown_penalty"].to_numpy(dtype=np.float64)
    r2 = curve["reward_r2"].to_numpy(dtype=np.float64)
    return {
        "cumulative_financial_r1": float(r1.sum()),
        "cumulative_turnover_penalty": float(tp.sum()),
        "cumulative_drawdown_penalty": float(dp.sum()),
        "cumulative_reward_r2": float(r2.sum()),
        "n_turnover_penalised_steps": int((tp > 0).sum()),
        "n_drawdown_penalised_steps": int((dp > 0).sum()),
        "max_step_drawdown_penalty": float(dp.max()) if len(dp) else 0.0,
        "log_final_equity": float(np.log(curve["equity"].to_numpy(dtype=np.float64)[-1] / initial_equity)),
    }


def run_episode_e3(env: BtcUsdtTradingEnvR2, policy: Policy, seed: int | None = None,
                   options: dict[str, Any] | None = None, periods_per_year: int = 365) -> EpisodeResult:
    """
    ``evaluation.run_episode`` with the R2 decomposition columns appended.
    Financial metrics come from ``evaluation.compute_metrics`` on the equity
    path exactly as in E1; the reward diagnostics are added under separate,
    explicitly named keys.  ``reward`` in the curve is the RETURNED (R2) reward;
    ``financial_reward_r1`` is the E1 reward of the same step.
    """
    policy.reset(seed)
    obs, info = env.reset(seed=seed, options=options)
    rows: list[dict[str, Any]] = []
    done = False
    while not done:
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        row = {
            "decision_date": info["settled_decision_date"],
            "fill_date": info["fill_date"],
            "mark_date": info["mark_date"],
            "action": info["action"],
            "position": info["position"],
            "legs": info["legs"],
            "cost_fraction": info["cost_fraction"],
            "gross_simple_return": info["gross_simple_return"],
            "position_return": info["position_return"],
            "reward": reward,
            "equity": info["equity"],
            "drawdown": info["drawdown"],
        }
        for k in R2_CURVE_COLUMNS:
            row[k] = info[k]
        if info["reward_definition"] != REWARD_DEFINITION or reward != info["reward_r2"]:
            raise RuntimeError("returned reward is not the recorded R2 of the same step")
        rows.append(row)
        done = terminated or truncated
    curve = pd.DataFrame(rows).set_index("decision_date")
    metrics = compute_metrics(curve, env.initial_equity, periods_per_year)   # financial metrics from the equity path
    metrics["policy"] = policy.name
    metrics["reward_definition_of_sum_reward"] = REWARD_DEFINITION       # sum_reward is cumulative R2, not a return
    metrics.update(reward_diagnostics(curve, env.initial_equity))
    if abs(metrics["cumulative_financial_r1"] - metrics["log_final_equity"]) > 1e-9:
        raise RuntimeError("cumulative R1 does not equal log(final equity): accounting/reward inconsistency")
    return EpisodeResult(policy=policy.name, curve=curve, metrics=metrics,
                         start_index=info["episode_start_index"], end_index=info["episode_end_index"])


# ───────────────────────────────────────────── evaluation (validation only)
def evaluate_e3(model, frame: pd.DataFrame, cfg: FoundationConfig, obs_cfg: ObservationConfig,
                split: Any = EVAL_SPLIT, name: str = "ppo_e3") -> EpisodeResult:
    env = make_e3_env(frame, split, cfg, obs_cfg, role="evaluation")
    policy = SB3DeterministicPolicyE3(model, env, role="evaluation", name=name, obs_cfg=obs_cfg)
    result = run_episode_e3(env, policy, periods_per_year=cfg.periods_per_year)
    _assert_finite_curve(result)
    result.metrics.update(state_distribution(result))
    return result


def check_finite_policy_outputs(model, env: Any, obs_cfg: ObservationConfig | None = None) -> dict[str, Any]:
    """One deterministic episode: every observation, R2 reward, decomposition term, logit and value finite."""
    import torch

    window = verify_e3_env(env, "diagnostic", obs_cfg)
    base = getattr(env, "unwrapped", env)
    if model.observation_space != base.observation_space or model.action_space != base.action_space:
        raise ForbiddenSplitError("E3 diagnostic: model spaces differ from the environment's spaces")
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
    return {"steps_checked": n, "window": window.name, "reward": REWARD_DEFINITION}


# ───────────────────────────────────────────── provenance / metadata
def provenance_e3(root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    files = source_files(root)
    key = {}
    for label, rel in KEY_SOURCE_FILES_E3.items():
        if rel not in files:
            raise RuntimeError(f"provenance: key source file missing: {rel}")
        key[label] = files[rel]
    return {"git": git_identity(root), "source_fingerprint": source_fingerprint(files),
            "source_fingerprint_globs": list(SOURCE_FINGERPRINT_GLOBS), "source_files": files, "key_hashes": key}


def reward_metadata() -> dict[str, Any]:
    return {
        "definition": REWARD_DEFINITION,
        "formula": R2_FORMULA,
        "terms": R2_TERMS,
        "turnover_penalty_lambda": TURNOVER_PENALTY_LAMBDA,
        "drawdown_penalty_lambda": DRAWDOWN_PENALTY_LAMBDA,
        "coefficients": FROZEN_R2.to_json(),
        "coefficient_policy": "fixed prospectively; never tuned; unchanged after VALIDATION; reused by E4",
        "financial_reward_r1": "r1_t = legs_t * log(1 - c) + p_{t+1} * log(open[t+2] / open[t+1]) (accepted E1 reward)",
        "timing": R2_TIMING,
        "reset": R2_RESET,
        "accounting_separation": R2_ACCOUNTING_SEPARATION,
        "shaping": "turnover penalty (per changed-position leg) and incremental-drawdown penalty; no Sharpe term",
        "decomposition_recorded": list(R2_CURVE_COLUMNS),
        "sb3_training_target_note": ("at the TRAIN window end (truncated=True) SB3 bootstraps the last rollout "
                                     "reward with gamma * V(terminal_observation); environment reward unchanged"),
    }


def collect_metadata(config: E3Config, model, frame: pd.DataFrame, dataset_report: dict[str, Any],
                     cfg: FoundationConfig, obs_cfg: ObservationConfig, scale_fit: ScaleFit | None,
                     train_seconds: float, preflight: dict[str, Any] | None = None,
                     source_before: dict[str, Any] | None = None, model_sha256: str | None = None) -> dict[str, Any]:
    import gymnasium
    import stable_baselines3
    import torch

    cost = cfg.costs[COST_PROFILE]
    train = resolve_e3_window(TRAIN_SPLIT, cfg, "train")
    validation = resolve_e3_window(EVAL_SPLIT, cfg, "evaluation")
    train_rec = _split_record(frame, train)
    prov = provenance_e3()
    if scale_fit is not None and scale_fit.scale != obs_cfg.return_scale:
        raise RuntimeError("observation scale differs from the recorded fit")
    source = {k: prov[k] for k in ("source_fingerprint", "source_fingerprint_globs", "key_hashes", "source_files")}
    if source_before is not None:
        source["fingerprint_before_training"] = source_before["source_fingerprint"]
        source["source_stable_during_run"] = source_before["source_fingerprint"] == prov["source_fingerprint"]
        if not source["source_stable_during_run"]:
            raise RuntimeError("source files changed while the run was executing; the run is not attributable")
    meta = {
        "experiment": EXPERIMENT,
        "control_experiment": CONTROL_EXPERIMENT,
        "differs_from_control_in": ["reward"],
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
        "hyperparameter_rationale": HYPERPARAMETER_RATIONALE_E3,
        "total_timesteps_requested": int(config.total_timesteps),
        "total_timesteps_trained": int(model.num_timesteps),
        "train_split": train_rec,
        "validation_split": _split_record(frame, validation),
        "forbidden_splits": list(FORBIDDEN_SPLITS),
        "observation": {   # identical to the accepted E1 record
            "definition": OBSERVATION_DEFINITION,
            "content": "10 most recent daily log close returns (scaled) + current portfolio state (0=CASH, 1=LONG)",
            "lookback": int(obs_cfg.lookback),
            "return_scale": float(obs_cfg.return_scale),
            "scale_fit": (scale_fit.to_json() if scale_fit is not None
                          else {"scale": 1.0, "fit_start": None, "fit_end": None, "statistic": "not scaled"}),
            "size": int(obs_cfg.size),
        },
        "reward": reward_metadata(),
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
            # SHA-256 of the saved checkpoint bytes: an E3 record is bound to exactly this checkpoint, so an E1
            # checkpoint (same 11-d space, same seed and hyperparameters) can never be reported as an E3 run
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


REQUIRED_REWARD_KEYS: tuple[str, ...] = ("definition", "formula", "turnover_penalty_lambda", "drawdown_penalty_lambda",
                                         "coefficients", "timing", "reset", "accounting_separation",
                                         "decomposition_recorded")


def validate_metadata(meta: dict[str, Any]) -> list[str]:
    """Missing/empty required keys of an E3 record (empty list = complete)."""
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
        problems.append("observation.definition is not O1")
    if isinstance(obs, dict) and obs.get("scale_fit", {}).get("statistic") != "not scaled":
        for k in ("fit_start", "fit_end", "scale"):
            if not obs.get("scale_fit", {}).get(k):
                problems.append(f"missing observation.scale_fit.{k}")
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
class E3RunResult:
    config: E3Config
    model: Any
    obs_config: ObservationConfig
    scale_fit: ScaleFit | None
    metadata: dict[str, Any]
    validation: EpisodeResult
    artifacts_dir: Path


def run_e3(config: E3Config, frame: pd.DataFrame | None = None, cfg: FoundationConfig | None = None,
           dataset_report: dict[str, Any] | None = None, preflight: dict[str, Any] | None = None,
           write_artifacts: bool = True) -> E3RunResult:
    """Complete E3 workflow for one seed (mirror of ``ppo_e1.run_e1`` with the R2 reward)."""
    # destination guard FIRST — before config/dataset loading, mkdir, training, any write.
    # The config already validated it at construction; this re-check covers a config altered afterwards.
    validate_run_group(config.run_group)
    out = verify_e3_output_dir(config.artifacts_dir)
    cfg = cfg or load_foundation_config()
    verify_split_table(cfg.splits)
    if frame is None:
        frame, dataset_report = load_dataset(cfg.dataset)
    if dataset_report is None:
        raise ValueError("dataset_report is required when frame is supplied")
    obs_cfg, scale_fit = resolve_observation_config(frame, cfg, config.scale_returns_on_train)  # E1's O1 + TRAIN scale
    if write_artifacts:
        out.mkdir(parents=True, exist_ok=True)

    source_before = provenance_e3()
    train_env = make_train_env(frame, cfg, obs_cfg)
    model = build_model(train_env, config, log_dir=(out / "train_log") if write_artifacts else None, obs_cfg=obs_cfg)
    t0 = time.time()
    model.learn(total_timesteps=int(config.total_timesteps), progress_bar=False)
    train_seconds = time.time() - t0

    model_sha256 = None
    if write_artifacts:
        model.save(str(out / "model.zip"))
        model_sha256 = sha256_of(out / "model.zip")
        model = load_e3_model(out / "model.zip")

    finite = check_finite_policy_outputs(model, make_e3_env(frame, EVAL_SPLIT, cfg, obs_cfg, role="diagnostic"), obs_cfg)
    validation = evaluate_e3(model, frame, cfg, obs_cfg, EVAL_SPLIT, name=f"ppo_e3_seed{config.seed}")
    meta = collect_metadata(config, model, frame, dataset_report, cfg, obs_cfg, scale_fit, train_seconds, preflight,
                            source_before=source_before, model_sha256=model_sha256)
    meta["finite_output_check"] = finite
    if write_artifacts:
        write_json(out / "metadata.json", meta)
        write_json(out / "validation_metrics.json", validation.metrics)
        validation.curve.to_csv(out / "validation_curve.csv")
    return E3RunResult(config, model, obs_cfg, scale_fit, meta, validation, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E3: PPO with the R2 risk-aware reward (O1; modelled costs: 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale), "
                                             "train-only, validation-only evaluation.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=FULL_BUDGET_TIMESTEPS)
    ap.add_argument("--smoke", action="store_true", help=f"{SMOKE_BUDGET_TIMESTEPS}-step integration run into artifacts/ppo/e3_smoke/")
    ap.add_argument("--skip-preflight", action="store_true", help="do not re-run the certification gate in-process")
    ap.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args(argv)

    config = E3Config(seed=args.seed, total_timesteps=SMOKE_BUDGET_TIMESTEPS if args.smoke else args.timesteps,
                      artifacts_root=args.artifacts_root, run_group=E3_SMOKE_RUN_GROUP if args.smoke else E3_RUN_GROUP,
                      verbose=args.verbose)
    preflight = None if args.skip_preflight else _preflight_or_die()
    res = run_e3(config, preflight=preflight)
    m = res.validation.metrics
    fit = res.metadata["observation"]["scale_fit"]
    print(f"\nE3 seed={config.seed} timesteps={res.metadata['total_timesteps_trained']} "
          f"train {res.metadata['train_seconds']:.0f}s -> {res.artifacts_dir}")
    print(f"scale {fit['scale']:.10f} fitted on TRAIN closes {fit['fit_start']}..{fit['fit_end']} "
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
