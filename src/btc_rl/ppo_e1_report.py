"""
ppo_e1_report.py — validation result table for E1.

    python -m btc_rl.ppo_e1_report                          # canonical report: seeds 42, 123, 2026 exactly
    python -m btc_rl.ppo_e1_report --non-canonical --seeds 42 123   # explicitly labelled subset report

Canonical report (default) — hardening
---------------------------------------------------
A canonical report is produced only when every one of the following holds;
otherwise ``ReportIntegrityError`` is raised and nothing is written.

1. Cohort: exactly the development seeds 42, 123 and 2026 (no subset, no
   extra, no duplicate); random baseline seeds exactly 0-4.
2. Schema: every ``metadata.json`` has the complete current schema
   (superseded artifacts fail with a descriptive error, never a KeyError).
3. Frozen specification (``ppo_e1.CANONICAL_E1_SPEC``): each record must
   declare the frozen E1 configuration — requested 200,000 / trained 200,704
   timesteps, every PPO hyperparameter (gamma 0.99, lr 3e-4, n_steps 2048,
   batch 64, 10 epochs, GAE 0.95, clip 0.2, ent 0, vf 0.5, grad-norm 0.5,
   [64, 64] Tanh MLP), CPU, O1 (lookback 10), R1, CASH/LONG, 0.10% fee + 0.05% slippage allowance per trade,
   the TRAIN-only scaler interval (2018-03-04 .. 2020-12-29, 1032/1031),
   the frozen dataset identity and split records, Python 3.12.14.
   Smoke, tiny and alternative runs are therefore NON-CANONICAL.
4. Checkpoint binding: each ``model.zip`` is loaded and its actual seed,
   hyperparameters (including target_kl None, clip_range_vf None, use_sde
   False), architecture, activation, device, requested and trained
   timesteps and observation/action spaces are compared with the record and
   the specification.  Uniform false metadata cannot pass.
5. Provenance: the runs must share branch, git HEAD, tracked-diff hash,
   source fingerprint, source file map and key hashes, Python / Torch / SB3
   versions, dataset SHA-256 and scaler fit; the fingerprint is recomputed
   from the recorded file map and the key hashes are checked against it.
   The training-relevant core sources of the CURRENT tree must still hash
   to the recorded values; post-run edits of enforcement/report/test files
   are reported, not rejected, and recorded hashes are never rewritten.
6. Scaler: the fit is recomputed from the frozen dataset and must equal the
   recorded interval and value.
7. Trajectory: each checkpoint is re-evaluated deterministically on
   VALIDATION and the complete saved ``validation_curve.csv`` must match:
   actions, positions, legs, fill dates, mark dates exactly; equity, reward,
   drawdown, cost fraction, gross return and position return within 1e-12.
   Discrete columns (action, position, legs) are validated BEFORE any
   integer conversion: finite, exactly integer-valued and inside {0, 1};
   fractional values such as 0.25 or 1.25 are rejected, never truncated.
8. Metrics: the recomputed metrics are authoritative; every key of the saved
   ``validation_metrics.json`` must equal the recomputed value.

Cost drag (``total_cost_fraction``) is 1 - prod_t (1 - cost_fraction_t): the
proportional terminal-equity reduction caused by transaction costs for the
same action path, i.e. 1 - (1 - c)^legs.

The ML Variant A replay is NOT reported: the committed v5B position series
covers only the test window (2021-08-30 onward), so it cannot be re-accounted
on validation.  The test split is refused by every E1 entry point.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FoundationConfig, load_foundation_config
from .data import load_dataset
from .evaluation import run_episode
from .observations import observation_space
from .policies import AlwaysFlat, AlwaysLong, RandomPolicy
from .ppo_e1 import (
    CANONICAL_E1_SPEC,
    DEFAULT_ARTIFACTS_ROOT,
    DEVELOPMENT_SEEDS,
    EVAL_SPLIT,
    IDENTITY_KEYS,
    KEY_SOURCE_FILES,
    PACKAGE_ROOT,
    REQUIRED_METADATA_KEYS,
    evaluate_e1,
    load_e1_model,
    make_e1_env,
    resolve_e1_window,
    resolve_observation_config,
    sha256_of,
    source_fingerprint,
    state_distribution,
    validate_metadata,
)

CANONICAL_SEEDS: tuple[int, ...] = DEVELOPMENT_SEEDS
CANONICAL_RANDOM_SEEDS: tuple[int, ...] = CANONICAL_E1_SPEC["random_baseline_seeds"]
REPORT_COLUMNS = [
    "policy", "seed", "n_steps", "total_return", "final_equity", "sharpe_ratio", "max_drawdown",
    "n_entries", "n_legs", "exposure", "fraction_CASH", "total_cost_fraction", "final_position",
]
AGG_COLUMNS = ["total_return", "final_equity", "sharpe_ratio", "max_drawdown", "n_entries", "n_legs",
               "exposure", "fraction_CASH", "total_cost_fraction"]
CURVE_EXACT_COLUMNS = ("action", "position", "legs", "fill_date", "mark_date")
CURVE_TOLERANT_COLUMNS = ("equity", "reward", "drawdown", "cost_fraction", "gross_simple_return", "position_return")
CURVE_ATOL = 1e-12
# run-level provenance/runtime fields that every run of one cohort must share
COHORT_PROVENANCE: tuple[tuple[str, ...], ...] = (
    ("git", "branch"), ("git", "commit"), ("git", "tracked_diff_sha256"),
    ("source", "source_fingerprint"), ("source", "source_files"), ("source", "key_hashes"),
    ("python_version",), ("packages", "torch"), ("packages", "stable_baselines3"),
    ("dataset", "sha256"), ("observation", "scale_fit"),
)
# key sources whose CURRENT hashes must equal the recorded ones for a canonical report (training-relevant)
TRAINING_CORE_SOURCES: tuple[str, ...] = (
    "env_source", "splits_source", "observation_source", "scaler_source", "costs_source", "evaluation_source",
    "data_source", "foundation_config", "pyproject_toml", "uv_lock",
)


class ReportIntegrityError(RuntimeError):
    """The cohort is not the canonical E1 cohort, or a record, checkpoint or saved artifact fails verification."""


def _row(metrics: dict[str, Any], policy: str, seed: Any) -> dict[str, Any]:
    r = {k: metrics.get(k) for k in REPORT_COLUMNS if k in metrics}
    r["policy"] = policy
    r["seed"] = seed
    return r


def _get(meta: dict[str, Any], path: tuple[str, ...]) -> Any:
    v: Any = meta
    for k in path:
        if not isinstance(v, dict) or k not in v:
            raise ReportIntegrityError(f"metadata lacks {'.'.join(path)}")
        v = v[k]
    return v


# ───────────────────────────────────────────── schema
def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Load ``metadata.json`` and reject superseded or incomplete schemas with a descriptive error."""
    p = Path(run_dir) / "metadata.json"
    if not p.exists():
        raise ReportIntegrityError(f"missing run metadata: {p}")
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportIntegrityError(f"{p}: metadata is not valid JSON ({exc})") from exc
    if not isinstance(meta, dict):
        raise ReportIntegrityError(f"{p}: metadata is not a JSON object")
    problems = validate_metadata(meta)
    missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
    if problems:
        hint = ("superseded or unsupported metadata schema (earlier runs lack source provenance and "
                "the TRAIN-only scaler record)" if missing else "incomplete metadata")
        raise ReportIntegrityError(f"{p}: {hint}: {'; '.join(problems)}")
    for path in COHORT_PROVENANCE:
        _get(meta, path)
    for k in ("model", "metadata", "validation_metrics", "validation_curve"):
        if k not in meta.get("artifacts", {}):
            raise ReportIntegrityError(f"{p}: artifacts record lacks {k!r}")
    return meta


# ───────────────────────────────────────────── canonical specification
def _spec_mismatch(label: str, expected: Any, actual: Any) -> str:
    return f"{label}: expected {expected!r}, recorded {actual!r}"


def verify_canonical_spec(meta: dict[str, Any]) -> None:
    """The record must declare exactly the frozen E1 specification (``CANONICAL_E1_SPEC``)."""
    spec = CANONICAL_E1_SPEC
    problems: list[str] = []

    def check(label: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            problems.append(_spec_mismatch(label, expected, actual))

    check("experiment", spec["experiment"], meta.get("experiment"))
    check("total_timesteps_requested", spec["total_timesteps_requested"], meta.get("total_timesteps_requested"))
    check("total_timesteps_trained", spec["total_timesteps_trained"], meta.get("total_timesteps_trained"))
    hp = meta.get("ppo_hyperparameters", {})
    for k, v in spec["ppo_hyperparameters"].items():
        check(f"ppo_hyperparameters.{k}", v, hp.get(k))
    extra = set(hp) - set(spec["ppo_hyperparameters"])
    if extra:
        problems.append(f"ppo_hyperparameters has unexpected keys {sorted(extra)}")
    check("device", spec["device"], meta.get("device"))
    check("torch_threads", spec["torch_threads"], meta.get("torch_threads"))
    obs = meta.get("observation", {})
    for k, v in spec["observation"].items():
        check(f"observation.{k}", v, obs.get(k))
    fit = obs.get("scale_fit", {})
    for k, v in spec["scale_fit"].items():
        check(f"observation.scale_fit.{k}", v, fit.get(k))
    if obs.get("return_scale") != fit.get("scale"):
        problems.append("observation.return_scale differs from observation.scale_fit.scale")
    check("reward.definition", spec["reward"], meta.get("reward", {}).get("definition"))
    costs = meta.get("costs", {})
    for k, v in spec["costs"].items():
        check(f"costs.{k}", v, costs.get(k))
    check("action_semantics", spec["action_semantics"], meta.get("action_semantics"))
    ds = meta.get("dataset", {})
    for k, v in spec["dataset"].items():
        check(f"dataset.{k}", v, ds.get(k))
    for split in ("train_split", "validation_split"):
        rec = meta.get(split, {})
        for k, v in spec[split].items():
            check(f"{split}.{k}", v, rec.get(k))
    ep = meta.get("training_episodes", {})
    for k, v in spec["training_episodes"].items():
        check(f"training_episodes.{k}", v, ep.get(k))
    check("training_episodes.vec_envs", spec["vec_envs"], ep.get("vec_envs"))
    check("python_version", spec["python_version"], meta.get("python_version"))
    if meta.get("forbidden_splits") != ["test"]:
        problems.append(_spec_mismatch("forbidden_splits", ["test"], meta.get("forbidden_splits")))
    if problems:
        raise ReportIntegrityError(
            f"run seed {meta.get('seed')} does not declare the frozen canonical E1 specification "
            f"(smoke/tiny/alternative runs are NON-CANONICAL): " + "; ".join(problems)
        )


# ───────────────────────────────────────────── checkpoint binding
def _schedule_value(x: Any) -> Any:
    return x(1.0) if callable(x) else x


def verify_checkpoint_binding(model, meta: dict[str, Any], expected_obs_space, canonical: bool = True) -> dict[str, Any]:
    """
    The loaded checkpoint must agree with its metadata record (seed, every
    hyperparameter, architecture, activation, device, requested and trained
    timesteps, spaces) and, for canonical cohorts, with the frozen spec.
    Returns the actual values read from the model.
    """
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    import torch.nn as nn

    hp = meta["ppo_hyperparameters"]
    actual = {
        "algorithm": type(model).__name__,
        "policy_class": model.policy_class.__name__ if hasattr(model, "policy_class") else type(model.policy).__name__,
        "seed": model.seed,
        "learning_rate": _schedule_value(model.learning_rate),
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "clip_range": _schedule_value(model.clip_range),
        "clip_range_vf": _schedule_value(model.clip_range_vf),
        "ent_coef": model.ent_coef,
        "vf_coef": model.vf_coef,
        "max_grad_norm": model.max_grad_norm,
        "target_kl": model.target_kl,
        "normalize_advantage": model.normalize_advantage,
        "use_sde": model.use_sde,
        "net_arch": model.policy.net_arch,
        "activation": model.policy.activation_fn.__name__,
        "device": str(model.device),
        "n_envs": model.n_envs,
        "total_timesteps_requested": int(getattr(model, "_total_timesteps", -1)),
        "total_timesteps_trained": int(model.num_timesteps),
    }
    problems: list[str] = []

    def check(label: str, expected: Any, got: Any) -> None:
        if expected != got:
            problems.append(f"{label}: metadata {expected!r} != checkpoint {got!r}")

    if not isinstance(model, PPO):
        problems.append(f"checkpoint is {type(model).__name__}, not PPO")
    if not (model.policy_class is ActorCriticPolicy or isinstance(model.policy, ActorCriticPolicy)):
        problems.append("checkpoint policy is not the MlpPolicy / ActorCriticPolicy")
    check("seed", int(meta["seed"]), actual["seed"])
    for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef",
              "vf_coef", "max_grad_norm", "normalize_advantage"):
        check(k, hp[k], actual[k])
    check("net_arch", {"pi": list(hp["net_arch"]), "vf": list(hp["net_arch"])}, actual["net_arch"])
    check("activation", hp["activation"], actual["activation"])
    check("device", meta["device"], actual["device"])
    check("n_envs", meta["training_episodes"]["vec_envs"], actual["n_envs"])
    check("total_timesteps_requested", int(meta["total_timesteps_requested"]), actual["total_timesteps_requested"])
    check("total_timesteps_trained", int(meta["total_timesteps_trained"]), actual["total_timesteps_trained"])
    if model.observation_space != expected_obs_space:
        problems.append("checkpoint observation space differs from the O1 space of the recorded scaler")
    if model.action_space != spaces.Discrete(2):
        problems.append(f"checkpoint action space {model.action_space} is not Discrete(2)")
    if canonical:
        spec = CANONICAL_E1_SPEC
        for k in ("target_kl", "clip_range_vf", "use_sde"):
            if actual[k] != spec[k]:
                problems.append(f"{k}: checkpoint {actual[k]!r} != canonical {spec[k]!r}")
        if actual["activation"] != "Tanh" or model.policy.activation_fn is not nn.Tanh:
            problems.append("checkpoint activation is not Tanh")
    if problems:
        raise ReportIntegrityError(f"checkpoint/metadata mismatch for seed {meta.get('seed')}: " + "; ".join(problems))
    return actual


# ───────────────────────────────────────────── cohort / provenance
def verify_record_provenance(meta: dict[str, Any]) -> None:
    """Internal consistency of one record's provenance block."""
    src = meta["source"]
    files = src["source_files"]
    if not isinstance(files, dict) or not files:
        raise ReportIntegrityError(f"seed {meta['seed']}: source file map is empty")
    recomputed = source_fingerprint(files)
    if recomputed != src["source_fingerprint"]:
        raise ReportIntegrityError(
            f"seed {meta['seed']}: source fingerprint {src['source_fingerprint'][:16]} does not equal the "
            f"fingerprint recomputed from its recorded file map ({recomputed[:16]})"
        )
    for label, rel in KEY_SOURCE_FILES.items():
        if rel not in files:
            raise ReportIntegrityError(f"seed {meta['seed']}: recorded file map lacks key source {rel}")
        if src["key_hashes"].get(label) != files[rel]:
            raise ReportIntegrityError(f"seed {meta['seed']}: key hash {label} disagrees with the recorded file map")
    if src.get("key_hashes", {}).get("uv_lock") != meta["uv_lock_sha256"]:
        raise ReportIntegrityError(f"seed {meta['seed']}: uv_lock_sha256 disagrees with the recorded key hash")
    if src.get("key_hashes", {}).get("pyproject_toml") != meta.get("pyproject_sha256"):
        raise ReportIntegrityError(f"seed {meta['seed']}: pyproject_sha256 disagrees with the recorded key hash")
    if src.get("key_hashes", {}).get("foundation_config") != meta.get("foundation_config_sha256"):
        raise ReportIntegrityError(f"seed {meta['seed']}: foundation_config_sha256 disagrees with the recorded key hash")
    if "source_stable_during_run" in src and src["source_stable_during_run"] is not True:
        raise ReportIntegrityError(f"seed {meta['seed']}: source changed during the run")


def current_source_status(meta: dict[str, Any], root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Compare the recorded key hashes with the current tree (informational, plus the core-source requirement)."""
    recorded = meta["source"]["key_hashes"]
    changed: dict[str, dict[str, str]] = {}
    for label, rel in KEY_SOURCE_FILES.items():
        p = Path(root) / rel
        now = sha256_of(p) if p.exists() else "<missing>"
        if now != recorded.get(label):
            changed[label] = {"recorded": recorded.get(label, "")[:16], "current": now[:16]}
    core_changed = sorted(k for k in changed if k in TRAINING_CORE_SOURCES)
    return {"changed_key_sources": changed, "training_core_changed": core_changed,
            "matches_recorded_fingerprint": not changed}


def verify_cohort(metas: list[dict[str, Any]], seeds: tuple[int, ...], canonical: bool) -> None:
    """Seed set, canonical specification, cross-run identity and provenance consistency."""
    if len(set(seeds)) != len(seeds):
        raise ReportIntegrityError(f"duplicate seeds in cohort: {list(seeds)}")
    if canonical and set(seeds) != set(CANONICAL_SEEDS):
        raise ReportIntegrityError(
            f"canonical E1 report requires exactly seeds {list(CANONICAL_SEEDS)}, got {list(seeds)}; "
            "use --non-canonical for an explicitly labelled subset"
        )
    if len(metas) != len(seeds):
        raise ReportIntegrityError("one metadata record per seed is required")
    for meta, seed in zip(metas, seeds):
        if int(meta["seed"]) != int(seed):
            raise ReportIntegrityError(f"metadata seed {meta['seed']} != requested seed {seed}")
        verify_record_provenance(meta)
        if canonical:
            verify_canonical_spec(meta)
    ref = metas[0]
    for meta in metas[1:]:
        for key in IDENTITY_KEYS:
            if meta.get(key) != ref.get(key):
                raise ReportIntegrityError(
                    f"runs {ref['seed']} and {meta['seed']} differ in {key!r}: not one experiment"
                )
        for path in COHORT_PROVENANCE:
            if _get(meta, path) != _get(ref, path):
                raise ReportIntegrityError(
                    f"runs {ref['seed']} and {meta['seed']} differ in {'.'.join(path)}: not one source/runtime/dataset"
                )


# ───────────────────────────────────────────── saved artifacts
DISCRETE_DOMAIN: dict[str, frozenset[int]] = {"action": frozenset({0, 1}), "position": frozenset({0, 1}),
                                              "legs": frozenset({0, 1})}


def validated_discrete(values: Any, column: str, where: str) -> np.ndarray:
    """
    Validate a discrete trajectory column BEFORE any integer conversion
    every value must be a finite number, exactly
    integer-valued and inside the legal domain (action/position/legs: {0, 1}).
    Fractional values such as 0.25 or 1.25 are rejected instead of being
    truncated.  Returns the validated values as int64.
    """
    domain = DISCRETE_DOMAIN[column]
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReportIntegrityError(f"{where}: column {column!r} contains non-numeric values") from exc
    if arr.ndim != 1:
        raise ReportIntegrityError(f"{where}: column {column!r} is not one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ReportIntegrityError(f"{where}: column {column!r} contains NaN/Inf or non-finite values")
    if not np.array_equal(arr, np.floor(arr)):
        bad = arr[arr != np.floor(arr)][:3].tolist()
        raise ReportIntegrityError(f"{where}: column {column!r} contains non-integer values {bad}")
    if np.any(np.abs(arr) > 2.0 ** 53):  # beyond exact float64 integer range: never cast, reject
        raise ReportIntegrityError(f"{where}: column {column!r} contains values outside the int64 range")
    ints = arr.astype(np.int64)
    illegal = sorted(set(ints.tolist()) - domain)
    if illegal:
        raise ReportIntegrityError(f"{where}: column {column!r} contains values {illegal} outside the legal domain {sorted(domain)}")
    return ints


def _compare_curve(saved: pd.DataFrame, fresh: pd.DataFrame, where: str) -> None:
    if len(saved) != len(fresh) or not saved.index.equals(fresh.index):
        raise ReportIntegrityError(f"{where}: re-evaluated trajectory length/dates differ from the saved curve")
    missing = [c for c in CURVE_EXACT_COLUMNS + CURVE_TOLERANT_COLUMNS if c not in saved.columns]
    if missing:
        raise ReportIntegrityError(f"{where}: saved curve lacks columns {missing}")
    for col in CURVE_EXACT_COLUMNS:
        if col in ("fill_date", "mark_date"):
            a = pd.to_datetime(saved[col]).to_numpy()
            b = pd.to_datetime(fresh[col]).to_numpy()
            ok = np.array_equal(a, b)
        else:
            # validate both sides (finite, integer-valued, legal domain) before the exact comparison
            a = validated_discrete(saved[col].to_numpy(), col, f"{where} (saved curve)")
            b = validated_discrete(fresh[col].to_numpy(), col, f"{where} (re-evaluated curve)")
            ok = np.array_equal(a, b)
        if not ok:
            raise ReportIntegrityError(f"{where}: re-evaluated trajectory differs from the saved curve in {col!r}")
    for col in CURVE_TOLERANT_COLUMNS:
        a = saved[col].to_numpy(dtype=np.float64)
        b = fresh[col].to_numpy(dtype=np.float64)
        if not (np.all(np.isfinite(a)) and np.allclose(a, b, rtol=0, atol=CURVE_ATOL)):
            raise ReportIntegrityError(f"{where}: re-evaluated trajectory differs from the saved curve in {col!r}")


def _metric_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=CURVE_ATOL)
    try:
        return pd.Timestamp(a) == pd.Timestamp(b)
    except (TypeError, ValueError):
        return a == b


def _compare_metrics(saved: dict[str, Any], fresh: dict[str, Any], where: str) -> None:
    """The recomputed metrics are authoritative; the saved set must equal them key for key."""
    if set(saved) != set(fresh):
        raise ReportIntegrityError(f"{where}: saved metric keys {sorted(set(saved) ^ set(fresh))} differ from the recomputed set")
    bad = [k for k in fresh if not _metric_equal(saved[k], fresh[k])]
    if bad:
        raise ReportIntegrityError(f"{where}: saved validation metrics differ from the recomputed metrics in {bad}")


def evaluate_saved_seeds(seeds: tuple[int, ...], frame: pd.DataFrame, cfg: FoundationConfig,
                         artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT, run_group: str = "e1",
                         split: Any = EVAL_SPLIT, canonical: bool = True
                         ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Verify the cohort, bind every checkpoint to its record, re-evaluate on VALIDATION, cross-check saved artifacts."""
    resolve_e1_window(split, cfg, "evaluation")
    metas = [load_run_metadata(Path(artifacts_root) / run_group / str(s)) for s in seeds]
    verify_cohort(metas, tuple(seeds), canonical)
    if canonical:
        status = current_source_status(metas[0])
        if status["training_core_changed"]:
            raise ReportIntegrityError(
                "training-relevant core sources of the current tree differ from the recorded hashes: "
                f"{status['training_core_changed']}; the canonical cohort is not reproducible from this tree"
            )
    rows = []
    for seed, meta in zip(seeds, metas):
        d = Path(artifacts_root) / run_group / str(seed)
        scaled = meta["observation"]["scale_fit"]["statistic"] != "not scaled"
        obs_cfg, fit = resolve_observation_config(frame, cfg, scaled)
        rec = meta["observation"]["scale_fit"]
        if scaled:
            if fit is None or fit.scale != rec["scale"] or str(fit.fit_start.date()) != rec["fit_start"] \
                    or str(fit.fit_end.date()) != rec["fit_end"] or fit.n_closes != rec["n_closes"] \
                    or fit.n_returns != rec["n_returns"]:
                raise ReportIntegrityError(f"{d}: recomputed scaler fit differs from the run metadata")
        if obs_cfg.return_scale != meta["observation"]["return_scale"] or obs_cfg.lookback != meta["observation"]["lookback"]:
            raise ReportIntegrityError(f"{d}: observation config differs from metadata")
        model_path = d / meta["artifacts"]["model"]
        if not model_path.exists():
            raise ReportIntegrityError(f"{d}: checkpoint {model_path.name} missing")
        model = load_e1_model(model_path)
        meta["_checkpoint"] = verify_checkpoint_binding(model, meta, observation_space(obs_cfg), canonical=canonical)
        res = evaluate_e1(model, frame, cfg, obs_cfg, split, name=f"ppo_e1_seed{seed}")
        curve_path = d / meta["artifacts"]["validation_curve"]
        if not curve_path.exists():
            raise ReportIntegrityError(f"{d}: saved validation curve missing")
        saved_curve = pd.read_csv(curve_path, index_col=0, parse_dates=True)
        _compare_curve(saved_curve, res.curve, str(d))
        metrics_path = d / meta["artifacts"]["validation_metrics"]
        if not metrics_path.exists():
            raise ReportIntegrityError(f"{d}: saved validation metrics missing")
        _compare_metrics(json.loads(metrics_path.read_text(encoding="utf-8")), res.metrics, str(d))
        rows.append(_row(res.metrics, "ppo_e1", int(seed)))
    return pd.DataFrame(rows), metas


def baseline_rows(frame: pd.DataFrame, cfg: FoundationConfig, random_seeds: tuple[int, ...],
                  split: Any = EVAL_SPLIT, scale_on_train: bool = True) -> pd.DataFrame:
    obs_cfg, _ = resolve_observation_config(frame, cfg, scale_on_train)
    env = make_e1_env(frame, split, cfg, obs_cfg, role="evaluation")
    rows = []
    for policy, label in ((AlwaysFlat(), "always_CASH"), (AlwaysLong(), "buy_and_hold_always_LONG")):
        res = run_episode(env, policy, periods_per_year=cfg.periods_per_year)
        res.metrics.update(state_distribution(res))
        rows.append(_row(res.metrics, label, "-"))
    for s in random_seeds:
        res = run_episode(env, RandomPolicy(s), periods_per_year=cfg.periods_per_year)
        res.metrics.update(state_distribution(res))
        rows.append(_row(res.metrics, "random", int(s)))
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {"policy": label, "seed": f"n={len(df)}", "n_steps": int(df["n_steps"].iloc[0])}
    for k in AGG_COLUMNS:
        vals = df[k].astype(float).to_numpy()
        out[k] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=0):.4f}"
    out["final_position"] = "-"
    return out


def build_report(seeds: tuple[int, ...] = CANONICAL_SEEDS, random_seeds: tuple[int, ...] = CANONICAL_RANDOM_SEEDS,
                 artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT, run_group: str = "e1",
                 frame: pd.DataFrame | None = None, cfg: FoundationConfig | None = None,
                 canonical: bool = True) -> dict[str, Any]:
    if canonical and tuple(random_seeds) != tuple(CANONICAL_RANDOM_SEEDS):
        raise ReportIntegrityError(
            f"canonical report uses random baseline seeds {list(CANONICAL_RANDOM_SEEDS)}; "
            f"got {list(random_seeds)} (use --non-canonical)"
        )
    cfg = cfg or load_foundation_config()
    if frame is None:
        frame, _ = load_dataset(cfg.dataset)
    ppo, metas = evaluate_saved_seeds(tuple(seeds), frame, cfg, artifacts_root, run_group, canonical=canonical)
    base = baseline_rows(frame, cfg, tuple(random_seeds))
    summary = pd.DataFrame([
        aggregate(ppo, "ppo_e1 mean ± std (population)"),
        aggregate(base[base["policy"] == "random"], "random mean ± std (population)"),
    ])
    return {"ppo": ppo, "baselines": base, "summary": summary, "metadata": metas, "split": EVAL_SPLIT,
            "canonical": canonical, "seeds": tuple(seeds), "source_status": current_source_status(metas[0])}


def format_markdown(report: dict[str, Any]) -> str:
    fmt = {"total_return": "{:+.4f}", "final_equity": "{:.4f}", "sharpe_ratio": "{:.3f}", "max_drawdown": "{:+.4f}",
           "exposure": "{:.3f}", "fraction_CASH": "{:.3f}", "total_cost_fraction": "{:.4f}"}

    def table(df: pd.DataFrame) -> str:
        d = df.copy()
        for k, f in fmt.items():
            if k in d:
                d[k] = d[k].map(lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v))
                                else (f.format(v) if isinstance(v, (int, float, np.floating)) else v))
        cols = [c for c in REPORT_COLUMNS if c in d.columns]
        head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
        return head + "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in d[cols].iterrows())

    metas = report["metadata"]
    m0 = metas[0]
    fit = m0["observation"]["scale_fit"]
    label = "CANONICAL" if report["canonical"] else "NON-CANONICAL (explicit seed subset or configuration; not the E1 result)"
    status = report.get("source_status", {})
    if status.get("matches_recorded_fingerprint", True):
        source_note = "Current source tree matches the recorded key-source hashes."
    else:
        changed = ", ".join(f"{k} ({v['recorded']}→{v['current']})" for k, v in status["changed_key_sources"].items())
        source_note = ("Post-run source edits detected in non-core files (recorded hashes retained, not rewritten): "
                       f"{changed}. Training-relevant core sources are unchanged.")
    lines = [
        f"# E1 validation results — {label} (split = {report['split']})",
        "",
        f"Validation window {m0['validation_split']['window_start']} .. {m0['validation_split']['window_end']}, "
        f"usable decisions {m0['validation_split']['first_usable_decision']} .. "
        f"{m0['validation_split']['last_usable_decision']} (n={m0['validation_split']['n_usable_decisions']}). "
        f"Modelled costs {m0['costs']['bps_per_leg'] / 100:.10g}% of the traded value on each purchase or sale (trading fee plus slippage allowance). Observation {m0['observation']['definition']} with return "
        f"scale {fit['scale']:.10f} fitted on TRAIN closes {fit['fit_start']} .. {fit['fit_end']} "
        f"({fit['n_returns']} returns). Reward {m0['reward']['definition']}. Actions 0=CASH, 1=LONG. "
        "Deterministic (argmax) evaluation of the final checkpoint.",
        "",
        "Verification performed for every run: metadata schema; " +
        ("frozen canonical E1 specification; " if report["canonical"] else "") +
        "checkpoint bound to metadata (seed, hyperparameters, architecture, timesteps, spaces); provenance "
        "consistency (branch, HEAD, tracked diff, source fingerprint recomputed from the recorded file map, key "
        "hashes, Python/Torch/SB3, dataset, scaler); scaler fit recomputed from the frozen dataset; complete saved "
        "trajectory (actions, positions, legs, fill/mark dates exact; equity, reward, drawdown, cost fraction, gross "
        "and position return within 1e-12); every saved metric equals the recomputed (authoritative) metric.",
        "",
        source_note,
        "",
        "Cost drag = `total_cost_fraction` = 1 - prod(1 - cost_fraction_t) = 1 - (1 - 0.0015)^legs: the "
        "proportional terminal-equity reduction from transaction costs for the same action path.",
        "",
        "## PPO E1 per seed",
        "",
        table(report["ppo"]),
        "",
        "## Mean ± std across seeds",
        "",
        table(report["summary"]),
        "",
        "## Baselines (same environment and accounting)",
        "",
        table(report["baselines"]),
        "",
        "ML Variant A replay: not available on validation (the committed v5B position series starts 2021-08-30); "
        "historical v5B metrics use different accounting and are not comparable.",
        "",
        "## Runs (shared identity verified: source fingerprint, git HEAD, dataset, hyperparameters, budget, "
        "observation/scaler, reward, costs; checkpoint bound to metadata)",
        "",
        "| seed | timesteps (ckpt) | git HEAD | source fingerprint | dataset sha256 | timestamp |",
        "|---|---|---|---|---|---|",
    ]
    for m in metas:
        ck = m.get("_checkpoint", {})
        lines.append(f"| {m['seed']} | {m['total_timesteps_trained']} ({ck.get('total_timesteps_trained', '-')}) | "
                     f"{m['git']['commit'][:12]}{'+dirty' if m['git']['dirty'] else ''} | "
                     f"{m['source']['source_fingerprint'][:16]} | {m['dataset']['sha256'][:12]} | {m['timestamp_utc']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS))
    ap.add_argument("--random-seeds", type=int, nargs="+", default=list(CANONICAL_RANDOM_SEEDS))
    ap.add_argument("--run-group", default="e1")
    ap.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--non-canonical", action="store_true",
                    help="allow a seed subset or a non-frozen configuration; the output is labelled NON-CANONICAL "
                         "and written to a separate file")
    args = ap.parse_args(argv)
    canonical = not args.non_canonical
    report = build_report(tuple(args.seeds), tuple(args.random_seeds), args.artifacts_root, args.run_group,
                          canonical=canonical)
    md = format_markdown(report)
    out_dir = args.artifacts_root / args.run_group
    stem = "validation_results" if canonical else "validation_results_NONCANONICAL_" + "_".join(map(str, args.seeds))
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    pd.concat([report["ppo"], report["baselines"]], ignore_index=True).to_csv(out_dir / f"{stem}.csv", index=False)
    print(md)
    print(f"written: {out_dir / (stem + '.md')} and {stem}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
