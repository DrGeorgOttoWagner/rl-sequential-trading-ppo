"""
ppo_e3_report.py — canonical E3 validation report and the E1-vs-E3 VALIDATION comparison (RQ3).

    python -m btc_rl.ppo_e3_report                          # canonical E3 report: seeds 42, 123, 2026 exactly
    python -m btc_rl.ppo_e3_report --non-canonical --seeds 42 123   # explicitly labelled subset report
    python -m btc_rl.ppo_e3_report --compare-e1             # canonical E3 report + E1-vs-E3 comparison

Canonical E3 report — same enforcement philosophy as the E1/E2 reports
----------------------------------------------------------------------
Produced only when every one of the following holds; otherwise
``ReportIntegrityError`` is raised and nothing is written.

1. Cohort: exactly seeds 42, 123, 2026 (no subset, extra or duplicate); random
   baseline seeds exactly 0-4.
2. Schema: complete E3 metadata (E1/E2 records and superseded schemas are
   rejected with a descriptive error: an E3 report never reads E1 or E2
   artifacts as E3 runs).
3. Frozen specification ``ppo_e3.CANONICAL_E3_SPEC`` (derived from the E1
   spec): 200,000 / 200,704 timesteps, every PPO hyperparameter, [64, 64] Tanh,
   CPU, O1 (lookback 10, TRAIN-only scale interval), R2 with the FROZEN
   coefficients (turnover 0.0005, drawdown 0.10), CASH/LONG, 0.10% fee + 0.05% slippage allowance per trade,
   dataset identity, split records, Python 3.12.14.  Smoke, tiny and
   alternative runs are NON-CANONICAL.
4. Checkpoint binding: actual seed, hyperparameters (target_kl None,
   clip_range_vf None, use_sde False), architecture, activation, device,
   timesteps and the 11-d O1 observation space are read from ``model.zip``.
5. Provenance: shared branch, HEAD, tracked-diff hash, source fingerprint
   (recomputed from the recorded file map), key hashes (E1 keys + R2/E3
   sources), Python / Torch / SB3, dataset SHA-256, scale fit.  The
   training-relevant core sources of the current tree (E1 core + ``reward_r2.py``)
   must still hash to the recorded values.
6. Scale: the O1 return scale is recomputed from the frozen dataset and must
   equal the recorded fit exactly (as in E1).
7. Trajectory: identical checks to E1 (validators reused) PLUS the R2
   decomposition columns of the saved curve within 1e-12, and the identity
   reward == financial_reward_r1 - turnover_penalty - drawdown_penalty.
8. Metrics: every saved metric (financial metrics AND reward diagnostics)
   equals the recomputed one.

E1-vs-E3 comparison (``--compare-e1``): VALIDATION only.  The E1 side is the
canonical E1 report path (``ppo_e1_report.build_report``, read-only), taken
only after the accepted E1 AND E2 files are verified byte-identical to the
frozen anchors (``ppo_e3.verify_accepted_artifacts_unchanged``) and again
afterwards.  The comparison lists within-seed differences E3(s) - E1(s) for
every canonical seed and the cohort mean differences, financial metrics only;
the reward diagnostics of E3 are reported separately and never compared with
a return.  With three seeds no significance is claimed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FoundationConfig, load_foundation_config
from .data import load_dataset
from .observations import observation_space
from .policies import AlwaysFlat, AlwaysLong, RandomPolicy
from .ppo_e1 import IDENTITY_KEYS, REQUIRED_METADATA_KEYS, resolve_observation_config, sha256_of, source_fingerprint
from .ppo_e1_report import (
    AGG_COLUMNS,
    COHORT_PROVENANCE,
    CURVE_ATOL,
    REPORT_COLUMNS,
    TRAINING_CORE_SOURCES as E1_TRAINING_CORE_SOURCES,
    ReportIntegrityError,
    _compare_curve,
    _compare_metrics,
    _get,
    _row,
    _schedule_value,
    aggregate,
)
from .ppo_e3 import (
    CANONICAL_E3_SPEC,
    DEFAULT_ARTIFACTS_ROOT,
    DEVELOPMENT_SEEDS,
    E3_RUN_GROUP,
    EVAL_SPLIT,
    EXPERIMENT,
    KEY_SOURCE_FILES_E3,
    PACKAGE_ROOT,
    R2_CURVE_COLUMNS,
    REWARD_DIAGNOSTIC_KEYS,
    evaluate_e3,
    load_e3_model,
    make_e3_env,
    resolve_e3_window,
    run_episode_e3,
    state_distribution,
    validate_metadata,
    verify_accepted_artifacts_unchanged,
    verify_accepted_core_sources_unchanged,
)

CANONICAL_SEEDS: tuple[int, ...] = DEVELOPMENT_SEEDS
CANONICAL_RANDOM_SEEDS: tuple[int, ...] = CANONICAL_E3_SPEC["random_baseline_seeds"]
TRAINING_CORE_SOURCES: tuple[str, ...] = E1_TRAINING_CORE_SOURCES + ("reward_r2_source",)
COMPARE_COLUMNS = ["total_return", "sharpe_ratio", "max_drawdown", "n_entries", "n_legs", "exposure",
                   "total_cost_fraction"]
DIAG_COLUMNS = ["cumulative_financial_r1", "cumulative_turnover_penalty", "cumulative_drawdown_penalty",
                "cumulative_reward_r2", "n_turnover_penalised_steps", "n_drawdown_penalised_steps"]
R2_CURVE_ATOL = CURVE_ATOL


# ───────────────────────────────────────────── schema
def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    p = Path(run_dir) / "metadata.json"
    if not p.exists():
        raise ReportIntegrityError(f"missing run metadata: {p}")
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportIntegrityError(f"{p}: metadata is not valid JSON ({exc})") from exc
    if not isinstance(meta, dict):
        raise ReportIntegrityError(f"{p}: metadata is not a JSON object")
    if meta.get("experiment") != EXPERIMENT:
        raise ReportIntegrityError(
            f"{p}: record declares experiment {meta.get('experiment')!r}, not {EXPERIMENT!r}; E1/E2 (or other) "
            "artifacts are never reported as E3 runs"
        )
    problems = validate_metadata(meta)
    if problems:
        missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
        hint = "superseded or unsupported metadata schema" if missing else "incomplete E3 metadata"
        raise ReportIntegrityError(f"{p}: {hint}: {'; '.join(problems)}")
    for path in COHORT_PROVENANCE:
        _get(meta, path)
    for k in ("model", "metadata", "validation_metrics", "validation_curve"):
        if k not in meta.get("artifacts", {}):
            raise ReportIntegrityError(f"{p}: artifacts record lacks {k!r}")
    return meta


# ───────────────────────────────────────────── canonical specification
def verify_canonical_spec(meta: dict[str, Any]) -> None:
    spec = CANONICAL_E3_SPEC
    problems: list[str] = []

    def check(label: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            problems.append(f"{label}: expected {expected!r}, recorded {actual!r}")

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
    rew = meta.get("reward", {})
    check("reward.definition", spec["reward"], rew.get("definition"))
    for k in ("formula", "turnover_penalty_lambda", "drawdown_penalty_lambda"):
        check(f"reward.{k}", spec["reward_r2"][k], rew.get(k))
    check("reward.coefficients", {"turnover_penalty_lambda": spec["reward_r2"]["turnover_penalty_lambda"],
                                  "drawdown_penalty_lambda": spec["reward_r2"]["drawdown_penalty_lambda"]},
          rew.get("coefficients"))
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
        problems.append(f"forbidden_splits: expected ['test'], recorded {meta.get('forbidden_splits')!r}")
    if not meta.get("artifacts", {}).get("model_sha256"):
        problems.append("artifacts.model_sha256: expected the saved checkpoint hash, recorded nothing")
    if problems:
        raise ReportIntegrityError(
            f"run seed {meta.get('seed')} does not declare the frozen canonical E3 specification "
            f"(smoke/tiny/alternative runs are NON-CANONICAL): " + "; ".join(problems)
        )


# ───────────────────────────────────────────── checkpoint binding
def verify_checkpoint_binding(model, meta: dict[str, Any], expected_obs_space, canonical: bool = True) -> dict[str, Any]:
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
        "observation_shape": tuple(model.observation_space.shape),
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
        problems.append("checkpoint observation space differs from the O1 space of the recorded scale")
    if model.action_space != spaces.Discrete(2):
        problems.append(f"checkpoint action space {model.action_space} is not Discrete(2)")
    if canonical:
        for k in ("target_kl", "clip_range_vf", "use_sde"):
            if actual[k] != CANONICAL_E3_SPEC[k]:
                problems.append(f"{k}: checkpoint {actual[k]!r} != canonical {CANONICAL_E3_SPEC[k]!r}")
        if actual["activation"] != "Tanh" or model.policy.activation_fn is not nn.Tanh:
            problems.append("checkpoint activation is not Tanh")
        if actual["observation_shape"] != (CANONICAL_E3_SPEC["observation"]["size"],):
            problems.append(f"checkpoint observation shape {actual['observation_shape']} is not the O1 shape")
    if problems:
        raise ReportIntegrityError(f"checkpoint/metadata mismatch for seed {meta.get('seed')}: " + "; ".join(problems))
    return actual


def verify_checkpoint_identity(model_path: Path, meta: dict[str, Any], canonical: bool = True) -> str:
    """
    The checkpoint BYTES must be the ones the record was written for
    (``artifacts.model_sha256``), and must not be an accepted E1 or E2
    checkpoint: E1 checkpoints share the 11-d O1 space, the seed and every
    hyperparameter with E3, so only the byte identity separates them.
    """
    from .ppo_e2 import E1_FROZEN_ARTIFACTS
    from .ppo_e3 import E2_FROZEN_ARTIFACTS

    digest = sha256_of(model_path)
    recorded = meta.get("artifacts", {}).get("model_sha256")
    if recorded is None:
        if canonical:
            raise ReportIntegrityError(f"{model_path}: record carries no checkpoint hash (artifacts.model_sha256); "
                                       "not a canonical E3 run")
    elif recorded != digest:
        raise ReportIntegrityError(f"{model_path}: checkpoint bytes ({digest[:16]}) are not the checkpoint the record "
                                   f"was written for ({str(recorded)[:16]}); E3 records bind to exactly one checkpoint")
    for label, anchors in (("E1", E1_FROZEN_ARTIFACTS), ("E2", E2_FROZEN_ARTIFACTS)):
        for seed, files in anchors.items():
            if digest == files["model.zip"]:
                raise ReportIntegrityError(f"{model_path}: checkpoint bytes are the accepted {label} seed {seed} "
                                           "checkpoint; E1/E2 checkpoints are never E3 runs")
    return digest


# ───────────────────────────────────────────── provenance / cohort
def verify_record_provenance(meta: dict[str, Any]) -> None:
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
    for label, rel in KEY_SOURCE_FILES_E3.items():
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
    recorded = meta["source"]["key_hashes"]
    changed: dict[str, dict[str, str]] = {}
    for label, rel in KEY_SOURCE_FILES_E3.items():
        p = Path(root) / rel
        now = sha256_of(p) if p.exists() else "<missing>"
        if now != recorded.get(label):
            changed[label] = {"recorded": recorded.get(label, "")[:16], "current": now[:16]}
    core_changed = sorted(k for k in changed if k in TRAINING_CORE_SOURCES)
    return {"changed_key_sources": changed, "training_core_changed": core_changed,
            "matches_recorded_fingerprint": not changed}


def verify_cohort(metas: list[dict[str, Any]], seeds: tuple[int, ...], canonical: bool) -> None:
    if len(set(seeds)) != len(seeds):
        raise ReportIntegrityError(f"duplicate seeds in cohort: {list(seeds)}")
    if canonical and set(seeds) != set(CANONICAL_SEEDS):
        raise ReportIntegrityError(
            f"canonical E3 report requires exactly seeds {list(CANONICAL_SEEDS)}, got {list(seeds)}; "
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
                raise ReportIntegrityError(f"runs {ref['seed']} and {meta['seed']} differ in {key!r}: not one experiment")
        for path in COHORT_PROVENANCE:
            if _get(meta, path) != _get(ref, path):
                raise ReportIntegrityError(
                    f"runs {ref['seed']} and {meta['seed']} differ in {'.'.join(path)}: not one source/runtime/dataset"
                )


# ───────────────────────────────────────────── saved artifacts
def _compare_r2_curve(saved: pd.DataFrame, fresh: pd.DataFrame, where: str) -> None:
    """The saved decomposition must equal the recomputed one and satisfy the R2 identity exactly (within 1e-12)."""
    missing = [c for c in R2_CURVE_COLUMNS if c not in saved.columns]
    if missing:
        raise ReportIntegrityError(f"{where}: saved curve lacks R2 decomposition columns {missing}")
    for col in R2_CURVE_COLUMNS:
        a = saved[col].to_numpy(dtype=np.float64)
        b = fresh[col].to_numpy(dtype=np.float64)
        if not (np.all(np.isfinite(a)) and np.allclose(a, b, rtol=0, atol=R2_CURVE_ATOL)):
            raise ReportIntegrityError(f"{where}: re-evaluated trajectory differs from the saved curve in {col!r}")
    r1 = saved["financial_reward_r1"].to_numpy(dtype=np.float64)
    tp = saved["turnover_penalty"].to_numpy(dtype=np.float64)
    dp = saved["drawdown_penalty"].to_numpy(dtype=np.float64)
    r2 = saved["reward_r2"].to_numpy(dtype=np.float64)
    if not np.allclose(r2, r1 - tp - dp, rtol=0, atol=R2_CURVE_ATOL):
        raise ReportIntegrityError(f"{where}: saved curve violates the R2 identity reward_r2 = r1 - turnover - drawdown")
    if not np.allclose(saved["reward"].to_numpy(dtype=np.float64), r2, rtol=0, atol=R2_CURVE_ATOL):
        raise ReportIntegrityError(f"{where}: saved 'reward' column is not the recorded R2")
    if np.any(tp < 0) or np.any(dp < 0):
        raise ReportIntegrityError(f"{where}: negative penalty in the saved curve")


def evaluate_saved_seeds(seeds: tuple[int, ...], frame: pd.DataFrame, cfg: FoundationConfig,
                         artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT, run_group: str = E3_RUN_GROUP,
                         split: Any = EVAL_SPLIT, canonical: bool = True
                         ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    resolve_e3_window(split, cfg, "evaluation")
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
        obs_cfg, fit = resolve_observation_config(frame, cfg, scaled)     # recomputed from the frozen dataset (TRAIN only)
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
        meta["_checkpoint_sha256"] = verify_checkpoint_identity(model_path, meta, canonical=canonical)
        model = load_e3_model(model_path)
        meta["_checkpoint"] = verify_checkpoint_binding(model, meta, observation_space(obs_cfg), canonical=canonical)
        res = evaluate_e3(model, frame, cfg, obs_cfg, split, name=f"ppo_e3_seed{seed}")
        curve_path = d / meta["artifacts"]["validation_curve"]
        if not curve_path.exists():
            raise ReportIntegrityError(f"{d}: saved validation curve missing")
        saved_curve = pd.read_csv(curve_path, index_col=0, parse_dates=True)
        _compare_curve(saved_curve, res.curve, str(d))
        _compare_r2_curve(saved_curve, res.curve, str(d))
        metrics_path = d / meta["artifacts"]["validation_metrics"]
        if not metrics_path.exists():
            raise ReportIntegrityError(f"{d}: saved validation metrics missing")
        _compare_metrics(json.loads(metrics_path.read_text(encoding="utf-8")), res.metrics, str(d))
        rows.append(_row_with_diagnostics(res.metrics, "ppo_e3", int(seed)))
    return pd.DataFrame(rows), metas


def _row_with_diagnostics(metrics: dict[str, Any], policy: str, seed: Any) -> dict[str, Any]:
    r = _row(metrics, policy, seed)
    for k in REWARD_DIAGNOSTIC_KEYS:
        r[k] = metrics.get(k)
    return r


def baseline_rows(frame: pd.DataFrame, cfg: FoundationConfig, random_seeds: tuple[int, ...],
                  split: Any = EVAL_SPLIT) -> pd.DataFrame:
    """Always CASH, buy-and-hold and random 0-4 through the R2 environment (identical accounting to E1's)."""
    obs_cfg, _ = resolve_observation_config(frame, cfg, True)
    env = make_e3_env(frame, split, cfg, obs_cfg, role="evaluation")
    rows = []
    for policy, label in ((AlwaysFlat(), "always_CASH"), (AlwaysLong(), "buy_and_hold_always_LONG")):
        res = run_episode_e3(env, policy, periods_per_year=cfg.periods_per_year)
        res.metrics.update(state_distribution(res))
        rows.append(_row_with_diagnostics(res.metrics, label, "-"))
    for s in random_seeds:
        res = run_episode_e3(env, RandomPolicy(s), periods_per_year=cfg.periods_per_year)
        res.metrics.update(state_distribution(res))
        rows.append(_row_with_diagnostics(res.metrics, "random", int(s)))
    return pd.DataFrame(rows)


def aggregate_diagnostics(df: pd.DataFrame, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {"policy": label, "seed": f"n={len(df)}"}
    for k in DIAG_COLUMNS:
        vals = df[k].astype(float).to_numpy()
        out[k] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=0):.4f}"
    return out


def build_report(seeds: tuple[int, ...] = CANONICAL_SEEDS, random_seeds: tuple[int, ...] = CANONICAL_RANDOM_SEEDS,
                 artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT, run_group: str = E3_RUN_GROUP,
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
        aggregate(ppo, "ppo_e3 mean ± std (population)"),
        aggregate(base[base["policy"] == "random"], "random mean ± std (population)"),
    ])
    diag_summary = pd.DataFrame([
        aggregate_diagnostics(ppo, "ppo_e3 mean ± std (population)"),
        aggregate_diagnostics(base[base["policy"] == "random"], "random mean ± std (population)"),
    ])
    return {"ppo": ppo, "baselines": base, "summary": summary, "diagnostics_summary": diag_summary,
            "metadata": metas, "split": EVAL_SPLIT, "canonical": canonical, "seeds": tuple(seeds),
            "source_status": current_source_status(metas[0])}


FMT = {"total_return": "{:+.4f}", "final_equity": "{:.4f}", "sharpe_ratio": "{:.3f}", "max_drawdown": "{:+.4f}",
       "exposure": "{:.3f}", "fraction_CASH": "{:.3f}", "total_cost_fraction": "{:.4f}",
       "cumulative_financial_r1": "{:+.4f}", "cumulative_turnover_penalty": "{:.4f}",
       "cumulative_drawdown_penalty": "{:.4f}", "cumulative_reward_r2": "{:+.4f}"}
DIAG_TABLE_COLUMNS = ["policy", "seed"] + DIAG_COLUMNS


def _table(df: pd.DataFrame, columns: list[str] | None = None, fmt: dict[str, str] | None = None) -> str:
    fmt = FMT if fmt is None else fmt
    d = df.copy()
    for k, f in fmt.items():
        if k in d:
            d[k] = d[k].map(lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v))
                            else (f.format(v) if isinstance(v, (int, float, np.floating)) else v))
    cols = [c for c in (columns or REPORT_COLUMNS) if c in d.columns]
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in d[cols].iterrows())


def format_markdown(report: dict[str, Any]) -> str:
    metas = report["metadata"]
    m0 = metas[0]
    fit = m0["observation"]["scale_fit"]
    rew = m0["reward"]
    label = "CANONICAL" if report["canonical"] else "NON-CANONICAL (explicit seed subset or configuration; not the E3 result)"
    status = report.get("source_status", {})
    if status.get("matches_recorded_fingerprint", True):
        source_note = "Current source tree matches the recorded key-source hashes."
    else:
        changed = ", ".join(f"{k} ({v['recorded']}→{v['current']})" for k, v in status["changed_key_sources"].items())
        source_note = ("Post-run source edits detected in non-core files (recorded hashes retained, not rewritten): "
                       f"{changed}. Training-relevant core sources are unchanged.")
    lines = [
        f"# E3 validation results — {label} (split = {report['split']})",
        "",
        f"Validation window {m0['validation_split']['window_start']} .. {m0['validation_split']['window_end']}, "
        f"usable decisions {m0['validation_split']['first_usable_decision']} .. "
        f"{m0['validation_split']['last_usable_decision']} (n={m0['validation_split']['n_usable_decisions']}). "
        f"Modelled costs {m0['costs']['bps_per_leg'] / 100:.10g}% of the traded value on each purchase or sale (trading fee plus slippage allowance). Observation {m0['observation']['definition']} with return "
        f"scale {fit['scale']:.10f} fitted on TRAIN closes {fit['fit_start']} .. {fit['fit_end']} "
        f"({fit['n_returns']} returns). Reward {rew['definition']}: `{rew['formula']}` with "
        f"TURNOVER_PENALTY_LAMBDA = {rew['turnover_penalty_lambda']} and DRAWDOWN_PENALTY_LAMBDA = "
        f"{rew['drawdown_penalty_lambda']} (frozen prospectively). Actions 0=CASH, 1=LONG. Deterministic (argmax) "
        f"evaluation of the final checkpoint. Control experiment: {m0.get('control_experiment', 'E1')}; E3 differs "
        "from it only in the training reward.",
        "",
        "Verification performed for every run: E3 metadata schema; " +
        ("frozen canonical E3 specification (derived from the E1 specification, reward block replaced, frozen R2 "
         "coefficients); " if report["canonical"] else "") +
        "checkpoint bound to metadata (seed, hyperparameters, architecture, timesteps, 11-d O1 space); provenance "
        "consistency (branch, HEAD, tracked diff, source fingerprint recomputed from the recorded file map, key hashes "
        "incl. R2/E3 sources, Python/Torch/SB3, dataset, scale fit); scale fit recomputed from the frozen dataset; "
        "complete saved trajectory (actions, positions, legs, fill/mark dates exact; equity, reward, drawdown, cost "
        "fraction, gross and position return AND every R2 decomposition column within 1e-12; R2 identity "
        "reward = r1 - turnover - drawdown); every saved metric and reward diagnostic equals the recomputed one.",
        "",
        source_note,
        "",
        "Financial metrics (return, equity, Sharpe, max drawdown, entries, legs, exposure, cost drag) are computed from "
        "the financial equity path under the unchanged E1 accounting. Reward diagnostics (cumulative R1 / penalties / "
        "R2) are TRAINING-REWARD sums and are never returns; cumulative R1 = log(final equity) by identity.",
        "",
        "Cost drag = `total_cost_fraction` = 1 - prod(1 - cost_fraction_t) = 1 - (1 - 0.0015)^legs.",
        "",
        "## PPO E3 per seed (financial metrics)",
        "",
        _table(report["ppo"]),
        "",
        "## Mean ± std across seeds",
        "",
        _table(report["summary"]),
        "",
        "## PPO E3 reward diagnostics per seed (training reward, NOT return)",
        "",
        _table(report["ppo"], DIAG_TABLE_COLUMNS),
        "",
        "## Reward diagnostics mean ± std across seeds",
        "",
        _table(report["diagnostics_summary"], DIAG_TABLE_COLUMNS),
        "",
        "## Baselines (same environment and accounting; R2 environment)",
        "",
        _table(report["baselines"]),
        "",
        "## Baseline reward diagnostics (what R2 would have paid these fixed policies)",
        "",
        _table(report["baselines"], DIAG_TABLE_COLUMNS),
        "",
        "ML Variant A replay: not available on validation (the committed v5B position series starts 2021-08-30); "
        "historical v5B metrics use different accounting and are not comparable.",
        "",
        "## Runs (shared identity verified: source fingerprint, git HEAD, dataset, hyperparameters, budget, "
        "observation/scaler, reward incl. coefficients, costs; checkpoint bound to metadata)",
        "",
        "| seed | timesteps (ckpt) | git HEAD | source fingerprint | dataset sha256 | checkpoint sha256 | timestamp |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in metas:
        ck = m.get("_checkpoint", {})
        lines.append(f"| {m['seed']} | {m['total_timesteps_trained']} ({ck.get('total_timesteps_trained', '-')}) | "
                     f"{m['git']['commit'][:12]}{'+dirty' if m['git']['dirty'] else ''} | "
                     f"{m['source']['source_fingerprint'][:16]} | {m['dataset']['sha256'][:12]} | "
                     f"{str(m.get('_checkpoint_sha256', '-'))[:16]} | {m['timestamp_utc']} |")
    return "\n".join(lines) + "\n"


# ───────────────────────────────────────────── E1 vs E3 (VALIDATION only)
def build_comparison(e3_report: dict[str, Any], frame: pd.DataFrame, cfg: FoundationConfig,
                     artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, Any]:
    """
    Canonical E1 (read-only; E1 AND E2 frozen anchors verified before and
    after) versus canonical E3 on VALIDATION: within-seed differences and
    cohort mean differences of the FINANCIAL metrics.
    """
    from . import ppo_e1_report as e1rep

    if not e3_report["canonical"]:
        raise ReportIntegrityError("the E1-vs-E3 comparison requires the CANONICAL E3 report")
    anchors = verify_accepted_artifacts_unchanged(artifacts_root)
    verify_accepted_core_sources_unchanged()
    e1_report = e1rep.build_report(artifacts_root=artifacts_root, frame=frame, cfg=cfg, canonical=True)
    after = verify_accepted_artifacts_unchanged(artifacts_root)
    for label in ("E1", "E2"):
        if after[label]["present"] != anchors[label]["present"]:
            raise ReportIntegrityError(f"{label} artifacts changed while the comparison was running")
    e1 = e1_report["ppo"].set_index("seed")
    e3 = e3_report["ppo"].set_index("seed")
    if list(e1.index) != list(CANONICAL_SEEDS) or list(e3.index) != list(CANONICAL_SEEDS):
        raise ReportIntegrityError("both cohorts must contain exactly the canonical seeds in canonical order")
    b1 = e1_report["baselines"]
    b3 = e3_report["baselines"]
    for col in COMPARE_COLUMNS + ["final_equity"]:
        a = b1[col].astype(float).to_numpy()
        b = b3[col].astype(float).to_numpy()
        if not np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True):
            raise ReportIntegrityError(f"baseline {col} differs between the E1 and E3 environments: accounting is not identical")
    within = []
    for s in CANONICAL_SEEDS:
        row: dict[str, Any] = {"seed": s}
        for col in COMPARE_COLUMNS:
            row[f"E1_{col}"] = float(e1.loc[s, col])
            row[f"E3_{col}"] = float(e3.loc[s, col])
            row[f"diff_{col}"] = float(e3.loc[s, col]) - float(e1.loc[s, col])
        within.append(row)
    within_df = pd.DataFrame(within)
    cohort = []
    for col in COMPARE_COLUMNS:
        a = e1[col].astype(float).to_numpy()
        b = e3[col].astype(float).to_numpy()
        d = b - a
        cohort.append({
            "metric": col,
            "E1_mean": float(np.mean(a)), "E1_std": float(np.std(a, ddof=0)),
            "E3_mean": float(np.mean(b)), "E3_std": float(np.std(b, ddof=0)),
            "mean_diff_E3_minus_E1": float(np.mean(d)),
            "within_seed_diff_min": float(np.min(d)), "within_seed_diff_max": float(np.max(d)),
            "seeds_where_E3_higher": int(np.sum(d > 0)),
        })
    diag = e3_report["ppo"][["seed"] + DIAG_COLUMNS].copy()
    return {"within_seed": within_df, "cohort": pd.DataFrame(cohort), "e3_diagnostics": diag,
            "e1_report": e1_report, "e3_report": e3_report, "anchor_status": anchors, "baselines_identical": True}


def format_comparison_markdown(comp: dict[str, Any]) -> str:
    e1m = comp["e1_report"]["metadata"][0]
    e3m = comp["e3_report"]["metadata"][0]
    fmt_w = {c: "{:+.4f}" for c in comp["within_seed"].columns if c != "seed"}
    fmt_c = {c: "{:+.4f}" for c in comp["cohort"].columns if c not in ("metric", "seeds_where_E3_higher")}
    n_files = sum(len(comp["anchor_status"][k]["present"]) for k in ("E1", "E2"))
    lines = [
        "# E1 (O1 + R1) vs E3 (O1 + R2 risk-aware reward) — VALIDATION ONLY (RQ3)",
        "",
        "Both cohorts: canonical seeds 42, 123, 2026; identical dataset, splits, timeline, O1 observation and "
        "TRAIN-only scale, CASH/LONG actions, modelled costs of 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale, PPO hyperparameters, [64, 64] Tanh MLP, CPU, one torch "
        "thread, 200,704 trained timesteps, deterministic full-TRAIN-window episodes, validation accounting and "
        "baselines.  The ONLY difference is the training reward: R1 (portfolio log return after costs) vs "
        f"R2 = R1 - {e3m['reward']['turnover_penalty_lambda']} * legs - {e3m['reward']['drawdown_penalty_lambda']} "
        "* max(0, D_(t+1) - D_t) (coefficients frozen prospectively, unchanged after VALIDATION).  All metrics below "
        "are FINANCIAL metrics of the realised equity path under the same accounting; no cumulative reward is "
        "compared with a return.  TEST performance of any learned policy was NOT computed.",
        "",
        f"E1 and E2 canonical artifacts verified byte-identical to the frozen anchors before and after reading "
        f"({n_files} files).  E1 source fingerprint {e1m['source']['source_fingerprint'][:16]}, E3 source fingerprint "
        f"{e3m['source']['source_fingerprint'][:16]}.  Baselines (always CASH, buy-and-hold, random 0-4) recomputed "
        "through the R1 and the R2 environment are identical to 1e-12: the accounting is the same code.",
        "",
        "Differences are E3 minus E1.  Three seeds: no statistical significance is claimed; population std across seeds.",
        "",
        "## Within-seed differences (same seed, same everything except the training reward)",
        "",
        _table(comp["within_seed"], list(comp["within_seed"].columns), fmt_w),
        "",
        "## Cohort mean ± std and mean difference",
        "",
        _table(comp["cohort"], list(comp["cohort"].columns), fmt_c),
        "",
        "## E3 reward diagnostics per seed (training reward sums; NOT returns; E1 was trained on R1 = cumulative_financial_r1 only)",
        "",
        _table(comp["e3_diagnostics"], list(comp["e3_diagnostics"].columns)),
        "",
        "## Validation baselines (identical under both environments)",
        "",
        _table(comp["e3_report"]["baselines"]),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS))
    ap.add_argument("--random-seeds", type=int, nargs="+", default=list(CANONICAL_RANDOM_SEEDS))
    ap.add_argument("--run-group", default=E3_RUN_GROUP)
    ap.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--non-canonical", action="store_true",
                    help="allow a seed subset or a non-frozen configuration; the output is labelled NON-CANONICAL "
                         "and written to a separate file")
    ap.add_argument("--compare-e1", action="store_true", help="also write the canonical E1-vs-E3 VALIDATION comparison")
    args = ap.parse_args(argv)
    canonical = not args.non_canonical
    cfg = load_foundation_config()
    frame, _ = load_dataset(cfg.dataset)
    report = build_report(tuple(args.seeds), tuple(args.random_seeds), args.artifacts_root, args.run_group,
                          frame=frame, cfg=cfg, canonical=canonical)
    md = format_markdown(report)
    out_dir = args.artifacts_root / args.run_group
    stem = "validation_results" if canonical else "validation_results_NONCANONICAL_" + "_".join(map(str, args.seeds))
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    pd.concat([report["ppo"], report["baselines"]], ignore_index=True).to_csv(out_dir / f"{stem}.csv", index=False)
    print(md)
    print(f"written: {out_dir / (stem + '.md')} and {stem}.csv")
    if args.compare_e1:
        comp = build_comparison(report, frame, cfg, args.artifacts_root)
        cmd = format_comparison_markdown(comp)
        (out_dir / "e1_vs_e3_validation_comparison.md").write_text(cmd, encoding="utf-8")
        comp["within_seed"].to_csv(out_dir / "e1_vs_e3_validation_within_seed.csv", index=False)
        comp["cohort"].to_csv(out_dir / "e1_vs_e3_validation_cohort.csv", index=False)
        comp["e3_diagnostics"].to_csv(out_dir / "e3_validation_reward_diagnostics.csv", index=False)
        print(cmd)
        print(f"written: {out_dir / 'e1_vs_e3_validation_comparison.md'} (+ within_seed / cohort / diagnostics csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
