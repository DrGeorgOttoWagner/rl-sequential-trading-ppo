"""
ppo_e4_report.py — canonical E4 validation report, E2-vs-E4 and E3-vs-E4 comparisons, E1–E4 matrix.

    python -m btc_rl.ppo_e4_report                          # canonical E4 report: seeds 42, 123, 2026 exactly
    python -m btc_rl.ppo_e4_report --non-canonical --seeds 42 123
    python -m btc_rl.ppo_e4_report --compare                # + E2-vs-E4, E3-vs-E4 and the full E1/E2/E3/E4 matrix

Canonical E4 report: the E2 enforcement (O2 schema, scaler recomputed from the
frozen dataset with every mean/std exact, 9-d space) AND the E3 enforcement
(frozen R2 coefficients in the spec, decomposition columns within 1e-12, the
R2 identity, reward diagnostics, checkpoint byte binding against the record
and against every accepted E1/E2/E3 checkpoint hash).  Cohort: exactly seeds
42/123/2026, random seeds 0-4; provenance, key hashes (E1 keys + O2 + R2 + E4
sources), Python/Torch/SB3, dataset; training-core sources of the current tree
(E1 core + O2 + R2) unchanged.

Comparisons (``--compare``): VALIDATION only.  E2, E3 and E1 sides are the
canonical read-only report paths of their own modules, taken only after the
accepted E1/E2/E3 files are verified byte-identical to the anchors (and again
afterwards).  Financial metrics only; E4 reward diagnostics are listed
separately.  Three seeds: no significance is claimed.
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
from .observations_o2 import o2_observation_space
from .policies import AlwaysFlat, AlwaysLong, RandomPolicy
from .ppo_e1 import IDENTITY_KEYS, REQUIRED_METADATA_KEYS, sha256_of, source_fingerprint
from .ppo_e1_report import (
    ReportIntegrityError,
    _compare_curve,
    _compare_metrics,
    _get,
    _schedule_value,
    aggregate,
)
from .ppo_e2_report import COHORT_PROVENANCE, _scaler_matches
from .ppo_e2_report import TRAINING_CORE_SOURCES as E2_TRAINING_CORE_SOURCES
from .ppo_e3_report import (
    COMPARE_COLUMNS,
    DIAG_COLUMNS,
    DIAG_TABLE_COLUMNS,
    _compare_r2_curve,
    _row_with_diagnostics,
    _table,
    aggregate_diagnostics,
)
from .ppo_e4 import (
    ACCEPTED_ANCHORS,
    CANONICAL_E4_SPEC,
    DEFAULT_ARTIFACTS_ROOT,
    DEVELOPMENT_SEEDS,
    E4_RUN_GROUP,
    EVAL_SPLIT,
    EXPERIMENT,
    KEY_SOURCE_FILES_E4,
    PACKAGE_ROOT,
    evaluate_e4,
    load_e4_model,
    make_e4_env,
    resolve_e4_window,
    resolve_o2_config,
    run_episode_e4,
    state_distribution,
    validate_metadata,
    verify_accepted_artifacts_unchanged,
    verify_accepted_core_sources_unchanged,
)

CANONICAL_SEEDS: tuple[int, ...] = DEVELOPMENT_SEEDS
CANONICAL_RANDOM_SEEDS: tuple[int, ...] = CANONICAL_E4_SPEC["random_baseline_seeds"]
TRAINING_CORE_SOURCES: tuple[str, ...] = E2_TRAINING_CORE_SOURCES + ("reward_r2_source",)
MATRIX = (("E1", "O1", "R1"), ("E2", "O2", "R1"), ("E3", "O1", "R2"), ("E4", "O2", "R2"))


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
            f"{p}: record declares experiment {meta.get('experiment')!r}, not {EXPERIMENT!r}; E1/E2/E3 (or other) "
            "artifacts are never reported as E4 runs"
        )
    problems = validate_metadata(meta)
    if problems:
        missing = [k for k in REQUIRED_METADATA_KEYS if k not in meta]
        hint = "superseded or unsupported metadata schema" if missing else "incomplete E4 metadata"
        raise ReportIntegrityError(f"{p}: {hint}: {'; '.join(problems)}")
    for path in COHORT_PROVENANCE:
        _get(meta, path)
    for k in ("model", "metadata", "validation_metrics", "validation_curve"):
        if k not in meta.get("artifacts", {}):
            raise ReportIntegrityError(f"{p}: artifacts record lacks {k!r}")
    return meta


# ───────────────────────────────────────────── canonical specification (E2 observation checks + E3 reward checks)
def verify_canonical_spec(meta: dict[str, Any]) -> None:
    spec = CANONICAL_E4_SPEC
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
    fit = obs.get("scaler_fit", {})
    for k, v in spec["scaler_fit"].items():
        check(f"observation.scaler_fit.{k}", v, fit.get(k))
    if "scale_fit" in obs:
        problems.append("observation carries an O1 scale_fit block")
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
            f"run seed {meta.get('seed')} does not declare the frozen canonical E4 specification "
            f"(smoke/tiny/alternative runs are NON-CANONICAL): " + "; ".join(problems)
        )


# ───────────────────────────────────────────── checkpoint binding (configuration, 9-d O2 space, bytes)
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
        problems.append("checkpoint observation space differs from the O2 space of the recorded scaler")
    if model.action_space != spaces.Discrete(2):
        problems.append(f"checkpoint action space {model.action_space} is not Discrete(2)")
    if canonical:
        for k in ("target_kl", "clip_range_vf", "use_sde"):
            if actual[k] != CANONICAL_E4_SPEC[k]:
                problems.append(f"{k}: checkpoint {actual[k]!r} != canonical {CANONICAL_E4_SPEC[k]!r}")
        if actual["activation"] != "Tanh" or model.policy.activation_fn is not nn.Tanh:
            problems.append("checkpoint activation is not Tanh")
        if actual["observation_shape"] != (CANONICAL_E4_SPEC["observation"]["size"],):
            problems.append(f"checkpoint observation shape {actual['observation_shape']} is not the O2 shape")
    if problems:
        raise ReportIntegrityError(f"checkpoint/metadata mismatch for seed {meta.get('seed')}: " + "; ".join(problems))
    return actual


def verify_checkpoint_identity(model_path: Path, meta: dict[str, Any], canonical: bool = True) -> str:
    """Checkpoint bytes must be the record's ``model_sha256`` and must not be any accepted E1/E2/E3 checkpoint."""
    digest = sha256_of(model_path)
    recorded = meta.get("artifacts", {}).get("model_sha256")
    if recorded is None:
        if canonical:
            raise ReportIntegrityError(f"{model_path}: record carries no checkpoint hash (artifacts.model_sha256); "
                                       "not a canonical E4 run")
    elif recorded != digest:
        raise ReportIntegrityError(f"{model_path}: checkpoint bytes ({digest[:16]}) are not the checkpoint the record "
                                   f"was written for ({str(recorded)[:16]}); E4 records bind to exactly one checkpoint")
    for label, anchors, _ in ACCEPTED_ANCHORS:
        for seed, files in anchors.items():
            if digest == files["model.zip"]:
                raise ReportIntegrityError(f"{model_path}: checkpoint bytes are the accepted {label} seed {seed} "
                                           "checkpoint; E1/E2/E3 checkpoints are never E4 runs")
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
    for label, rel in KEY_SOURCE_FILES_E4.items():
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
    for label, rel in KEY_SOURCE_FILES_E4.items():
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
            f"canonical E4 report requires exactly seeds {list(CANONICAL_SEEDS)}, got {list(seeds)}; "
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
def evaluate_saved_seeds(seeds: tuple[int, ...], frame: pd.DataFrame, cfg: FoundationConfig,
                         artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT, run_group: str = E4_RUN_GROUP,
                         split: Any = EVAL_SPLIT, canonical: bool = True
                         ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    resolve_e4_window(split, cfg, "evaluation")
    metas = [load_run_metadata(Path(artifacts_root) / run_group / str(s)) for s in seeds]
    verify_cohort(metas, tuple(seeds), canonical)
    if canonical:
        status = current_source_status(metas[0])
        if status["training_core_changed"]:
            raise ReportIntegrityError(
                "training-relevant core sources of the current tree differ from the recorded hashes: "
                f"{status['training_core_changed']}; the canonical cohort is not reproducible from this tree"
            )
    obs_cfg, fit = resolve_o2_config(frame, cfg)     # recomputed from the frozen dataset (TRAIN only)
    rows = []
    for seed, meta in zip(seeds, metas):
        d = Path(artifacts_root) / run_group / str(seed)
        if not _scaler_matches(fit, meta["observation"]["scaler_fit"]):
            raise ReportIntegrityError(f"{d}: recomputed O2 scaler fit differs from the run metadata")
        model_path = d / meta["artifacts"]["model"]
        if not model_path.exists():
            raise ReportIntegrityError(f"{d}: checkpoint {model_path.name} missing")
        meta["_checkpoint_sha256"] = verify_checkpoint_identity(model_path, meta, canonical=canonical)
        model = load_e4_model(model_path)
        meta["_checkpoint"] = verify_checkpoint_binding(model, meta, o2_observation_space(obs_cfg), canonical=canonical)
        res = evaluate_e4(model, frame, cfg, obs_cfg, split, name=f"ppo_e4_seed{seed}")
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
        rows.append(_row_with_diagnostics(res.metrics, "ppo_e4", int(seed)))
    return pd.DataFrame(rows), metas


def baseline_rows(frame: pd.DataFrame, cfg: FoundationConfig, random_seeds: tuple[int, ...],
                  split: Any = EVAL_SPLIT) -> pd.DataFrame:
    """Always CASH, buy-and-hold and random 0-4 through the O2+R2 environment (identical accounting to E1's)."""
    obs_cfg, _ = resolve_o2_config(frame, cfg)
    env = make_e4_env(frame, split, cfg, obs_cfg, role="evaluation")
    rows = []
    for policy, label in ((AlwaysFlat(), "always_CASH"), (AlwaysLong(), "buy_and_hold_always_LONG")):
        res = run_episode_e4(env, policy, periods_per_year=cfg.periods_per_year)
        res.metrics.update(state_distribution(res))
        rows.append(_row_with_diagnostics(res.metrics, label, "-"))
    for s in random_seeds:
        res = run_episode_e4(env, RandomPolicy(s), periods_per_year=cfg.periods_per_year)
        res.metrics.update(state_distribution(res))
        rows.append(_row_with_diagnostics(res.metrics, "random", int(s)))
    return pd.DataFrame(rows)


def build_report(seeds: tuple[int, ...] = CANONICAL_SEEDS, random_seeds: tuple[int, ...] = CANONICAL_RANDOM_SEEDS,
                 artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT, run_group: str = E4_RUN_GROUP,
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
        aggregate(ppo, "ppo_e4 mean ± std (population)"),
        aggregate(base[base["policy"] == "random"], "random mean ± std (population)"),
    ])
    diag_summary = pd.DataFrame([
        aggregate_diagnostics(ppo, "ppo_e4 mean ± std (population)"),
        aggregate_diagnostics(base[base["policy"] == "random"], "random mean ± std (population)"),
    ])
    return {"ppo": ppo, "baselines": base, "summary": summary, "diagnostics_summary": diag_summary,
            "metadata": metas, "split": EVAL_SPLIT, "canonical": canonical, "seeds": tuple(seeds),
            "source_status": current_source_status(metas[0])}


def format_markdown(report: dict[str, Any]) -> str:
    metas = report["metadata"]
    m0 = metas[0]
    fit = m0["observation"]["scaler_fit"]
    rew = m0["reward"]
    label = "CANONICAL" if report["canonical"] else "NON-CANONICAL (explicit seed subset or configuration; not the E4 result)"
    status = report.get("source_status", {})
    if status.get("matches_recorded_fingerprint", True):
        source_note = "Current source tree matches the recorded key-source hashes."
    else:
        changed = ", ".join(f"{k} ({v['recorded']}→{v['current']})" for k, v in status["changed_key_sources"].items())
        source_note = ("Post-run source edits detected in non-core files (recorded hashes retained, not rewritten): "
                       f"{changed}. Training-relevant core sources are unchanged.")
    scaler_lines = ["| feature | train mean | train std | fit rows |", "|---|---|---|---|"]
    for name in fit["features"]:
        pf = fit["per_feature"][name]
        scaler_lines.append(f"| {name} | {pf['mean']:.6g} | {pf['std']:.6g} | {fit['fit_row_start']} .. {fit['fit_row_end']} ({fit['n_rows']}) |")
    lines = [
        f"# E4 validation results — {label} (split = {report['split']})",
        "",
        f"Validation window {m0['validation_split']['window_start']} .. {m0['validation_split']['window_end']}, "
        f"usable decisions {m0['validation_split']['first_usable_decision']} .. "
        f"{m0['validation_split']['last_usable_decision']} (n={m0['validation_split']['n_usable_decisions']}). "
        f"Modelled costs {m0['costs']['bps_per_leg'] / 100:.10g}% of the traded value on each purchase or sale (trading fee plus slippage allowance). Observation {m0['observation']['definition']} "
        f"({m0['observation']['n_features']} engineered features + position, size {m0['observation']['size']}; the "
        f"accepted E2 definition and TRAIN-only scaler, rows {fit['fit_row_start']} .. {fit['fit_row_end']}, {fit['n_rows']} "
        f"rows). Reward {rew['definition']}: `{rew['formula']}` with TURNOVER_PENALTY_LAMBDA = "
        f"{rew['turnover_penalty_lambda']} and DRAWDOWN_PENALTY_LAMBDA = {rew['drawdown_penalty_lambda']} (the accepted E3 "
        "reward, frozen). Actions 0=CASH, 1=LONG. Deterministic (argmax) evaluation of the final checkpoint. Controls: "
        "E2 (same O2, reward R1) and E3 (same R2, observation O1).",
        "",
        "Verification performed for every run: E4 metadata schema (O2 + R2); " +
        ("frozen canonical E4 specification (E2 spec with E3's frozen reward block); " if report["canonical"] else "") +
        "checkpoint bound to metadata (seed, hyperparameters, architecture, timesteps, 9-d O2 space) and to its bytes; "
        "provenance consistency; O2 scaler recomputed from the frozen dataset (every mean/std exact); complete saved "
        "trajectory incl. every R2 decomposition column (1e-12) and the R2 identity; every saved metric and reward "
        "diagnostic equals the recomputed one.",
        "",
        source_note,
        "",
        "Financial metrics are computed from the financial equity path under the unchanged E1 accounting. Reward "
        "diagnostics are training-reward sums, never returns; cumulative R1 = log(final equity) by identity.",
        "",
        "## O2 scaler (TRAIN only; identical to the accepted E2 scaler)",
        "",
        *scaler_lines,
        "",
        "## PPO E4 per seed (financial metrics)",
        "",
        _table(report["ppo"]),
        "",
        "## Mean ± std across seeds",
        "",
        _table(report["summary"]),
        "",
        "## PPO E4 reward diagnostics per seed (training reward, NOT return)",
        "",
        _table(report["ppo"], DIAG_TABLE_COLUMNS),
        "",
        "## Reward diagnostics mean ± std across seeds",
        "",
        _table(report["diagnostics_summary"], DIAG_TABLE_COLUMNS),
        "",
        "## Baselines (same environment and accounting; O2+R2 environment)",
        "",
        _table(report["baselines"]),
        "",
        "## Runs (shared identity verified; checkpoint bound to metadata and bytes)",
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


# ───────────────────────────────────────────── comparisons (VALIDATION only)
def _pairwise(control: pd.DataFrame, treat: pd.DataFrame, c_label: str, t_label: str) -> dict[str, Any]:
    c = control.set_index("seed")
    t = treat.set_index("seed")
    if list(c.index) != list(CANONICAL_SEEDS) or list(t.index) != list(CANONICAL_SEEDS):
        raise ReportIntegrityError("both cohorts must contain exactly the canonical seeds in canonical order")
    within = []
    for s in CANONICAL_SEEDS:
        row: dict[str, Any] = {"seed": s}
        for col in COMPARE_COLUMNS:
            row[f"{c_label}_{col}"] = float(c.loc[s, col])
            row[f"{t_label}_{col}"] = float(t.loc[s, col])
            row[f"diff_{col}"] = float(t.loc[s, col]) - float(c.loc[s, col])
        within.append(row)
    cohort = []
    for col in COMPARE_COLUMNS:
        a = c[col].astype(float).to_numpy()
        b = t[col].astype(float).to_numpy()
        d = b - a
        cohort.append({
            "metric": col,
            f"{c_label}_mean": float(np.mean(a)), f"{c_label}_std": float(np.std(a, ddof=0)),
            f"{t_label}_mean": float(np.mean(b)), f"{t_label}_std": float(np.std(b, ddof=0)),
            f"mean_diff_{t_label}_minus_{c_label}": float(np.mean(d)),
            "within_seed_diff_min": float(np.min(d)), "within_seed_diff_max": float(np.max(d)),
            f"seeds_where_{t_label}_higher": int(np.sum(d > 0)),
        })
    return {"within_seed": pd.DataFrame(within), "cohort": pd.DataFrame(cohort), "control": c_label, "treatment": t_label}


def build_comparisons(e4_report: dict[str, Any], frame: pd.DataFrame, cfg: FoundationConfig,
                      artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, Any]:
    """E2-vs-E4, E3-vs-E4 and the E1/E2/E3/E4 matrix; every control is its own canonical read-only report path."""
    from . import ppo_e1_report as e1rep
    from . import ppo_e2_report as e2rep
    from . import ppo_e3_report as e3rep

    if not e4_report["canonical"]:
        raise ReportIntegrityError("the E4 comparisons require the CANONICAL E4 report")
    anchors = verify_accepted_artifacts_unchanged(artifacts_root)
    verify_accepted_core_sources_unchanged()
    reports = {
        "E1": e1rep.build_report(artifacts_root=artifacts_root, frame=frame, cfg=cfg, canonical=True),
        "E2": e2rep.build_report(artifacts_root=artifacts_root, frame=frame, cfg=cfg, canonical=True),
        "E3": e3rep.build_report(artifacts_root=artifacts_root, frame=frame, cfg=cfg, canonical=True),
        "E4": e4_report,
    }
    after = verify_accepted_artifacts_unchanged(artifacts_root)
    for label in ("E1", "E2", "E3"):
        if after[label]["present"] != anchors[label]["present"]:
            raise ReportIntegrityError(f"{label} artifacts changed while the comparison was running")
    b4 = reports["E4"]["baselines"]
    for label in ("E1", "E2", "E3"):
        b = reports[label]["baselines"]
        for col in COMPARE_COLUMNS + ["final_equity"]:
            if not np.allclose(b[col].astype(float).to_numpy(), b4[col].astype(float).to_numpy(), rtol=0, atol=1e-12, equal_nan=True):
                raise ReportIntegrityError(f"baseline {col} differs between the {label} and E4 environments: accounting is not identical")
    e2_vs_e4 = _pairwise(reports["E2"]["ppo"], reports["E4"]["ppo"], "E2", "E4")
    e3_vs_e4 = _pairwise(reports["E3"]["ppo"], reports["E4"]["ppo"], "E3", "E4")
    rows = []
    for label, obs, rew in MATRIX:
        df = reports[label]["ppo"].set_index("seed")
        for s in CANONICAL_SEEDS:
            r = {"experiment": label, "observation": obs, "reward": rew, "seed": s}
            r.update({col: float(df.loc[s, col]) for col in COMPARE_COLUMNS + ["final_equity"]})
            r["final_position"] = int(df.loc[s, "final_position"])
            rows.append(r)
    per_seed = pd.DataFrame(rows)
    agg = []
    for label, obs, rew in MATRIX:
        df = reports[label]["ppo"]
        r = {"experiment": label, "observation": obs, "reward": rew}
        for col in COMPARE_COLUMNS:
            v = df[col].astype(float).to_numpy()
            r[f"{col}_mean"] = float(np.mean(v))
            r[f"{col}_std"] = float(np.std(v, ddof=0))
        agg.append(r)
    matrix = pd.DataFrame(agg)
    diag = reports["E4"]["ppo"][["seed"] + DIAG_COLUMNS].copy()
    e3_diag = reports["E3"]["ppo"][["seed"] + DIAG_COLUMNS].copy()
    return {"e2_vs_e4": e2_vs_e4, "e3_vs_e4": e3_vs_e4, "matrix_per_seed": per_seed, "matrix_cohort": matrix,
            "e4_diagnostics": diag, "e3_diagnostics": e3_diag, "reports": reports, "anchor_status": anchors,
            "baselines_identical": True}


def format_comparisons_markdown(comp: dict[str, Any]) -> str:
    fps = {k: comp["reports"][k]["metadata"][0]["source"]["source_fingerprint"][:16] for k in ("E1", "E2", "E3", "E4")}
    n_files = sum(len(comp["anchor_status"][k]["present"]) for k in ("E1", "E2", "E3"))

    def pair_block(p: dict[str, Any], title: str, difference: str) -> list[str]:
        w = p["within_seed"]
        c = p["cohort"]
        fmt_w = {k: "{:+.4f}" for k in w.columns if k != "seed"}
        fmt_c = {k: "{:+.4f}" for k in c.columns if k not in ("metric",) and not k.startswith("seeds_where")}
        return [
            f"## {title}",
            "",
            f"Identical everything except {difference}.  Differences are {p['treatment']} minus {p['control']}.",
            "",
            "### Within-seed differences",
            "",
            _table(w, list(w.columns), fmt_w),
            "",
            "### Cohort mean ± std and mean difference",
            "",
            _table(c, list(c.columns), fmt_c),
            "",
        ]

    ms = comp["matrix_per_seed"]
    mc = comp["matrix_cohort"]
    fmt_ms = {k: "{:+.4f}" for k in ms.columns if k not in ("experiment", "observation", "reward", "seed", "final_position", "n_entries", "n_legs")}
    fmt_mc = {k: "{:+.4f}" for k in mc.columns if k not in ("experiment", "observation", "reward")}
    lines = [
        "# E4 (O2 + R2) comparisons and the complete E1/E2/E3/E4 matrix — VALIDATION ONLY",
        "",
        "All cohorts: canonical seeds 42, 123, 2026; identical dataset, splits, timeline, CASH/LONG actions, modelled costs of 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale, "
        "PPO hyperparameters, [64, 64] Tanh MLP, CPU, one torch thread, 200,704 trained timesteps, deterministic "
        "full-TRAIN-window episodes, validation accounting and baselines.  E1 = O1+R1, E2 = O2+R1, E3 = O1+R2, "
        "E4 = O2+R2.  All metrics are FINANCIAL metrics of the realised equity path under the same accounting; no "
        "cumulative reward is compared with a return.  TEST performance of any learned policy was NOT computed.",
        "",
        f"E1, E2 and E3 canonical artifacts verified byte-identical to the frozen anchors before and after reading "
        f"({n_files} files).  Source fingerprints: E1 {fps['E1']}, E2 {fps['E2']}, E3 {fps['E3']}, E4 {fps['E4']}.  "
        "Baselines recomputed through the E1, E2, E3 and E4 environments are identical to 1e-12: the accounting is the "
        "same code.  Three seeds: no statistical significance is claimed; population std across seeds.",
        "",
        *pair_block(comp["e2_vs_e4"], "E2 (O2 + R1) vs E4 (O2 + R2) — RQ3 under engineered observations", "the training reward"),
        *pair_block(comp["e3_vs_e4"], "E3 (O1 + R2) vs E4 (O2 + R2) — observation representation under R2", "the observation"),
        "## Complete matrix per seed",
        "",
        _table(ms, list(ms.columns), fmt_ms),
        "",
        "## Complete matrix, cohort mean ± population std",
        "",
        _table(mc, list(mc.columns), fmt_mc),
        "",
        "## Reward diagnostics of the R2 cohorts (training reward sums; NOT returns)",
        "",
        "E3:",
        "",
        _table(comp["e3_diagnostics"], list(comp["e3_diagnostics"].columns)),
        "",
        "E4:",
        "",
        _table(comp["e4_diagnostics"], list(comp["e4_diagnostics"].columns)),
        "",
        "## Validation baselines (identical under all four environments)",
        "",
        _table(comp["reports"]["E4"]["baselines"]),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS))
    ap.add_argument("--random-seeds", type=int, nargs="+", default=list(CANONICAL_RANDOM_SEEDS))
    ap.add_argument("--run-group", default=E4_RUN_GROUP)
    ap.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--non-canonical", action="store_true")
    ap.add_argument("--compare", action="store_true", help="also write E2-vs-E4, E3-vs-E4 and the E1/E2/E3/E4 matrix")
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
    if args.compare:
        comp = build_comparisons(report, frame, cfg, args.artifacts_root)
        cmd = format_comparisons_markdown(comp)
        (out_dir / "e4_comparisons_validation.md").write_text(cmd, encoding="utf-8")
        comp["e2_vs_e4"]["within_seed"].to_csv(out_dir / "e2_vs_e4_validation_within_seed.csv", index=False)
        comp["e2_vs_e4"]["cohort"].to_csv(out_dir / "e2_vs_e4_validation_cohort.csv", index=False)
        comp["e3_vs_e4"]["within_seed"].to_csv(out_dir / "e3_vs_e4_validation_within_seed.csv", index=False)
        comp["e3_vs_e4"]["cohort"].to_csv(out_dir / "e3_vs_e4_validation_cohort.csv", index=False)
        comp["matrix_per_seed"].to_csv(out_dir / "e1_e2_e3_e4_validation_matrix_per_seed.csv", index=False)
        comp["matrix_cohort"].to_csv(out_dir / "e1_e2_e3_e4_validation_matrix_cohort.csv", index=False)
        comp["e4_diagnostics"].to_csv(out_dir / "e4_validation_reward_diagnostics.csv", index=False)
        print(cmd)
        print(f"written: {out_dir / 'e4_comparisons_validation.md'} (+ csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
