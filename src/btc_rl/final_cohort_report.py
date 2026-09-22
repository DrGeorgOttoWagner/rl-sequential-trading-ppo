"""
final_cohort_report.py — five-seed VALIDATION reports for E1–E4 and the five-seed matrix.

    python -m btc_rl.final_cohort_report            # writes artifacts/ppo/final5/reports/

For every experiment the five-seed cohort is assembled from exactly

    accepted 42 / 123 / 2026   read through the experiment's own canonical three-seed report path
                               (``evaluate_saved_seeds(..., canonical=True)`` on artifacts/ppo/<group>/) — the
                               accepted reports therefore still reproduce, read-only, and their numbers are reused
    +
    new 31415 / 271828         read from artifacts/ppo/final5/<group>/ through the same experiment report functions
                               (schema, provenance, checkpoint binding, saved trajectory, metrics) PLUS the
                               final-cohort checks: seed ∈ NEW_SEEDS, no other seed directory present, canonical
                               specification of the accepted experiment satisfied (everything except the seed set),
                               ``final_cohort.json`` sidecar consistent (seed, experiment, cohort, checkpoint
                               SHA-256, TEST_EVALUATED false), checkpoint bytes not equal to any accepted checkpoint.

The accepted E1/E2/E3/E4 artifacts and sources are verified against their
anchors before and after.  A cohort that is not exactly
(42, 123, 2026, 31415, 271828) in that order fails.  Financial metrics are the
accepted ``compute_metrics`` output; R2 reward diagnostics are reported
separately and never as returns.  Baselines are recomputed through all four
environments and must agree to 1e-12.  VALIDATION only; the split is fixed
and TEST is never read.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FoundationConfig, load_foundation_config
from .data import load_dataset
from .final_cohort import (
    ACCEPTED_ARTIFACT_ANCHORS,
    ACCEPTED_CORE_SOURCE_HASHES,
    ACCEPTED_SEEDS,
    COMPONENT_SOURCES,
    DEFINITIONS,
    EXPERIMENTS,
    FINAL5_REPORTS_DIR,
    FINAL5_ROOT,
    FINAL_CANONICAL_SEEDS,
    NEW_SEEDS,
    RUN_GROUPS,
    STUDY_STAGE,
    FinalCohortError,
    _RUNNERS,
    verify_accepted_artifacts_unchanged,
    verify_accepted_sources_unchanged,
    verify_final5_root,
    verify_recorded_sources_anchored,
)
from .ppo_e1 import DEFAULT_ARTIFACTS_ROOT, EVAL_SPLIT, sha256_of
from .ppo_e4 import KEY_SOURCE_FILES_E4
from .ppo_e1_report import AGG_COLUMNS, REPORT_COLUMNS, ReportIntegrityError, aggregate
from .ppo_e3_report import DIAG_COLUMNS, DIAG_TABLE_COLUMNS, _table, aggregate_diagnostics

R2_EXPERIMENTS: tuple[str, ...] = ("E3", "E4")
COMPARE_COLUMNS = ["total_return", "final_equity", "sharpe_ratio", "max_drawdown", "n_entries", "n_legs", "exposure",
                   "total_cost_fraction"]


def _accepted_checkpoint_hashes() -> set[str]:
    return {files["model.zip"] for anchors in ACCEPTED_ARTIFACT_ANCHORS.values() for files in anchors.values()}


REQUIRED_SIDECAR_FIELDS: tuple[str, ...] = (
    "study_stage", "experiment", "definition", "seed", "final_canonical_seeds", "accepted_seeds", "new_seeds",
    "run_group", "artifacts_root", "runner", "config_class", "git", "source_fingerprint", "dataset_sha256",
    "total_timesteps_requested", "total_timesteps_trained", "model_sha256", "component_provenance",
    "accepted_core_sources_verified", "canonical_spec_verified", "canonical_spec_error", "validation_metrics",
    "evaluation_split", "TEST_EVALUATED", "timestamp_utc",
)
SIDECAR_METRIC_KEYS: tuple[str, ...] = ("total_return", "final_equity", "sharpe_ratio", "max_drawdown", "n_entries",
                                        "n_legs", "exposure", "total_cost_fraction", "final_position")


def _metric_close(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return a == b


def verify_new_seed_record(experiment: str, seed: int, run_dir: Path, meta: dict[str, Any],
                           recomputed_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Final-cohort checks on one new-seed record, in addition to the accepted
    experiment report's own canonical checks:

    * seed ∈ NEW_SEEDS, experiment / run group, accepted canonical specification;
    * recorded training provenance anchored to the frozen accepted sources;
    * every ``final_cohort.json`` field present and equal to its authoritative
      counterpart: metadata (experiment, seed, run group, git, fingerprint,
      dataset, budgets), the frozen constants (cohort, study stage, definition,
      runner, config class), the actual location (resolved root), the actual
      checkpoint bytes, the frozen component anchors, the saved AND recomputed
      VALIDATION metrics, ``evaluation_split == validation`` and
      ``TEST_EVALUATED is False``;
    * checkpoint bytes distinct from every accepted checkpoint.
    """
    rep = _RUNNERS[experiment][2]
    group = RUN_GROUPS[experiment]
    if seed not in NEW_SEEDS:
        raise FinalCohortError(f"{run_dir}: seed {seed} is not a new final-cohort seed {NEW_SEEDS}")
    if int(meta["seed"]) != seed:
        raise FinalCohortError(f"{run_dir}: metadata seed {meta['seed']} != {seed}")
    if meta.get("experiment") != experiment or meta.get("run_group") != group:
        raise FinalCohortError(f"{run_dir}: record is not an {experiment} run in run group {group}")
    try:
        rep.verify_canonical_spec(meta)            # everything of the accepted specification except the seed set
    except ReportIntegrityError as exc:
        raise FinalCohortError(f"{run_dir}: new seed does not match the accepted {experiment} specification: {exc}") from exc
    verify_recorded_sources_anchored(meta)
    sidecar_path = run_dir / "final_cohort.json"
    if not sidecar_path.exists():
        raise FinalCohortError(f"{run_dir}: final_cohort.json sidecar missing")
    side = json.loads(sidecar_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_SIDECAR_FIELDS if k not in side]
    if missing:
        raise FinalCohortError(f"{run_dir}: final-cohort sidecar lacks fields {missing}")
    model_sha = sha256_of(run_dir / "model.zip")
    metrics_path = run_dir / meta["artifacts"]["validation_metrics"]
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    problems: list[str] = []

    def expect(label: str, expected: Any, actual: Any) -> None:
        if expected != actual:
            problems.append(f"{label}: sidecar {str(actual)[:40]!r} != authoritative {str(expected)[:40]!r}")

    expect("study_stage", STUDY_STAGE, side["study_stage"])
    expect("experiment", experiment, side["experiment"])
    expect("definition", DEFINITIONS[experiment], side["definition"])
    expect("seed", seed, side["seed"])
    expect("final_canonical_seeds", list(FINAL_CANONICAL_SEEDS), list(side["final_canonical_seeds"]))
    expect("accepted_seeds", list(ACCEPTED_SEEDS), list(side["accepted_seeds"]))
    expect("new_seeds", list(NEW_SEEDS), list(side["new_seeds"]))
    expect("run_group", group, side["run_group"])
    expect("artifacts_root (resolved record location)", str(Path(run_dir).resolve().parent.parent), side["artifacts_root"])
    if Path(run_dir).resolve().parent.name != group or Path(run_dir).resolve().name != str(seed):
        problems.append("record location is not <root>/<group>/<seed>")
    expect("runner", f"btc_rl.ppo_{group}.run_{group}", side["runner"])
    expect("config_class", f"{experiment}Config", side["config_class"])
    expect("git", meta["git"], side["git"])
    expect("source_fingerprint", meta["source"]["source_fingerprint"], side["source_fingerprint"])
    expect("dataset_sha256", meta["dataset"]["sha256"], side["dataset_sha256"])
    expect("total_timesteps_requested", int(meta["total_timesteps_requested"]), side["total_timesteps_requested"])
    expect("total_timesteps_trained", int(meta["total_timesteps_trained"]), side["total_timesteps_trained"])
    expect("model_sha256 (checkpoint bytes)", model_sha, side["model_sha256"])
    recorded = meta.get("artifacts", {}).get("model_sha256")
    if recorded is not None and recorded != model_sha:
        problems.append("metadata model_sha256 != checkpoint bytes")
    anchors_by_path = {KEY_SOURCE_FILES_E4[label]: digest for label, digest in ACCEPTED_CORE_SOURCE_HASHES.items()}
    prov = side["component_provenance"]
    if not isinstance(prov, dict) or set(prov) != set(COMPONENT_SOURCES):
        problems.append("component_provenance: unexpected component set")
    else:
        for name, rel in COMPONENT_SOURCES.items():
            entry = prov[name]
            if not isinstance(entry, dict) or entry.get("path") != rel or entry.get("sha256") != anchors_by_path[rel]:
                problems.append(f"component_provenance.{name}: not the frozen accepted anchor of {rel}")
            if meta["source"]["source_files"].get(rel) != anchors_by_path[rel]:
                problems.append(f"component_provenance.{name}: recorded training source differs from the anchor")
    if side["accepted_core_sources_verified"] is not True:
        problems.append("accepted_core_sources_verified is not true")
    if side["canonical_spec_verified"] is not True or side["canonical_spec_error"] is not None:
        problems.append("canonical_spec_verified is not true / canonical_spec_error is not null")
    if side["evaluation_split"] != EVAL_SPLIT:
        problems.append("evaluation_split is not validation")
    if side["TEST_EVALUATED"] is not False:
        problems.append("TEST_EVALUATED is not false")
    vm = side["validation_metrics"]
    if not isinstance(vm, dict) or any(k not in vm for k in SIDECAR_METRIC_KEYS):
        problems.append("validation_metrics: missing financial keys")
    else:
        if saved_metrics is None:
            problems.append("saved validation_metrics.json missing")
        else:
            for k, v in vm.items():
                if not _metric_close(saved_metrics.get(k), v):
                    problems.append(f"validation_metrics.{k}: sidecar != saved validation_metrics.json")
        if recomputed_metrics is not None:
            for k, v in vm.items():
                if k in recomputed_metrics and not _metric_close(recomputed_metrics[k], v):
                    problems.append(f"validation_metrics.{k}: sidecar != recomputed VALIDATION evaluation")
    if not isinstance(side["timestamp_utc"], str) or not side["timestamp_utc"]:
        problems.append("timestamp_utc missing")
    if model_sha in _accepted_checkpoint_hashes():
        problems.append("checkpoint bytes equal an accepted E1/E2/E3/E4 checkpoint")
    if problems:
        raise FinalCohortError(f"{run_dir}: final-cohort record inconsistent: {'; '.join(problems)}")
    return {"model_sha256": model_sha, "sidecar": side}


@contextmanager
def _new_seed_set(rep):
    """
    Narrow adapter: the accepted report's canonical cohort check
    requires the historical seed set 42/123/2026; for the frozen new pair the
    ONLY relaxation is that constant.  Every other canonical check of
    ``evaluate_saved_seeds(canonical=True)`` — frozen specification, checkpoint
    binding incl. target_kl / clip_range_vf / use_sde / Tanh / shape,
    checkpoint bytes (E3/E4), provenance, current-tree core sources, scaler
    recomputation, saved trajectory and metrics — runs unchanged.
    """
    saved = rep.CANONICAL_SEEDS
    rep.CANONICAL_SEEDS = tuple(NEW_SEEDS)
    try:
        yield
    finally:
        rep.CANONICAL_SEEDS = saved


def evaluate_new_seeds(experiment: str, frame: pd.DataFrame, cfg: FoundationConfig, final_root: Path = FINAL5_ROOT
                       ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """The two new seeds through the accepted report's CANONICAL path (seed set relaxed to the frozen new pair only)."""
    rep = _RUNNERS[experiment][2]
    with _new_seed_set(rep):
        return rep.evaluate_saved_seeds(NEW_SEEDS, frame, cfg, Path(final_root), RUN_GROUPS[experiment], canonical=True)


def _seed_dirs(group_dir: Path) -> list[int]:
    if not group_dir.exists():
        return []
    out = []
    for p in sorted(group_dir.iterdir()):
        if p.is_dir():
            try:
                out.append(int(p.name))
            except ValueError as exc:
                raise FinalCohortError(f"{group_dir}: unexpected non-seed directory {p.name!r}") from exc
    return out


def evaluate_experiment_final5(experiment: str, frame: pd.DataFrame, cfg: FoundationConfig,
                               accepted_root: Path = DEFAULT_ARTIFACTS_ROOT, final_root: Path = FINAL5_ROOT,
                               accepted_canonical: bool = True, require_new_canonical: bool = True
                               ) -> dict[str, Any]:
    """Accepted three seeds (their canonical report path) + new two seeds (final5), in the frozen order."""
    if experiment not in EXPERIMENTS:
        raise FinalCohortError(f"unknown experiment {experiment!r}")
    rep = _RUNNERS[experiment][2]
    group = RUN_GROUPS[experiment]
    verify_final5_root(Path(final_root)) if Path(final_root).resolve() != FINAL5_ROOT.resolve() else None
    found = _seed_dirs(Path(final_root) / group)
    if sorted(found) != sorted(NEW_SEEDS):
        raise FinalCohortError(f"{Path(final_root) / group}: expected exactly the new seeds {list(NEW_SEEDS)}, found {found}")
    accepted_rows, accepted_metas = rep.evaluate_saved_seeds(ACCEPTED_SEEDS, frame, cfg, Path(accepted_root), group,
                                                             canonical=accepted_canonical)
    if require_new_canonical:
        new_rows, new_metas = evaluate_new_seeds(experiment, frame, cfg, final_root)      # canonical, seed set relaxed only
    else:   # tiny test cohorts only: non-canonical budgets cannot satisfy the frozen specification
        new_rows, new_metas = rep.evaluate_saved_seeds(NEW_SEEDS, frame, cfg, Path(final_root), group, canonical=False)
    checks = {}
    for seed, meta, (_, row) in zip(NEW_SEEDS, new_metas, new_rows.iterrows()):
        run_dir = Path(final_root) / group / str(seed)
        if require_new_canonical:
            checks[seed] = verify_new_seed_record(experiment, seed, run_dir, meta, recomputed_metrics=row.to_dict())
        else:
            checks[seed] = {"model_sha256": sha256_of(run_dir / "model.zip"), "sidecar": None}
    rows = pd.concat([accepted_rows, new_rows], ignore_index=True)
    rows["policy"] = f"ppo_{group}"
    rows["cohort"] = ["accepted"] * len(ACCEPTED_SEEDS) + ["new"] * len(NEW_SEEDS)
    seeds = tuple(int(s) for s in rows["seed"])
    if seeds != FINAL_CANONICAL_SEEDS:
        raise FinalCohortError(f"{experiment}: assembled cohort {seeds} is not the frozen final cohort {FINAL_CANONICAL_SEEDS}")
    summary = aggregate(rows, f"ppo_{group} five-seed mean ± std (population)")
    dev_summary = aggregate(rows.iloc[: len(ACCEPTED_SEEDS)], f"ppo_{group} three-seed (accepted) mean ± std")
    out: dict[str, Any] = {"experiment": experiment, "definition": DEFINITIONS[experiment], "rows": rows,
                           "summary": pd.DataFrame([dev_summary, summary]), "accepted_metadata": accepted_metas,
                           "new_metadata": new_metas, "new_checks": checks}
    if experiment in R2_EXPERIMENTS:
        out["diagnostics_summary"] = pd.DataFrame([
            aggregate_diagnostics(rows.iloc[: len(ACCEPTED_SEEDS)], f"ppo_{group} three-seed (accepted)"),
            aggregate_diagnostics(rows, f"ppo_{group} five-seed"),
        ])
    return out


def baseline_rows_all(frame: pd.DataFrame, cfg: FoundationConfig, random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Baselines through every accepted environment; they must agree to 1e-12 (same accounting); E3's rows carry diagnostics."""
    tables = {e: _RUNNERS[e][2].baseline_rows(frame, cfg, random_seeds) for e in EXPERIMENTS}
    ref = tables["E1"]
    for e, t in tables.items():
        if t["policy"].tolist() != ref["policy"].tolist():
            raise FinalCohortError(f"baseline policies differ between E1 and {e}")
        for col in COMPARE_COLUMNS:
            if not np.allclose(t[col].astype(float).to_numpy(), ref[col].astype(float).to_numpy(), rtol=0, atol=1e-12, equal_nan=True):
                raise FinalCohortError(f"baseline {col} differs between the E1 and {e} environments")
    return tables["E3"]   # identical financial columns, plus the R2 diagnostics of the fixed policies


def build_final_report(frame: pd.DataFrame | None = None, cfg: FoundationConfig | None = None,
                       accepted_root: Path = DEFAULT_ARTIFACTS_ROOT, final_root: Path = FINAL5_ROOT,
                       accepted_canonical: bool = True, require_new_canonical: bool = True) -> dict[str, Any]:
    cfg = cfg or load_foundation_config()
    if frame is None:
        frame, _ = load_dataset(cfg.dataset)
    anchors_before = None
    if accepted_canonical:
        anchors_before = verify_accepted_artifacts_unchanged(Path(accepted_root))
        verify_accepted_sources_unchanged()
    experiments = {e: evaluate_experiment_final5(e, frame, cfg, accepted_root, final_root, accepted_canonical,
                                                 require_new_canonical) for e in EXPERIMENTS}
    baselines = baseline_rows_all(frame, cfg)
    if accepted_canonical:
        after = verify_accepted_artifacts_unchanged(Path(accepted_root))
        for e in EXPERIMENTS:
            if after[e]["present"] != anchors_before[e]["present"]:
                raise FinalCohortError(f"{e} accepted artifacts changed while the report was running")
    rows = []
    for e in EXPERIMENTS:
        df = experiments[e]["rows"]
        r = {"experiment": e, "definition": DEFINITIONS[e], "n_seeds": len(df)}
        for col in AGG_COLUMNS:
            v = df[col].astype(float).to_numpy()
            r[f"{col}_mean"] = float(np.mean(v))
            r[f"{col}_std"] = float(np.std(v, ddof=0))
        r["ending_LONG"] = int((df["final_position"] == 1).sum())
        rows.append(r)
    matrix = pd.DataFrame(rows)
    per_seed = pd.concat([experiments[e]["rows"].assign(experiment=e, definition=DEFINITIONS[e]) for e in EXPERIMENTS],
                         ignore_index=True)
    disp = []
    for e in EXPERIMENTS:
        df = experiments[e]["rows"]
        for col in ("total_return", "sharpe_ratio", "max_drawdown", "n_legs", "total_cost_fraction"):
            v3 = df[col].astype(float).to_numpy()[: len(ACCEPTED_SEEDS)]
            v5 = df[col].astype(float).to_numpy()
            disp.append({"experiment": e, "metric": col, "mean_3": float(np.mean(v3)), "std_3": float(np.std(v3, ddof=0)),
                         "mean_5": float(np.mean(v5)), "std_5": float(np.std(v5, ddof=0)),
                         "new_seed_31415": float(df.set_index("seed").loc[31415, col]),
                         "new_seed_271828": float(df.set_index("seed").loc[271828, col])})
    return {"experiments": experiments, "baselines": baselines, "matrix": matrix, "per_seed": per_seed,
            "dispersion": pd.DataFrame(disp), "seeds": FINAL_CANONICAL_SEEDS, "split": EVAL_SPLIT,
            "test_evaluated": False, "anchor_status": anchors_before}


FMT = {"total_return": "{:+.4f}", "final_equity": "{:.4f}", "sharpe_ratio": "{:.3f}", "max_drawdown": "{:+.4f}",
       "exposure": "{:.3f}", "fraction_CASH": "{:.3f}", "total_cost_fraction": "{:.4f}",
       "cumulative_financial_r1": "{:+.4f}", "cumulative_turnover_penalty": "{:.4f}",
       "cumulative_drawdown_penalty": "{:.4f}", "cumulative_reward_r2": "{:+.4f}"}
ROW_COLUMNS = ["policy", "cohort"] + [c for c in REPORT_COLUMNS if c not in ("policy",)]


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Final five-seed cohort — VALIDATION ONLY ({STUDY_STAGE})",
        "",
        f"Seeds {list(report['seeds'])}: accepted development seeds 42 / 123 / 2026 (read through the accepted "
        "canonical three-seed report paths, artifacts byte-identical to their anchors before and after) + new seeds "
        "31415 / 271828 (artifacts/ppo/final5, verified against the accepted experiment specification, checkpoint "
        "bytes bound and distinct from every accepted checkpoint, sidecar TEST_EVALUATED = false).  Same dataset, "
        "splits, modelled costs of 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale, PPO configuration, [64, 64] Tanh, CPU, one thread, one environment, 200,704 trained "
        "timesteps, deterministic full-TRAIN-window episodes, deterministic (argmax) VALIDATION evaluation.  Financial "
        "metrics from the equity path under the unchanged accounting; R2 reward diagnostics separate, never returns.  "
        f"TEST evaluated: {report['test_evaluated']}.  Three-seed rows are the accepted numbers; five-seed statistics are "
        "an uncertainty / reproducibility expansion, not a model-selection stage.",
        "",
    ]
    for e in EXPERIMENTS:
        ex = report["experiments"][e]
        lines += [f"## {e} = {ex['definition']} — five seeds", "", _table(ex["rows"], ROW_COLUMNS, FMT), "",
                  "Mean ± population std (accepted three seeds, then all five):", "", _table(ex["summary"], REPORT_COLUMNS, FMT), ""]
        if "diagnostics_summary" in ex:
            lines += ["Reward diagnostics per seed (training reward, NOT return):", "",
                      _table(ex["rows"], ["policy", "seed", "cohort"] + DIAG_COLUMNS, FMT), "",
                      "Reward diagnostics mean ± std:", "", _table(ex["diagnostics_summary"], DIAG_TABLE_COLUMNS, FMT), ""]
        lines += ["New-seed checkpoints: " + ", ".join(f"{s} `{ex['new_checks'][s]['model_sha256'][:16]}`" for s in NEW_SEEDS), ""]
    m = report["matrix"]
    fmt_m = {k: "{:+.4f}" for k in m.columns if k not in ("experiment", "definition", "n_seeds", "ending_LONG")}
    d = report["dispersion"]
    fmt_d = {k: "{:+.4f}" for k in d.columns if k not in ("experiment", "metric")}
    lines += ["## Five-seed E1–E4 matrix (mean ± population std over 42, 123, 2026, 31415, 271828)", "",
              _table(m, list(m.columns), fmt_m), "",
              "## Seed dispersion: accepted three seeds vs final five seeds", "",
              _table(d, list(d.columns), fmt_d), "",
              "## Baselines (same environment and accounting; verified identical through the E1, E2, E3 and E4 environments)", "",
              _table(report["baselines"], REPORT_COLUMNS, FMT), "",
              "Baseline reward diagnostics (what R2 would have paid these fixed policies):", "",
              _table(report["baselines"], DIAG_TABLE_COLUMNS, FMT), ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accepted-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--final-root", type=Path, default=FINAL5_ROOT)
    ap.add_argument("--out-dir", type=Path, default=FINAL5_REPORTS_DIR)
    args = ap.parse_args(argv)
    verify_final5_root(args.out_dir)
    report = build_final_report(accepted_root=args.accepted_root, final_root=args.final_root)
    md = format_markdown(report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "final5_validation_results.md").write_text(md, encoding="utf-8")
    report["per_seed"].to_csv(args.out_dir / "final5_validation_per_seed.csv", index=False)
    report["matrix"].to_csv(args.out_dir / "final5_validation_matrix.csv", index=False)
    report["dispersion"].to_csv(args.out_dir / "final5_seed_dispersion.csv", index=False)
    report["baselines"].to_csv(args.out_dir / "final5_validation_baselines.csv", index=False)
    print(md)
    print(f"written: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
