"""
final_test_report.py — the final TEST report builder, reconstructed from persisted evidence only.

    python -m btc_rl.final_test_report          # requires the completed one-shot run in artifacts/ppo/final_test/

The builder never evaluates, never loads a model, never constructs an environment and never repairs anything.  It
requires the completion marker and verifies the complete identity chain by EXACT equality across the durable run
state (the authority), the manifest, every ``item.json`` and the persisted evidence: run id, canonical root,
authorization SHA-256 (record present at the recorded path, bytes equal, contents valid), cohort manifest + hash,
the 20 checkpoints, TEST window and grid, dataset / runtime / source / harness identity, context kind, the exact
canonical 48-item execution plan (order, identities, checkpoint hashes, canonical baseline kind / seed / name,
COMMITTED, recovery flags, evidence hashes), and the summary rows (exact ordered identities, no duplicates, values
equal to the evidence).  Every curve is validated over the full frozen timeline and the accepted accounting
identities (R1 and the frozen R2 decomposition); financial metrics are recomputed from the curves and required to
equal the persisted metrics.  Report structure: per-seed table (20 rows, R2 diagnostics separate), per-experiment
mean ± population std over the exact five seeds, E1–E4 matrix, baselines (identical through the four environments),
seed dispersion, descriptive RQ1 / RQ2 / RQ3; no pre-specified significance test, no profitability or
generalisation claim, no post-TEST retuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .final_cohort import DEFINITIONS, EXPERIMENTS, FINAL_CANONICAL_SEEDS
from .final_test import (
    AUTHORIZATION_NAME,
    BASELINE_IDENTITY,
    BASELINE_LABELS,
    CANONICAL_PLAN_IDENTITY,
    COHORT_ORDER,
    DIAG_KEYS,
    EVIDENCE_FILES,
    FINAL_TEST_ROOT,
    FROZEN_CHECKPOINTS,
    FROZEN_DATASET_SHA256,
    FROZEN_RUNTIME_IDENTITY,
    FROZEN_TEST_GRID_SHA256,
    HARNESS_FILES,
    MANIFEST_NAME,
    MARKER_NAME,
    METRIC_KEYS,
    N_TEST_TRANSITIONS,
    POST_TEST_RULES,
    PRODUCTION_FINAL_TEST,
    R2_EXPERIMENTS,
    RUN_STATE_NAME,
    SYNTHETIC_TEST_FIXTURE,
    TEST_WINDOW_IDENTITY,
    FinalTestError,
    _HEX40,
    _close,
    _plan_projection,
    _validate_record_contents,
    cohort_manifest_sha256,
    expected_item_identity,
    verify_source_identity,
    item_identity,
    validate_item_evidence,
)
from .ppo_e1 import sha256_of, sha256_text
from .ppo_e3_report import _table

FIN_COLUMNS = ["total_return", "final_equity", "sharpe_ratio", "max_drawdown", "n_entries", "n_legs", "exposure",
               "total_cost_fraction", "final_position"]
AGG_COLUMNS = ["total_return", "final_equity", "sharpe_ratio", "max_drawdown", "n_entries", "n_legs", "exposure", "total_cost_fraction"]
FMT = {"total_return": "{:+.4f}", "final_equity": "{:.4f}", "sharpe_ratio": "{:.3f}", "max_drawdown": "{:+.4f}",
       "exposure": "{:.3f}", "total_cost_fraction": "{:.4f}", "cumulative_financial_r1": "{:+.4f}",
       "cumulative_turnover_penalty": "{:.4f}", "cumulative_drawdown_penalty": "{:.4f}", "cumulative_reward_r2": "{:+.4f}"}
STATE_MANIFEST_EQUAL_KEYS: tuple[str, ...] = ("run_id", "context_kind", "execution_commit", "cohort_manifest", "cohort_manifest_sha256", "checkpoints",
                                              "test_window", "test_grid_sha256", "source_identity", "dataset", "runtime", "results_dir",
                                              "post_test_rules", "started_utc", "recovery_events")


def load_completed_run(root: Path = FINAL_TEST_ROOT) -> dict[str, Any]:
    """Marker → manifest → run state → authorization → plan / identity chain.  Raises on any inconsistency; creates nothing."""
    root = Path(root)
    results = root / "results"
    marker, manifest_path, state_path, auth_path = results / MARKER_NAME, results / MANIFEST_NAME, root / RUN_STATE_NAME, root / AUTHORIZATION_NAME
    if not marker.exists():
        raise FinalTestError(f"final TEST report refused: {marker} does not exist — the one-time TEST evaluation has not been completed")
    if not manifest_path.exists():
        raise FinalTestError(f"final TEST report refused: {manifest_path} missing")
    manifest_sha = sha256_of(manifest_path)
    marker_text = marker.read_text(encoding="utf-8")
    if f"manifest_sha256 {manifest_sha}" not in marker_text:
        raise FinalTestError("final TEST report refused: completion marker does not bind to the manifest hash")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if m.get("TEST_OPENED") is not True:
        problems.append("manifest does not record TEST_OPENED = true")
    if f"run_id {m.get('run_id')}" not in marker_text or f"cohort_manifest_sha256 {m.get('cohort_manifest_sha256')}" not in marker_text:
        problems.append("completion marker does not bind to the manifest run id / cohort hash")
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if not isinstance(state, dict):
        problems.append("run state missing")
        state = None
    else:
        if state.get("status") != "COMPLETE" or state.get("run_id") != m.get("run_id") or state.get("manifest_sha256") != manifest_sha:
            problems.append("run state is not COMPLETE for this manifest / run id")
        if state.get("authorization_sha256") != m.get("authorization", {}).get("sha256") or state.get("authorization_path") != m.get("authorization", {}).get("path"):
            problems.append("run state authorization identity differs from the manifest")
        for k in STATE_MANIFEST_EQUAL_KEYS:
            if state.get(k) != m.get(k):
                problems.append(f"run state and manifest differ in {k}")
        if _plan_projection(state.get("plan", [])) != m.get("items"):
            problems.append("manifest items are not the exact projection of the durable plan")
        if [item_identity(p) for p in state.get("plan", [])] != list(CANONICAL_PLAN_IDENTITY):
            problems.append("durable plan is not the canonical 48-item execution plan (order, identities, checkpoint hashes)")
        for p in state.get("plan", []):
            if p.get("state") != "COMMITTED" or int(p.get("attempts", 0)) < 1 or not isinstance(p.get("evidence_sha256"), dict) \
                    or set(p["evidence_sha256"]) != set(EVIDENCE_FILES) or not isinstance(p.get("recovery_retry"), bool):
                problems.append(f"plan item {p.get('key')} is not COMMITTED with complete evidence hashes / recovery flags")
        if state.get("results_dir") != str(results.resolve()):
            problems.append("run state canonical root differs from the reported root")
        if state.get("runtime") != FROZEN_RUNTIME_IDENTITY:
            problems.append("runtime identity differs from the frozen runtime identity")
        if state.get("test_grid_sha256") != FROZEN_TEST_GRID_SHA256:
            problems.append("TEST grid identity differs from the frozen grid")
        kind = state.get("context_kind")
        is_production_root = root.resolve() == FINAL_TEST_ROOT.resolve()
        if kind not in (PRODUCTION_FINAL_TEST, SYNTHETIC_TEST_FIXTURE) or (kind == PRODUCTION_FINAL_TEST) != is_production_root:
            problems.append("context kind does not match the root (production evidence only under the canonical root)")
        src = state.get("source_identity") or {}
        if set(src.get("harness_files", {})) != set(HARNESS_FILES) or not src.get("harness_anchors_sha256") or not src.get("source_fingerprint"):
            problems.append("source identity is incomplete")
        anchors_path = Path(src.get("harness_anchors_path", ""))
        if not anchors_path.is_file() or sha256_of(anchors_path) != src.get("harness_anchors_sha256"):
            problems.append("reviewed harness anchors recorded by the run are missing or changed")
        ec = state.get("execution_commit")
        if not isinstance(ec, str) or not _HEX40.match(ec) or ec != src.get("execution_commit"):
            problems.append("run state execution_commit is missing / malformed / different from the recorded source identity")
        if kind == PRODUCTION_FINAL_TEST:
            ds = state.get("dataset") or {}
            if ds.get("sha256") != FROZEN_DATASET_SHA256 or src.get("two_commit_rule_verified") is not True or not src.get("reviewed_code_commit") \
                    or not src.get("execution_commit"):
                problems.append("production run does not bind the frozen dataset / reviewed two-commit source identity")
            else:
                try:
                    current = verify_source_identity()["execution_commit"]       # read-only A→B re-verification, no TEST access
                except FinalTestError as exc:
                    current = None
                    problems.append(f"current reviewed execution identity cannot be established: {exc}")
                if current is not None and current != ec:
                    problems.append("manifest / run state execution_commit differs from the current reviewed execution identity B")
    if m.get("cohort_manifest_sha256") != cohort_manifest_sha256():
        problems.append("manifest cohort hash does not bind to the frozen cohort")
    if m.get("cohort_manifest") is None or sha256_text(json.dumps(m["cohort_manifest"], sort_keys=True, separators=(",", ":"))) != cohort_manifest_sha256():
        problems.append("embedded cohort manifest does not hash to the frozen cohort hash")
    got = [(c.get("experiment"), c.get("seed"), c.get("model_sha256")) for c in m.get("checkpoints", [])]
    if got != [(e, s, FROZEN_CHECKPOINTS[(e, s)]) for e, s in COHORT_ORDER]:
        problems.append("manifest checkpoints are not the 20 frozen checkpoints in the frozen order")
    if m.get("test_window") != TEST_WINDOW_IDENTITY:
        problems.append("manifest TEST window differs from the frozen identity")
    auth = m.get("authorization", {})
    if not auth_path.exists():
        problems.append("authorization record missing at the canonical path")
    else:
        raw = auth_path.read_bytes()
        if sha256_of(auth_path) != auth.get("sha256") or str(auth_path.resolve()) != auth.get("path"):
            problems.append("authorization record bytes/path differ from the manifest")
        else:
            try:
                rec = json.loads(raw.decode("utf-8"))
                _validate_record_contents(rec, "report")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"authorization record invalid: {exc}")
            else:
                if rec.get("execution_commit") != m.get("execution_commit") or (state is not None and rec.get("execution_commit") != state.get("execution_commit")):
                    problems.append("authorization execution_commit differs from the manifest / run state execution commit")
    items = m.get("items", [])
    if [item_identity(it) for it in items] != list(CANONICAL_PLAN_IDENTITY):
        problems.append("manifest items are not the canonical plan: 20 learned checkpoints in order and exactly CASH, BUY_AND_HOLD, "
                        "RANDOM_0..4 (canonical kind / seed / name) per experiment")
    if any(it.get("state") != "COMMITTED" for it in items):
        problems.append("an item is not COMMITTED")
    # summary rows: exact ordered identities, no duplicates (checked as lists, never through dictionary keys)
    learned_ids = [(r.get("key"), r.get("experiment"), r.get("seed"), r.get("policy"), r.get("policy_kind"), r.get("policy_name"), r.get("model_sha256"))
                   for r in m.get("learned_rows", [])]
    expected_learned = [(it["key"], it["experiment"], it["seed"], it["policy"], it["policy_kind"], it["policy_name"], it["model_sha256"])
                        for it in items if it.get("kind") == "learned"]
    base_ids = [(r.get("key"), r.get("experiment"), r.get("policy"), r.get("policy_kind"), r.get("random_seed"), r.get("policy_name"))
                for r in m.get("baseline_rows", [])]
    expected_base = [(it["key"], it["experiment"], it["policy"], it["policy_kind"], it["random_seed"], it["policy_name"])
                     for it in items if it.get("kind") == "baseline"]
    if learned_ids != expected_learned or len(set(learned_ids)) != 20:
        problems.append("learned summary rows are not exactly the 20 ordered learned identities (missing / duplicate / foreign)")
    if base_ids != expected_base or len(set(base_ids)) != 28 or [(e, lab, *BASELINE_IDENTITY[lab][:2], BASELINE_IDENTITY[lab][2]) for e in EXPERIMENTS for lab in BASELINE_LABELS] \
            != [(b[1], b[2], b[3], b[4], b[5]) for b in base_ids]:
        problems.append("baseline summary rows are not exactly the 7 canonical identities × 4 experiments (kind / seed / name)")
    if problems:
        raise FinalTestError("final TEST report refused: " + "; ".join(problems))
    return {"manifest": m, "state": state, "results_dir": results, "manifest_sha256": manifest_sha}


def reconstruct_rows(loaded: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows from persisted evidence only: identities against the run state (authority), hashes, full timeline and
    accounting, metrics recomputed from the curves; manifest summary rows required to equal the evidence."""
    m, state, results = loaded["manifest"], loaded["state"], loaded["results_dir"]
    learned_rows, base_rows = [], []
    for it, plan_item in zip(m["items"], state["plan"]):
        d = results / it["key"]
        expected = expected_item_identity(state, plan_item)
        checked = validate_item_evidence(d, expected, it.get("evidence_sha256") or {})
        ev, metrics = checked["item"], checked["metrics"]
        if ev.get("evidence_sha256", {}).get("test_curve.csv") != plan_item["evidence_sha256"]["test_curve.csv"]:
            raise FinalTestError(f"{it['key']}: item.json evidence hashes differ from the run state")
        row = {"experiment": it["experiment"], "definition": DEFINITIONS[it["experiment"]], "seed": it.get("seed"),
               "policy": ev["policy"], "policy_kind": ev["policy_kind"], "policy_name": ev["policy_name"], "key": it["key"],
               **{k: metrics.get(k) for k in METRIC_KEYS}}
        if it["experiment"] in R2_EXPERIMENTS:
            row.update(checked["diagnostics"])
        rows = m["learned_rows"] if it["kind"] == "learned" else m["baseline_rows"]
        matching = [r for r in rows if r.get("key") == it["key"]]
        if len(matching) != 1:
            raise FinalTestError(f"{it['key']}: manifest summary row missing or duplicated")
        summary = matching[0]
        for k in list(METRIC_KEYS) + (list(DIAG_KEYS) if it["experiment"] in R2_EXPERIMENTS else []):
            if not _close(summary.get(k), row.get(k)):
                raise FinalTestError(f"{it['key']}: manifest summary row differs from the persisted evidence in {k}")
        (learned_rows if it["kind"] == "learned" else base_rows).append(row)
    return pd.DataFrame(learned_rows), pd.DataFrame(base_rows)


def build_report(root: Path = FINAL_TEST_ROOT) -> dict[str, Any]:
    loaded = load_completed_run(root)
    per_seed, base = reconstruct_rows(loaded)
    if len(per_seed) != 20 or len(base) != 4 * len(BASELINE_LABELS):
        raise FinalTestError("reconstructed evidence is not the exact learned/baseline set")
    ref = base[base["experiment"] == "E1"].reset_index(drop=True)
    for e in EXPERIMENTS[1:]:
        other = base[base["experiment"] == e].reset_index(drop=True)
        if other["policy"].tolist() != list(BASELINE_LABELS) or ref["policy"].tolist() != list(BASELINE_LABELS):
            raise FinalTestError("baseline rows are not the canonical identities in order")
        for col in AGG_COLUMNS:
            if not np.allclose(ref[col].astype(float).to_numpy(), other[col].astype(float).to_numpy(), rtol=0, atol=1e-12, equal_nan=True):
                raise FinalTestError(f"baseline {col} differs between the E1 and {e} TEST environments")
    summary_rows, matrix_rows, disp_rows = [], [], []
    for e in EXPERIMENTS:
        df = per_seed[per_seed["experiment"] == e]
        if df["seed"].tolist() != list(FINAL_CANONICAL_SEEDS):
            raise FinalTestError(f"{e}: rows are not the frozen five seeds in order")
        agg: dict[str, Any] = {"experiment": e, "definition": DEFINITIONS[e], "n_seeds": len(df)}
        for col in AGG_COLUMNS + (list(DIAG_KEYS) if e in R2_EXPERIMENTS else []):
            v = df[col].astype(float).to_numpy()
            agg[f"{col}_mean"], agg[f"{col}_std"] = float(np.mean(v)), float(np.std(v, ddof=0))
        agg["ending_LONG"] = int((df["final_position"] == 1).sum())
        summary_rows.append(agg)
        matrix_rows.append({k: agg[k] for k in agg if not any(k.startswith(d) for d in DIAG_KEYS)})
        for col in ("total_return", "sharpe_ratio", "max_drawdown", "n_legs", "total_cost_fraction"):
            v = df[col].astype(float).to_numpy()
            disp_rows.append({"experiment": e, "metric": col, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=0)),
                              "min": float(np.min(v)), "max": float(np.max(v)), **{f"seed_{s}": float(x) for s, x in zip(df["seed"], v)}})
    return {"manifest": loaded["manifest"], "manifest_sha256": loaded["manifest_sha256"], "per_seed": per_seed,
            "summary": pd.DataFrame(summary_rows), "matrix": pd.DataFrame(matrix_rows), "baselines": ref,
            "dispersion": pd.DataFrame(disp_rows)}


def _pair(per_seed: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    x, y = per_seed[per_seed["experiment"] == a].set_index("seed"), per_seed[per_seed["experiment"] == b].set_index("seed")
    rows = []
    for col in AGG_COLUMNS:
        d = y[col].astype(float).to_numpy() - x[col].astype(float).to_numpy()
        rows.append({"metric": col, f"{a}_mean": float(x[col].astype(float).mean()), f"{b}_mean": float(y[col].astype(float).mean()),
                     f"mean_diff_{b}_minus_{a}": float(np.mean(d)), "paired_min": float(np.min(d)), "paired_max": float(np.max(d)),
                     f"seeds_{b}_higher": int(np.sum(d > 0))})
    return pd.DataFrame(rows)


def _tbl(df: pd.DataFrame) -> str:
    return _table(df, list(df.columns), {k: "{:+.4f}" for k in df.columns if k not in ("experiment", "definition", "metric", "seed", "policy",
                                                                                          "n_seeds", "ending_LONG", "n_steps", "final_position",
                                                                                          "n_entries", "n_legs") and not str(k).startswith("seeds_")})


def format_markdown(report: dict[str, Any]) -> str:
    m, ps = report["manifest"], report["per_seed"]
    lines = [
        "# Final one-time TEST evaluation — frozen cohort (VALIDATION-frozen E1–E4, 5 seeds each)",
        "",
        f"Run {m['run_id']}, completed {m['finished_utc']}; manifest SHA-256 {report['manifest_sha256'][:16]}; authorization record "
        f"SHA-256 {m['authorization']['sha256'][:16]}; cohort manifest SHA-256 {m['cohort_manifest_sha256'][:16]}; TEST window "
        f"{m['test_window']['window_start']} .. {m['test_window']['window_end']} ({N_TEST_TRANSITIONS} transitions, last usable decision "
        f"{m['test_window']['last_usable_decision']}); modelled costs of 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale; 365 periods/year; deterministic argmax; every row reconstructed and "
        "verified from persisted evidence (identity chain, curve hashes, full timeline, metrics recomputed from the curves, R2 decomposition).",
        "",
        "Descriptive results for the frozen TEST period only. No statistical significance test was pre-specified and none is "
        "computed. No profitability claim. No generalisation beyond the frozen period. No post-TEST retuning. All 20 learned "
        "checkpoints are reported; none was selected, dropped or reweighted. R2 reward diagnostics are training-reward sums, never returns.",
        "",
        "## Per-seed table (20 learned rows)", "", _table(ps, ["experiment", "definition", "seed", "policy_name", "n_steps"] + FIN_COLUMNS, FMT), "",
        "R2 experiments — reward diagnostics (separate from financial metrics):", "",
        _table(ps[ps["experiment"].isin(R2_EXPERIMENTS)], ["experiment", "seed"] + list(DIAG_KEYS), FMT), "",
        "## Per-experiment summary (mean ± population std over the exact five seeds)", "", _tbl(report["summary"]), "",
        "## Final E1–E4 matrix", "", _tbl(report["matrix"]), "",
        "## Baselines (same TEST environment; identical through the E1, E2, E3 and E4 environments)", "",
        _table(report["baselines"], ["policy", "policy_kind", "seed", "n_steps"] + FIN_COLUMNS, FMT), "",
        "## Seed dispersion", "", _tbl(report["dispersion"]), "",
        "## RQ1 — learned policies vs baselines (descriptive)", "",
        "Whether learned policies show economically meaningful out-of-sample behaviour is read from the per-seed and baseline tables "
        "(return, Sharpe, drawdown, turnover and cost drag relative to always CASH, buy-and-hold and the random policy), per seed and "
        "per configuration; no claim beyond the frozen period.", "",
        "## RQ2 — O2 vs O1 under both rewards (descriptive, paired by seed)", "", _tbl(_pair(ps, "E1", "E2")), "", _tbl(_pair(ps, "E3", "E4")), "",
        "## RQ3 — R2 vs R1 under both observation spaces (descriptive, paired by seed)", "", _tbl(_pair(ps, "E1", "E3")), "", _tbl(_pair(ps, "E2", "E4")), "",
        "## Post-TEST rules", "", *[f"- {r}" for r in POST_TEST_RULES], "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=FINAL_TEST_ROOT)
    args = ap.parse_args(argv)
    report = build_report(args.root)
    md = format_markdown(report)
    out = Path(args.root) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "final_test_results.md").write_text(md, encoding="utf-8")
    report["per_seed"].to_csv(out / "final_test_per_seed.csv", index=False)
    report["matrix"].to_csv(out / "final_test_matrix.csv", index=False)
    report["baselines"].to_csv(out / "final_test_baselines.csv", index=False)
    report["dispersion"].to_csv(out / "final_test_seed_dispersion.csv", index=False)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
