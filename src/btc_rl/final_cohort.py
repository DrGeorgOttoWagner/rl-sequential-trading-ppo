"""
final_cohort.py — completion of the prospectively frozen final five-seed cohort (VALIDATION only).

    python -m btc_rl.final_cohort --experiment E1 --seed 31415      # one new run (gate first)
    python -m btc_rl.final_cohort --all                             # the eight required runs, sequentially, one gate

Governance (docs/methodology.md, amendment 2026-09-16; docs/evidence-provenance.md):

    FINAL_CANONICAL_SEEDS = (42, 123, 2026, 31415, 271828)
    ACCEPTED_SEEDS        = (42, 123, 2026)      accepted development runs, part of the final cohort, immutable
    NEW_SEEDS             = (31415, 271828)      the only seeds this module may train

This module is a HARNESS.  It defines no environment, observation, reward,
accounting, PPO configuration or metric: each new run is the accepted
experiment runner (``ppo_e1.run_e1`` .. ``ppo_e4.run_e4``) with its own
accepted ``E?Config`` and run group, differing from the accepted runs only in
the seed and in the artifacts root, which is the separate namespace

    artifacts/ppo/final5/<run_group>/<seed>/

so that the accepted three-seed canonical reports (which read
``artifacts/ppo/<run_group>/``) never see the new seeds and keep working
unchanged.  Every accepted E1/E2/E3/E4 run file and implementation file is
anchored by SHA-256 and verified before and after training; the final5 root
may never equal, descend into or contain an accepted artifact directory; a
new run refuses to overwrite an existing checkpoint; seeds 42/123/2026 and
any seed outside the frozen cohort are refused before anything is built.

Each new run additionally writes ``final_cohort.json`` next to the accepted
metadata: experiment, seed, the frozen cohort, checkpoint SHA-256, the
provenance of the reused O1/O2/R1/R2 components, the canonical-spec
verification of the record against the accepted experiment specification,
and ``TEST_EVALUATED = false``.  TEST is never built, evaluated or read.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ppo_e1, ppo_e2, ppo_e3, ppo_e4
from . import ppo_e1_report as rep1
from . import ppo_e2_report as rep2
from . import ppo_e3_report as rep3
from . import ppo_e4_report as rep4
from .config import FoundationConfig, load_foundation_config
from .data import load_dataset
from .ppo_e1 import DEFAULT_ARTIFACTS_ROOT, FULL_BUDGET_TIMESTEPS, PPOHyperparameters, sha256_of, write_json
from .ppo_e2 import E1_FROZEN_ARTIFACTS
from .ppo_e3 import E2_FROZEN_ARTIFACTS, artifact_status, verify_e3_output_dir
from .ppo_e4 import E3_FROZEN_ARTIFACTS, E3_FROZEN_CORE_SOURCE_HASHES, KEY_SOURCE_FILES_E4
from .preflight import PACKAGE_ROOT

STUDY_STAGE = "final-cohort"
FINAL_CANONICAL_SEEDS: tuple[int, ...] = (42, 123, 2026, 31415, 271828)
ACCEPTED_SEEDS: tuple[int, ...] = (42, 123, 2026)
NEW_SEEDS: tuple[int, ...] = (31415, 271828)
assert FINAL_CANONICAL_SEEDS == ACCEPTED_SEEDS + NEW_SEEDS and len(set(FINAL_CANONICAL_SEEDS)) == 5
assert ACCEPTED_SEEDS == ppo_e1.DEVELOPMENT_SEEDS

EXPERIMENTS: tuple[str, ...] = ("E1", "E2", "E3", "E4")
RUN_GROUPS: dict[str, str] = {"E1": "e1", "E2": "e2", "E3": "e3", "E4": "e4"}
DEFINITIONS: dict[str, str] = {"E1": "O1 + R1", "E2": "O2 + R1", "E3": "O1 + R2", "E4": "O2 + R2"}
FINAL5_ROOT: Path = DEFAULT_ARTIFACTS_ROOT / "final5"
FINAL5_REPORTS_DIR: Path = FINAL5_ROOT / "reports"
TEST_EVALUATED = False

# SHA-256 anchors of the accepted E4 artifacts (closure, commit db455ed5; checkpoint hashes equal those in
# the independent review record).  Taken before any final-cohort training.
E4_FROZEN_ARTIFACTS: dict[int, dict[str, str]] = {
    42: {
        "model.zip": "676aeba16a9040acc7c9952c5823ce5c5a00988843843c612d86939159fa0b83",
        "metadata.json": "247aa8ebb2a2c9a42c06a64b767621fc047268b3013cf7545c477ded1a6dee36",
        "validation_curve.csv": "7db3981cbc81e4e5a58dfd42e4c0091fabf37b2338ed399fc84c3ea7072287cd",
        "validation_metrics.json": "1074725631ee5d81fd9a949b47861fbe7835e585535d74b9805fc4b33eadeda6",
    },
    123: {
        "model.zip": "542f421804c701e2dd49defc5b637378e2ce455a7450b66e1a965fd2a5347b85",
        "metadata.json": "60133cdb7376ba316c3c3ca4a78a6e3a7908eea54326cff032a6f9f429754fc6",
        "validation_curve.csv": "bc98454c78da1b426a596a90230c6138fa3e36cdc67fdd1e49777d52cf25e22c",
        "validation_metrics.json": "ebf682c999cfda460f2075702a65a0d0def3d59327b23e14ac1d4480da1c4863",
    },
    2026: {
        "model.zip": "27747348676da025fd8f290eaefe7a5f7b2b4e816249ce00ac62a21379de8286",
        "metadata.json": "62a2949ccf8ccf4bb40f792d6eb49fdb0f1d9acf2cfc01c7e3af14fd95fb7342",
        "validation_curve.csv": "a832d295f8ffba5aaee256e4ff1e7e2ab5ff564aa5314fe6dec00aab3d205994",
        "validation_metrics.json": "7f4fd2769b92ccf53471ee2059be28946a996c1c1165d8f55edf31f218c1c324",
    },
}
ACCEPTED_ARTIFACT_ANCHORS: dict[str, dict[int, dict[str, str]]] = {
    "E1": E1_FROZEN_ARTIFACTS, "E2": E2_FROZEN_ARTIFACTS, "E3": E3_FROZEN_ARTIFACTS, "E4": E4_FROZEN_ARTIFACTS,
}
# The accepted implementation files (trainers and canonical reports) as of the E4 closure commit db455ed5.  The new
# seeds must be trained by exactly this code.
ACCEPTED_IMPLEMENTATION_HASHES: dict[str, str] = {
    "src/btc_rl/ppo_e1.py": "5c44cf54de1244417ec55dd4a4e747d75393409e9c843ee3a059ef4f0474f9e6",
    "src/btc_rl/ppo_e2.py": "b3702f68bb6d37e1f3f0a2a81c99331b5c6ee2f5f487f77effa8f7a698a221c7",
    "src/btc_rl/ppo_e3.py": "c098652eb60aefd1e392dd5fd277c4da98ca384a53180d52f11e9650786c2cd8",
    "src/btc_rl/ppo_e4.py": "ed487e58d5c83f55fc801063a5e9e825e3bdd51fc56321da5f3aab94c48d7f66",
    "src/btc_rl/ppo_e1_report.py": "f8dcee0fa36e952c7149dd1c0dffb67368e8aec2ae5af66989e4d01a6270d449",
    "src/btc_rl/ppo_e2_report.py": "75b251077172e778572cd5d3788cda21fb987e44b3c9749bad06b78694ead196",
    "src/btc_rl/ppo_e3_report.py": "b57e89beafc0d59b8711af0053095b80411f18a122356e9c6bd35f0d1e656c94",
    "src/btc_rl/ppo_e4_report.py": "b4d5dac6e73cfb43a9f898d21675b0118b7f3a8127a57d6c75b3eb65590dbc33",
}
# E1 core + config/lock + O2 source + R2 source, as recorded by the accepted runs (12 files)
ACCEPTED_CORE_SOURCE_HASHES: dict[str, str] = dict(E3_FROZEN_CORE_SOURCE_HASHES)
COMPONENT_SOURCES: dict[str, str] = {
    "O1": "src/btc_rl/observations.py", "O1_scale": "src/btc_rl/scaling.py", "O2": "src/btc_rl/observations_o2.py",
    "R1_accounting": "src/btc_rl/env.py", "R2": "src/btc_rl/reward_r2.py",
}
PROTECTED_ACCEPTED_DIRS: tuple[Path, ...] = tuple((DEFAULT_ARTIFACTS_ROOT / g).resolve() for g in RUN_GROUPS.values())


class FinalCohortError(RuntimeError):
    """A governance rule of the final five-seed cohort was violated."""


# ───────────────────────────────────────────── seed authorisation
def authorize_new_seed(seed: Any) -> int:
    """Only 31415 and 271828 may be trained.  Accepted seeds are refused as retraining; anything else as an unfrozen seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FinalCohortError(f"seed {seed!r} is not an integer")
    if seed in ACCEPTED_SEEDS:
        raise FinalCohortError(f"seed {seed} is an accepted development seed; retraining it is refused "
                               f"(accepted artifacts are immutable)")
    if seed not in NEW_SEEDS:
        raise FinalCohortError(f"seed {seed} is not in the prospectively frozen final cohort {FINAL_CANONICAL_SEEDS}; "
                               f"only {NEW_SEEDS} may be trained")
    return int(seed)


def require_experiment(experiment: Any) -> str:
    if experiment not in EXPERIMENTS:
        raise FinalCohortError(f"unknown experiment {experiment!r}; expected one of {EXPERIMENTS}")
    return str(experiment)


# ───────────────────────────────────────────── accepted-artifact immutability
def verify_accepted_artifacts_unchanged(artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT) -> dict[str, Any]:
    out = {}
    for label, anchors in ACCEPTED_ARTIFACT_ANCHORS.items():
        status = artifact_status(anchors, artifacts_root, RUN_GROUPS[label])
        if not status["all_present"]:
            raise FinalCohortError(f"{label} accepted artifacts missing: {status['missing']}")
        if not status["unchanged"]:
            raise FinalCohortError(f"{label} accepted artifacts differ from the frozen anchors: {status['changed']}")
        out[label] = status
    return out


def verify_accepted_sources_unchanged(root: Path = PACKAGE_ROOT) -> dict[str, str]:
    """Core/config/dependency files (12) and the accepted trainer/report implementations (8) must hash as accepted."""
    hashes = {}
    changed = []
    for label, digest in ACCEPTED_CORE_SOURCE_HASHES.items():
        rel = KEY_SOURCE_FILES_E4[label]
        now = sha256_of(Path(root) / rel)
        hashes[rel] = now
        if now != digest:
            changed.append(rel)
    for rel, digest in ACCEPTED_IMPLEMENTATION_HASHES.items():
        now = sha256_of(Path(root) / rel)
        hashes[rel] = now
        if now != digest:
            changed.append(rel)
    if changed:
        raise FinalCohortError(f"accepted sources changed since the E1–E4 closures: {sorted(changed)}")
    return hashes


def verify_recorded_sources_anchored(meta: dict[str, Any]) -> dict[str, str]:
    """
    the training provenance RECORDED in a new-seed
    record (its ``source.source_files`` map, captured before learning and
    verified stable after it) must hash every accepted core/config/dependency
    file and every accepted trainer/report file to the frozen anchors.  This
    is independent of the current working tree: a correct tree with a record
    that claims different training sources is rejected.
    """
    files = meta.get("source", {}).get("source_files")
    if not isinstance(files, dict) or not files:
        raise FinalCohortError(f"seed {meta.get('seed')}: record carries no source file map")
    expected: dict[str, str] = {KEY_SOURCE_FILES_E4[label]: digest for label, digest in ACCEPTED_CORE_SOURCE_HASHES.items()}
    expected.update(ACCEPTED_IMPLEMENTATION_HASHES)
    bad = {rel: (files.get(rel, "<missing>"), digest) for rel, digest in expected.items() if files.get(rel) != digest}
    if bad:
        raise FinalCohortError(
            f"seed {meta.get('seed')}: recorded training provenance does not match the frozen accepted source anchors: "
            + "; ".join(f"{rel} recorded {str(rec)[:12]} != accepted {dig[:12]}" for rel, (rec, dig) in sorted(bad.items()))
        )
    return {rel: files[rel] for rel in expected}


def verify_final5_root(root: Path, protected: tuple[Path, ...] | None = None) -> Path:
    """The final5 namespace must be disjoint from the accepted artifact directories (and not the accepted root)."""
    protected = PROTECTED_ACCEPTED_DIRS if protected is None else protected
    try:
        resolved = verify_e3_output_dir(root, protected)
    except ValueError as exc:
        raise FinalCohortError(str(exc)) from exc
    if resolved == DEFAULT_ARTIFACTS_ROOT.resolve():
        raise FinalCohortError("the final5 root must be a separate namespace, not the accepted artifacts root")
    return resolved


def verify_final_destination(destination: Path, protected: tuple[Path, ...] | None = None) -> Path:
    """
    the ACTUAL resolved run directory the accepted
    runner will use (``<root>/<group>/<seed>``), resolved through every
    existing symlink and ``..`` component at call time, must not equal,
    descend into or contain any accepted E1/E2/E3/E4 artifact directory, and
    no path component below the root may itself be a symlink.  Pure: creates
    nothing.  Called at configuration time and again immediately before the
    dataset loader and before the runner, so a filesystem change after
    configuration (e.g. ``final5/e1 -> accepted/e1``) is caught.
    """
    protected = PROTECTED_ACCEPTED_DIRS if protected is None else protected
    dest = Path(destination)
    resolved = dest.resolve()
    for prot in protected:
        if resolved == prot or resolved.is_relative_to(prot) or prot.is_relative_to(resolved):
            raise FinalCohortError(
                f"final-cohort destination {dest} resolves to {resolved}, which aliases the protected accepted "
                f"artifacts {prot}; the run is refused before any loader, training or write"
            )
    # the group and seed components below the root must be real directories (or absent), never symlinks
    for component in (dest, dest.parent):
        if component.is_symlink():
            raise FinalCohortError(f"final-cohort destination component {component} is a symlink; refused")
    if resolved == DEFAULT_ARTIFACTS_ROOT.resolve() or DEFAULT_ARTIFACTS_ROOT.resolve().is_relative_to(resolved):
        raise FinalCohortError(f"final-cohort destination {resolved} contains the accepted artifacts root; refused")
    return resolved


# ───────────────────────────────────────────── the accepted runners, unchanged
_RUNNERS: dict[str, tuple[Any, Any, Any]] = {   # experiment -> (Config class, run function, canonical report module)
    "E1": (ppo_e1.E1Config, ppo_e1.run_e1, rep1),
    "E2": (ppo_e2.E2Config, ppo_e2.run_e2, rep2),
    "E3": (ppo_e3.E3Config, ppo_e3.run_e3, rep3),
    "E4": (ppo_e4.E4Config, ppo_e4.run_e4, rep4),
}


def make_config(experiment: str, seed: int, artifacts_root: Path = FINAL5_ROOT,
                total_timesteps: int = FULL_BUDGET_TIMESTEPS, hyperparameters: PPOHyperparameters | None = None):
    """The accepted experiment's own config class, own run group, new seed, final5 root."""
    experiment = require_experiment(experiment)
    seed = authorize_new_seed(seed)
    root = verify_final5_root(Path(artifacts_root))
    cls = _RUNNERS[experiment][0]
    kwargs: dict[str, Any] = {"seed": seed, "total_timesteps": total_timesteps, "artifacts_root": root,
                              "run_group": RUN_GROUPS[experiment]}
    if hyperparameters is not None:
        kwargs["hyperparameters"] = hyperparameters
    config = cls(**kwargs)
    verify_final_destination(config.artifacts_dir)      # the actual <root>/<group>/<seed> destination, not only the root
    return config


def component_provenance(root: Path = PACKAGE_ROOT) -> dict[str, dict[str, str]]:
    return {name: {"path": rel, "sha256": sha256_of(Path(root) / rel)} for name, rel in COMPONENT_SOURCES.items()}


def train_final_seed(experiment: str, seed: int, artifacts_root: Path = FINAL5_ROOT, frame=None,
                     cfg: FoundationConfig | None = None, dataset_report: dict[str, Any] | None = None,
                     preflight: dict[str, Any] | None = None, total_timesteps: int = FULL_BUDGET_TIMESTEPS,
                     hyperparameters: PPOHyperparameters | None = None, require_canonical: bool = True,
                     write_artifacts: bool = True) -> dict[str, Any]:
    """
    Train ONE new final-cohort seed with the accepted runner of ``experiment``
    and write the ``final_cohort.json`` sidecar.  Refuses accepted seeds,
    unfrozen seeds, aliased roots, changed accepted sources/artifacts and an
    existing checkpoint at the destination.  With ``require_canonical`` the
    record must satisfy the accepted experiment's canonical specification
    (everything except the seed set); the sidecar records the outcome.
    """
    experiment = require_experiment(experiment)
    config = make_config(experiment, seed, artifacts_root, total_timesteps, hyperparameters)
    verify_accepted_sources_unchanged()
    if write_artifacts and (config.artifacts_dir / "model.zip").exists():
        raise FinalCohortError(f"{config.artifacts_dir}: a checkpoint already exists; final-cohort runs are never overwritten")
    _, run, rep = _RUNNERS[experiment]
    verify_final_destination(config.artifacts_dir)      # F1: re-resolved immediately before the loader ...
    cfg = cfg or load_foundation_config()
    if frame is None:
        frame, dataset_report = load_dataset(cfg.dataset)
    destination = verify_final_destination(config.artifacts_dir)   # ... and again immediately before the runner
    if config.artifacts_dir.resolve() != destination:
        raise FinalCohortError("final-cohort destination changed between verification and the runner call")
    res = run(config, frame=frame, cfg=cfg, dataset_report=dataset_report, preflight=preflight,
              write_artifacts=write_artifacts)
    if write_artifacts and res.artifacts_dir.resolve() != destination:
        raise FinalCohortError(f"runner wrote to {res.artifacts_dir}, not the verified destination {destination}")
    meta = res.metadata
    spec_ok: bool | None
    spec_error = None
    try:
        rep.verify_canonical_spec(meta)
        spec_ok = True
    except rep1.ReportIntegrityError as exc:
        spec_ok = False
        spec_error = str(exc)
        if require_canonical:
            raise FinalCohortError(f"{experiment} seed {seed}: record does not match the accepted specification: {exc}") from exc
    model_sha = sha256_of(res.artifacts_dir / "model.zip") if write_artifacts else None
    sidecar = {
        "study_stage": STUDY_STAGE,
        "experiment": experiment,
        "definition": DEFINITIONS[experiment],
        "seed": int(seed),
        "final_canonical_seeds": list(FINAL_CANONICAL_SEEDS),
        "accepted_seeds": list(ACCEPTED_SEEDS),
        "new_seeds": list(NEW_SEEDS),
        "run_group": RUN_GROUPS[experiment],
        "artifacts_root": str(Path(artifacts_root).resolve()),
        "runner": f"{run.__module__}.{run.__name__}",
        "config_class": type(config).__name__,
        "git": meta["git"],
        "source_fingerprint": meta["source"]["source_fingerprint"],
        "dataset_sha256": meta["dataset"]["sha256"],
        "total_timesteps_requested": meta["total_timesteps_requested"],
        "total_timesteps_trained": meta["total_timesteps_trained"],
        "model_sha256": model_sha,
        "component_provenance": component_provenance(),
        "accepted_core_sources_verified": True,
        "canonical_spec_verified": spec_ok,
        "canonical_spec_error": spec_error,
        "validation_metrics": {k: v for k, v in res.validation.metrics.items()
                               if k in ("total_return", "final_equity", "sharpe_ratio", "max_drawdown", "n_entries",
                                        "n_legs", "exposure", "total_cost_fraction", "final_position",
                                        "cumulative_financial_r1", "cumulative_turnover_penalty",
                                        "cumulative_drawdown_penalty", "cumulative_reward_r2")},
        "evaluation_split": "validation",
        "TEST_EVALUATED": TEST_EVALUATED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if write_artifacts:
        write_json(res.artifacts_dir / "final_cohort.json", sidecar)
    return {"result": res, "sidecar": sidecar, "config": config}


REQUIRED_RUNS: tuple[tuple[str, int], ...] = tuple((e, s) for e in EXPERIMENTS for s in NEW_SEEDS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="train the frozen final-cohort seeds 31415 / 271828 with the accepted E1–E4 runners.")
    ap.add_argument("--experiment", choices=EXPERIMENTS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true", help="the eight required runs, sequentially, one gate")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--artifacts-root", type=Path, default=FINAL5_ROOT)
    args = ap.parse_args(argv)
    runs = list(REQUIRED_RUNS) if args.all else [(args.experiment, args.seed)]
    if not args.all and (args.experiment is None or args.seed is None):
        ap.error("--experiment and --seed are required unless --all")
    for e, s in runs:
        require_experiment(e)
        authorize_new_seed(s)
    verify_accepted_artifacts_unchanged()
    verify_accepted_sources_unchanged()
    preflight = None if args.skip_preflight else ppo_e1._preflight_or_die()
    cfg = load_foundation_config()
    frame, report = load_dataset(cfg.dataset)
    for e, s in runs:
        out = train_final_seed(e, s, args.artifacts_root, frame=frame, cfg=cfg, dataset_report=report, preflight=preflight)
        m = out["sidecar"]["validation_metrics"]
        print(f"{e} seed={s} trained={out['sidecar']['total_timesteps_trained']} -> {out['result'].artifacts_dir} | "
              f"model {out['sidecar']['model_sha256'][:12]} | spec ok={out['sidecar']['canonical_spec_verified']} | "
              f"VALIDATION return={m['total_return']:+.4f} sharpe={m['sharpe_ratio']:.3f} max_dd={m['max_drawdown']:+.4f} "
              f"legs={m['n_legs']} | TEST_EVALUATED={out['sidecar']['TEST_EVALUATED']}")
    verify_accepted_artifacts_unchanged()
    print("accepted E1/E2/E3/E4 artifacts verified unchanged after training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
