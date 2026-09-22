"""
final_test.py — the prospective ONE-TIME final TEST harness.
BUILD ONLY: TEST is not opened here; no authorization, harness anchors, run state or result exists.

    python -m btc_rl.final_test --verify-cohort               # read-only: bind the 20 frozen checkpoints (no TEST)
    python -m btc_rl.final_test --status                      # read-only: gate / run-state / result status (no TEST)
    python -m btc_rl.final_test --preflight-open              # read-only: every pre-opening prerequisite except the authorization
    python -m btc_rl.final_test --print-harness-anchors A     # read-only: the anchor record for reviewed commit A (printed, never written)
    python -m btc_rl.final_test --open-test                   # the one-shot run; REFUSED unless the human gate exists

Frozen TEST protocol (docs/methodology.md, docs/evidence-provenance.md)
---------------------------------------------------------------------------------------------------
E1 = O1+R1, E2 = O2+R1, E3 = O1+R2, E4 = O2+R2 in that order; seeds (42, 123, 2026, 31415, 271828) in that order;
exactly 20 learned checkpoints bound by SHA-256; the frozen TEST split 2021-08-30 .. 2026-02-04 (usable decisions
2021-08-30 .. 2026-02-02, 1618 transitions, last mark 2026-02-04, the daily decision / fill / mark grid frozen by
hash); 0.10% fee + 0.05% slippage allowance per trade; 365 periods/year; 0 = CASH / 1 = LONG; deterministic argmax; accepted timing / accounting /
observation / reward / scaler implementations, unchanged.  Baselines: always CASH, buy-and-hold, random seeds 0–4
(7 canonical identities × 4 environments).  All 20 proceed; nothing is selected, filtered or retrained.

Execution confinement
---------------------------------------
There is exactly one episode dispatcher, ``_dispatch_certified_episode``; it runs only under a ``CertifiedRoute``
of kind ``AUTHORIZED_TEST`` (constructed by ``_authorized_test_route``: canonical production context, active
authorization, active plan item, certified TEST environment, bound checkpoint) or ``VALIDATION_CERTIFIED``
(constructed by ``_validation_certified_route``: the accepted VALIDATION window only, TEST dates rejected).  The
dispatcher re-establishes the route's proof at the lowest runner boundary; a constructed route object proves nothing.
Contexts carry a kind: ``PRODUCTION_FINAL_TEST`` (only ``production_context()``, sealed, canonical root
``artifacts/ppo/final_test``) or ``SYNTHETIC_TEST_FIXTURE`` (``test_only_context``: disposable roots outside the
package / artifacts tree, synthetic evaluators; ``_real_item_evaluator`` and any callable of this module are
refused).  Every real TEST boundary (frame extraction, environment construction, adapter binding, each prediction,
evaluation, dispatch, the real evaluator) validates the context AT USE TIME and refuses synthetic contexts and any
mutated / manually constructed production context.  Every TEST evaluation is bound to the single IN_PROGRESS item
of the durable plan (``_require_active_item``); the adapter revalidates authorization AND active item before every
prediction.

Durable run state, transactional persistence, reconciliation, ownership
----------------------------------------------------------------------------------
``RUN_STATE.json`` is published transactionally (temp file + fsync + no-overwrite hard link + directory fsync): no
partial JSON can ever appear at the state path, and exclusivity is kept.  An exclusive crash-releasing OS advisory
lock (``RUN.lock``, ``flock``) is held by the single executor from opening / resumption through evaluation,
persistence and finalization; a second live executor is refused immediately; the lock is never scientific state.
Each item: durable IN_PROGRESS → evaluate → evidence to a temporary item directory (fsync) → hashes → atomic
promotion → COMMITTED with the hashes.  Recovery rule (prospectively fixed): COMMITTED items are re-verified, never
re-evaluated; PENDING items execute; for an IN_PROGRESS item, the harness FIRST inspects the promoted evidence
directory: if it exists and is complete and valid (identities, recomputed hashes, full timeline, recomputed metrics,
R2 decomposition) it is adopted durably as COMMITTED with the event ``PROMOTED_EVIDENCE_RECONCILED`` and no model is
evaluated; if it exists but is partial / invalid the run is ABORTED (``PROMOTED_EVIDENCE_INVALID``; nothing is deleted
or overwritten, no re-prediction); only when no promoted evidence exists does the single deterministic
``RECOVERY_RETRY`` occur, and a second interruption of the same item aborts the run.  Finalization is deterministic
and prediction-free: ``FINAL_TEST_MANIFEST.json`` + ``FINAL_TEST_COMPLETE`` (manifest SHA-256), read-only results.

Frozen runtime, data and source identity; two-commit anchors
-----------------------------------------------------------------------
Production opening verifies: exact CPython / .python-version and locked dependencies, the frozen foundation
configuration, the frozen dataset (SHA-256) and its TEST grid, all accepted source anchors, the complete final5
closure (hardened verifier + frozen closure-file anchors + recomputed metrics from the persisted VALIDATION curves +
the authority documents and commits), and the reviewed harness identity under the two-commit rule:
reviewed code commit A (anchors record A, A's tree, and the blob hashes of every anchored file at A), execution
HEAD B with exactly one parent A, ``git diff A..B`` containing only the anchors file, every anchored file at B equal
to its hash at A and to the working file, clean tree.  No later descendant is accepted.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import ppo_e1, ppo_e2, ppo_e3, ppo_e4
from .config import DEFAULT_CONFIG_PATH, FoundationConfig, load_foundation_config
from .costs import CONSERVATIVE_COST
from .data import load_dataset
from .env import BtcUsdtTradingEnv, make_split_env
from .evaluation import EpisodeResult, compute_metrics, run_episode
from .final_cohort import (
    ACCEPTED_ARTIFACT_ANCHORS,
    ACCEPTED_CORE_SOURCE_HASHES,
    ACCEPTED_IMPLEMENTATION_HASHES,
    ACCEPTED_SEEDS,
    DEFINITIONS,
    EXPERIMENTS,
    FINAL5_ROOT,
    FINAL_CANONICAL_SEEDS,
    NEW_SEEDS,
    RUN_GROUPS,
    verify_accepted_artifacts_unchanged,
    verify_accepted_sources_unchanged,
)
from .observations import ObservationConfig, observation_space
from .observations_o2 import O2ObservationConfig, o2_observation_space
from .policies import AlwaysFlat, AlwaysLong, Policy, RandomPolicy
from .ppo_e1 import CANONICAL_E1_SPEC, DEFAULT_ARTIFACTS_ROOT, PPOHyperparameters, load_e1_model, sha256_of, sha256_text
from .ppo_e2 import BtcUsdtTradingEnvO2
from .ppo_e3 import R2_CURVE_COLUMNS, run_episode_e3
from .ppo_e4 import KEY_SOURCE_FILES_E4, BtcUsdtTradingEnvO2R2
from .preflight import LOCK_PATH, PACKAGE_ROOT, check_locked_dependencies, check_runtime
from .reward_r2 import FROZEN_R2, BtcUsdtTradingEnvR2
from .splits import FROZEN_SPLITS, SETTLEMENT_BARS, SplitWindow, usable_decision_index_range

STUDY_STAGE = "final-test"
TEST_OPENED: bool = False
AUTHORIZATION_STATEMENT = "ONE-TIME TEST OPENING AUTHORIZED"

# ───────────────────────────────────────────── frozen scope
TEST_SPLIT = "test"
TEST_WINDOW: SplitWindow = FROZEN_SPLITS[TEST_SPLIT]
VALIDATION_WINDOW: SplitWindow = FROZEN_SPLITS["validation"]
N_VALIDATION_TRANSITIONS = 239
FROZEN_DATASET_SHA256 = CANONICAL_E1_SPEC["dataset"]["sha256"]
TEST_WINDOW_IDENTITY: dict[str, Any] = {
    "split": TEST_SPLIT, "window_start": "2021-08-30", "window_end": "2026-02-04",
    "first_usable_decision": "2021-08-30", "last_usable_decision": "2026-02-02", "n_usable_decisions": 1618,
    "last_mark_open": "2026-02-04", "bps_per_leg": 15.0, "periods_per_year": 365, "initial_equity": 1.0,
    "actions": {"0": "CASH", "1": "LONG"}, "policy": "deterministic argmax of the final checkpoint",
    "dataset_sha256": FROZEN_DATASET_SHA256,
}
N_TEST_TRANSITIONS = 1618
assert str(TEST_WINDOW.start.date()) == TEST_WINDOW_IDENTITY["window_start"]
assert str(TEST_WINDOW.end.date()) == TEST_WINDOW_IDENTITY["window_end"]
# The frozen TEST timeline: daily decision grid, fill = next open, mark = the open after (SETTLEMENT_BARS = 2).
# Derived from the frozen dataset's date index only (no prices, no policy) and frozen by hash before the protocol runs.
FROZEN_TEST_GRID_SHA256 = "c25d86a4d572d0d254e24dc4325eaa1b0b2aaab7e3c8a4e2f101680ba8f24bfc"
ONE_LEG_COST = 0.0015
OBSERVATION_OF: dict[str, str] = {"E1": "O1", "E2": "O2", "E3": "O1", "E4": "O2"}
REWARD_OF: dict[str, str] = {"E1": "R1", "E2": "R1", "E3": "R2", "E4": "R2"}
ENV_TYPE_OF: dict[str, type] = {"E1": BtcUsdtTradingEnv, "E2": BtcUsdtTradingEnvO2, "E3": BtcUsdtTradingEnvR2, "E4": BtcUsdtTradingEnvO2R2}
R2_EXPERIMENTS: tuple[str, ...] = ("E3", "E4")
RANDOM_BASELINE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
BASELINE_LABELS: tuple[str, ...] = ("CASH", "BUY_AND_HOLD") + tuple(f"RANDOM_{s}" for s in RANDOM_BASELINE_SEEDS)
# canonical baseline identity: label -> (policy kind, random seed, accepted policy name)
BASELINE_IDENTITY: dict[str, tuple[str, int | None, str]] = {
    "CASH": ("always_flat", None, AlwaysFlat.name), "BUY_AND_HOLD": ("always_long", None, AlwaysLong.name),
    **{f"RANDOM_{s}": ("random", s, f"random_seed{s}") for s in RANDOM_BASELINE_SEEDS},
}

_FINAL5_CHECKPOINTS: dict[tuple[str, int], str] = {
    ("E1", 31415): "bb0a6f8a5988b044b828ebd6042362a7d4212421011c77d7d2c2974e14c32c00",
    ("E1", 271828): "e2ec8d7c640e6f06584dfa21386b6b3e25df7314b975df39f146603c3953db24",
    ("E2", 31415): "b7571483190534345942191952aeb437a5fe0e88e739c40fa0f92f884d4c23e1",
    ("E2", 271828): "f3236f5e6e048d5bdb74d3e559bda894f9fbadc31cf6d253661317dd7ea6a1b6",
    ("E3", 31415): "86caa1068db3dd60acd7f70a2249a41202c0a8bbcda3ab499648e411f64c1869",
    ("E3", 271828): "4ac816a1071693f70b59d9ded00bec13be52ceeb21a351be2eb8af9e36e2614a",
    ("E4", 31415): "6bc39bc643d76e743ff175a8e6db1d90982a72b11ba93388a1091d23366a0844",
    ("E4", 271828): "8616ec01318edfab14d4de48fc057b09496953d4d0cb3a51ab710aa4aa1c236a",
}
FROZEN_CHECKPOINTS: dict[tuple[str, int], str] = {}
for _e in EXPERIMENTS:
    for _s in FINAL_CANONICAL_SEEDS:
        FROZEN_CHECKPOINTS[(_e, _s)] = (ACCEPTED_ARTIFACT_ANCHORS[_e][_s]["model.zip"] if _s in ACCEPTED_SEEDS
                                       else _FINAL5_CHECKPOINTS[(_e, _s)])
assert len(FROZEN_CHECKPOINTS) == 20 and len(set(FROZEN_CHECKPOINTS.values())) == 20
COHORT_ORDER: tuple[tuple[str, int], ...] = tuple((e, s) for e in EXPERIMENTS for s in FINAL_CANONICAL_SEEDS)

# closure authority: the eight new-seed records' closure files, the final5 reports, the closure
# documents and the closure commits, frozen by hash (taken from the accepted, unchanged 136-file freeze).
FINAL5_CLOSURE_ANCHORS: dict[str, str] = {
    "e1/31415/metadata.json": "dc42bc256b36d4d72d923e6ce7e7c5a268ad9ee9e724dc0b19b5faf1131cb711",
    "e1/31415/validation_curve.csv": "e767b36b03ebb708a619bbf40ed164f1b3a9d82afeaf1ca1ae5c604ea57d20b1",
    "e1/31415/validation_metrics.json": "fcce81ffe01a2b65f82ab5382d89ce344deaaa0988194c8216ac2f988de91670",
    "e1/31415/final_cohort.json": "f3d12b539732e2e1958f8d835cb8fca65d8bd66ebef9a1df9a195e68419d375a",
    "e1/31415/train_log/progress.csv": "bcd22a512926efd7426d7fcde971cdae86e172d6f844f233335eae798d85804d",
    "e1/271828/metadata.json": "1c1c34a59a9c34e60d9fb98061689dc82e4913c4edc953bd9890120f6a672d6d",
    "e1/271828/validation_curve.csv": "71cc9bf043f07cb1084ef1e17fd8e093c0ecee26ee025083876ce6bc034e858c",
    "e1/271828/validation_metrics.json": "dbe7faaeadba6eb5df346d7b5439a9ef671ce9f0089c8f9b0a0d23532203a50b",
    "e1/271828/final_cohort.json": "68df766bf03d6152ffd5ea1b669a021a157452ba260b137638ae66b4e7f1fcfb",
    "e1/271828/train_log/progress.csv": "532c8cdb843e84e4053d9a8a5799310887eaaa4cadbc085cb7c01b6789bbcf92",
    "e2/31415/metadata.json": "003c4b9251fd3774c62e821a66ac392b54d14f74789dfd9a2f7c8f5679acf727",
    "e2/31415/validation_curve.csv": "412b9614acba1071aa6107355a72d60d621383712b3519cb49b298810f399611",
    "e2/31415/validation_metrics.json": "ca8679a57f24532017c12f34bdf722fba98bfb514fb8e2fd6ddeff74550f42ae",
    "e2/31415/final_cohort.json": "72893125259f94d00726e6e262715d5af339ac4ed173226b22daac91f8effe2e",
    "e2/31415/train_log/progress.csv": "a3f26e15fd5f7ae578e61f28e39b2d13efdf96c27b858c40ce414aef85e78afd",
    "e2/271828/metadata.json": "dcf7a8e3dc872810da0663df7674fbd5e466dff5bef19dbcc9b315b3fd439e15",
    "e2/271828/validation_curve.csv": "d17c8a2173544cf1ba47fe6fc97bb771d3226ebd31de3627838c747eb6f20940",
    "e2/271828/validation_metrics.json": "0ea56438c9c45f77666bbf0cbcb8d596dfae2aff56e4c74a8de2b086b48adb51",
    "e2/271828/final_cohort.json": "399156d22531077a0ebd9040cbcea0d313a17607383731b2ad67fed28f206eb5",
    "e2/271828/train_log/progress.csv": "9d90f40ad95f4e9f977c8cf59c48dba99cabfd731f6d769e7f1321bb4cea0663",
    "e3/31415/metadata.json": "36d188dc00f3adc973fbd96de4875e825e33b383640ffc85d294309faf9f5e1c",
    "e3/31415/validation_curve.csv": "8b2ad1325dd0cb43d00d17a67dce2fb6685036d6a0fee86408aa0bc15c907609",
    "e3/31415/validation_metrics.json": "c2fff0a895cd6cb30a5130f0ef39b151d1b792143bb8ad198f5d2c868b767fbd",
    "e3/31415/final_cohort.json": "bcac7ab939413072c7b463728bd88da0772c7d8b061c1f70cc19195d0cf183dc",
    "e3/31415/train_log/progress.csv": "77baf1a7b8ebb7dd337ac67fe4bb4d439b0f4a72d32a2ad04c8fa4df85700116",
    "e3/271828/metadata.json": "249ffe6370a3018aee3727c0815355d0b37f4ba647092222f5c9b5293e7039a5",
    "e3/271828/validation_curve.csv": "43bf830461fd8f80aeb001956ee9392a27f9a49f94c31860bbcfad40816faf1b",
    "e3/271828/validation_metrics.json": "7cedc498bf9184570fc2537b7fca321143b381237be0bfc4124952b267e65141",
    "e3/271828/final_cohort.json": "91ae62054adb44db1543799a14bb5ef39bfece4806f374a0613be606a8763f58",
    "e3/271828/train_log/progress.csv": "8b45e96cff94a9ba273080ea11844c3a97c83105efd7a5f9037916b889d75c01",
    "e4/31415/metadata.json": "942b5216d415487f140a8c2dab864e232127594a87193328b981799409d25d92",
    "e4/31415/validation_curve.csv": "5a045517cd4a8c5435e42ffef8a577d6cb088f1449b94c8cc0de30a653d576c3",
    "e4/31415/validation_metrics.json": "effb8d84a9c00ecb9ecb4f9a15e8d57990303f935e898ec8dcc22746d6b8f4d8",
    "e4/31415/final_cohort.json": "82ea8c708dae6d85102bb20ab43a24985fbe696fc18d149989f11933cbb9220c",
    "e4/31415/train_log/progress.csv": "4f4a59b4256348ca9af4b7926a4e9894d21a46843c84cefbade791ee40bf2b11",
    "e4/271828/metadata.json": "5e905c9fa6b452d3bda14136fe4094ac667adb4d597469f6c767d4b5da057493",
    "e4/271828/validation_curve.csv": "66326cdc5d6e47cef01c943c47e1a644a930197bdfa5b43ad6c0fade24f4f6b7",
    "e4/271828/validation_metrics.json": "e26c131fe28c14398e6e7666eea4ce286366d197af40bde1251c802010e10d64",
    "e4/271828/final_cohort.json": "2cda81328b0f2b93be9cf2da64b69e553d34097901016f36fa827e957fb6fa69",
    "e4/271828/train_log/progress.csv": "7d41013f9f771b83ef5b984d39a95c21260e902286d7020c0d6e3966957a3c75",
    "reports/final5_seed_dispersion.csv": "c1a361ddd4f05ca556fc1e24a04162edc1c100c8c186eabc6a78d3bdb4877ce9",
    "reports/final5_validation_baselines.csv": "f8c41c970fab86187c33b7567d58bc97cf48b5f0ab476a131048dfee4f33a4f2",
    "reports/final5_validation_matrix.csv": "96e27afa34c96e1770ae2fc4f023f98ae7bd6afe290ddcf864157eb09d4b0f26",
    "reports/final5_validation_per_seed.csv": "be4f8cc3f08ef00b4775815d648ac32ec41a593f85bbdb3a1f5db05fa4ede12f",
    "reports/final5_validation_results.md": "5aa824b08d28454f3c72b051865bef555dc93982a97b2345554937f05dd7b0d8",
}
CLOSURE_AUTHORITY_DOCUMENTS: dict[str, str] = {
    "authority/cohort-corrections.md": "08533c98c2762339cdb4b42928cdc353258f1acd807b8ba99fb8fb554aa35dff",
    "authority/validation-freeze.md": "0eb12d58ed747e7d113eb6afada226ee55752217cf7f5434adeb731c3262a9d1",
    "authority/cohort-review.md": "6da9d884f8559caae09517ea7d7488402744a7b44191cc05a15170df43fe2de8",
    "authority/cohort-closure-review.md": "9f89cc266b9b37adc4e87fe471dbd73c11a553ec34d9c956a319e11118d2f3b4",
}
CLOSURE_AUTHORITY_COMMITS: tuple[str, ...] = ("947c9055632d6157df27f76094fd090613a017cb",   # freeze commit
                                              "ce253e4307940d815783da123fced9e3af420487")   # record commit
VALIDATION_METRIC_KEYS: tuple[str, ...] = ("total_return", "final_equity", "sharpe_ratio", "max_drawdown", "n_entries",
                                           "n_legs", "exposure", "total_cost_fraction", "n_steps")

FINAL_TEST_ROOT: Path = DEFAULT_ARTIFACTS_ROOT / "final_test"          # the single canonical production root
AUTHORIZATION_NAME = "AUTHORIZATION.json"
RUN_STATE_NAME = "RUN_STATE.json"
MANIFEST_NAME = "FINAL_TEST_MANIFEST.json"
MARKER_NAME = "FINAL_TEST_COMPLETE"
LOCK_NAME = "RUN.lock"
HARNESS_ANCHORS_REL = "configs/final_test_harness_anchors.json"
HARNESS_ANCHORS_PATH: Path = PACKAGE_ROOT / HARNESS_ANCHORS_REL
HARNESS_ANCHORS_SCHEMA = "btc_rl.final_test_harness_anchors/2"
# Reviewed TEST-harness sources: the harness itself plus every directly imported runtime file that the accepted
# 20 anchors (12 core/config/dependency + 8 trainer/report files) do not already cover.
HARNESS_FILES: tuple[str, ...] = ("src/btc_rl/final_test.py", "src/btc_rl/final_test_report.py", "src/btc_rl/final_cohort.py",
                                  "src/btc_rl/final_cohort_report.py", "src/btc_rl/config.py", "src/btc_rl/policies.py",
                                  "src/btc_rl/preflight.py", "src/btc_rl/__init__.py", ".python-version")
ALLOWED_ADMINISTRATIVE_DELTA: tuple[str, ...] = (HARNESS_ANCHORS_REL,)
RELEVANT_SOURCE_PREFIXES: tuple[str, ...] = ("src/btc_rl/", "configs/", "pyproject.toml", "uv.lock", ".python-version")
FROZEN_RUNTIME_IDENTITY: dict[str, Any] = {
    "initial_equity": 1.0, "bps_per_leg": 15.0, "periods_per_year": 365, "python": "3.12.14", "implementation": "CPython",
    "foundation_config_sha256": ACCEPTED_CORE_SOURCE_HASHES["foundation_config"],
    "uv_lock_sha256": ACCEPTED_CORE_SOURCE_HASHES["uv_lock"], "pyproject_sha256": ACCEPTED_CORE_SOURCE_HASHES["pyproject_toml"],
}

PRODUCTION_FINAL_TEST = "PRODUCTION_FINAL_TEST"
SYNTHETIC_TEST_FIXTURE = "SYNTHETIC_TEST_FIXTURE"
AUTHORIZED_TEST = "AUTHORIZED_TEST"
VALIDATION_CERTIFIED = "VALIDATION_CERTIFIED"

POST_TEST_RULES: tuple[str, ...] = (
    "no feature / observation change (O1, O2, scalers frozen)",
    "no reward change (R1, R2 and the R2 coefficients frozen)",
    "no PPO / hyperparameter / architecture change",
    "no seed change; the cohort is exactly (42, 123, 2026, 31415, 271828) per configuration",
    "no retraining of any checkpoint",
    "no model selection followed by retuning",
    "no second TEST opening; one run id, one canonical root, one executor, one-shot read-only results",
    "interrupted IN_PROGRESS item: promoted valid evidence is reconciled (never re-evaluated); promoted invalid evidence "
    "aborts the run; otherwise exactly one deterministic RECOVERY_RETRY of the same item, then abort",
    "results are interpreted descriptively only, for the frozen TEST period, with no profitability or generalisation claim",
)

# Test seam for synthetic interruption probes (None in production; never influences results).
_INTERRUPTION_PROBE: Callable[[str], None] | None = None


class TestNotAuthorized(RuntimeError):
    """A learned-policy TEST boundary was reached without a currently valid one-time authorization / production context."""


class FinalTestError(RuntimeError):
    """A frozen-cohort, window, runtime-identity, evidence or one-shot rule of the final TEST harness was violated."""


# ───────────────────────────────────────────── durable helpers
def _probe(point: str) -> None:
    if _INTERRUPTION_PROBE is not None:
        _INTERRUPTION_PROBE(point)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=ppo_e1._json_default)


def _fsync_dir(d: Path) -> None:
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _fsync_file(p: Path) -> None:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_durable(path: Path, text: str, readonly: bool = False) -> None:
    """Rewrite: tmp + fsync + atomic replace + directory fsync.  Never exposes a partial file at ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    if readonly:
        os.chmod(tmp, 0o444)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _publish_exclusive(path: Path, text: str) -> None:
    """
    Transactional exclusive creation: complete JSON is written to a temp file in the same directory and fsynced;
    publication is a no-overwrite hard link (fails if ``path`` exists), then the temp name is removed and the
    directory fsynced.  A crash at any point leaves either nothing at ``path`` or the complete file; never partial JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text[: len(text) // 2])
        fh.flush()
        _probe("state_tmp_write")              # interruption here leaves a partial TEMP file only, never a partial state
        fh.write(text[len(text) // 2:])
        fh.flush()
        os.fsync(fh.fileno())
    _probe("state_before_publish")
    try:
        os.link(tmp, path)                       # atomic, no overwrite: exclusivity without exposing a partial file
    except FileExistsError as exc:
        os.unlink(tmp)
        raise FinalTestError(f"{path}: a run state already exists; a second opening is refused") from exc
    _probe("state_after_publish_before_dirsync")
    os.unlink(tmp)
    _fsync_dir(path.parent)


def _read_json(path: Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FinalTestError(f"{p}: not valid JSON ({exc})") from exc
    return obj if isinstance(obj, dict) else None


# ───────────────────────────────────────────── cohort binding (read-only, no TEST)
def checkpoint_path(experiment: str, seed: int, accepted_root: Path = DEFAULT_ARTIFACTS_ROOT,
                    final5_root: Path = FINAL5_ROOT) -> Path:
    if experiment not in EXPERIMENTS:
        raise FinalTestError(f"unknown experiment {experiment!r}")
    if seed not in FINAL_CANONICAL_SEEDS:
        raise FinalTestError(f"seed {seed} is not in the frozen final cohort {FINAL_CANONICAL_SEEDS}")
    root = Path(accepted_root) if seed in ACCEPTED_SEEDS else Path(final5_root)
    return root / RUN_GROUPS[experiment] / str(seed) / "model.zip"


def cohort_manifest() -> dict[str, Any]:
    return {
        "study_stage": STUDY_STAGE,
        "experiments": [{"experiment": e, "definition": DEFINITIONS[e], "observation": OBSERVATION_OF[e], "reward": REWARD_OF[e]}
                        for e in EXPERIMENTS],
        "seeds": list(FINAL_CANONICAL_SEEDS),
        "checkpoints": [{"experiment": e, "seed": s, "model_sha256": FROZEN_CHECKPOINTS[(e, s)]} for e, s in COHORT_ORDER],
        "test_window": TEST_WINDOW_IDENTITY,
        "test_grid_sha256": FROZEN_TEST_GRID_SHA256,
        "ppo_hyperparameters": PPOHyperparameters().to_json(),
        "baselines": list(BASELINE_LABELS),
        "post_test_rules": list(POST_TEST_RULES),
    }


def cohort_manifest_sha256() -> str:
    return sha256_text(_canonical_json(cohort_manifest()))


def _expected_component_paths(experiment: str) -> tuple[str, ...]:
    core = ("src/btc_rl/env.py", "src/btc_rl/observations.py", "src/btc_rl/scaling.py", "src/btc_rl/splits.py",
            "src/btc_rl/costs.py", "src/btc_rl/evaluation.py", "src/btc_rl/data.py", "configs/foundation.toml",
            "pyproject.toml", "uv.lock")
    if OBSERVATION_OF[experiment] == "O2":
        core += ("src/btc_rl/observations_o2.py",)
    if REWARD_OF[experiment] == "R2":
        core += ("src/btc_rl/reward_r2.py",)
    return core


def verify_checkpoint_record(experiment: str, seed: int, accepted_root: Path = DEFAULT_ARTIFACTS_ROOT,
                             final5_root: Path = FINAL5_ROOT) -> dict[str, Any]:
    """Bytes, metadata identity, observation / reward / PPO / dataset / costs / splits / provenance; closure identity."""
    path = checkpoint_path(experiment, seed, accepted_root, final5_root)
    if not path.exists():
        raise FinalTestError(f"missing checkpoint: {experiment} seed {seed} at {path}")
    digest = sha256_of(path)
    if digest != FROZEN_CHECKPOINTS[(experiment, seed)]:
        raise FinalTestError(f"{path}: checkpoint bytes {digest[:16]} are not the frozen {experiment} seed {seed} checkpoint")
    meta_path = path.parent / "metadata.json"
    if not meta_path.exists():
        raise FinalTestError(f"{path.parent}: metadata.json missing")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if meta.get("experiment") != experiment:
        problems.append(f"experiment {meta.get('experiment')!r} != {experiment}")
    if int(meta.get("seed", -1)) != seed:
        problems.append(f"seed {meta.get('seed')} != {seed}")
    if meta.get("observation", {}).get("definition") != OBSERVATION_OF[experiment]:
        problems.append(f"observation {meta.get('observation', {}).get('definition')!r} != {OBSERVATION_OF[experiment]}")
    if meta.get("reward", {}).get("definition") != REWARD_OF[experiment]:
        problems.append(f"reward {meta.get('reward', {}).get('definition')!r} != {REWARD_OF[experiment]}")
    if REWARD_OF[experiment] == "R2" and meta.get("reward", {}).get("coefficients") != FROZEN_R2.to_json():
        problems.append("R2 coefficients are not the frozen values")
    if meta.get("ppo_hyperparameters") != PPOHyperparameters().to_json():
        problems.append("PPO hyperparameters differ from the canonical configuration")
    if (meta.get("total_timesteps_requested"), meta.get("total_timesteps_trained")) != (200_000, 200_704):
        problems.append("training budget differs from 200,000 / 200,704")
    if meta.get("dataset", {}).get("sha256") != FROZEN_DATASET_SHA256:
        problems.append("dataset identity differs from the frozen dataset")
    costs = meta.get("costs", {})
    if costs.get("bps_per_leg") != 15.0 or costs.get("profile") != "conservative":
        problems.append("costs are not the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    if meta.get("forbidden_splits") != ["test"]:
        problems.append("record does not declare the test split forbidden for training/validation")
    for split, expected in (("train_split", CANONICAL_E1_SPEC["train_split"]), ("validation_split", CANONICAL_E1_SPEC["validation_split"])):
        if meta.get(split) != expected:
            problems.append(f"{split} differs from the frozen split record")
    files = meta.get("source", {}).get("source_files", {})
    anchors = {KEY_SOURCE_FILES_E4[label]: d for label, d in ACCEPTED_CORE_SOURCE_HASHES.items()}
    for rel in _expected_component_paths(experiment):
        if files.get(rel) != anchors[rel]:
            problems.append(f"recorded source {rel} is not the accepted anchor")
    if problems:
        raise FinalTestError(f"{experiment} seed {seed} ({path}): " + "; ".join(problems))
    if seed in ACCEPTED_SEEDS:
        for name, digest_expected in ACCEPTED_ARTIFACT_ANCHORS[experiment][seed].items():
            p = path.parent / name
            if not p.exists() or sha256_of(p) != digest_expected:
                raise FinalTestError(f"{experiment} seed {seed}: accepted closure file {name} missing or changed")
    else:
        from .final_cohort import FinalCohortError
        from .final_cohort_report import verify_new_seed_record

        try:
            verify_new_seed_record(experiment, seed, path.parent, meta)
        except FinalCohortError as exc:
            raise FinalTestError(f"{experiment} seed {seed}: final5 closure verification failed: {exc}") from exc
    return {"experiment": experiment, "seed": seed, "path": str(path), "model_sha256": digest,
            "observation": OBSERVATION_OF[experiment], "reward": REWARD_OF[experiment]}


def _seed_dirs(group_dir: Path) -> list[int]:
    out = []
    if group_dir.exists():
        for p in sorted(group_dir.iterdir()):
            if p.is_dir():
                try:
                    out.append(int(p.name))
                except ValueError:
                    continue
    return out


def verify_cohort(accepted_root: Path = DEFAULT_ARTIFACTS_ROOT, final5_root: Path = FINAL5_ROOT) -> dict[str, Any]:
    records, seen = [], {}
    for e, s in COHORT_ORDER:
        rec = verify_checkpoint_record(e, s, accepted_root, final5_root)
        if rec["model_sha256"] in seen:
            raise FinalTestError(f"duplicate checkpoint bytes: {e} seed {s} equals {seen[rec['model_sha256']]}")
        seen[rec["model_sha256"]] = (e, s)
        records.append(rec)
    for e in EXPERIMENTS:
        acc, new = _seed_dirs(Path(accepted_root) / RUN_GROUPS[e]), _seed_dirs(Path(final5_root) / RUN_GROUPS[e])
        if sorted(acc) != sorted(ACCEPTED_SEEDS):
            raise FinalTestError(f"{e}: accepted run group must contain exactly seeds {list(ACCEPTED_SEEDS)}, found {acc}")
        if sorted(new) != sorted(NEW_SEEDS):
            raise FinalTestError(f"{e}: final5 run group must contain exactly seeds {list(NEW_SEEDS)}, found {new}")
    if len(records) != 20:
        raise FinalTestError(f"expected 20 checkpoints, bound {len(records)}")
    return {"checkpoints": records, "cohort_manifest_sha256": cohort_manifest_sha256(), "n": len(records)}


def verify_final5_closure(final5_root: Path = FINAL5_ROOT, package_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """
    The accepted closure authority, bound without any replay: frozen closure-file anchors of the
    eight new records and the final5 reports, the hardened final5 verifier, the financial metrics recomputed from the
    PERSISTED VALIDATION curves (identity of the persisted evidence, not a fresh evaluation), the closure documents
    and the closure commits reachable from HEAD.
    """
    from .final_cohort import FinalCohortError
    from .final_cohort_report import verify_new_seed_record

    problems: list[str] = []
    for rel, digest in FINAL5_CLOSURE_ANCHORS.items():
        p = Path(final5_root) / rel
        if not p.exists() or sha256_of(p) != digest:
            problems.append(f"final5 closure file {rel} missing or changed")
    for rel, digest in CLOSURE_AUTHORITY_DOCUMENTS.items():
        p = Path(package_root) / rel
        if not p.exists() or sha256_of(p) != digest:
            problems.append(f"closure authority document {rel} missing or changed")
    for c in CLOSURE_AUTHORITY_COMMITS:
        try:
            if ppo_e1._git("cat-file", "-t", c, root=package_root) != "commit":
                problems.append(f"closure commit {c[:12]} is not a commit")
            subprocess.run(["git", "merge-base", "--is-ancestor", c, "HEAD"], cwd=package_root, check=True, capture_output=True)
        except (RuntimeError, subprocess.CalledProcessError):
            problems.append(f"closure commit {c[:12]} is not an ancestor of HEAD")
    if problems:
        raise FinalTestError("final5 closure authority not established: " + "; ".join(problems))
    out = {}
    for e in EXPERIMENTS:
        for s in NEW_SEEDS:
            d = Path(final5_root) / RUN_GROUPS[e] / str(s)
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
            curve = pd.read_csv(d / "validation_curve.csv", index_col=0, parse_dates=True)
            saved = json.loads((d / "validation_metrics.json").read_text(encoding="utf-8"))
            recomputed = compute_metrics(curve, 1.0, 365)
            for k in VALIDATION_METRIC_KEYS:
                a, b = recomputed.get(k), saved.get(k)
                if not ((a is None and b is None) or (a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12))):
                    raise FinalTestError(f"{e} seed {s}: persisted VALIDATION curve does not reproduce validation_metrics.json ({k})")
            if len(curve) != N_VALIDATION_TRANSITIONS:
                raise FinalTestError(f"{e} seed {s}: persisted VALIDATION curve is not the frozen 239-transition window")
            try:
                verify_new_seed_record(e, s, d, meta, recomputed_metrics=recomputed)
            except FinalCohortError as exc:
                raise FinalTestError(f"{e} seed {s}: final5 closure verification failed: {exc}") from exc
            out[f"{RUN_GROUPS[e]}/{s}"] = {"model_sha256": sha256_of(d / "model.zip"), "validation_curve_sha256": sha256_of(d / "validation_curve.csv")}
    return {"records": out, "closure_anchors": dict(FINAL5_CLOSURE_ANCHORS), "authority_documents": dict(CLOSURE_AUTHORITY_DOCUMENTS),
            "authority_commits": list(CLOSURE_AUTHORITY_COMMITS)}


# ───────────────────────────────────────────── runtime / data / source identity (before any TEST access)
def verify_runtime_config(cfg: FoundationConfig) -> dict[str, Any]:
    ppo_e1.verify_split_table(cfg.splits)
    cost = cfg.costs.get("conservative")
    problems = []
    if cost != CONSERVATIVE_COST or abs(cost.one_leg_fraction - ONE_LEG_COST) > 1e-15:
        problems.append("cost profile is not the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    if cfg.initial_equity != 1.0:
        problems.append(f"initial equity {cfg.initial_equity} != 1")
    if cfg.periods_per_year != 365:
        problems.append(f"periods per year {cfg.periods_per_year} != 365")
    if cfg.observation != ObservationConfig(lookback=10, return_scale=1.0):
        problems.append("foundation observation config is not the frozen O1 base config")
    if sha256_of(DEFAULT_CONFIG_PATH) != ACCEPTED_CORE_SOURCE_HASHES["foundation_config"]:
        problems.append("configs/foundation.toml is not the accepted anchored file")
    if problems:
        raise FinalTestError("runtime configuration is not the frozen protocol: " + "; ".join(problems))
    return runtime_identity()


def runtime_identity() -> dict[str, Any]:
    """The actual runtime identity; must equal ``FROZEN_RUNTIME_IDENTITY`` (checked by ``verify_runtime_environment``)."""
    return {"initial_equity": 1.0, "bps_per_leg": 15.0, "periods_per_year": 365,
            "python": ".".join(map(str, sys.version_info[:3])), "implementation": platform.python_implementation(),
            "foundation_config_sha256": sha256_of(DEFAULT_CONFIG_PATH), "uv_lock_sha256": sha256_of(LOCK_PATH),
            "pyproject_sha256": sha256_of(PACKAGE_ROOT / "pyproject.toml")}


def verify_runtime_environment() -> dict[str, Any]:
    """Exact CPython / .python-version, locked dependencies installed, and the frozen runtime identity."""
    try:
        check_runtime()
        check_locked_dependencies()
    except Exception as exc:  # noqa: BLE001
        raise FinalTestError(f"runtime environment is not the certified one: {exc}") from exc
    ident = runtime_identity()
    if ident != FROZEN_RUNTIME_IDENTITY:
        raise FinalTestError(f"runtime identity differs from the frozen identity: {ident} != {FROZEN_RUNTIME_IDENTITY}")
    return ident


def load_frozen_dataset(cfg: FoundationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Only the frozen dataset path, identity verified by the loader and re-checked here."""
    frame, report = load_dataset(cfg.dataset)
    if report.get("sha256") != FROZEN_DATASET_SHA256 or int(report.get("rows", 0)) != CANONICAL_E1_SPEC["dataset"]["rows_after_cutoff"]:
        raise FinalTestError("the loaded dataset is not the frozen dataset")
    return frame, {"path": str(cfg.dataset.raw_csv), "sha256": report["sha256"], "rows_after_cutoff": int(report["rows"])}


@functools.lru_cache(maxsize=1)
def frozen_test_grid() -> pd.DataFrame:
    """The frozen TEST timeline (decision / fill / mark dates), hash-frozen; derived from dates only."""
    dec = pd.date_range(TEST_WINDOW_IDENTITY["first_usable_decision"], periods=N_TEST_TRANSITIONS, freq="D")
    grid = pd.DataFrame({"fill_date": dec + pd.Timedelta(days=1), "mark_date": dec + pd.Timedelta(days=SETTLEMENT_BARS)},
                        index=pd.DatetimeIndex(dec, name="decision_date"))
    txt = "\n".join(f"{d.date()},{f.date()},{m.date()}" for d, f, m in zip(grid.index, grid["fill_date"], grid["mark_date"]))
    if hashlib.sha256(txt.encode()).hexdigest() != FROZEN_TEST_GRID_SHA256:
        raise FinalTestError("the constructed TEST grid does not hash to the frozen grid identity")
    if str(grid.index[-1].date()) != TEST_WINDOW_IDENTITY["last_usable_decision"] or str(grid["mark_date"].iloc[-1].date()) != TEST_WINDOW_IDENTITY["last_mark_open"]:
        raise FinalTestError("the frozen TEST grid does not end at the frozen last usable decision / last mark")
    return grid


def current_harness_hashes(root: Path = PACKAGE_ROOT) -> dict[str, str]:
    return {rel: sha256_of(Path(root) / rel) for rel in HARNESS_FILES}


def _git_blob_sha256(commit: str, rel: str, root: Path) -> str:
    proc = subprocess.run(["git", "show", f"{commit}:./{rel}"], cwd=root, capture_output=True)
    if proc.returncode != 0:
        raise FinalTestError(f"{rel} does not exist at commit {commit[:12]}: {proc.stderr.decode(errors='replace').strip()}")
    return hashlib.sha256(proc.stdout).hexdigest()


def build_harness_anchors(reviewed_code_commit: str, repo_root: Path = PACKAGE_ROOT, files: tuple[str, ...] = HARNESS_FILES) -> dict[str, Any]:
    """The anchor record for reviewed code commit A, computed from A's committed blobs.  Returned, never written."""
    commit = ppo_e1._git("rev-parse", f"{reviewed_code_commit}^{{commit}}", root=repo_root)
    tree = ppo_e1._git("rev-parse", f"{commit}^{{tree}}", root=repo_root)
    return {
        "schema": HARNESS_ANCHORS_SCHEMA, "study_stage": STUDY_STAGE,
        "reviewed_code_commit": commit, "reviewed_code_tree": tree,
        "files": {rel: _git_blob_sha256(commit, rel, repo_root) for rel in files},
        "parent_of_execution_commit": commit, "allowed_administrative_delta": list(ALLOWED_ADMINISTRATIVE_DELTA),
        "accepted_source_anchors": {**{KEY_SOURCE_FILES_E4[k]: v for k, v in ACCEPTED_CORE_SOURCE_HASHES.items()}, **ACCEPTED_IMPLEMENTATION_HASHES},
        "cohort_manifest_sha256": cohort_manifest_sha256(), "test_grid_sha256": FROZEN_TEST_GRID_SHA256,
    }


_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def verify_reviewed_commit_chain(anchors: dict[str, Any], repo_root: Path = PACKAGE_ROOT, files: tuple[str, ...] = HARNESS_FILES,
                                 anchors_rel: str = HARNESS_ANCHORS_REL) -> dict[str, Any]:
    """
    The two-commit rule at execution HEAD B: clean tree; HEAD has exactly one parent A; A == reviewed_code_commit;
    A's tree == reviewed_code_tree; ``git diff A..B`` (repository-wide) contains ONLY the anchors file; every anchored
    file hashes at A to the recorded value and equals the working file; no later descendant is accepted.
    """
    problems: list[str] = []
    if anchors.get("schema") != HARNESS_ANCHORS_SCHEMA:
        problems.append("anchors schema is not the reviewed schema")
    a = anchors.get("reviewed_code_commit")
    if not isinstance(a, str) or not _HEX40.match(a):
        raise FinalTestError("anchors do not carry a non-optional 40-hex reviewed_code_commit")
    if anchors.get("parent_of_execution_commit") != a:
        problems.append("parent_of_execution_commit != reviewed_code_commit")
    if list(anchors.get("allowed_administrative_delta", [])) != [anchors_rel]:
        problems.append("allowed_administrative_delta is not exactly the anchors file")
    git = ppo_e1.git_identity(repo_root)
    dirty = [p for p in git["untracked_files"] if p.startswith(RELEVANT_SOURCE_PREFIXES)]
    if git["tracked_changes"] or dirty:
        raise FinalTestError("relevant sources are not committed/clean; the TEST opening requires the reviewed committed state "
                             f"(tracked changes: {git['tracked_changes']}, untracked relevant: {dirty})")
    head = git["commit"]
    parents = ppo_e1._git("rev-list", "--parents", "-n", "1", "HEAD", root=repo_root).split()[1:]
    if parents != [a]:
        problems.append(f"HEAD {head[:12]} must have exactly one parent equal to the reviewed code commit {a[:12]} (parents: {[p[:12] for p in parents]})")
    else:
        try:
            tree_a = ppo_e1._git("rev-parse", f"{a}^{{tree}}", root=repo_root)
        except RuntimeError as exc:
            raise FinalTestError(f"reviewed code commit {a[:12]} is not present: {exc}") from exc
        if anchors.get("reviewed_code_tree") != tree_a:
            problems.append("reviewed_code_tree does not equal the tree of the reviewed code commit")
        prefix = ppo_e1._git("rev-parse", "--show-prefix", root=repo_root)
        delta = sorted(ln for ln in ppo_e1._git("diff", "--name-only", a, head, root=repo_root).splitlines() if ln)
        if delta != [prefix + anchors_rel]:
            problems.append(f"diff A..B must contain only {anchors_rel}; found {delta}")
        recorded = anchors.get("files", {})
        if set(recorded) != set(files):
            problems.append(f"anchored file set {sorted(recorded)} != reviewed set {sorted(files)}")
        for rel in files:
            at_a = _git_blob_sha256(a, rel, repo_root)
            now = sha256_of(Path(repo_root) / rel)
            if recorded.get(rel) != at_a or now != at_a:
                problems.append(f"{rel}: anchored {str(recorded.get(rel))[:12]} / at A {at_a[:12]} / working {now[:12]} differ")
    if problems:
        raise FinalTestError("reviewed harness identity (two-commit rule) not established: " + "; ".join(problems))
    return {"reviewed_code_commit": a, "reviewed_code_tree": anchors["reviewed_code_tree"], "execution_commit": head,
            "administrative_delta": [anchors_rel], "branch": git["branch"]}


def verify_source_identity(harness_anchors_path: Path = HARNESS_ANCHORS_PATH, require_clean_tree: bool = True,
                           repo_root: Path = PACKAGE_ROOT) -> dict[str, Any]:
    """Accepted anchors + reviewed harness anchors (files) + (production) the two-commit chain."""
    accepted = verify_accepted_sources_unchanged(repo_root)
    anchors = _read_json(harness_anchors_path)
    if anchors is None:
        raise FinalTestError(f"reviewed harness anchors missing at {harness_anchors_path}; the TEST opening requires the "
                             "harness identity frozen by the reviewed closure and authorization procedure")
    now = current_harness_hashes(repo_root)
    bad = [rel for rel in HARNESS_FILES if anchors.get("files", {}).get(rel) != now[rel]]
    if bad:
        raise FinalTestError(f"current harness files differ from the reviewed frozen anchors: {bad}")
    git = ppo_e1.git_identity(repo_root)
    chain = verify_reviewed_commit_chain(anchors, repo_root) if require_clean_tree else None
    return {"harness_files": now, "harness_anchors_sha256": sha256_of(harness_anchors_path), "harness_anchors_path": str(Path(harness_anchors_path).resolve()),
            "git": git, "source_fingerprint": ppo_e1.provenance(repo_root)["source_fingerprint"], "accepted_source_anchors": accepted,
            "reviewed_code_commit": chain["reviewed_code_commit"] if chain else None,
            "reviewed_code_tree": chain["reviewed_code_tree"] if chain else None,
            "execution_commit": chain["execution_commit"] if chain else None,
            "two_commit_rule_verified": chain is not None}


# ───────────────────────────────────────────── run contexts: sealed production or explicit synthetic fixture
@dataclass
class RunContext:
    root: Path
    production: bool
    kind: str = SYNTHETIC_TEST_FIXTURE
    frame: pd.DataFrame | None = None
    cfg: FoundationConfig | None = None
    dataset_identity: dict[str, Any] | None = None
    item_evaluator: Callable[..., dict[str, Any]] | None = None     # synthetic evaluator (test fixtures only)
    harness_anchors_path: Path = HARNESS_ANCHORS_PATH
    require_clean_tree: bool = True
    accepted_root: Path = DEFAULT_ARTIFACTS_ROOT
    final5_root: Path = FINAL5_ROOT
    synthetic_execution_commit: str | None = None                     # fixtures only; production derives B from A→B
    lock_fd: int | None = field(default=None, repr=False)

    @property
    def authorization_path(self) -> Path:
        return self.root / AUTHORIZATION_NAME

    @property
    def run_state_path(self) -> Path:
        return self.root / RUN_STATE_NAME

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def lock_path(self) -> Path:
        return self.root / LOCK_NAME


class _ProductionContext(RunContext):
    """Sealed: identity fields cannot be reassigned after construction (mutation attempts raise)."""

    _SEALED = ("root", "production", "kind", "item_evaluator", "harness_anchors_path", "require_clean_tree", "accepted_root", "final5_root",
               "synthetic_execution_commit")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._SEALED and name in self.__dict__:
            raise FinalTestError(f"production context is sealed; {name} cannot be changed")
        object.__setattr__(self, name, value)


def production_context() -> RunContext:
    """The single canonical production root; no override is possible through this API."""
    return _ProductionContext(root=FINAL_TEST_ROOT, production=True, kind=PRODUCTION_FINAL_TEST)


_REAL_MACHINERY_NAMES: frozenset[str] = frozenset({"_real_item_evaluator", "evaluate_test_item", "run_final_test", "build_test_env",
                                                    "extract_test_frame", "TestDeterministicPolicy", "_dispatch_certified_episode",
                                                    "_authorized_test_route", "load_e1_model"})


def _is_real_evaluator(fn: Any) -> bool:
    """The real evaluator, any wrapper / partial / bound method around it, and any callable defined in this module."""
    seen: set[int] = set()
    stack = [fn]
    while stack:
        f = stack.pop()
        if f is None or id(f) in seen:
            continue
        seen.add(id(f))
        if f is _real_item_evaluator or f is evaluate_test_item or f is run_final_test or f is sys.modules.get(__name__):
            return True
        if getattr(f, "__module__", None) == __name__ and callable(f):
            return True
        code = getattr(f, "__code__", None)
        if code is not None:                      # a lambda / def that names the real machinery (by global or attribute)
            names = set(code.co_names) | {c for c in code.co_consts if isinstance(c, str)}
            if names & _REAL_MACHINERY_NAMES:
                return True
            g = getattr(f, "__globals__", {})
            stack.extend(g.get(n) for n in code.co_names if n in g)
        stack.extend([getattr(f, "__wrapped__", None), getattr(f, "func", None), getattr(f, "__func__", None),
                      getattr(f, "__self__", None)])
        for cell in getattr(f, "__closure__", None) or ():
            try:
                stack.append(cell.cell_contents)
            except ValueError:
                continue
        if isinstance(getattr(f, "__defaults__", None), tuple):
            stack.extend(f.__defaults__)
    return False


def test_only_context(root: Path, frame: pd.DataFrame | None = None, cfg: FoundationConfig | None = None,
                      item_evaluator: Callable[..., dict[str, Any]] | None = None,
                      harness_anchors_path: Path | None = None, accepted_root: Path = DEFAULT_ARTIFACTS_ROOT,
                      final5_root: Path = FINAL5_ROOT, execution_commit: str | None = None) -> RunContext:
    """
    SYNTHETIC_TEST_FIXTURE: a disposable root outside the artifacts tree and outside the package, a synthetic item
    evaluator (never the real TEST evaluation, nor anything wrapping it), synthetic harness anchors and a synthetic
    execution identity (``execution_commit``, 40-hex; e.g. the B of a synthetic A→B repository).  Such a context can
    exercise persistence / state / report logic only; every real TEST boundary refuses it at use time.
    """
    if not isinstance(execution_commit, str) or not _HEX40.match(execution_commit):
        raise FinalTestError("test-only contexts require an explicit 40-hex synthetic execution identity")
    root = Path(root).resolve()
    for forbidden in (DEFAULT_ARTIFACTS_ROOT.resolve(), PACKAGE_ROOT.resolve()):
        if root == forbidden or root.is_relative_to(forbidden):
            raise FinalTestError("test-only contexts must live outside the package and artifacts tree")
    if item_evaluator is None:
        raise FinalTestError("test-only contexts require a synthetic item evaluator; the real TEST evaluation is never run in tests")
    if _is_real_evaluator(item_evaluator):
        raise FinalTestError("test-only contexts refuse the real TEST evaluator (or anything wrapping / aliasing it)")
    return RunContext(root=root, production=False, kind=SYNTHETIC_TEST_FIXTURE, frame=frame, cfg=cfg, item_evaluator=item_evaluator,
                      harness_anchors_path=Path(harness_anchors_path) if harness_anchors_path else root / "harness_anchors.json",
                      require_clean_tree=False, accepted_root=Path(accepted_root), final5_root=Path(final5_root),
                      synthetic_execution_commit=execution_commit)


def require_context(ctx: Any, boundary: str, production_required: bool = False) -> RunContext:
    """
    USE-TIME context validation.  Production authority exists only for the sealed canonical production context on
    the canonical root with no injection; synthetic fixtures must be confined and are refused wherever
    ``production_required`` (every real TEST boundary).  A manually constructed or mutated context is refused.
    """
    if not isinstance(ctx, RunContext):
        raise TestNotAuthorized(f"{boundary}: not a run context")
    kind = getattr(ctx, "kind", None)
    if kind == PRODUCTION_FINAL_TEST:
        ok = (type(ctx) is _ProductionContext and ctx.production is True and Path(ctx.root).resolve() == FINAL_TEST_ROOT.resolve()
              and ctx.item_evaluator is None and Path(ctx.harness_anchors_path).resolve() == HARNESS_ANCHORS_PATH.resolve()
              and ctx.require_clean_tree is True and Path(ctx.accepted_root).resolve() == DEFAULT_ARTIFACTS_ROOT.resolve()
              and Path(ctx.final5_root).resolve() == FINAL5_ROOT.resolve() and ctx.synthetic_execution_commit is None)
        if not ok:
            raise TestNotAuthorized(f"{boundary}: context claims production authority but is not the sealed canonical production context")
        return ctx
    if kind == SYNTHETIC_TEST_FIXTURE:
        if type(ctx) is not RunContext or ctx.production is not False:
            raise TestNotAuthorized(f"{boundary}: malformed synthetic context")
        root = Path(ctx.root).resolve()
        for forbidden in (DEFAULT_ARTIFACTS_ROOT.resolve(), PACKAGE_ROOT.resolve()):
            if root == forbidden or root.is_relative_to(forbidden):
                raise TestNotAuthorized(f"{boundary}: synthetic context inside the package / artifacts tree")
        if ctx.item_evaluator is None or _is_real_evaluator(ctx.item_evaluator):
            raise TestNotAuthorized(f"{boundary}: synthetic context without a purely synthetic evaluator")
        if production_required:
            raise TestNotAuthorized(f"{boundary}: SYNTHETIC_TEST_FIXTURE context refused at a real TEST boundary")
        return ctx
    raise TestNotAuthorized(f"{boundary}: unknown context kind {kind!r}")


# ───────────────────────────────────────────── active authorization (re-validated at every boundary)
@dataclass(frozen=True)
class ActiveAuthorization:
    """Opaque capability.  Its fields are never trusted: every use re-reads and re-validates the external record and the
    durable run state (``revalidate``).  A constructed instance without a matching record and run state is worthless."""

    record_path: str
    record_sha256: str
    cohort_manifest_sha256: str
    run_id: str
    execution_commit: str
    _nonce: str


def _validate_record_contents(rec: dict[str, Any] | None, where: str) -> None:
    if rec is None:
        raise TestNotAuthorized(f"{where}: no valid authorization record")
    expected = cohort_manifest_sha256()
    problems = []
    if rec.get("statement") != AUTHORIZATION_STATEMENT:
        problems.append("statement is not the exact authorization statement")
    if not isinstance(rec.get("authorized_by"), str) or not rec["authorized_by"].strip():
        problems.append("authorized_by missing")
    if not isinstance(rec.get("date"), str) or not rec["date"].strip():
        problems.append("date missing")
    if rec.get("cohort_manifest_sha256") != expected:
        problems.append("cohort_manifest_sha256 does not bind to the frozen cohort")
    if rec.get("test_window") != TEST_WINDOW_IDENTITY:
        problems.append("test_window does not equal the frozen TEST window identity")
    ec = rec.get("execution_commit")
    if ec is None:
        problems.append("execution_commit missing: the authorization must name the exact execution commit B")
    elif not isinstance(ec, str) or not _HEX40.match(ec):
        problems.append("execution_commit is not a 40-hex commit id")
    if problems:
        raise TestNotAuthorized(f"{where}: authorization record invalid: " + "; ".join(problems))


def _read_authorization_record(ctx: RunContext) -> tuple[dict[str, Any], str]:
    p = ctx.authorization_path
    if not p.exists():
        raise TestNotAuthorized(f"no authorization record at the canonical path {p}; the one-time TEST opening requires "
                                "a separate explicit human authorization")
    raw = p.read_bytes()
    try:
        rec = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TestNotAuthorized(f"{p}: authorization record is not valid JSON ({exc})") from exc
    if not isinstance(rec, dict):
        raise TestNotAuthorized(f"{p}: authorization record is not a JSON object")
    return rec, hashlib.sha256(raw).hexdigest()


def revalidate(ctx: RunContext, token: Any, boundary: str) -> ActiveAuthorization:
    """
    The F1 rule: at EVERY protected boundary re-establish that (1) the context is a valid context, (2) the token is
    an ActiveAuthorization, (3) the record still exists at the canonical path, its bytes hash to the token's
    recorded hash and its contents are valid, (4) the token's recorded path is exactly the canonical path, (5) the
    durable run state exists, is OPENED / FINALIZING, and carries this run id and this record hash.
    """
    if TEST_OPENED is not False:
        raise FinalTestError("TEST_OPENED must be False in source; the gate is decided by the authorization record only")
    require_context(ctx, boundary)
    if not isinstance(token, ActiveAuthorization):
        raise TestNotAuthorized(f"{boundary}: learned-policy TEST boundary reached without an active authorization")
    if token.record_path != str(ctx.authorization_path.resolve()):
        raise TestNotAuthorized(f"{boundary}: token record path is not the canonical authorization path")
    rec, digest = _read_authorization_record(ctx)
    if digest != token.record_sha256:
        raise TestNotAuthorized(f"{boundary}: authorization record bytes changed since the token was issued (deleted/replaced/modified)")
    _validate_record_contents(rec, boundary)
    if token.cohort_manifest_sha256 != cohort_manifest_sha256():
        raise TestNotAuthorized(f"{boundary}: token does not bind to the frozen cohort")
    state = _read_json(ctx.run_state_path)
    if state is None:
        raise TestNotAuthorized(f"{boundary}: no durable run state; the token belongs to no active run")
    if state.get("run_id") != token.run_id or state.get("authorization_sha256") != digest \
            or state.get("cohort_manifest_sha256") != token.cohort_manifest_sha256 or state.get("status") not in ("OPENED", "FINALIZING"):
        raise TestNotAuthorized(f"{boundary}: token does not belong to the active OPENED run recorded in the run state")
    if state.get("context_kind") != ctx.kind or state.get("results_dir") != str(ctx.results_dir.resolve()):
        raise TestNotAuthorized(f"{boundary}: the run state belongs to another context / root")
    current = current_execution_identity(ctx, state)
    if not (rec.get("execution_commit") == token.execution_commit == state.get("execution_commit") == current):
        raise TestNotAuthorized(f"{boundary}: execution identity mismatch (record {str(rec.get('execution_commit'))[:12]}, token "
                                f"{token.execution_commit[:12]}, run state {str(state.get('execution_commit'))[:12]}, current {str(current)[:12]}); "
                                "the authorization binds exactly the verified execution commit B")
    return token


def current_execution_identity(ctx: RunContext, state: dict[str, Any] | None = None) -> str:
    """
    The CURRENT verified execution identity, re-established at every boundary (never taken from the token alone).
    Production: HEAD must be the execution commit B recorded by the opened run, the reviewed harness files and the
    anchors file must still hash as recorded, and the tracked package tree must be clean.  Synthetic fixtures: the
    synthetic execution identity supplied through the explicitly synthetic ``test_only_context`` API.
    """
    if ctx.kind == SYNTHETIC_TEST_FIXTURE:
        ec = ctx.synthetic_execution_commit
        if not isinstance(ec, str) or not _HEX40.match(ec):
            raise TestNotAuthorized("synthetic fixture without a 40-hex synthetic execution identity")
        return ec
    require_context(ctx, "execution identity", production_required=True)
    git = ppo_e1.git_identity(PACKAGE_ROOT)
    if git["tracked_changes"]:
        raise TestNotAuthorized("execution identity: the tracked package tree changed after the opening")
    if state is not None:
        src = state.get("source_identity") or {}
        if current_harness_hashes() != src.get("harness_files") or not Path(HARNESS_ANCHORS_PATH).is_file() \
                or sha256_of(HARNESS_ANCHORS_PATH) != src.get("harness_anchors_sha256") or git["commit"] != src.get("execution_commit"):
            raise TestNotAuthorized("execution identity: the reviewed harness / anchors / HEAD changed after the opening")
    return git["commit"]


def bind_execution_commit(ctx: RunContext, rec_execution_commit: Any, source_identity: dict[str, Any]) -> str:
    """
    Opening-time binding: the authorization's execution_commit must equal the execution identity established by the
    A→B source verification (production) or the synthetic execution identity (fixtures), BEFORE dataset access,
    TEST resolution, environment construction, model loading, prediction or run-state publication.
    """
    verified = source_identity.get("execution_commit") if ctx.kind == PRODUCTION_FINAL_TEST else current_execution_identity(ctx)
    if not isinstance(verified, str) or not _HEX40.match(verified):
        raise TestNotAuthorized("no verified execution identity B (the two-commit source verification did not establish one)")
    if rec_execution_commit != verified:
        raise TestNotAuthorized(f"authorization names execution commit {str(rec_execution_commit)[:12]} but the verified execution "
                                f"identity is {verified[:12]}; the human authorization must name exactly B")
    if ctx.kind == PRODUCTION_FINAL_TEST and verified != ppo_e1.git_identity(PACKAGE_ROOT)["commit"]:
        raise TestNotAuthorized("verified execution identity B is not the current HEAD")
    return verified


def require_test_authorization(open_test: bool, ctx: RunContext, run_id: str | None = None) -> ActiveAuthorization:
    """The gate.  Issues a capability only for a valid record at the canonical path; ``run_id`` binds it to a run."""
    if TEST_OPENED is not False:
        raise FinalTestError("TEST_OPENED must be False in source; the gate is decided by the authorization record only")
    require_context(ctx, "authorization gate")
    if open_test is not True:
        raise TestNotAuthorized("learned-policy TEST evaluation refused: the one-time opening was not requested "
                                "(open_test=True is required together with the human-gate authorization record)")
    rec, digest = _read_authorization_record(ctx)
    _validate_record_contents(rec, "authorization")
    return ActiveAuthorization(str(ctx.authorization_path.resolve()), digest, cohort_manifest_sha256(),
                              run_id or "", rec["execution_commit"], secrets.token_hex(8))


def status(ctx: RunContext | None = None) -> dict[str, Any]:
    ctx = ctx or production_context()
    state = _read_json(ctx.run_state_path) if ctx.run_state_path.exists() else None
    return {
        "TEST_OPENED_default": TEST_OPENED,
        "canonical_root": str(ctx.root), "context_kind": ctx.kind,
        "authorization_record_present": ctx.authorization_path.exists(),
        "run_state_present": state is not None, "run_status": None if state is None else state.get("status"),
        "run_lock_present": ctx.lock_path.exists(),
        "results_dir_exists": ctx.results_dir.exists(),
        "results_populated": ctx.results_dir.exists() and any(ctx.results_dir.iterdir()),
        "final_marker_present": (ctx.results_dir / MARKER_NAME).exists(),
        "harness_anchors_present": Path(ctx.harness_anchors_path).exists(),
        "cohort_manifest_sha256": cohort_manifest_sha256(), "test_grid_sha256": FROZEN_TEST_GRID_SHA256,
    }


# ───────────────────────────────────────────── execution plan, run state, active-item binding, run lock
IDENTITY_KEYS: tuple[str, ...] = ("key", "kind", "experiment", "seed", "policy", "policy_kind", "random_seed", "policy_name", "model_sha256")


def execution_plan() -> list[dict[str, Any]]:
    plan = [{"key": f"{RUN_GROUPS[e]}/{s}", "kind": "learned", "experiment": e, "seed": s, "policy": f"ppo_{RUN_GROUPS[e]}",
             "policy_kind": "ppo_deterministic_argmax", "random_seed": None, "policy_name": f"ppo_{RUN_GROUPS[e]}_seed{s}",
             "model_sha256": FROZEN_CHECKPOINTS[(e, s)], "state": "PENDING", "attempts": 0, "recovery_retry": False}
            for e, s in COHORT_ORDER]
    for e in EXPERIMENTS:
        for label in BASELINE_LABELS:
            kind, rseed, name = BASELINE_IDENTITY[label]
            plan.append({"key": f"{RUN_GROUPS[e]}/baseline/{label}", "kind": "baseline", "experiment": e, "seed": rseed,
                         "policy": label, "policy_kind": kind, "random_seed": rseed, "policy_name": name, "model_sha256": None,
                         "state": "PENDING", "attempts": 0, "recovery_retry": False})
    return plan


def item_identity(it: dict[str, Any]) -> tuple:
    return tuple(it.get(k) for k in IDENTITY_KEYS)


CANONICAL_PLAN_IDENTITY: tuple[tuple, ...] = tuple(item_identity(p) for p in execution_plan())
assert len(CANONICAL_PLAN_IDENTITY) == 48 and len(set(CANONICAL_PLAN_IDENTITY)) == 48


def _new_run_state(ctx: RunContext, digest: str, cohort: dict[str, Any], source_identity: dict[str, Any],
                   runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_stage": STUDY_STAGE, "status": "OPENED", "context_kind": ctx.kind,
        "run_id": f"final-test-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(6)}",
        "authorization_path": str(ctx.authorization_path.resolve()), "authorization_sha256": digest,
        "cohort_manifest": cohort_manifest(), "cohort_manifest_sha256": cohort["cohort_manifest_sha256"],
        "checkpoints": cohort["checkpoints"], "test_window": TEST_WINDOW_IDENTITY, "test_grid_sha256": FROZEN_TEST_GRID_SHA256,
        "source_identity": source_identity, "dataset": ctx.dataset_identity, "runtime": runtime,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "plan": execution_plan(),
        "post_test_rules": list(POST_TEST_RULES), "results_dir": str(ctx.results_dir.resolve()), "recovery_events": [],
    }


def _save_state(ctx: RunContext, state: dict[str, Any]) -> None:
    _write_durable(ctx.run_state_path, json.dumps(state, indent=1, sort_keys=True, default=ppo_e1._json_default))


def _load_state(ctx: RunContext) -> dict[str, Any]:
    state = _read_json(ctx.run_state_path)
    if state is None:
        raise FinalTestError(f"{ctx.run_state_path}: run state missing or invalid")
    return state


def _require_active_item(ctx: RunContext, token: ActiveAuthorization, item: dict[str, Any], boundary: str) -> dict[str, Any]:
    """The requested evaluation must be the single IN_PROGRESS item of the OPENED durable plan, identity-exact."""
    state = _load_state(ctx)
    if state.get("status") != "OPENED" or state.get("run_id") != token.run_id:
        raise TestNotAuthorized(f"{boundary}: the run is not OPENED under this authorization")
    plan = state.get("plan", [])
    active = [p for p in plan if p.get("state") == "IN_PROGRESS"]
    if len(active) != 1:
        raise TestNotAuthorized(f"{boundary}: exactly one plan item must be IN_PROGRESS (found {len(active)})")
    a = active[0]
    idx = plan.index(a)
    if item_identity(a) != item_identity(item) or idx >= len(CANONICAL_PLAN_IDENTITY) or item_identity(a) != CANONICAL_PLAN_IDENTITY[idx]:
        raise TestNotAuthorized(f"{boundary}: requested item {item.get('key')!r} is not the active canonical plan item {a.get('key')!r}")
    if (a.get("recovery_retry"), a.get("attempts")) != (item.get("recovery_retry"), item.get("attempts")):
        raise TestNotAuthorized(f"{boundary}: requested item recovery state differs from the durable plan")
    return a


def _acquire_run_lock(ctx: RunContext) -> None:
    """Exclusive crash-releasing OS advisory lock on the canonical lock file (never scientific state)."""
    if ctx.lock_fd is not None:
        return
    ctx.root.mkdir(parents=True, exist_ok=True)
    fd = os.open(ctx.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise FinalTestError(f"{ctx.lock_path}: another live executor holds the exclusive run lock; a second executor is refused") from exc
    ctx.lock_fd = fd


def _release_run_lock(ctx: RunContext) -> None:
    if ctx.lock_fd is not None:
        try:
            fcntl.flock(ctx.lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(ctx.lock_fd)
            ctx.lock_fd = None


# ───────────────────────────────────────────── real TEST boundaries (production context + token + active item)
def extract_test_frame(ctx: RunContext, token: ActiveAuthorization) -> tuple[int, int]:
    require_context(ctx, "TEST frame extraction", production_required=True)
    revalidate(ctx, token, "TEST frame extraction")
    frame = ctx.frame
    if frame is None or ctx.dataset_identity is None or ctx.dataset_identity.get("sha256") != FROZEN_DATASET_SHA256:
        raise FinalTestError("TEST frame extraction requires the frozen dataset loaded by the production opening")
    i0, last = usable_decision_index_range(frame, TEST_WINDOW)
    grid = frozen_test_grid()
    if last - i0 + 1 != N_TEST_TRANSITIONS or not frame.index[i0:last + 1].equals(grid.index) \
            or not frame.index[i0 + 1:last + 2].equals(pd.DatetimeIndex(grid["fill_date"])) \
            or not frame.index[i0 + SETTLEMENT_BARS:last + SETTLEMENT_BARS + 1].equals(pd.DatetimeIndex(grid["mark_date"])):
        raise FinalTestError("the frozen TEST window does not resolve to the frozen decision / fill / mark grid")
    return i0, last


def build_test_env(ctx: RunContext, token: ActiveAuthorization, experiment: str) -> BtcUsdtTradingEnv:
    """The accepted mechanical environment of ``experiment`` on the frozen TEST window — gated and certified."""
    require_context(ctx, "TEST environment construction", production_required=True)
    revalidate(ctx, token, "TEST environment construction")
    extract_test_frame(ctx, token)
    frame, cfg = ctx.frame, ctx.cfg
    verify_runtime_config(cfg)
    cost = cfg.costs["conservative"]
    if experiment == "E1":
        obs_cfg, _ = ppo_e1.resolve_observation_config(frame, cfg, True)
        env = make_split_env(frame, TEST_WINDOW, cost_config=cost, obs_config=obs_cfg, splits=FROZEN_SPLITS, initial_equity=1.0)
    elif experiment == "E2":
        o2, _ = ppo_e2.resolve_o2_config(frame, cfg)
        env = ppo_e2.make_o2_split_env(frame, TEST_WINDOW, o2, cost_config=cost, initial_equity=1.0)
    elif experiment == "E3":
        obs_cfg, _ = ppo_e1.resolve_observation_config(frame, cfg, True)
        env = ppo_e3.make_r2_split_env(frame, TEST_WINDOW, obs_cfg, cost_config=cost, initial_equity=1.0, r2_coefficients=FROZEN_R2)
    elif experiment == "E4":
        o2, _ = ppo_e2.resolve_o2_config(frame, cfg)
        env = ppo_e4.make_o2r2_split_env(frame, TEST_WINDOW, o2, cost_config=cost, initial_equity=1.0, r2_coefficients=FROZEN_R2)
    else:
        raise FinalTestError(f"unknown experiment {experiment!r}")
    certify_test_env(env, experiment, frame, cfg)
    return env


def certify_test_env(env: Any, experiment: str, frame: pd.DataFrame, cfg: FoundationConfig) -> dict[str, Any]:
    """The environment must be exactly the frozen experiment's certified type on exactly the frozen TEST window."""
    if experiment not in EXPERIMENTS:
        raise FinalTestError(f"unknown experiment {experiment!r}")
    if type(env) is not ENV_TYPE_OF[experiment]:
        raise FinalTestError(f"{experiment}: environment type {type(env).__name__} is not the certified {ENV_TYPE_OF[experiment].__name__}")
    problems = []
    first, last = env.decision_window
    dates = env._dates
    if first != TEST_WINDOW.start or env.price_end != TEST_WINDOW.end or dates[-1] != TEST_WINDOW.end \
            or dates[env._i1 + SETTLEMENT_BARS] != TEST_WINDOW.end or str(last.date()) != TEST_WINDOW_IDENTITY["last_usable_decision"]:
        problems.append("decision window / price boundary is not the frozen TEST window")
    if env.n_decisions_available != N_TEST_TRANSITIONS:
        problems.append(f"{env.n_decisions_available} decisions != {N_TEST_TRANSITIONS}")
    if env.initial_equity != 1.0:
        problems.append("initial equity != 1")
    if env.cost_config != CONSERVATIVE_COST:
        problems.append("cost profile is not the frozen conservative cost profile (modelled 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale)")
    if env.random_start is not False or env.episode_length is not None:
        problems.append("episode options are not deterministic full-window")
    from gymnasium import spaces

    if env.action_space != spaces.Discrete(2):
        problems.append("action space is not Discrete(2) CASH/LONG")
    if OBSERVATION_OF[experiment] == "O1":
        expected_cfg, _ = ppo_e1.resolve_observation_config(frame, cfg, True)
        if not isinstance(env.obs_config, ObservationConfig) or env.obs_config != expected_cfg or env.observation_space != observation_space(expected_cfg):
            problems.append("observation is not the accepted TRAIN-scaled O1")
    else:
        expected_o2, _ = ppo_e2.resolve_o2_config(frame, cfg)
        if not isinstance(env.obs_config, O2ObservationConfig) or env.obs_config != expected_o2 or env.observation_space != o2_observation_space(expected_o2):
            problems.append("observation is not the accepted TRAIN-scaled O2")
    if REWARD_OF[experiment] == "R2" and (not hasattr(env, "r2_coefficients") or env.r2_coefficients != FROZEN_R2):
        problems.append("R2 coefficients are not the frozen values")
    if problems:
        raise FinalTestError(f"{experiment}: TEST environment not certified: " + "; ".join(problems))
    return {"type": type(env).__name__, "n_decisions": env.n_decisions_available, "obs_shape": tuple(env.observation_space.shape)}


def certify_validation_env(env: Any, experiment: str) -> None:
    """VALIDATION_CERTIFIED: the accepted VALIDATION window only; every TEST date is outside the environment."""
    if experiment not in EXPERIMENTS or type(env) is not ENV_TYPE_OF[experiment]:
        raise FinalTestError(f"{experiment}: environment is not the accepted VALIDATION environment type")
    first, last = env.decision_window
    if first != VALIDATION_WINDOW.start or env.price_end != VALIDATION_WINDOW.end or env._dates[-1] != VALIDATION_WINDOW.end \
            or env.n_decisions_available != N_VALIDATION_TRANSITIONS or env._dates[-1] >= TEST_WINDOW.start or last >= TEST_WINDOW.start:
        raise FinalTestError("validation regression route is bound to the frozen VALIDATION window only; TEST dates are rejected")


class TestDeterministicPolicy:
    """Deterministic (argmax) adapter bound to the production context, active authorization, active plan item, checkpoint
    and certified env; authorization AND active item are revalidated before EVERY prediction."""

    def __init__(self, ctx: RunContext, token: ActiveAuthorization, item: dict[str, Any], model, env: BtcUsdtTradingEnv) -> None:
        require_context(ctx, "TEST policy binding", production_required=True)
        revalidate(ctx, token, "TEST policy binding")
        _require_active_item(ctx, token, item, "TEST policy binding")
        experiment, seed, model_sha256 = item["experiment"], item["seed"], item["model_sha256"]
        if item.get("kind") != "learned" or FROZEN_CHECKPOINTS.get((experiment, seed)) != model_sha256:
            raise FinalTestError(f"adapter: item is not a frozen learned checkpoint of {experiment} seed {seed}")
        certify_test_env(env, experiment, ctx.frame, ctx.cfg)
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise FinalTestError("model spaces differ from the certified TEST environment's spaces")
        self.ctx, self.token, self.item, self.model, self.env = ctx, token, dict(item), model, env
        self.experiment, self.seed, self.model_sha256, self.name = experiment, seed, model_sha256, item["policy_name"]
        self._first, self._last = env.decision_window
        self._end, self._obs_space, self._predictions = TEST_WINDOW.end, env.observation_space, 0

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        require_context(self.ctx, "TEST prediction", production_required=True)
        revalidate(self.ctx, self.token, "TEST prediction")
        _require_active_item(self.ctx, self.token, self.item, "TEST prediction")
        decision, instant = pd.Timestamp(info["decision_date"]), pd.Timestamp(info["state_instant_date"])
        if not (self._first <= decision <= self._last) or instant > self._end:
            raise FinalTestError(f"TEST adapter refused decision {decision.date()} outside the frozen window before prediction")
        if not self._obs_space.contains(np.asarray(obs, dtype=np.float32)):
            raise FinalTestError("observation outside the model's observation space")
        action, _ = self.model.predict(obs, deterministic=True)
        self._predictions += 1
        return int(np.asarray(action).item())


@dataclass(frozen=True)
class CertifiedRoute:
    """Proof carrier for the single episode dispatcher.  Never trusted by itself: the dispatcher re-establishes the
    proof (production context, authorization, active item, certified env / VALIDATION window) at the runner boundary."""

    kind: str
    experiment: str
    env_id: int
    ctx: RunContext | None
    token: ActiveAuthorization | None
    item: dict[str, Any] | None
    _nonce: str


def _authorized_test_route(ctx: RunContext, token: ActiveAuthorization, item: dict[str, Any], env: Any) -> CertifiedRoute:
    require_context(ctx, "TEST route", production_required=True)
    revalidate(ctx, token, "TEST route")
    active = _require_active_item(ctx, token, item, "TEST route")
    certify_test_env(env, item["experiment"], ctx.frame, ctx.cfg)
    return CertifiedRoute(AUTHORIZED_TEST, item["experiment"], id(env), ctx, token, dict(active), secrets.token_hex(8))


def _validation_certified_route(env: Any, experiment: str) -> CertifiedRoute:
    certify_validation_env(env, experiment)
    return CertifiedRoute(VALIDATION_CERTIFIED, experiment, id(env), None, None, None, secrets.token_hex(8))


def _dispatch_certified_episode(route: Any, env: Any, policy: Any) -> EpisodeResult:
    """THE ONLY episode dispatcher.  Runs an accepted episode runner only under a re-verified certified route."""
    if not isinstance(route, CertifiedRoute):
        raise TestNotAuthorized("episode dispatch refused: no certified route (AUTHORIZED_TEST or VALIDATION_CERTIFIED)")
    if route.env_id != id(env) or route.experiment not in EXPERIMENTS:
        raise TestNotAuthorized("episode dispatch refused: the environment is not the one certified for this route")
    if route.kind == AUTHORIZED_TEST:
        ctx, token, item = route.ctx, route.token, route.item
        require_context(ctx, "episode dispatch", production_required=True)
        revalidate(ctx, token, "episode dispatch")
        _require_active_item(ctx, token, item, "episode dispatch")
        certify_test_env(env, route.experiment, ctx.frame, ctx.cfg)
        if item["kind"] == "learned":
            if not (isinstance(policy, TestDeterministicPolicy) and policy.ctx is ctx and policy.token is token and policy.env is env
                    and item_identity(policy.item) == item_identity(item)):
                raise TestNotAuthorized("episode dispatch refused: the policy is not the adapter bound to this route's active item")
        else:
            expected_type = {"always_flat": AlwaysFlat, "always_long": AlwaysLong, "random": RandomPolicy}[item["policy_kind"]]
            if type(policy) is not expected_type or policy.name != item["policy_name"]:
                raise TestNotAuthorized("episode dispatch refused: the baseline policy is not the canonical policy of the active item")
    elif route.kind == VALIDATION_CERTIFIED:
        certify_validation_env(env, route.experiment)
    else:
        raise TestNotAuthorized(f"episode dispatch refused: unknown route kind {route.kind!r}")
    runner = run_episode_e3 if route.experiment in R2_EXPERIMENTS else run_episode
    res = runner(env, policy, periods_per_year=365)
    num = res.curve.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(num)):
        raise FinalTestError("non-finite value in the evaluation curve")
    res.metrics.update(ppo_e1.state_distribution(res))
    return res


def evaluate_test_item(ctx: RunContext, token: ActiveAuthorization, item: dict[str, Any]) -> EpisodeResult:
    """The ONLY TEST evaluation route: production context, active authorization, active plan item, certified TEST
    environment, bound checkpoint identity, certified-route dispatch."""
    require_context(ctx, "TEST evaluation", production_required=True)
    revalidate(ctx, token, "TEST evaluation")
    active = _require_active_item(ctx, token, item, "TEST evaluation")
    e = active["experiment"]
    env = build_test_env(ctx, token, e)
    if active["kind"] == "learned":
        path = checkpoint_path(e, active["seed"], ctx.accepted_root, ctx.final5_root)
        if active["model_sha256"] != FROZEN_CHECKPOINTS[(e, active["seed"])] or sha256_of(path) != active["model_sha256"]:
            raise FinalTestError(f"{active['key']}: checkpoint bytes do not match the frozen identity")
        model = load_e1_model(path)
        policy: Policy = TestDeterministicPolicy(ctx, token, active, model, env)
    else:
        kind, rseed, _ = BASELINE_IDENTITY[active["policy"]]
        policy = AlwaysFlat() if kind == "always_flat" else AlwaysLong() if kind == "always_long" else RandomPolicy(int(rseed))
    route = _authorized_test_route(ctx, token, active, env)
    res = _dispatch_certified_episode(route, env, policy)
    if res.metrics["n_steps"] != N_TEST_TRANSITIONS or (active["kind"] == "learned" and policy._predictions != N_TEST_TRANSITIONS):
        raise FinalTestError(f"{active['key']}: evaluation did not cover exactly the {N_TEST_TRANSITIONS} frozen transitions")
    revalidate(ctx, token, "TEST evaluation (after episode)")
    return res


def evaluate_validation_regression(model, experiment: str, frame: pd.DataFrame, cfg: FoundationConfig) -> EpisodeResult:
    """VALIDATION-only regression route: the accepted VALIDATION environments and the accepted TEST-refusing adapters."""
    if experiment == "E1":
        oc, _ = ppo_e1.resolve_observation_config(frame, cfg, True)
        env = ppo_e1.make_e1_env(frame, "validation", cfg, oc, role="evaluation")
        policy = ppo_e1.SB3DeterministicPolicy(model, env, role="evaluation", obs_cfg=oc)
    elif experiment == "E2":
        o2, _ = ppo_e2.resolve_o2_config(frame, cfg)
        env = ppo_e2.make_e2_env(frame, "validation", cfg, o2, role="evaluation")
        policy = ppo_e2.SB3DeterministicPolicyE2(model, env, role="evaluation", obs_cfg=o2)
    elif experiment == "E3":
        oc, _ = ppo_e1.resolve_observation_config(frame, cfg, True)
        env = ppo_e3.make_e3_env(frame, "validation", cfg, oc, role="evaluation")
        policy = ppo_e3.SB3DeterministicPolicyE3(model, env, role="evaluation", obs_cfg=oc)
    elif experiment == "E4":
        o2, _ = ppo_e2.resolve_o2_config(frame, cfg)
        env = ppo_e4.make_e4_env(frame, "validation", cfg, o2, role="evaluation")
        policy = ppo_e4.SB3DeterministicPolicyE4(model, env, role="evaluation", obs_cfg=o2)
    else:
        raise FinalTestError(f"unknown experiment {experiment!r}")
    route = _validation_certified_route(env, experiment)
    return _dispatch_certified_episode(route, env, policy)


# ───────────────────────────────────────────── evidence: identity, full validation, transactional persistence
METRIC_KEYS: tuple[str, ...] = ("total_return", "final_equity", "sharpe_ratio", "max_drawdown", "n_entries", "n_legs",
                                "exposure", "total_cost_fraction", "final_position", "n_steps")
DIAG_KEYS: tuple[str, ...] = ("cumulative_financial_r1", "cumulative_turnover_penalty", "cumulative_drawdown_penalty",
                              "cumulative_reward_r2")
CURVE_COLUMNS: tuple[str, ...] = ("fill_date", "mark_date", "action", "position", "legs", "cost_fraction", "gross_simple_return",
                                  "position_return", "reward", "equity", "drawdown")
EVIDENCE_FILES: tuple[str, ...] = ("test_curve.csv", "test_metrics.json", "item.json")
TOL = 1e-12


def _close(a: Any, b: Any, tol: float = TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    except (TypeError, ValueError):
        return a == b


def _item_dir(ctx: RunContext, item: dict[str, Any]) -> Path:
    return ctx.results_dir / item["key"]


def _compact_source_identity(source_identity: dict[str, Any]) -> dict[str, Any]:
    return {"harness_files": source_identity["harness_files"], "harness_anchors_sha256": source_identity["harness_anchors_sha256"],
            "source_fingerprint": source_identity["source_fingerprint"], "commit": source_identity["git"]["commit"],
            "reviewed_code_commit": source_identity.get("reviewed_code_commit"), "execution_commit": source_identity.get("execution_commit")}


def expected_item_identity(state: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """Everything an item's evidence must carry, derived from the durable run state (the authority) and the plan item."""
    return {**{k: item.get(k) for k in IDENTITY_KEYS}, "run_id": state["run_id"], "context_kind": state["context_kind"],
            "execution_commit": state["execution_commit"],
            "results_dir": state["results_dir"], "authorization_sha256": state["authorization_sha256"],
            "cohort_manifest_sha256": state["cohort_manifest_sha256"], "test_window": TEST_WINDOW_IDENTITY,
            "test_grid_sha256": FROZEN_TEST_GRID_SHA256, "n_transitions": N_TEST_TRANSITIONS,
            "source_identity": _compact_source_identity(state["source_identity"]), "dataset": state["dataset"], "runtime": state["runtime"],
            "recovery_retry": bool(item.get("recovery_retry")), "attempts": int(item.get("attempts", 0))}


def validate_curve_timeline_and_accounting(curve: pd.DataFrame, experiment: str, key: str) -> dict[str, float]:
    """
    The full persisted timeline and the accepted accounting identities, from the curve alone: exactly the frozen
    decision / fill / mark grid (1618 rows, unique, monotonic), finite values, binary actions / positions, legs,
    cost fraction, equity recurrence, R1 reward, drawdown; for R2 experiments the frozen decomposition
    (turnover = 0.0005·legs, drawdown penalty = 0.10·max(0, D_{t+1} − D_t), R2 = R1 − penalties, penalties ≥ 0).
    Returns the R2 diagnostics (empty for R1 experiments).
    """
    grid = frozen_test_grid()
    missing = [c for c in CURVE_COLUMNS if c not in curve.columns]
    if missing:
        raise FinalTestError(f"{key}: persisted curve lacks columns {missing}")
    idx = pd.DatetimeIndex(curve.index)
    if len(curve) != N_TEST_TRANSITIONS or not idx.is_unique or not idx.is_monotonic_increasing or not idx.equals(grid.index):
        raise FinalTestError(f"{key}: persisted curve is not exactly the frozen TEST decision grid ({len(curve)} rows)")
    if not pd.DatetimeIndex(pd.to_datetime(curve["fill_date"])).equals(pd.DatetimeIndex(grid["fill_date"])) \
            or not pd.DatetimeIndex(pd.to_datetime(curve["mark_date"])).equals(pd.DatetimeIndex(grid["mark_date"])):
        raise FinalTestError(f"{key}: persisted fill / mark dates are not the frozen settlement grid")
    num_cols = [c for c in curve.columns if c not in ("fill_date", "mark_date")]
    num = curve[num_cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(num)):
        raise FinalTestError(f"{key}: non-finite value in the persisted curve")
    pos = curve["position"].to_numpy(dtype=np.float64)
    act = curve["action"].to_numpy(dtype=np.float64)
    legs = curve["legs"].to_numpy(dtype=np.float64)
    if not (np.isin(pos, (0.0, 1.0)).all() and np.isin(act, (0.0, 1.0)).all() and np.array_equal(act, pos)):
        raise FinalTestError(f"{key}: actions / positions are not binary CASH/LONG target states")
    prev = np.concatenate([[0.0], pos[:-1]])
    if not np.array_equal(legs, np.abs(pos - prev)):
        raise FinalTestError(f"{key}: legs do not equal |position change| from the initial CASH state")
    cost = curve["cost_fraction"].to_numpy(dtype=np.float64)
    if not np.allclose(cost, 1.0 - (1.0 - ONE_LEG_COST) ** legs, rtol=0, atol=TOL):
        raise FinalTestError(f"{key}: cost fractions do not match 0.15% of the traded value per purchase or sale (modelled 0.10% trading fee plus a 0.05% allowance for slippage)")
    equity = curve["equity"].to_numpy(dtype=np.float64)
    eq_prev = np.concatenate([[1.0], equity[:-1]])
    pret = curve["position_return"].to_numpy(dtype=np.float64)
    gross = curve["gross_simple_return"].to_numpy(dtype=np.float64)
    if not np.allclose(pret, pos * gross, rtol=0, atol=TOL) or not np.allclose(equity, eq_prev * (1.0 - cost) * (1.0 + pret), rtol=0, atol=1e-10) or np.any(equity <= 0):
        raise FinalTestError(f"{key}: equity path violates the accepted open-to-open accounting recurrence")
    r1 = np.log(equity / eq_prev)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))
    dd = equity / peak[1:] - 1.0
    if not np.allclose(curve["drawdown"].to_numpy(dtype=np.float64), dd, rtol=0, atol=TOL):
        raise FinalTestError(f"{key}: drawdown column is not equity / running peak − 1")
    reward = curve["reward"].to_numpy(dtype=np.float64)
    if experiment not in R2_EXPERIMENTS:
        if not np.allclose(reward, r1, rtol=0, atol=TOL):
            raise FinalTestError(f"{key}: R1 reward is not log(E_t+1 / E_t)")
        return {}
    missing = [c for c in R2_CURVE_COLUMNS if c not in curve.columns]
    if missing:
        raise FinalTestError(f"{key}: R2 evidence lacks decomposition columns {missing}")
    fr1, tp, dp, r2 = (curve[c].to_numpy(dtype=np.float64) for c in ("financial_reward_r1", "turnover_penalty", "drawdown_penalty", "reward_r2"))
    inc, d_before, d_after = (curve[c].to_numpy(dtype=np.float64) for c in ("incremental_drawdown", "drawdown_before", "drawdown_after"))
    pk_before, pk_after = (curve[c].to_numpy(dtype=np.float64) for c in ("running_peak_before", "running_peak_after"))
    exp_pk_before, exp_pk_after = peak[:-1], np.maximum(peak[:-1], equity)
    exp_d_before, exp_d_after = 1.0 - eq_prev / exp_pk_before, 1.0 - equity / exp_pk_after
    problems = []
    if not np.allclose(fr1, r1, rtol=0, atol=TOL):
        problems.append("financial_reward_r1 != log(E_t+1 / E_t)")
    if not (np.allclose(pk_before, exp_pk_before, rtol=0, atol=TOL) and np.allclose(pk_after, exp_pk_after, rtol=0, atol=TOL)):
        problems.append("running peaks differ from the equity path")
    if not (np.allclose(d_before, exp_d_before, rtol=0, atol=TOL) and np.allclose(d_after, exp_d_after, rtol=0, atol=TOL)):
        problems.append("drawdown magnitudes differ from the equity path")
    if not np.allclose(inc, np.maximum(0.0, d_after - d_before), rtol=0, atol=TOL):
        problems.append("incremental_drawdown != max(0, D_t+1 − D_t)")
    if not np.allclose(tp, FROZEN_R2.turnover_penalty_lambda * legs, rtol=0, atol=TOL) or np.any(tp < 0):
        problems.append("turnover penalty != 0.0005·legs or negative")
    if not np.allclose(dp, FROZEN_R2.drawdown_penalty_lambda * inc, rtol=0, atol=TOL) or np.any(dp < 0):
        problems.append("drawdown penalty != 0.10·incremental drawdown or negative")
    if not np.allclose(r2, fr1 - tp - dp, rtol=0, atol=TOL) or not np.allclose(reward, r2, rtol=0, atol=TOL):
        problems.append("R2 != R1 − turnover − drawdown penalty, or the returned reward is not R2")
    if problems:
        raise FinalTestError(f"{key}: R2 decomposition invalid: " + "; ".join(problems))
    diag = {"cumulative_financial_r1": float(fr1.sum()), "cumulative_turnover_penalty": float(tp.sum()),
            "cumulative_drawdown_penalty": float(dp.sum()), "cumulative_reward_r2": float(r2.sum())}
    if not _close(diag["cumulative_financial_r1"], math.log(float(equity[-1])), 1e-9):
        raise FinalTestError(f"{key}: cumulative R1 != log(final equity)")
    return diag


def validate_item_evidence(d: Path, expected: dict[str, Any], recorded_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    """
    Complete validation of one promoted evidence directory against the expected identity: files present, hashes
    (recomputed; equal to ``recorded_hashes`` when given and to item.json's own record), item.json identity exact,
    the full timeline and accounting, metrics recomputed from the curve equal to test_metrics.json and item.json.
    Never predicts, never repairs.
    """
    d = Path(d)
    key = expected.get("key", str(d))
    for name in EVIDENCE_FILES:
        if not (d / name).is_file():
            raise FinalTestError(f"{key}: promoted evidence {name} missing")
    hashes = {name: sha256_of(d / name) for name in EVIDENCE_FILES}
    if recorded_hashes is not None and any(hashes[n] != recorded_hashes.get(n) for n in EVIDENCE_FILES):
        raise FinalTestError(f"{key}: evidence hash mismatch against the recorded hashes")
    try:
        ev = json.loads((d / "item.json").read_text(encoding="utf-8"))
        metrics = json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
        curve = pd.read_csv(d / "test_curve.csv", index_col=0, parse_dates=True)
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        raise FinalTestError(f"{key}: evidence unreadable: {exc}") from exc
    if not isinstance(ev, dict) or not isinstance(metrics, dict):
        raise FinalTestError(f"{key}: evidence records are not JSON objects")
    for k, v in expected.items():
        if ev.get(k) != v:
            raise FinalTestError(f"{key}: item evidence identity differs from the authority in {k}")
    if ev.get("evidence_sha256", {}).get("test_curve.csv") != hashes["test_curve.csv"] or ev.get("evidence_sha256", {}).get("test_metrics.json") != hashes["test_metrics.json"]:
        raise FinalTestError(f"{key}: item.json evidence hashes differ from the persisted files")
    diag = validate_curve_timeline_and_accounting(curve, expected["experiment"], key)
    recomputed = compute_metrics(curve, 1.0, 365)
    if int(metrics.get("n_steps", -1)) != N_TEST_TRANSITIONS or metrics.get("policy") != expected["policy_name"]:
        raise FinalTestError(f"{key}: saved metrics do not carry the frozen transition count / canonical policy name")
    for k in METRIC_KEYS:
        if not _close(recomputed.get(k), metrics.get(k)) or not _close((ev.get("financial_metrics") or {}).get(k), metrics.get(k)):
            raise FinalTestError(f"{key}: metric {k} in the saved metrics/item does not equal the value recomputed from the persisted curve")
    if str(pd.Timestamp(metrics.get("first_decision_date")).date()) != TEST_WINDOW_IDENTITY["first_usable_decision"] \
            or str(pd.Timestamp(metrics.get("last_decision_date")).date()) != TEST_WINDOW_IDENTITY["last_usable_decision"]:
        raise FinalTestError(f"{key}: saved metrics do not carry the frozen first / last decision dates")
    if expected["experiment"] in R2_EXPERIMENTS:
        for k, v in diag.items():
            if not _close(metrics.get(k), v) or not _close((ev.get("r2_diagnostics") or {}).get(k), v):
                raise FinalTestError(f"{key}: R2 diagnostic {k} does not equal the value recomputed from the persisted curve")
    elif ev.get("r2_diagnostics") is not None:
        raise FinalTestError(f"{key}: R1 evidence carries R2 diagnostics")
    return {"item": ev, "metrics": metrics, "curve": curve, "hashes": hashes, "diagnostics": diag}


def persist_item_evidence(ctx: RunContext, token: ActiveAuthorization, state: dict[str, Any], item: dict[str, Any],
                          result: dict[str, Any]) -> dict[str, str]:
    """
    ``result``: {"curve": DataFrame, "metrics": dict}.  Writes to ``<item>.tmp/``, fsyncs, hashes, atomically
    promotes to ``<item>/`` and returns the evidence hashes.  A crash before promotion leaves only a ``.tmp`` directory.
    """
    revalidate(ctx, token, "evidence commit")
    _require_active_item(ctx, token, item, "evidence commit")
    final_dir = _item_dir(ctx, item)
    if final_dir.exists():
        raise FinalTestError(f"{item['key']}: promoted evidence already exists; it must be reconciled, never overwritten")
    tmp_dir = final_dir.with_name(final_dir.name + ".tmp")
    _probe("evidence_before_tmp_create")
    if tmp_dir.exists():
        for p in sorted(tmp_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        tmp_dir.rmdir()
    tmp_dir.mkdir(parents=True, exist_ok=False)
    curve: pd.DataFrame = result["curve"]
    metrics: dict[str, Any] = result["metrics"]
    if int(metrics.get("n_steps", -1)) != N_TEST_TRANSITIONS or len(curve) != N_TEST_TRANSITIONS:
        raise FinalTestError(f"{item['key']}: evidence does not cover exactly {N_TEST_TRANSITIONS} transitions")
    curve.to_csv(tmp_dir / "test_curve.csv")
    (tmp_dir / "test_metrics.json").write_text(json.dumps(metrics, default=ppo_e1._json_default, indent=1, sort_keys=True), encoding="utf-8")
    _probe("evidence_after_tmp_files_written")
    for name in ("test_curve.csv", "test_metrics.json"):
        _fsync_file(tmp_dir / name)
    _probe("evidence_after_tmp_fsync")
    hashes = {"test_curve.csv": sha256_of(tmp_dir / "test_curve.csv"), "test_metrics.json": sha256_of(tmp_dir / "test_metrics.json")}
    evidence = {
        **expected_item_identity(state, item), "evidence_sha256": hashes,
        "financial_metrics": {k: metrics.get(k) for k in METRIC_KEYS},
        "r2_diagnostics": ({k: metrics.get(k) for k in DIAG_KEYS} if item["experiment"] in R2_EXPERIMENTS else None),
        "committed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (tmp_dir / "item.json").write_text(json.dumps(evidence, default=ppo_e1._json_default, indent=1, sort_keys=True), encoding="utf-8")
    _fsync_file(tmp_dir / "item.json")
    _fsync_dir(tmp_dir)
    hashes["item.json"] = sha256_of(tmp_dir / "item.json")
    validate_item_evidence(tmp_dir, expected_item_identity(state, item), hashes)      # the evidence must be valid BEFORE promotion
    for p in tmp_dir.iterdir():
        os.chmod(p, 0o444)
    _probe("evidence_before_rename")
    os.rename(tmp_dir, final_dir)          # atomic promotion
    _probe("evidence_after_rename")
    _fsync_dir(final_dir.parent)
    _probe("evidence_after_parent_fsync")
    return hashes


def verify_item_evidence(ctx: RunContext, state: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """A COMMITTED item's evidence must exist, hash exactly as recorded in the run state and validate completely."""
    return validate_item_evidence(_item_dir(ctx, item), expected_item_identity(state, item), item.get("evidence_sha256") or {})["item"]


def _record_event(state: dict[str, Any], item: dict[str, Any], event: str, detail: str | None = None) -> None:
    item["recovery_event"] = event
    state.setdefault("recovery_events", []).append({"key": item["key"], "event": event, "detail": detail,
                                                    "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})


def _reconcile_promoted_evidence(ctx: RunContext, state: dict[str, Any], item: dict[str, Any]) -> None:
    """Promoted evidence for an IN_PROGRESS item: adopt if complete and valid, otherwise ABORT.  Never re-predict, never delete."""
    d = _item_dir(ctx, item)
    try:
        checked = validate_item_evidence(d, expected_item_identity(state, item))
    except FinalTestError as exc:
        _record_event(state, item, "PROMOTED_EVIDENCE_INVALID", str(exc))
        state["status"] = "ABORTED"
        _save_state(ctx, state)
        raise FinalTestError(f"{item['key']}: promoted evidence exists but is partial / invalid; the prospectively fixed rule aborts the run "
                             f"without deleting, overwriting or re-predicting ({exc})") from exc
    item["state"], item["evidence_sha256"] = "COMMITTED", checked["hashes"]
    item["committed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _record_event(state, item, "PROMOTED_EVIDENCE_RECONCILED")
    _save_state(ctx, state)


# ───────────────────────────────────────────── the one-shot run (refused without the human gate); resumable
def _real_item_evaluator(ctx: RunContext, token: ActiveAuthorization, item: dict[str, Any]) -> dict[str, Any]:
    require_context(ctx, "real item evaluator", production_required=True)
    res = evaluate_test_item(ctx, token, item)
    return {"curve": res.curve, "metrics": res.metrics}


def _open_or_resume(ctx: RunContext, token_probe: ActiveAuthorization, source_identity: dict[str, Any],
                    execution_commit: str) -> tuple[ActiveAuthorization, dict[str, Any]]:
    """Prerequisites → (publish exclusive run state | resume the same run).  Lock held by the caller; the source identity
    and the bound execution commit B were established by the caller BEFORE the lock.  No TEST access here."""
    production = ctx.kind == PRODUCTION_FINAL_TEST
    marker = ctx.results_dir / MARKER_NAME
    state = _read_json(ctx.run_state_path) if ctx.run_state_path.exists() else None
    if state is not None and state.get("status") == "COMPLETE":
        raise FinalTestError("the one-time TEST evaluation has already been completed for this frozen cohort; a second execution is refused")
    if marker.exists() and (state is None or state.get("status") != "FINALIZING"):
        raise FinalTestError("a completion marker exists without a resumable FINALIZING run state; a second execution is refused")
    if state is None:
        if ctx.results_dir.exists() and any(ctx.results_dir.iterdir()):
            raise FinalTestError(f"{ctx.results_dir}: result namespace is not empty but no run state exists; refusing to start")
        cohort = verify_cohort(ctx.accepted_root, ctx.final5_root)
        if production:
            verify_final5_closure()
            verify_accepted_artifacts_unchanged()
        runtime = verify_runtime_environment()
        if production:
            ctx.cfg = load_foundation_config()
            verify_runtime_config(ctx.cfg)
            ctx.frame, ctx.dataset_identity = load_frozen_dataset(ctx.cfg)
            frozen_test_grid()
        else:
            ctx.dataset_identity = {"path": "synthetic", "sha256": None, "rows_after_cutoff": 0}
        state = _new_run_state(ctx, token_probe.record_sha256, cohort, source_identity, runtime)
        state["execution_commit"] = execution_commit
        _publish_exclusive(ctx.run_state_path, json.dumps(state, indent=1, sort_keys=True, default=ppo_e1._json_default))
    else:
        if state.get("authorization_sha256") != token_probe.record_sha256:
            raise TestNotAuthorized("the authorization record differs from the one bound to the active run; refused")
        if state.get("cohort_manifest_sha256") != cohort_manifest_sha256() or state.get("results_dir") != str(ctx.results_dir.resolve()) \
                or state.get("context_kind") != ctx.kind or state.get("test_grid_sha256") != FROZEN_TEST_GRID_SHA256:
            raise FinalTestError("the active run state does not belong to this frozen cohort / canonical root / context kind; refused")
        if state.get("status") not in ("OPENED", "FINALIZING"):
            raise FinalTestError(f"run state status {state.get('status')!r} cannot be resumed")
        if [item_identity(p) for p in state.get("plan", [])] != list(CANONICAL_PLAN_IDENTITY):
            raise FinalTestError("the durable plan is not the canonical 48-item plan; resumption refused")
        if not (state.get("execution_commit") == execution_commit == token_probe.execution_commit):
            raise TestNotAuthorized(f"the run was opened under execution commit {str(state.get('execution_commit'))[:12]}; the current verified "
                                    f"execution identity is {execution_commit[:12]} and the authorization names {token_probe.execution_commit[:12]}; "
                                    "resumption under a different HEAD / authorization is refused")
        verify_cohort(ctx.accepted_root, ctx.final5_root)
        if _compact_source_identity(source_identity) != _compact_source_identity(state["source_identity"]):
            raise FinalTestError("the harness / source identity changed since the run was opened; resumption refused")
        runtime = verify_runtime_environment()
        if runtime != state.get("runtime"):
            raise FinalTestError("runtime identity differs from the opened run; resumption refused")
        if production:
            verify_final5_closure()
            verify_accepted_artifacts_unchanged()
            ctx.cfg = load_foundation_config()
            verify_runtime_config(ctx.cfg)
            ctx.frame, ctx.dataset_identity = load_frozen_dataset(ctx.cfg)
            frozen_test_grid()
        else:
            ctx.dataset_identity = {"path": "synthetic", "sha256": None, "rows_after_cutoff": 0}
        if ctx.dataset_identity != state.get("dataset"):
            raise FinalTestError("dataset identity differs from the opened run; resumption refused")
    token = ActiveAuthorization(token_probe.record_path, token_probe.record_sha256, token_probe.cohort_manifest_sha256,
                                state["run_id"], token_probe.execution_commit, secrets.token_hex(8))
    revalidate(ctx, token, "run opening")
    return token, state


def run_final_test(open_test: bool = False, ctx: RunContext | None = None) -> dict[str, Any]:
    """
    Production: ``run_final_test(open_test=True)`` on the single canonical root.  Synthetic fixtures (synthetic
    evaluators, temp roots) are the only way to pass a context.  Prospectively fixed recovery rule: see module doc.
    """
    if ctx is None:
        ctx = production_context()
    elif getattr(ctx, "kind", None) == PRODUCTION_FINAL_TEST or getattr(ctx, "production", None) is True:
        raise FinalTestError("a production context must not be supplied explicitly; call run_final_test(open_test=True)")
    require_context(ctx, "run")
    token_probe = require_test_authorization(open_test, ctx)         # gate first: no data / env / model / file work before it
    source_identity = verify_source_identity(ctx.harness_anchors_path, ctx.require_clean_tree)   # A→B (production) before anything else
    execution_commit = bind_execution_commit(ctx, token_probe.execution_commit, source_identity)  # authorization must name exactly B
    source_identity = {**source_identity, "execution_commit": execution_commit}                  # recorded identity carries B
    _acquire_run_lock(ctx)                                           # exclusive ownership for the whole execution
    try:
        token, state = _open_or_resume(ctx, token_probe, source_identity, execution_commit)
        evaluator = ctx.item_evaluator if ctx.kind == SYNTHETIC_TEST_FIXTURE else _real_item_evaluator
        ctx.results_dir.mkdir(parents=True, exist_ok=True)
        for item in state["plan"]:
            if item["state"] == "COMMITTED":
                verify_item_evidence(ctx, state, item)                        # reuse, never re-evaluate
                continue
            if item["state"] == "IN_PROGRESS":
                if _item_dir(ctx, item).exists():                             # promoted evidence FIRST: reconcile, never re-predict
                    _reconcile_promoted_evidence(ctx, state, item)
                    continue
                if item.get("recovery_retry"):
                    state["status"] = "ABORTED"
                    _record_event(state, item, "ABORTED_SECOND_INTERRUPTION")
                    _save_state(ctx, state)
                    raise FinalTestError(f"{item['key']}: interrupted twice (IN_PROGRESS after its single RECOVERY_RETRY); "
                                         "the prospectively fixed policy aborts the run — no further discretionary rerun")
                item["recovery_retry"] = True                                 # the one pre-authorized deterministic retry
                item["recovery_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                _record_event(state, item, "RECOVERY_RETRY")
            elif item["state"] != "PENDING":
                raise FinalTestError(f"{item['key']}: unknown item state {item['state']!r}")
            item["state"], item["attempts"] = "IN_PROGRESS", int(item.get("attempts", 0)) + 1
            item["started_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _save_state(ctx, state)                                           # durable IN_PROGRESS before evaluation
            revalidate(ctx, token, f"item {item['key']}")
            _require_active_item(ctx, token, item, f"item {item['key']}")
            result = evaluator(ctx, token, item)
            hashes = persist_item_evidence(ctx, token, state, item, result)
            item["state"], item["evidence_sha256"] = "COMMITTED", hashes
            item["committed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _probe("state_before_committed_save")
            _save_state(ctx, state)
            _probe("state_after_committed_save")
        return finalize_run(ctx, token, state)
    finally:
        _release_run_lock(ctx)


def _plan_projection(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**{k: it.get(k) for k in IDENTITY_KEYS}, "state": it["state"], "attempts": it["attempts"],
             "recovery_retry": bool(it.get("recovery_retry", False)), "recovery_event": it.get("recovery_event"),
             "evidence_sha256": it.get("evidence_sha256")} for it in plan]


def build_manifest(ctx: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, prediction-free: assembled from the run state and the completely verified evidence only."""
    if any(item["state"] != "COMMITTED" for item in state["plan"]):
        raise FinalTestError("finalization requires every plan item COMMITTED")
    if [item_identity(p) for p in state["plan"]] != list(CANONICAL_PLAN_IDENTITY):
        raise FinalTestError("the durable plan is not the canonical 48-item plan")
    evidence = [verify_item_evidence(ctx, state, item) for item in state["plan"]]
    learned = [ev for ev in evidence if ev["kind"] == "learned"]
    baselines = [ev for ev in evidence if ev["kind"] == "baseline"]
    if [(ev["experiment"], ev["seed"]) for ev in learned] != list(COHORT_ORDER) or len(baselines) != 4 * len(BASELINE_LABELS):
        raise FinalTestError("evidence set is not the exact frozen learned/baseline set")
    return {
        "study_stage": STUDY_STAGE, "TEST_OPENED": True, "run_id": state["run_id"], "context_kind": state["context_kind"],
        "execution_commit": state["execution_commit"],
        "started_utc": state["started_utc"], "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authorization": {"path": state["authorization_path"], "sha256": state["authorization_sha256"]},
        "run_state_sha256_at_finalization": None, "results_dir": state["results_dir"],
        "cohort_manifest": state["cohort_manifest"], "cohort_manifest_sha256": state["cohort_manifest_sha256"],
        "checkpoints": state["checkpoints"], "test_window": state["test_window"], "test_grid_sha256": state["test_grid_sha256"],
        "source_identity": state["source_identity"], "dataset": state["dataset"], "runtime": state["runtime"],
        "recovery_events": state.get("recovery_events", []),
        "items": _plan_projection(state["plan"]),
        "learned_rows": [{**ev["financial_metrics"], **(ev["r2_diagnostics"] or {}), "experiment": ev["experiment"], "seed": ev["seed"],
                          "policy": ev["policy"], "policy_kind": ev["policy_kind"], "policy_name": ev["policy_name"], "model_sha256": ev["model_sha256"],
                          "key": ev["key"]} for ev in learned],
        "baseline_rows": [{**ev["financial_metrics"], **(ev["r2_diagnostics"] or {}), "experiment": ev["experiment"], "seed": ev["seed"],
                           "policy": ev["policy"], "policy_kind": ev["policy_kind"], "random_seed": ev["random_seed"], "policy_name": ev["policy_name"],
                           "key": ev["key"]} for ev in baselines],
        "post_test_rules": list(POST_TEST_RULES),
    }


def finalize_run(ctx: RunContext, token: ActiveAuthorization, state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic finalization: manifest (if absent), marker bound to the manifest hash, run state COMPLETE, read-only."""
    revalidate(ctx, token, "finalization")
    manifest_path, marker = ctx.results_dir / MANIFEST_NAME, ctx.results_dir / MARKER_NAME
    if state["status"] != "FINALIZING":
        state["status"] = "FINALIZING"
        _save_state(ctx, state)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_id") != state["run_id"]:
            raise FinalTestError("an existing manifest belongs to a different run; finalization refused")
    else:
        manifest = build_manifest(ctx, state)
        manifest["run_state_sha256_at_finalization"] = sha256_of(ctx.run_state_path)
        _write_durable(manifest_path, json.dumps(manifest, indent=1, sort_keys=True, default=ppo_e1._json_default), readonly=True)
    manifest_sha = sha256_of(manifest_path)
    if not marker.exists():
        _write_durable(marker, f"{STUDY_STAGE} one-time TEST evaluation completed\nrun_id {state['run_id']}\n"
                               f"manifest_sha256 {manifest_sha}\ncohort_manifest_sha256 {state['cohort_manifest_sha256']}\n", readonly=True)
    for p in ctx.results_dir.rglob("*"):          # results read-only BEFORE the run is declared COMPLETE
        if p.is_file():
            os.chmod(p, 0o444)
    state["status"], state["manifest_sha256"] = "COMPLETE", manifest_sha
    state["finished_utc"] = manifest["finished_utc"]
    _save_state(ctx, state)
    os.chmod(ctx.run_state_path, 0o444)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="prospective one-time final TEST harness (refuses without the human gate).")
    ap.add_argument("--verify-cohort", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--preflight-open", action="store_true", help="read-only: cohort, final5 closure, sources, harness anchors, runtime, dataset, grid")
    ap.add_argument("--print-harness-anchors", metavar="COMMIT", help="read-only: print the anchor record for reviewed commit A (never written)")
    ap.add_argument("--open-test", action="store_true", help="request the ONE-TIME TEST evaluation (requires AUTHORIZATION.json)")
    args = ap.parse_args(argv)
    if args.verify_cohort:
        c = verify_cohort()
        print(f"cohort bound: {c['n']} checkpoints; manifest sha256 {c['cohort_manifest_sha256']}; test grid sha256 {FROZEN_TEST_GRID_SHA256}")
    if args.status:
        print(json.dumps(status(), indent=1))
    if args.print_harness_anchors:
        print(json.dumps(build_harness_anchors(args.print_harness_anchors), indent=1, sort_keys=True))
    if args.preflight_open:
        verify_cohort(); verify_final5_closure(); verify_accepted_artifacts_unchanged()
        verify_source_identity()
        verify_runtime_environment()
        cfg = load_foundation_config(); verify_runtime_config(cfg); load_frozen_dataset(cfg); frozen_test_grid()
        print("pre-opening prerequisites satisfied (authorization not checked, TEST not touched)")
    if args.open_test:
        m = run_final_test(open_test=True)
        print(f"ONE-TIME TEST EVALUATION COMPLETED: run {m['run_id']}, {len(m['learned_rows'])} learned items")
    if not (args.verify_cohort or args.status or args.preflight_open or args.open_test or args.print_harness_anchors):
        ap.error("choose --verify-cohort, --status, --preflight-open, --print-harness-anchors or --open-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
