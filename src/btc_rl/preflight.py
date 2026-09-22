"""
preflight.py — the certification gate that must pass before any PPO run.

    python -m btc_rl.preflight                # complete certification
    python -m btc_rl.preflight --diagnostic   # partial: never certifies
    python -m btc_rl.preflight --skip-replay  # partial: never certifies

Exit codes: 0 = CERTIFIED (complete gate, every check PASS);
            1 = FAIL (at least one check failed);
            2 = PARTIAL / NOT CERTIFIED (a component was skipped by request;
                usable for diagnosis only, never as PPO approval).

Checks
------
  1. runtime      exact CPython 3.12.14, equal to ``.python-version``
  2. dependencies ``uv.lock`` present; every locked runtime + dev dependency
                  applicable to this platform is installed at exactly the
                  locked version (markers evaluated); requires-python pinned
  3. config       ``configs/foundation.toml`` equals the frozen constants
  4. dataset      raw file present; SHA-256, first date, last retained date and
                  row count equal the frozen identity; integrity validation
  5. layout       split contiguity, warm-up and purge rule
  6. artifacts    ML Variant A replay artifact present, unique, binary, and
                  covering every fill date of the test split
  7. environment  gymnasium check_env on train/validation/test, deterministic
                  and random-start
  8. baselines    always-flat stays at initial equity with zero legs;
                  buy-and-hold equals the analytic open-to-open path (1e-12);
                  random policy is seed-deterministic; every observation of
                  every split (raw and train-scaled) lies in observation_space;
                  replay runs end to end on the test split
  8b. O2          O2 scaler fitted on TRAIN-only feature rows,
                  warm-up covers the first TRAIN decision, every TRAIN and
                  VALIDATION O2 observation is finite and inside the Box
  8c. R2          R2 coefficients are the frozen values; on one
                  seeded fixed action path over TRAIN and over VALIDATION the
                  R2 environment reproduces the certified environment's
                  observations, equity, costs, positions and drawdown exactly,
                  every R2 term is finite and non-negative, and the identity
                  R2 = R1 - turnover - drawdown holds (no learned policy, no TEST)
  9. tests        the mandatory suite in a scrubbed subprocess (no PYTEST_*
                  environment, ini addopts overridden, explicit plugin):
                  collected == EXPECTED_MANDATORY_TESTS, 0 deselected,
                  passed == expected, 0 failed/error/skipped/xfailed/xpassed

Test-window policy: the test-split runs here are mechanical integrity and
baseline verification.  They are not model selection.  PPO hyperparameters
and reward coefficients must never be chosen on test performance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = PACKAGE_ROOT / "tests"
LOCK_PATH = PACKAGE_ROOT / "uv.lock"
PYTHON_VERSION_FILE = PACKAGE_ROOT / ".python-version"
PROJECT_NAME = "btc-usdt-rl-trading"

REQUIRED_PYTHON = (3, 12, 14)          # exact validated CPython patch release
REQUIRED_IMPLEMENTATION = "CPython"
LOCK_EXTRAS = ("dev",)

# Exact number of mandatory tests.  Update ONLY when tests are legitimately added
# or removed; tests/test_certification.py verifies it against a fresh collection.
EXPECTED_MANDATORY_TESTS = 651  # 121 foundation + 103 PPO E1 + 66 PPO E2 + 96 PPO E3 + 70 PPO E4 + 102 final cohort + 93 final TEST harness (tests/test_final_test.py)


# ────────────────────────────────────────────────────────────── results
@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class PreflightResult:
    checks: list[Check] = field(default_factory=list)
    partial_reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def complete(self) -> bool:
        return not self.partial_reasons

    @property
    def certified(self) -> bool:
        """True only for the complete gate with every check passing."""
        return self.ok and self.complete

    @property
    def verdict(self) -> str:
        if self.certified:
            return "CERTIFIED"
        if not self.ok:
            return "FAIL"
        return "PARTIAL / NOT CERTIFIED"

    @property
    def exit_code(self) -> int:
        if self.certified:
            return 0
        if not self.ok:
            return 1
        return 2

    def add(self, name: str, fn: Callable[[], str]) -> bool:
        try:
            detail = fn()
            self.checks.append(Check(name, True, detail))
            return True
        except Exception as exc:  # noqa: BLE001 - every failure must be reported, never raised
            tb = traceback.format_exc().strip().splitlines()[-1]
            self.checks.append(Check(name, False, f"{type(exc).__name__}: {exc}" if str(exc) else tb))
            return False


# ────────────────────────────────────────────────────────────── 1. runtime
def check_runtime(
    version_info: tuple[int, int, int] | None = None,
    implementation: str | None = None,
    python_version_file: Path = PYTHON_VERSION_FILE,
) -> str:
    vi = tuple(version_info) if version_info is not None else tuple(sys.version_info[:3])
    impl = implementation if implementation is not None else platform.python_implementation()
    want = ".".join(map(str, REQUIRED_PYTHON))
    got = ".".join(map(str, vi))
    if impl != REQUIRED_IMPLEMENTATION:
        raise RuntimeError(f"interpreter is {impl}, required {REQUIRED_IMPLEMENTATION}")
    if vi != REQUIRED_PYTHON:
        raise RuntimeError(f"Python {got} is not the exact validated {want}")
    if not Path(python_version_file).exists():
        raise FileNotFoundError(f".python-version missing at {python_version_file}")
    pinned = Path(python_version_file).read_text(encoding="utf-8").strip()
    if pinned != want:
        raise RuntimeError(f".python-version pins {pinned!r}, required {want!r}")
    return f"{impl} {got} == .python-version {pinned}"


# ────────────────────────────────────────────────────────────── 2. locked dependencies
def locked_requirements(lock_path: Path = LOCK_PATH, extras: tuple[str, ...] = LOCK_EXTRAS) -> dict[str, str]:
    """
    Name -> locked version for the project's dependency closure applicable to
    this platform, resolved from uv.lock with environment-marker evaluation.
    """
    from packaging.markers import Marker  # locked dependency of pytest; present in every certified env

    if not Path(lock_path).exists():
        raise FileNotFoundError(f"uv.lock missing at {lock_path}; run `uv lock`")
    with open(lock_path, "rb") as fh:
        lock = tomllib.load(fh)
    pkgs = _applicable_lock_packages(lock, Marker)
    if PROJECT_NAME not in pkgs:
        raise RuntimeError(f"uv.lock has no entry for {PROJECT_NAME}")
    root = pkgs[PROJECT_NAME]
    queue = list(root.get("dependencies", []))
    for extra in extras:
        queue.extend(root.get("optional-dependencies", {}).get(extra, []))
    required: dict[str, str] = {}
    while queue:
        dep = queue.pop()
        marker = dep.get("marker")
        if marker and not Marker(marker).evaluate():
            continue
        name = dep["name"]
        if name in required:
            continue
        if name not in pkgs:
            raise RuntimeError(f"uv.lock dependency {name} has no package entry")
        required[name] = pkgs[name]["version"]
        queue.extend(pkgs[name].get("dependencies", []))
    return required


def _applicable_lock_packages(lock: Mapping, marker_cls) -> dict[str, dict]:
    """
    Name -> lock entry applicable to this platform.  A package may appear more
    than once in ``uv.lock`` with disjoint ``resolution-markers`` (
    ``torch`` 2.14.0 for darwin and 2.14.0+cpu elsewhere); exactly one entry
    must apply here.  Entries without resolution markers apply everywhere.
    """
    chosen: dict[str, dict] = {}
    for entry in lock.get("package", []):
        markers = entry.get("resolution-markers")
        if markers and not any(marker_cls(m).evaluate() for m in markers):
            continue
        name = entry["name"]
        if name in chosen:
            raise RuntimeError(
                f"uv.lock has more than one applicable entry for {name}: "
                f"{chosen[name]['version']} and {entry['version']}"
            )
        chosen[name] = entry
    return chosen


def check_locked_dependencies(
    lock_path: Path = LOCK_PATH,
    installed_version: Callable[[str], str] | None = None,
    pyproject_path: Path = PACKAGE_ROOT / "pyproject.toml",
) -> str:
    from packaging.markers import Marker
    from packaging.version import Version

    get = installed_version or importlib.metadata.version
    with open(lock_path, "rb") as fh:
        lock = tomllib.load(fh)
    with open(pyproject_path, "rb") as fh:
        project = tomllib.load(fh)["project"]
    if project.get("requires-python") != ">=3.12,<3.13":
        raise RuntimeError(f"pyproject requires-python {project.get('requires-python')!r} != '>=3.12,<3.13'")
    if lock.get("requires-python") != "==3.12.*":
        raise RuntimeError(f"uv.lock requires-python {lock.get('requires-python')!r} != '==3.12.*'")

    required = locked_requirements(lock_path)
    required[PROJECT_NAME] = _applicable_lock_packages(lock, Marker)[PROJECT_NAME]["version"]
    problems = []
    for name, locked in sorted(required.items()):
        try:
            inst = get(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name}: locked {locked}, NOT INSTALLED")
            continue
        if Version(inst) != Version(locked):
            problems.append(f"{name}: installed {inst} != locked {locked}")
    if problems:
        raise RuntimeError("installed packages differ from uv.lock: " + "; ".join(problems))
    return f"{len(required)} locked packages match installed versions (platform markers evaluated)"


# ────────────────────────────────────────────────────────────── 9. mandatory tests
@dataclass
class TestRunSummary:
    collected: int
    deselected: int
    executed: int
    counts: dict[str, int]
    exitstatus: int
    expected: int
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def describe(self) -> str:
        c = self.counts
        base = (f"collected {self.collected}/{self.expected}, deselected {self.deselected}, passed {c['passed']}, "
                f"failed {c['failed']}, errors {c['error']}, skipped {c['skipped']}, "
                f"xfailed {c['xfailed']}, xpassed {c['xpassed']}")
        return base if self.ok else base + " | " + "; ".join(self.problems)


def scrubbed_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy of the environment with every PYTEST_* variable removed."""
    src = os.environ if base_env is None else base_env
    return {k: v for k, v in src.items() if not k.upper().startswith("PYTEST_")}


def run_mandatory_tests(
    tests_dir: Path = TESTS_DIR,
    expected_count: int | None = None,
    base_env: Mapping[str, str] | None = None,
    python: str = sys.executable,
) -> TestRunSummary:
    """
    Run the mandatory suite in a subprocess immune to inherited pytest
    selection: PYTEST_* removed, ini addopts overridden, explicit rootdir and
    config, cache disabled, certification plugin loaded explicitly.
    """
    expected = EXPECTED_MANDATORY_TESTS if expected_count is None else int(expected_count)
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "certify.json"
        env = scrubbed_env(base_env)
        env["BTC_RL_CERT_REPORT"] = str(report)
        cmd = [
            python, "-m", "pytest",
            "--rootdir", str(PACKAGE_ROOT),
            "-c", str(PACKAGE_ROOT / "pyproject.toml"),
            "-o", "addopts=",
            "-p", "no:cacheprovider",
            "-p", "btc_rl._pytest_certify",
            "-q",
            str(tests_dir),
        ]
        proc = subprocess.run(cmd, cwd=PACKAGE_ROOT, env=env, capture_output=True, text=True)
        if not report.exists():
            tail = (proc.stdout.strip().splitlines() or proc.stderr.strip().splitlines() or ["<no output>"])[-1]
            return TestRunSummary(0, 0, 0, dict.fromkeys(
                ("passed", "failed", "error", "skipped", "xfailed", "xpassed"), 0), proc.returncode, expected,
                [f"certification plugin produced no report (pytest exit {proc.returncode}): {tail}"])
        data = json.loads(report.read_text(encoding="utf-8"))
    c = data["counts"]
    problems = []
    if data["collected"] != expected:
        problems.append(f"collected {data['collected']} != expected {expected}")
    if data["deselected"]:
        problems.append(f"{data['deselected']} deselected")
    if c["passed"] != expected:
        problems.append(f"passed {c['passed']} != expected {expected}")
    for k in ("failed", "error", "skipped", "xfailed", "xpassed"):
        if c[k]:
            problems.append(f"{c[k]} {k}")
    if data["executed"] != expected:
        problems.append(f"executed {data['executed']} != expected {expected}")
    if proc.returncode != 0 or data["exitstatus"] != 0:
        problems.append(f"pytest exit {proc.returncode}/{data['exitstatus']}")
    return TestRunSummary(data["collected"], data["deselected"], data["executed"], c,
                          data["exitstatus"], expected, problems)


# ────────────────────────────────────────────────────────────── gate
def run_preflight(require_replay: bool = True, run_tests: bool = True) -> PreflightResult:
    from . import reference
    from .config import load_foundation_config
    from .data import load_dataset
    from .env import make_split_env
    from .evaluation import run_episode
    from .observations import ObservationConfig
    from .scaling import fit_train_return_scale
    from .policies import AlwaysFlat, AlwaysLong, RandomPolicy, make_variant_a_replay, v5b_artifact_path
    from .splits import check_split_layout, usable_decision_index_range

    res = PreflightResult()
    if not require_replay:
        res.partial_reasons.append("ML replay artifact check skipped by request (--skip-replay)")
    if not run_tests:
        res.partial_reasons.append("mandatory test suite not run (run_tests=False / --diagnostic)")
    state: dict = {}

    res.add("runtime", check_runtime)
    res.add("locked dependencies", check_locked_dependencies)

    def _config() -> str:
        state["cfg"] = load_foundation_config()
        return "foundation.toml equals frozen constants"
    if not res.add("config", _config):
        return res
    cfg = state["cfg"]

    def _dataset() -> str:
        frame, report = load_dataset(cfg.dataset)
        state["frame"] = frame
        return (f"{report['rows']} rows {report['first_date'].date()}..{report['last_date'].date()}, "
                f"sha256 {report['sha256'][:12]} verified")
    if not res.add("dataset identity + integrity", _dataset):
        return res
    frame = state["frame"]

    def _layout() -> str:
        layout = check_split_layout(frame, cfg.splits, lookback=cfg.observation.lookback)
        return "; ".join(f"{k}: {v['n_usable_decisions']} usable decisions, last mark {v['last_mark_date'].date()}"
                         for k, v in layout.items())
    res.add("split layout (purge rule)", _layout)

    if require_replay:
        def _artifact() -> str:
            p = v5b_artifact_path()
            if not p.exists():
                raise FileNotFoundError(f"mandatory ML replay artifact missing: {p}")
            state["replay"] = make_variant_a_replay(frame, cfg.splits["test"])
            return f"{p.name}: Variant A positions cover every test fill date"
        res.add("ML replay artifact", _artifact)
    else:
        res.checks.append(Check("ML replay artifact", True, "SKIPPED BY REQUEST (partial run)"))

    def _env_checks() -> str:
        from gymnasium.utils.env_checker import check_env
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*render modes.*")
            for name in ("train", "validation", "test"):
                check_env(make_split_env(frame, name, obs_config=cfg.observation, splits=cfg.splits))
                check_env(make_split_env(frame, name, obs_config=cfg.observation, splits=cfg.splits,
                                         episode_length=64, random_start=True))
        return "check_env passed on 3 splits x {deterministic, random-start}"
    res.add("gymnasium check_env", _env_checks)

    def _baselines() -> str:
        notes = []
        # TRAIN-only scaler: closes from the TRAIN start through the last usable TRAIN decision
        fit = fit_train_return_scale(frame, cfg.splits["train"])
        if fit.fit_start != cfg.splits["train"].start or fit.fit_end >= cfg.splits["train"].end:
            raise AssertionError("return scale fit interval is not inside TRAIN")
        scaled_cfg = ObservationConfig(lookback=cfg.observation.lookback, return_scale=fit.scale)
        notes.append(f"scale {fit.scale:.6g} fitted on TRAIN closes {fit.fit_start.date()}..{fit.fit_end.date()}")
        for name in ("train", "validation", "test"):
            for cost_name, cost in cfg.costs.items():
                env = make_split_env(frame, name, cost_config=cost, obs_config=cfg.observation,
                                     splits=cfg.splits, initial_equity=cfg.initial_equity)
                flat = run_episode(env, AlwaysFlat())
                if not ((flat.curve["equity"] == cfg.initial_equity).all() and flat.metrics["n_legs"] == 0):
                    raise AssertionError(f"{name}/{cost_name}: always-flat moved equity or paid legs")
                bh = run_episode(env, AlwaysLong())
                i0, last = usable_decision_index_range(frame, cfg.splits[name])
                ref = reference.buy_and_hold_equity(frame["open"].to_numpy(), i0, last - i0 + 1, cost,
                                                    cfg.initial_equity)
                if not np.allclose(bh.curve["equity"].to_numpy(), ref, rtol=1e-12, atol=0):
                    raise AssertionError(f"{name}/{cost_name}: buy-and-hold deviates from analytic path")
                if bh.metrics["n_legs"] != 1:
                    raise AssertionError(f"{name}/{cost_name}: buy-and-hold paid {bh.metrics['n_legs']} legs")
                r1 = run_episode(env, RandomPolicy(0), seed=0)
                r2 = run_episode(env, RandomPolicy(0), seed=0)
                if not r1.curve.equals(r2.curve):
                    raise AssertionError(f"{name}/{cost_name}: random policy not seed-deterministic")
            for oc in (cfg.observation, scaled_cfg):
                env = make_split_env(frame, name, obs_config=oc, splits=cfg.splits)
                if not (np.all(np.isfinite(env.observation_space.low)) and np.all(np.isfinite(env.observation_space.high))):
                    raise AssertionError(f"{name}: non-finite observation bounds")
                obs, _ = env.reset()
                n = 0
                done = False
                while True:
                    if not (np.all(np.isfinite(obs)) and env.observation_space.contains(obs)):
                        raise AssertionError(f"{name}: observation outside declared finite space at step {n}")
                    n += 1
                    if done:
                        break
                    obs, _, _, done, _ = env.step(1 if n % 2 else 0)
                notes.append(f"{name}:{n} obs finite+in space (scale={oc.return_scale:.4g})")
        if "replay" in state:
            env = make_split_env(frame, "test", obs_config=cfg.observation, splits=cfg.splits)
            rep = run_episode(env, state["replay"])
            i0, last = usable_decision_index_range(frame, cfg.splits["test"])
            if rep.metrics["n_steps"] != last - i0 + 1:
                raise AssertionError("replay did not cover the full test decision range")
            notes.append(f"replay {rep.metrics['n_steps']} steps")
        return "; ".join(notes)
    res.add("baseline sanity", _baselines)

    def _o2() -> str:
        # O2 engineered observation — TRAIN-only scaler interval, warm-up at the first TRAIN
        # decision, and every TRAIN / VALIDATION observation finite and inside the declared finite Box.
        # No environment is built and TEST is not walked (analytic bounds cover every validated frame).
        from .observations_o2 import (O2ObservationConfig, build_o2_observation, fit_o2_scaler,
                                      o2_observation_space, verify_o2_warmup)

        train = cfg.splits["train"]
        fit = fit_o2_scaler(frame, train)
        if fit.fit_close_start != train.start or fit.fit_close_end >= train.end:
            raise AssertionError("O2 scaler fit closes are not inside TRAIN")
        if fit.fit_row_start < fit.fit_close_start or fit.fit_row_end != fit.fit_close_end:
            raise AssertionError("O2 scaler fit rows are not inside the TRAIN fit closes")
        warm = verify_o2_warmup(frame, train)
        if warm["first_decision"] != str(train.start.date()) or warm["warmup_slack_rows"] < 0:
            raise AssertionError("O2 warm-up does not cover the first TRAIN decision")
        oc = O2ObservationConfig.from_fit(fit)
        space = o2_observation_space(oc)
        if not (np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high))):
            raise AssertionError("O2 observation bounds are not finite")
        closes = frame["close"].to_numpy(dtype=np.float64)
        notes = [f"O2 scaler fitted on TRAIN feature rows {fit.fit_row_start.date()}..{fit.fit_row_end.date()} "
                 f"(closes {fit.fit_close_start.date()}..{fit.fit_close_end.date()}, {fit.n_rows} rows)",
                 f"warm-up slack {warm['warmup_slack_rows']} rows at {warm['first_decision']}"]
        for name in ("train", "validation"):
            i0, last = usable_decision_index_range(frame, cfg.splits[name])
            n = 0
            for t in range(i0, last + 2):   # every decision observation plus the terminal observation
                for pos in (0, 1):
                    obs = build_o2_observation(closes[: t + 1], pos, oc)
                    if not (np.all(np.isfinite(obs)) and space.contains(obs)):
                        raise AssertionError(f"{name}: O2 observation outside the declared finite space at row {t}")
                n += 1
            notes.append(f"{name}:{n} O2 obs finite+in space")
        return "; ".join(notes)
    res.add("O2 engineered observation", _o2)

    def _r2() -> str:
        # R2 risk-aware reward — frozen coefficients; for one seeded fixed action path on TRAIN and on
        # VALIDATION the R2 environment reproduces the certified environment's observations, equity, costs,
        # positions and drawdown exactly, every R2 term is finite, no penalty is negative and the identity
        # R2 = R1 - turnover - drawdown holds.  No learned policy runs; TEST is not walked.
        from .reward_r2 import (DRAWDOWN_PENALTY_LAMBDA, FROZEN_R2, TURNOVER_PENALTY_LAMBDA, BtcUsdtTradingEnvR2,
                                compute_r2)

        if (TURNOVER_PENALTY_LAMBDA, DRAWDOWN_PENALTY_LAMBDA) != (0.0005, 0.10) or not FROZEN_R2.is_frozen:
            raise AssertionError("R2 coefficients are not the prospectively frozen values")
        fit = fit_train_return_scale(frame, cfg.splits["train"])
        oc = ObservationConfig(lookback=cfg.observation.lookback, return_scale=fit.scale)
        cost = cfg.costs["conservative"]
        notes = [f"R2 lambdas turnover={TURNOVER_PENALTY_LAMBDA} drawdown={DRAWDOWN_PENALTY_LAMBDA}"]
        for name in ("train", "validation"):
            w = cfg.splits[name]
            i0, last = usable_decision_index_range(frame, w)
            env1 = make_split_env(frame, w, cost_config=cost, obs_config=oc, splits=cfg.splits,
                                  initial_equity=cfg.initial_equity)
            env3 = BtcUsdtTradingEnvR2(frame, frame.index[i0], frame.index[last], cost_config=cost, obs_config=oc,
                                       initial_equity=cfg.initial_equity, price_end=w.end)
            rng = np.random.default_rng(3)
            o1, _ = env1.reset()
            o3, _ = env3.reset()
            n = 0
            penalised = 0
            while True:
                if not np.array_equal(o1, o3):
                    raise AssertionError(f"{name}: R2 environment observation differs from the certified O1 observation")
                a = int(rng.integers(0, 2))
                o1, r1, _, d1, i1 = env1.step(a)
                o3, r3, _, d3, i3 = env3.step(a)
                for k in ("equity", "position", "legs", "drawdown", "peak_equity", "cost_fraction",
                          "gross_simple_return", "fill_date", "mark_date"):
                    if i1[k] != i3[k]:
                        raise AssertionError(f"{name}: R2 environment changed the financial accounting ({k})")
                if i3["financial_reward_r1"] != r1 or d1 != d3:
                    raise AssertionError(f"{name}: R2 environment does not carry the exact R1 reward")
                terms = (i3["turnover_penalty"], i3["drawdown_penalty"], i3["incremental_drawdown"], r3)
                if not all(np.isfinite(v) for v in terms) or terms[0] < 0 or terms[1] < 0:
                    raise AssertionError(f"{name}: non-finite or negative R2 term at step {n}")
                if abs(r3 - (r1 - i3["turnover_penalty"] - i3["drawdown_penalty"])) > 1e-15:
                    raise AssertionError(f"{name}: R2 identity violated at step {n}")
                ref = compute_r2(r1, i3["legs"], i3["equity_before"], i3["running_peak_before"], i3["equity"],
                                 i3["peak_equity"])
                if ref.reward_r2 != r3:
                    raise AssertionError(f"{name}: R2 differs from the pure formula at step {n}")
                penalised += int(i3["drawdown_penalty"] > 0)
                n += 1
                if d3:
                    break
            if n != last - i0 + 1:
                raise AssertionError(f"{name}: fixed path did not cover the full decision range")
            notes.append(f"{name}:{n} steps R1/R2 accounting identical, R2 finite ({penalised} drawdown-penalised)")
        return "; ".join(notes)
    res.add("R2 risk-aware reward", _r2)

    if run_tests:
        def _tests() -> str:
            summary = run_mandatory_tests(TESTS_DIR, EXPECTED_MANDATORY_TESTS)
            if not summary.ok:
                raise AssertionError(summary.describe())
            return summary.describe()
        res.add("mandatory tests (exact count, no skip/deselect/xfail/xpass)", _tests)
    else:
        res.checks.append(Check("mandatory tests (exact count, no skip/deselect/xfail/xpass)", True,
                                "NOT RUN (partial run) -> NOT CERTIFIED"))
    return res


def format_report(res: PreflightResult) -> str:
    width = max(len(c.name) for c in res.checks)
    head = "certification gate" if res.complete else "PARTIAL DIAGNOSTIC RUN — NOT CERTIFIED"
    lines = [head]
    for c in res.checks:
        lines.append(f"  [{'PASS' if c.ok else 'FAIL'}] {c.name:<{width}}  {c.detail}")
    for r in res.partial_reasons:
        lines.append(f"  [PARTIAL] {r}")
    lines.append(f"preflight verdict: {res.verdict} (exit {res.exit_code}). Test-split runs above are mechanical "
                 "verification only; never tune PPO on test performance.")
    if not res.certified:
        lines.append("This result is NOT a PPO approval.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Certification gate for the BTC/USDT RL foundation (pre-PPO).")
    ap.add_argument("--skip-replay", action="store_true",
                    help="do not require the ML Variant A replay artifact (partial run; never certifies)")
    ap.add_argument("--diagnostic", action="store_true",
                    help="skip the mandatory test suite (partial run; never certifies)")
    args = ap.parse_args(argv)
    t0 = time.time()
    res = run_preflight(require_replay=not args.skip_replay, run_tests=not args.diagnostic)
    print(format_report(res))
    print(f"({time.time() - t0:.1f}s)")
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
