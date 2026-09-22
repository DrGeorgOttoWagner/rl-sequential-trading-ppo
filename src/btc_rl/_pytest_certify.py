"""
_pytest_certify.py — pytest plugin used only by the certification gate.

Loaded explicitly with ``-p btc_rl._pytest_certify`` in a subprocess whose
environment has every ``PYTEST_*`` variable removed and whose ini ``addopts``
is overridden to empty.  It records the exact number of collected items
before any deselection, the number deselected, and one outcome per test
(passed / failed / error / skipped / xfailed / xpassed) into the JSON file
named by ``BTC_RL_CERT_REPORT``.  The gate then requires collected ==
expected, deselected == 0, passed == expected, and every other bucket == 0.
"""

from __future__ import annotations

import json
import os

import pytest

REPORT_ENV = "BTC_RL_CERT_REPORT"


class _Recorder:
    def __init__(self, path: str) -> None:
        self.path = path
        self.collected = 0
        self.deselected = 0
        self.outcomes: dict[str, str] = {}
        self.exitstatus: int | None = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):
        self.collected = len(items)  # before -k / -m / --deselect filtering runs

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_runtest_logreport(self, report):
        nodeid = report.nodeid
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        elif report.failed:
            outcome = "failed" if report.when == "call" else "error"
        elif report.skipped:
            outcome = "skipped"
        elif report.when == "call":
            outcome = "passed"
        else:
            return  # setup/teardown passed: not an outcome
        prev = self.outcomes.get(nodeid)
        if prev in ("failed", "error", "skipped", "xfailed", "xpassed") and outcome == "passed":
            return  # never downgrade a bad outcome
        self.outcomes[nodeid] = outcome

    def pytest_sessionfinish(self, session, exitstatus):
        self.exitstatus = int(exitstatus)
        counts = {k: 0 for k in ("passed", "failed", "error", "skipped", "xfailed", "xpassed")}
        for o in self.outcomes.values():
            counts[o] += 1
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "collected": self.collected,
                    "deselected": self.deselected,
                    "executed": len(self.outcomes),
                    "counts": counts,
                    "exitstatus": self.exitstatus,
                    "nodeids": sorted(self.outcomes),
                },
                fh,
                indent=1,
            )


def pytest_configure(config):
    path = os.environ.get(REPORT_ENV)
    if path:
        config.pluginmanager.register(_Recorder(path), "btc_rl_certify_recorder")
