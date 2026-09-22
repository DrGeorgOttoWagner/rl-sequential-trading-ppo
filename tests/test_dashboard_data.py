"""Static checks of the read-only dashboard: no embedded data, no replay code path, fail-closed loader.

When an evidence package is installed under dashboard/data/, the structural checks of its entry point run as well;
without one, those checks skip. Nothing here fetches, trains or computes a result."""
import json
import re
from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "dashboard"
ENTRY = DASH / "data" / "provenance.json"


def test_dashboard_files_present():
    for rel in ("index.html", "css/app.css", "js/data.js", "js/charts.js", "js/app.js"):
        assert (DASH / rel).is_file(), rel


def test_no_replay_code_path():
    for rel in ("index.html", "js/data.js", "js/app.js"):
        text = (DASH / rel).read_text(encoding="utf-8")
        assert "replay.js" not in text, rel
        assert "data/replay" not in text, rel


def test_no_embedded_result_data():
    html = (DASH / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "application/json" not in html
    # no numeric result literal of the frozen tables may be embedded; design constants (0.10% fee + 0.05% slippage per trade, 0.9985, coefficients) are allowed
    for rel in ("index.html", "js/app.js", "js/data.js"):
        text = (DASH / rel).read_text(encoding="utf-8")
        assert not re.search(r"-?0\.\d{6,}", text), rel


def test_loader_fails_closed():
    js = (DASH / "js/data.js").read_text(encoding="utf-8")
    assert "NoPackageError" in js and "IntegrityError" in js
    assert "dataRoot: 'data/'" in js
    assert "fetch(CONTRACT.dataRoot + CONTRACT.entryPoint" in js
    assert "bindingAllowlist" in js
    assert re.search(r"fetch\(['\"]http", js) is None


@pytest.mark.skipif(not ENTRY.exists(), reason="no evidence package installed under dashboard/data/")
def test_entry_point_structure():
    m2 = json.loads(ENTRY.read_text(encoding="utf-8"))
    for key in ("dashboard_set", "dashboard_set_id", "evidence_set", "producer", "study_evidence_status", "public_provenance", "notices"):
        assert key in m2, key
    projections = m2["dashboard_set"]["descriptors"]
    assert [p["physical_copy_id"] for p in projections] == ["EC-7a#1", "EC-7b#1", "EC-7c#1", "EC-7d#1", "EC-11#dashboard"]
    for p in projections:
        assert sorted(p) == ["content_gates", "logical_payload_id", "path", "physical_copy_id", "reason_codes", "sha256", "size_bytes", "status"]
        if p["status"] == "INCLUDED":
            assert p["path"].startswith("dashboard/data/") and (DASH.parent / p["path"]).is_file()
