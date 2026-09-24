"""Dataset integrity, hard cutoff, and split boundaries."""
import numpy as np
import pandas as pd
import pytest

from btc_rl.data import (
    DataIntegrityError,
    apply_hard_cutoff,
    read_raw_csv,
    sha256_of_file,
    validate_ohlcv,
)
from btc_rl.splits import FROZEN_SPLITS, check_split_layout, decision_dates, usable_decision_dates

from conftest import make_synthetic_frame

from btc_rl.config import load_foundation_config as _load_cfg

if not _load_cfg().dataset.raw_csv.exists():
    pytest.skip("dataset file is missing; restore the complete repository checkout", allow_module_level=True)


def test_raw_file_unchanged(cfg):
    if not cfg.dataset.raw_csv.exists():
        pytest.fail("mandatory raw dataset not present")
    assert sha256_of_file(cfg.dataset.raw_csv) == cfg.dataset.expected_sha256


def test_frame_sorted_unique_contiguous(frame):
    idx = frame.index
    assert idx.is_monotonic_increasing
    assert not idx.has_duplicates
    assert (np.diff(idx.as_unit("ns").asi8) == 86_400 * 10**9).all()
    assert (idx == idx.normalize()).all()
    assert list(frame.columns) == ["open_time_ms", "open", "high", "low", "close", "volume"]


def test_hard_cutoff_respected(frame, cfg):
    assert frame.index.max() == pd.Timestamp("2026-04-29")
    assert frame.index.min() == pd.Timestamp("2017-08-17")
    assert len(frame) == 3178
    # the restored raw file contains one candle after the cutoff (2026-04-30); it must be dropped
    raw = read_raw_csv(cfg.dataset.raw_csv)
    assert raw.index.max() > cfg.dataset.hard_cutoff
    assert len(raw) - len(frame) == 1


def test_expected_split_boundaries(frame, cfg):
    layout = check_split_layout(frame, cfg.splits, lookback=cfg.observation.lookback)
    expected = {
        "train": ("2018-03-04", "2020-12-31", 1034, "2020-12-29"),
        "validation": ("2021-01-01", "2021-08-29", 241, "2021-08-27"),
        "test": ("2021-08-30", "2026-02-04", 1620, "2026-02-02"),
    }
    for name, (s, e, n, last_usable) in expected.items():
        assert layout[name]["start"] == pd.Timestamp(s)
        assert layout[name]["end"] == pd.Timestamp(e)
        assert layout[name]["n_dates"] == n
        assert layout[name]["n_usable_decisions"] == n - 2
        assert layout[name]["last_usable_decision"] == pd.Timestamp(last_usable)
        assert layout[name]["last_mark_date"] == pd.Timestamp(e)
        d = decision_dates(frame, cfg.splits[name])
        assert d[0] == pd.Timestamp(s) and d[-1] == pd.Timestamp(e) and len(d) == n
        u = usable_decision_dates(frame, cfg.splits[name])
        assert u[0] == pd.Timestamp(s) and u[-1] == pd.Timestamp(last_usable) and len(u) == n - 2
    # contiguous and non-overlapping
    assert layout["validation"]["first_index"] == layout["train"]["last_index"] + 1
    assert layout["test"]["first_index"] == layout["validation"]["last_index"] + 1
    assert FROZEN_SPLITS["test"].end == pd.Timestamp("2026-02-04")


def test_validate_rejects_duplicates():
    df = make_synthetic_frame(20)
    bad = pd.concat([df, df.iloc[[5]]]).sort_index()
    with pytest.raises(DataIntegrityError, match="duplicate"):
        validate_ohlcv(bad)


def test_validate_rejects_unsorted():
    df = make_synthetic_frame(20)
    with pytest.raises(DataIntegrityError, match="sorted"):
        validate_ohlcv(df.iloc[::-1])


def test_validate_rejects_gaps():
    df = make_synthetic_frame(20).drop(index=pd.Timestamp("2020-01-10"))
    with pytest.raises(DataIntegrityError, match="gap"):
        validate_ohlcv(df)


def test_validate_rejects_rows_after_cutoff():
    df = make_synthetic_frame(20)
    with pytest.raises(DataIntegrityError, match="cutoff"):
        validate_ohlcv(df, cutoff=pd.Timestamp("2020-01-10"))
    cut = apply_hard_cutoff(df, pd.Timestamp("2020-01-10"))
    assert cut.index.max() == pd.Timestamp("2020-01-10") and len(cut) == 10
    validate_ohlcv(cut, cutoff=pd.Timestamp("2020-01-10"))


def test_validate_rejects_bad_ohlc():
    df = make_synthetic_frame(20)
    df.loc[df.index[3], "high"] = df.loc[df.index[3], "low"] * 0.5
    with pytest.raises(DataIntegrityError, match="high"):
        validate_ohlcv(df)
