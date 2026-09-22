"""
data.py — deterministic loading and integrity validation of the frozen
BTC/USDT daily dataset.

Provenance
----------
The audit steps (sort, duplicate detection, monotone timestamps, daily-grid
gap detection, OHLC sanity, volume validity) are adapted from
the author's earlier supervised-learning study (not part of this repository).

Deliberate differences from the ML pipeline:

* No fetching.  The restored raw file is the only source (frozen study design (docs/methodology.md)).
* No wall-clock logic.  The ML ``remove_incomplete_candle`` step is replaced by
  a hard cutoff *by candle date* (last usable candle 2026-04-29).
* No silent repair.  The ML audit dropped duplicates and logged gaps; here
  every violation raises ``DataIntegrityError`` because the frozen file is the
  reproducibility anchor and must be exactly as expected.
* The raw file is never written to.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .observations import MAX_ABS_LOG_RETURN

# data.py(0=btc_rl) -> 1 src -> 2 <project root>
# the raw dataset is expected under data/raw/ (not distributed)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_CSV_RELATIVE = (
    "data/raw/btc-usdt-daily-raw-v4.csv"
)
HARD_CUTOFF = pd.Timestamp("2026-04-29")
RAW_CSV_ENV_VAR = "BTC_RL_RAW_CSV"
MS_PER_DAY = 86_400_000
NS_PER_DAY = 86_400 * 1_000_000_000

# Frozen dataset identity (F1).  Verified by default on every load.
EXPECTED_SHA256 = "7ff14ebd8f2236eb733073cbea7bfd36d2e1c6977aff89e00bdfde74f3af1305"
EXPECTED_FIRST_DATE = pd.Timestamp("2017-08-17")
EXPECTED_LAST_DATE = HARD_CUTOFF  # last retained candle
EXPECTED_ROWS_AFTER_CUTOFF = 3178

REQUIRED_RAW_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataIntegrityError(RuntimeError):
    """Raised when the dataset violates a frozen integrity requirement."""


@dataclass(frozen=True)
class DatasetConfig:
    """
    Dataset location, cutoff and frozen identity.

    ``verify_identity=True`` (default) requires SHA-256, first date, last
    retained date and row count to match.  Set it to False ONLY for synthetic
    or test data; the frozen research dataset must always be verified.
    """

    raw_csv: Path = field(default_factory=lambda: resolve_raw_csv_path())
    hard_cutoff: pd.Timestamp = HARD_CUTOFF
    expected_sha256: str | None = EXPECTED_SHA256
    expected_first_date: pd.Timestamp | None = EXPECTED_FIRST_DATE
    expected_last_date: pd.Timestamp | None = EXPECTED_LAST_DATE
    expected_rows_after_cutoff: int | None = EXPECTED_ROWS_AFTER_CUTOFF
    verify_identity: bool = True

    def __post_init__(self) -> None:
        if self.verify_identity:
            missing = [
                n for n in ("expected_sha256", "expected_first_date",
                            "expected_last_date", "expected_rows_after_cutoff")
                if getattr(self, n) is None
            ]
            if missing:
                raise ValueError(
                    f"verify_identity=True requires {missing}; "
                    "set verify_identity=False only for synthetic/test data"
                )


def resolve_raw_csv_path(relative: str = DEFAULT_RAW_CSV_RELATIVE) -> Path:
    """Locate the raw CSV: env override ``BTC_RL_RAW_CSV`` else repo-relative path."""
    override = os.environ.get(RAW_CSV_ENV_VAR)
    if override:
        return Path(override)
    return REPO_ROOT / relative


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read the raw CSV and return a ``date``-indexed OHLCV frame (unvalidated)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"raw dataset not found: {path}")
    # open_time is read as TEXT so that its tokens are validated verbatim; pandas'
    # numeric parsing would otherwise round e.g. 1599955200000.00001 to an integer float.
    raw = pd.read_csv(path, dtype={"open_time": "string"})
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise DataIntegrityError(f"raw CSV missing columns: {missing}")
    if len(raw) < 2:
        raise DataIntegrityError("raw CSV has fewer than 2 rows")

    # F6: exact UTC-midnight daily timestamps with exact 24 h spacing, checked on the
    # raw millisecond epoch BEFORE any normalisation, in file order.  The text tokens
    # must be exact integers; nothing passes through floating point.
    open_ms = _raw_open_time_ms(raw["open_time"])
    off = np.nonzero(open_ms % MS_PER_DAY != 0)[0]
    if off.size:
        raise DataIntegrityError(
            f"{off.size} open_time value(s) are not exact UTC midnight; first at row {int(off[0])}"
        )
    spacing = np.diff(open_ms)
    bad = np.nonzero(spacing != MS_PER_DAY)[0]
    if bad.size:
        raise DataIntegrityError(
            f"{bad.size} open_time spacing(s) differ from exactly 24 h; first between rows "
            f"{int(bad[0])} and {int(bad[0]) + 1} ({int(spacing[bad[0]])} ms)"
        )

    date = (
        pd.to_datetime(pd.Series(open_ms), unit="ms", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df = pd.DataFrame({c: raw[c].astype("float64").to_numpy() for c in OHLCV_COLUMNS})
    df.insert(0, "open_time_ms", open_ms)
    df.index = pd.DatetimeIndex(date, name="date")
    return df


_INT_TOKEN = re.compile(r"[0-9]+")


def _raw_open_time_ms(col: pd.Series) -> np.ndarray:
    """
    Exact integer millisecond epochs from the ORIGINAL CSV text tokens.

    Accepted: a token consisting solely of decimal digits (e.g. "1599955200000").
    Rejected: missing/empty values, any sign, decimal point or exponent
    (so "1599955200000.5", "1599955200000.00001", "1.5999552e12",
    "1599955200000.0" and non-numeric text all fail).  "…​.0" is rejected
    deliberately: the pinned source file uses integer tokens only, and accepting it
    would require a float-free decimal proof for no benefit.  Conversion is
    done with Python ``int`` on the validated token, which is exact; no
    floating-point value is ever formed.
    """
    if pd.api.types.is_integer_dtype(col.dtype):  # already exact integers (in-memory frames)
        return col.to_numpy(dtype="int64")
    if col.isna().any():
        first = int(np.nonzero(col.isna().to_numpy())[0][0])
        raise DataIntegrityError(f"open_time has a missing value at row {first}")
    tokens = [str(t).strip() for t in col.tolist()]
    for i, tok in enumerate(tokens):
        if not _INT_TOKEN.fullmatch(tok):
            raise DataIntegrityError(
                f"open_time token {tok!r} at row {i} is not an exact integer millisecond timestamp"
            )
    return np.array([int(t) for t in tokens], dtype=np.int64)


def apply_hard_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Keep candles with date <= cutoff.  Truncation by date only, never by wall clock."""
    cutoff = pd.Timestamp(cutoff)
    return df.loc[df.index <= cutoff].copy()


def validate_ohlcv(df: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> dict:
    """
    Validate a date-indexed daily OHLCV frame.  Raises ``DataIntegrityError``
    on the first violation; returns a small report dict when everything passes.
    """
    if len(df) < 3:
        raise DataIntegrityError("dataset has fewer than 3 rows")
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise DataIntegrityError("index must be a DatetimeIndex")
    if idx.has_duplicates:
        raise DataIntegrityError(f"duplicate dates: {idx[idx.duplicated()].tolist()[:5]}")
    if not idx.is_monotonic_increasing:
        raise DataIntegrityError("dates are not sorted ascending")
    if idx.tz is not None:
        raise DataIntegrityError("index must be tz-naive UTC dates")
    off = idx[idx != idx.normalize()]
    if len(off):
        raise DataIntegrityError(f"timestamps not at UTC midnight: {off[:5].tolist()}")
    # exact 24 h spacing in nanoseconds regardless of the index's storage unit;
    # no flooring to days (24 h + 1 min must fail)
    diffs_ns = np.diff(idx.as_unit("ns").asi8)
    bad = np.nonzero(diffs_ns != NS_PER_DAY)[0]
    if bad.size:
        first = idx[int(bad[0]) + 1]
        raise DataIntegrityError(
            f"daily grid has a gap or irregular spacing ({int(diffs_ns[bad[0]])} ns != 24 h) before {first}"
        )
    if cutoff is not None and idx.max() > pd.Timestamp(cutoff):
        raise DataIntegrityError(f"rows after hard cutoff {pd.Timestamp(cutoff).date()}")

    for col in OHLCV_COLUMNS:
        if col not in df.columns:
            raise DataIntegrityError(f"missing column {col}")
        if not np.isfinite(df[col].to_numpy()).all():
            raise DataIntegrityError(f"non-finite values in {col}")
    for col in ("open", "high", "low", "close"):
        if (df[col] <= 0).any():
            raise DataIntegrityError(f"non-positive prices in {col}")
    if (df["volume"] < 0).any():
        raise DataIntegrityError("negative volume")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise DataIntegrityError("high below max(open, close)")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise DataIntegrityError("low above min(open, close)")
    log_ret = np.abs(np.diff(np.log(df["close"].to_numpy(dtype=np.float64))))
    if log_ret.size and log_ret.max() > MAX_ABS_LOG_RETURN:
        raise DataIntegrityError(
            f"|log close return| {log_ret.max():.4f} exceeds bound {MAX_ABS_LOG_RETURN}"
        )

    return {
        "rows": int(len(df)),
        "first_date": idx[0],
        "last_date": idx[-1],
        "zero_volume_rows": int((df["volume"] == 0).sum()),
    }


def load_dataset(cfg: DatasetConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Load, cut, and validate the frozen dataset.

    Returns ``(frame, report)``.  ``frame`` is date-indexed with columns
    ``open_time_ms, open, high, low, close, volume`` and contains only candles
    with date <= hard cutoff.  The raw file is read-only.
    """
    cfg = cfg or DatasetConfig()
    path = Path(cfg.raw_csv)
    report: dict = {"raw_csv": str(path), "identity_verified": bool(cfg.verify_identity)}
    if not path.exists():
        raise FileNotFoundError(
            f"raw dataset not found: {path} (the file is not tracked in git; restore it or set "
            f"{RAW_CSV_ENV_VAR})"
        )

    if cfg.verify_identity:
        digest = sha256_of_file(path)
        if digest != cfg.expected_sha256:
            raise DataIntegrityError(
                f"raw file SHA-256 mismatch: got {digest}, expected {cfg.expected_sha256}"
            )
        report["sha256"] = digest

    raw = read_raw_csv(path)
    report["rows_raw"] = int(len(raw))
    df = apply_hard_cutoff(raw, cfg.hard_cutoff)
    report["rows_dropped_by_cutoff"] = int(len(raw) - len(df))
    report.update(validate_ohlcv(df, cutoff=cfg.hard_cutoff))

    if cfg.verify_identity:
        if df.index[0] != pd.Timestamp(cfg.expected_first_date):
            raise DataIntegrityError(
                f"first date {df.index[0].date()} != expected {pd.Timestamp(cfg.expected_first_date).date()}"
            )
        if df.index[-1] != pd.Timestamp(cfg.expected_last_date):
            raise DataIntegrityError(
                f"last retained date {df.index[-1].date()} != expected "
                f"{pd.Timestamp(cfg.expected_last_date).date()}"
            )
        if len(df) != cfg.expected_rows_after_cutoff:
            raise DataIntegrityError(
                f"{len(df)} rows after cutoff, expected {cfg.expected_rows_after_cutoff}"
            )
    return df, report


def log_close_returns(close: np.ndarray | pd.Series) -> np.ndarray:
    """r[t] = log(close[t] / close[t-1]); r[0] is NaN."""
    c = np.asarray(close, dtype=np.float64)
    out = np.full(c.shape, np.nan)
    out[1:] = np.log(c[1:] / c[:-1])
    return out
