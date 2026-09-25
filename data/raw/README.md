# Included BTC/USDT dataset

`btc-usdt-daily-raw-v4.csv` is the exact daily OHLCV input used by the study and remains included so that the instructor and other readers can run the integrity checks and reproduce the documented method.

## Provenance and identity

- Source: Binance historical spot-market data for BTC/USDT.
- Timestamps: UTC.
- Raw observations: 3,179 consecutive daily rows from 2017-08-17 through 2026-04-30.
- Frozen study cutoff: 3,178 retained rows through 2026-04-29.
- SHA-256 of the tracked file: `7ff14ebd8f2236eb733073cbea7bfd36d2e1c6977aff89e00bdfde74f3af1305`.
- Official historical-data service: <https://data.binance.vision/>.

The loader verifies the hash, dates and retained row count against `configs/foundation.toml` and rejects a changed or incomplete file.

## Rights and reuse

The dataset is third-party market data and is not covered by this repository's MIT or CC BY 4.0 licenses. It is included for academic transparency and exact reproducibility. The repository author claims no ownership of the underlying Binance data and grants no additional rights in it. Users are responsible for checking the terms that apply to their own use or redistribution of Binance data: <https://www.binance.com/en/terms>.
