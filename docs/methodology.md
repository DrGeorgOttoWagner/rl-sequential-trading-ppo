# Methodology — the frozen study design

This document describes the final frozen study design. Choices were frozen before the final TEST evaluation. The values below are mirrored by `configs/foundation.toml` and the source modules. It states design facts only; it contains no result.

## Data

- One asset and one historical daily spot-market dataset: Bitcoin priced in the US-dollar stablecoin Tether (BTC/USDT), obtained from Binance. Each daily record holds the opening, highest, lowest and closing price and the traded volume (OHLCV). Timestamps are expressed in UTC. No exchange, product or asset is recommended.
- The exact dataset is included at `data/raw/btc-usdt-daily-raw-v4.csv`. The raw file contains 3,179 consecutive daily observations from 2017-08-17 through 2026-04-30, with no duplicate timestamps, missing daily intervals or blank fields. The predetermined cutoff retains 3,178 rows through 2026-04-29, so no missing-data imputation was required. The retained input is identified by its SHA-256 (`configs/foundation.toml`, `[dataset].expected_sha256`).
- Hard cutoff by candle date (last usable candle 2026-04-29); no wall-clock logic; no fetching; no silent repair. Any integrity violation raises `DataIntegrityError`.
- Timestamps must be exact UTC midnight with exact 24-hour spacing; the checks run on the raw millisecond epoch before any normalisation.

## Splits and decision windows

| Split | Window | Last usable decision | Use |
|---|---|---|---|
| TRAIN | 2018-03-04 … 2020-12-31 | 2020-12-29 | PPO training episodes; scaling statistics |
| VALIDATION | 2021-01-01 … 2021-08-29 | 2021-08-27 | Development assessment and freezing of choices before TEST; no TEST-based selection |
| TEST | 2021-08-30 … 2026-02-04 | 2026-02-02 | one deterministic evaluation, opened once after every choice was frozen |

A decision at t consumes open[t+1] and open[t+2], so the last two dates of every window are settlement dates only (purge rule). No TRAIN reward uses VALIDATION prices; no VALIDATION reward uses TEST prices; TEST never consumes prices after its window end. Observation history is strictly causal and may use the previous split's closes as warm-up.

## Environment semantics

```
   close[t]              open[t+1]                       open[t+2]
     |                      |                               |
 observation formed     fill: position p_{t+1} takes    mark-to-market;
 action a_t chosen      effect; one cost leg per unit   next fill point
                        of |p_{t+1} - p_t|
```

- Action space `Discrete(2)`: 0 = CASH, 1 = INVESTED (internal identifiers `FLAT` and `LONG`). No short selling, no leverage, full capital.
- Open-to-open accounting keeps the overnight gap inside the held return. Equity is marked at execution points (opens); drawdown is `equity / running_peak − 1`.
- Costs in plain words: a **0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale**, 0.15% of the traded value in total per transaction leg. Slippage is the gap between the expected price and the price actually obtained. A purchase worth 10,000 USD is charged 10 USD fee and 5 USD slippage allowance, 15 USD in total. The cost is a multiplicative fraction of 0.0015 per leg, not a fixed amount in USD: a purchase followed later by a sale costs about 30 USD only if both legs are worth roughly 10,000 USD, and the cost of the sale varies with the portfolio value. These are modelled costs, a fixed assumption and not a model of market liquidity; they are not guaranteed real execution costs. In `configs/foundation.toml` the same two assumptions appear under the technical names `fee_bps = 10.0` and `slippage_bps = 5.0`; these mean 0.10% and 0.05%.
- With per-leg cost fraction c (conservative profile 0.0015) and position p ∈ {0, 1}:

```
legs_t         = |p_{t+1} - p_t|
equity_{t+1}   = equity_t · (1 - c)^{legs_t} · (1 + p_{t+1} · (open[t+2] / open[t+1] - 1))
reward_t (R1)  = log(equity_{t+1} / equity_t)
```

- Costs: CASH→INVESTED one entry leg, INVESTED→CASH one exit leg, a held state nothing. No forced liquidation at the end of a window; the final position is reported.
- Metrics use 365 periods per year; the Sharpe ratio is mean/std of simple step returns × √365 with zero risk-free rate. Cost drag is `1 − 0.9985^legs`.

## Observations

An **observation** is the vector of numbers handed to the agent before each decision. Two versions are compared. Below, c_t is the closing price of decision day t and r_k = ln(c_k / c_{k−1}) is the daily log close return, a standard way to express a daily price change.

**Main observation O1** (minimal; 11 numbers; uses the last 11 closes):

| Component | Definition | Lookback |
|---|---|---|
| 10 daily returns | r_{t−9}, …, r_t, each divided by one constant: the population standard deviation of daily log close returns over TRAIN closes only, fitted through the last usable TRAIN decision (2018-03-04 to 2020-12-29) | 10 days |
| current position | 0 = CASH, 1 = INVESTED, the state held during day t | none |

**Extended observation O2** (engineered; 9 numbers; uses the last 200 closes):

| Indicator | Definition | Lookback |
|---|---|---|
| `return_1d` | ln(c_t / c_{t−1}) | 1 day |
| `return_7d` | ln(c_t / c_{t−7}) | 7 days |
| `return_30d` | ln(c_t / c_{t−30}) | 30 days |
| `volatility_14` | sample standard deviation of the 14 daily returns ending at t | 14 days |
| `volatility_30` | sample standard deviation of the 30 daily returns ending at t | 30 days |
| `ma_50_ratio` | c_t divided by the mean of the last 50 closes, c_t included | 50 days |
| `ma_200_ratio` | c_t divided by the mean of the last 200 closes, c_t included | 200 days |
| `rsi_14` | relative strength index, simple rolling-mean form: 100 − 100 / (1 + mean gain / mean loss) over the last 14 daily price changes; 100 if there was no loss, 50 if the price did not move | 14 days |
| current position | 0 = CASH, 1 = INVESTED, the state held during day t | none |

Each of the eight indicators is standardised (mean subtracted, divided by the standard deviation) with statistics fitted on TRAIN only: the 833 decision rows 2018-09-19 to 2020-12-29 whose complete 200-close window lies inside the TRAIN closes.

**Relation between O1 and O2.** O2 is not a superset of O1. The two share the most recent daily return (r_t, which is `return_1d`, under a different scaling) and the current position. The nine older individual daily returns of O1 (r_{t−9}, …, r_{t−1}) are not components of O2; O2 carries multi-day summaries instead.

**What is not an agent input.** Both observations are functions of daily closing prices and the current position only. Not provided to the agent in either version: traded volume; opening, highest and lowest prices; the calendar date, day of week or any timing of the Bitcoin halving cycle; news or sentiment; blockchain statistics; macroeconomic series such as interest rates or inflation. Opening prices are used for trade execution and accounting, never as an agent input. The data-integrity checks read all market columns, including volume.

Observation history may reach back before the start of a split as warm-up (O2 needs 199 earlier closes, so the first TRAIN observation on 2018-03-04 uses closes from 2017-08-17 onwards). Warm-up closes never enter the scaling statistics. Labels and future returns never enter an observation.

## Rewards

A **reward** is the single number the agent receives after each decision; training adjusts the agent so that the sum of rewards grows.

- **Reward R1**: portfolio log return per step after transaction costs.
- **Reward R2**: R1 − 0.0005 · legs − 0.10 · max(0, D_{t+1} − D_t), where D is the **nonnegative drawdown magnitude** `D = 1 − equity / running_peak` (D ≥ 0; the running peak is the maximum realised equity so far, initial equity included). D is the negative of the signed drawdown metric `equity / running_peak − 1` defined above; the penalty applies only when the magnitude increases, so a new high, a recovery or an unchanged drawdown contributes 0. The implementation records describe these coefficients as fixed prospectively and do not document a data-driven coefficient-selection exercise. The design reserved any coefficient selection to the validation stage; that rule does not establish that a tuning search was performed. No coefficient was selected using TEST. R2 changes only the training incentive; the financial accounting is identical in all four configurations.

## Agent and experiment grid

- PPO (Stable-Baselines3) with separate actor and critic multilayer perceptrons of two 64-unit `tanh` hidden layers, trained on CPU. Fixed settings: learning rate `3e-4`, rollout length `2048`, batch size `64`, `10` epochs per update, discount factor `0.99`, GAE lambda `0.95`, clip range `0.2`, entropy coefficient `0`, value coefficient `0.5` and maximum gradient norm `0.5`. The requested budget was 200,000 steps; the rollout structure executed 200,704 steps.
- E1 = O1 + R1 (control for RQ1, RQ2 and RQ3), E2 = O2 + R1 (RQ2 treatment), E3 = O1 + R2 (RQ3 treatment), E4 = O2 + R2 (prospectively designated RQ1 main configuration and the second RQ2/RQ3 contrast). Every configuration shares the dataset, splits, timeline, action space, cost model and PPO settings; only the observation and/or reward differ.
- Training-seed cohort: 42, 123, 2026, 31415, 271828 (development subset 42, 123, 2026; the two further seeds were fixed prospectively before any additional training). Results are reported as mean and population standard deviation across seeds.

## Baselines

Always CASH, BUY_AND_HOLD (one entry leg, then held) and five seeded random policies (stream indices 0–4), all through the same environment and accounting. Random-stream indices are a separate identifier space from training seeds.

## Governance of the TEST window

All observation and reward definitions, PPO settings, seeds and split dates were fixed before the final TEST evaluation. The 20 trained policies were evaluated on TEST once, after development had ended. No policy or design choice was selected using TEST results. Once those results had been inspected, the same period could still document this historical study but could no longer serve as a new untouched hold-out for later model choices. How the resulting evidence is represented in this repository is described in `evidence-provenance.md`.
