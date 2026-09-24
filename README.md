# Reinforcement Learning for Sequential Trading Decisions: A Controlled PPO Study on State Representation and Reward Design

**A completed historical study, explained in plain language.** This repository contains the experiment code, the exact historical dataset used by the code, a summary of what happened in the final historical test, and the aggregate and per-training-start numerical reports behind the published tables. It does not contain trained models, training logs or day-by-day agent replay files.

**Start here:** Read sections 1–6 for the question and method, section 7 for the actual results, and the [illustrated results guide](results/README.md) for all nine charts with explanations. The underlying VALIDATION and TEST summary reports are available under [`results/data/`](results/data/). The figures and tables are readable on GitHub; the disabled dashboard and the private study website are not needed to understand the findings.

This is a research project, not financial advice and not a live trading system. Nothing here recommends buying or selling anything.

## 1. The question

Can a computer program learn, from historical prices alone, when it is better to hold an asset and when it is better to stay out of it? And if it can, what matters more for the outcome: **the information the program is given**, or **the way its behaviour is scored while it learns**?

The study answers this with a small, tightly controlled experiment. It uses reinforcement learning, a branch of machine learning in which a program, called the **agent**, learns by trial and error. The agent acts, sees what happens, receives a score, and gradually adjusts its behaviour to collect a higher score.

The asset is Bitcoin priced in the US-dollar stablecoin Tether (the BTC/USDT pair). The data are historical daily price records from the spot market.

## 2. One decision per day: CASH or INVESTED

Once a day, after the market's daily closing price is known, the agent chooses one of two states:

- **CASH**: hold none of the asset.
- **INVESTED**: hold the asset with the whole portfolio.

There is nothing in between, no borrowing (no leverage) and no betting on falling prices (no short selling).

Only a **change** of state is a trade. Moving from CASH to INVESTED is a purchase. Moving from INVESTED to CASH is a sale. Staying in the same state costs nothing. A decision made after the close of one day is executed at the opening price of the next day, so the decision uses only information available at that day's close; the next opening price is not known when the decision is made.

**Costs.** Every trade is charged a modelled cost: a **0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale**. Slippage is the small gap between the price one expects and the price one actually gets, because real orders are rarely filled at exactly the quoted price. Example: a purchase worth 10,000 USD is charged 10 USD as fee and 5 USD as slippage allowance, 15 USD in total. The cost is a fixed proportion of the traded value, not a fixed amount in USD: buying and later selling again costs about 30 USD only if both trades are worth roughly 10,000 USD. These are modelled costs, a fixed assumption of the study, not guaranteed real execution costs. An agent that trades often therefore pays visibly for it.

## 3. What the agent sees: the observation

The **observation** is the set of numbers handed to the agent before each decision. It is everything the agent knows about the world. The study compares two versions.

- **Main observation O1** is deliberately plain. It contains the daily price changes of the last 10 days, plus the agent's current state (CASH or INVESTED).
- **Extended observation O2** is a summary of the kind a human chart reader might prepare. It contains eight indicators computed from daily closing prices, plus the agent's current state: price changes over 1, 7 and 30 days; two measures of price fluctuation over 14 and 30 days; comparisons with the 50-day and 200-day average closing prices; and the 14-day relative strength index (RSI), which summarizes recent gains versus losses. The longest lookback is 200 days.

O2 is a different view, not O1 with extras. The two share the most recent daily price change and the current state. O2 does not contain the other nine individual daily changes that O1 shows. It replaces them with summaries over longer periods.

Both observations are computed from **daily closing prices only**. The following are **not** inputs to the agent in either version:

- trading volume, and the daily opening, highest and lowest prices;
- the calendar date, or the timing of the Bitcoin halving cycle;
- news, social-media sentiment or blockchain statistics;
- macroeconomic data such as interest rates or inflation.

Opening prices are used only to execute trades and to value the portfolio. The exact formulas and lookback periods are listed in [`docs/methodology.md`](docs/methodology.md#observations).

## 4. How the agent learns: the reward

After each decision the agent receives a **reward**, a single number that tells it how well that step went. Learning means adjusting behaviour so that the rewards add up to as much as possible. What is rewarded therefore shapes what is learned. The study compares two versions.

- **Reward R1** scores the portfolio's proportional growth after costs, using the logarithm of the ratio of its new value to its previous value. It is a growth rate, not an amount of money, and it rewards growth and nothing else.
- **Reward R2** is R1 with two deductions: a small penalty for every trade, and a penalty whenever the portfolio falls further below its previous highest value. A fall below the previous high is called a **drawdown**. R2 asks the agent to trade less and to avoid deepening losses.

R2 changes only the score used during learning. The bookkeeping of money is identical in all experiments, so their financial outcomes are measured in exactly the same way.

The learning method is **Proximal Policy Optimization (PPO)**, a widely used standard algorithm. The agent is a small neural network. It replays the training period many times, and after each batch of experience PPO adjusts the network. PPO discourages overly large changes to the policy during learning; this does not guarantee stable learning or profitable decisions. The same PPO settings are used in every experiment. The study does not try to find the best algorithm; PPO is held fixed so that the comparison below is fair.

## 5. The fixed 2 × 2 comparison

Two observations and two rewards give four experiments. Everything else is identical: data, dates, costs, learning method and settings.

| | Reward R1 (portfolio growth after costs) | Reward R2 (R1 minus trading and drawdown penalties) |
|---|---|---|
| **Main observation O1** (last 10 daily price changes) | Experiment E1, the reference case | Experiment E3 |
| **Extended observation O2** (eight indicators) | Experiment E2 | Experiment E4 |

Only one ingredient changes between neighbouring cells. These paired comparisons examine the effect of changing one ingredient, while retaining the limitations of five training seeds on one shared market history:

- **Does the observation matter?** Compare E2 with E1, and E4 with E3.
- **Does the reward matter?** Compare E3 with E1, and E4 with E2.
- **Can the agent learn a risk-aware policy at all?** Compare E4 with simple reference strategies that do not learn: always CASH, buy once and hold, and random decisions.

Learning starts from random initial settings, and a different start can lead to a different result. Each experiment was therefore trained five times with five fixed starting numbers, called **training seeds** (42, 123, 2026, 31415 and 271828). That gives 20 trained agents. All 20 see the same price history, so the five repetitions show how much the outcome depends on the random start. They are not five independent markets.

## 6. How the evaluation is protected: TRAIN, VALIDATION and an untouched TEST

A program that is graded on the same data it learned from can look excellent and still be useless, like a student who has seen the answers before the test. The study therefore cuts the history into three consecutive periods, in calendar order, and the agent never learns from a later period.

- **TRAIN, 2018-03-04 to 2020-12-31.** The agent learns here.
- **VALIDATION, 2021-01-01 to 2021-08-29.** The trained agents are assessed here during development, and choices are frozen before TEST. No choice uses TEST data.
- **TEST, 2021-08-30 to 2026-02-04.** The final, one-time evaluation.

Three safeguards apply:

1. **No look-ahead.** Every observation uses only prices that were already known on the decision day. Statistics used to scale the inputs come from TRAIN only. The last two dates of each period are used only to settle the final trade, so no period borrows prices from the next one.
2. **TEST stayed untouched.** No choice of any kind was made with TEST data. The evaluation procedure was frozen and independently reviewed before TEST was opened.
3. **TEST was used once.** All 20 trained agents were evaluated on TEST a single time. No policy was selected using TEST results, and no subsequent training, retuning or change to the frozen scientific evidence was permitted. The TEST evaluation is consumed; no second one is permitted.

An untouched TEST period is an evaluation safeguard. It is not proof that a model is free of overfitting, and it is not proof of future profitability: it describes one historical period.

Details: [`docs/methodology.md`](docs/methodology.md#splits-and-decision-windows) and [`docs/evidence-provenance.md`](docs/evidence-provenance.md).

## 7. What this repository contains today

This repository contains the study code and a plain-language summary of its historical findings.

It contains:

- the experiment source code (`src/btc_rl/`), prepared for publication;
- the frozen configuration (`configs/foundation.toml`) and the locked list of software dependencies;
- the exact historical daily dataset used by the study (`data/raw/btc-usdt-daily-raw-v4.csv`), pinned and checked by SHA-256;
- the complete test suite (`tests/`), including dataset-integrity checks;
- a read-only dashboard that is currently disabled: it displays no results and rejects every package (`dashboard/`);
- documentation of the method, the evidence rules, the limitations and the reproduction boundary (`docs/`);
- the frozen aggregate, per-training-start, baseline and seed-dispersion reports for VALIDATION and TEST (`results/data/`);
- the historical TEST and VALIDATION results, the simple reference strategies, and the outcomes of the three research questions, explained below.

It does not contain:

- day-by-day agent trajectories or portfolio replay series;
- trained models, checkpoints, training logs or full run records;
- any evidence package;
- a licence, citation metadata or a release.

### What the final historical test showed

The final TEST period ran from **30 August 2021 to 4 February 2026**. These are observations from that one past period, not forecasts or evidence that an agent would make money in live trading. The tables below present the historical results in a readable form; the separate dashboard remains disabled.

![Final historical TEST total returns for the four PPO designs, buy-and-hold, always CASH and five random references](results/figures/final-test-returns.svg)

The bars show total return over the whole final TEST period, after modelled trading costs—not a yearly return. E4 had the highest mean, but its five separate training starts varied substantially. The [illustrated results guide](results/README.md) explains this graph and eight further figures, including risk, seed sensitivity, the VALIDATION-to-TEST ranking reversal and the research-question comparisons.

#### Numbers from the final historical TEST

The table below summarizes the frozen final assessment: 1,618 daily decision steps from 30 August 2021 to 4 February 2026. Each E1–E4 value is the average of five separately trained agents tested on the **same** historical market path. “±” is the population standard deviation across those five training starts; it is not a confidence interval. Buy-and-hold is one fixed comparison strategy, not a five-run average.

| Strategy | Total return, mean ± seed SD | Sharpe ratio, mean | Largest fall from a previous peak, mean | Purchases or sales, mean | Modelled cost drag, mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1: main observation O1 + reward R1 | −7.07% ± 16.78% | +0.145 | 67.51% | 634.2 | 60.78% |
| E2: extended observation O2 + reward R1 | +56.30% ± 14.23% | +0.461 | 59.63% | 166.4 | 21.89% |
| E3: main observation O1 + reward R2 | −28.15% ± 24.84% | −0.037 | 72.47% | 595.2 | 58.25% |
| E4: extended observation O2 + reward R2 | +107.30% ± 58.56% | +0.632 | 46.17% | 180.0 | 23.41% |
| Buy once and hold throughout | +61.03% | +0.468 | 76.63% | 1 | 0.15% |

**How to read this:** Total return is the change in portfolio value over the whole TEST period after the study’s modelled transaction costs; it is not an annual return. A positive return means the ending value exceeded the starting value. The Sharpe ratio compares return with the variability of daily returns, and higher is better within this comparison. “Largest fall” is the maximum drawdown expressed as a positive loss magnitude, so smaller is better. The fractional number of purchases or sales is an average over five whole-number counts, not a partial trade. “Cost drag” is the compounded cost factor implied by repeated purchases and sales at 0.10% fee plus 0.05% slippage allowance each; it is **not** fees divided by starting capital and it is not an additional cost to subtract from the reported returns.

Both extended-observation designs ended above their starting value in **all five** training runs; the main-observation designs did so in only **two of five** runs for E1 and **one of five** for E3. E4 exceeded buy-and-hold return and Sharpe in **four of five** runs; E2 did so in **two of five**. Every O2 run had a smaller maximum drawdown than buy-and-hold, but the falls were still substantial. E4’s high mean was influenced by one run at +204.14%; its lowest run was +28.73%. The five runs share one market path, so these counts are descriptive, not a statistical test.

On the shorter development VALIDATION period, mean total returns were E1 +75.17%, E2 +75.80%, E3 +100.00% and E4 +53.58%; buy-and-hold returned +66.45%. E3 ranked first and E4 last there. On the longer final TEST period, the order reversed: E4 ranked first and E3 last. Because these periods have different lengths and market paths, their cumulative percentages are **not** directly comparable as if they covered the same interval. The ranking reversal is a warning against choosing a strategy from one short development period and assuming it will lead later.

For completeness, the VALIDATION summary is below. It covers **239 daily decisions from 1 January to 29 August 2021**. As in the TEST table, “±” is the standard deviation across five training starts, not a confidence interval. The buy-and-hold row is one fixed strategy.

| Strategy | Total return, mean ± seed SD | Sharpe ratio, mean ± seed SD | Largest fall from a previous peak, mean | Purchases or sales, mean | Modelled cost drag, mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1: main observation O1 + reward R1 | +75.17% ± 72.52% | +1.463 ± 0.775 | 41.41% | 108.8 | 15.06% |
| E2: extended observation O2 + reward R1 | +75.80% ± 35.08% | +1.538 ± 0.466 | 43.55% | 50.6 | 7.29% |
| E3: main observation O1 + reward R2 | +100.00% ± 71.16% | +1.806 ± 0.866 | 35.71% | 113.6 | 15.67% |
| E4: extended observation O2 + reward R2 | +53.58% ± 30.87% | +1.219 ± 0.395 | 42.21% | 51.0 | 7.36% |
| Buy once and hold throughout | +66.45% | +1.322 | 53.14% | 1 | 0.15% |

These figures come from the included frozen VALIDATION and TEST reports. The human-readable [VALIDATION report](results/data/validation/final5_validation_results.md) and [TEST report](results/data/test/final_test_results.md) are accompanied by the corresponding machine-readable CSV files in the same directories.

#### What each agent actually did

The TEST return table alone does not show how often an agent changed its mind. Each purchase **or** sale is one transaction leg. More legs mean more applications of the modelled 0.10% fee plus 0.05% slippage allowance. The table gives averages over five training starts; the percentage of time invested is an average over the daily decisions. “Ended invested” counts whole agents, not a probability.

| Experiment | Purchases or sales, mean ± seed SD | Modelled cost drag, mean ± seed SD | Time invested, mean ± seed SD | Ended INVESTED |
| --- | ---: | ---: | ---: | ---: |
| E1: main observation O1 + reward R1 | 634.2 ± 117.1 | 60.78% ± 7.32% | 62.9% ± 8.6% | 1 of 5 |
| E2: extended observation O2 + reward R1 | 166.4 ± 48.4 | 21.89% ± 5.81% | 60.7% ± 16.8% | 0 of 5 |
| E3: main observation O1 + reward R2 | 595.2 ± 131.9 | 58.25% ± 8.43% | 56.1% ± 10.5% | 2 of 5 |
| E4: extended observation O2 + reward R2 | 180.0 ± 55.5 | 23.41% ± 6.37% | 43.8% ± 3.3% | 0 of 5 |

For example, E2 did not simply stay in CASH more than E1: their average invested times were 60.7% and 62.9%. It changed state far less often. Cost drag is the compounded effect of the fixed cost factor on a sequence of trades, **not** an extra charge to subtract from the already net returns.

#### The simple reference strategies

The same TEST path was also evaluated with strategies that do not learn. **Always CASH** makes no trade. **Buy and hold** buys once and remains invested. Five **random** strategies make independent random state decisions. They are useful points of comparison, not practical recommendations. Their values are identical across the four experiment settings because they use the same market path and cost assumption.

| Reference strategy | Total return after modelled costs | Sharpe ratio | Largest fall from a previous peak | Purchases or sales | Time invested |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always CASH | 0.00% | not defined | 0.00% | 0 | 0.0% |
| Buy once and hold | +61.03% | +0.468 | 76.63% | 1 | 100.0% |
| Random 0 | −2.69% | +0.174 | 61.70% | 857 | 48.6% |
| Random 1 | −71.11% | −0.558 | 75.60% | 795 | 50.2% |
| Random 2 | −76.11% | −0.713 | 80.65% | 809 | 49.8% |
| Random 3 | −22.47% | +0.048 | 58.25% | 802 | 50.0% |
| Random 4 | −64.50% | −0.455 | 69.08% | 853 | 49.1% |

The random strategies traded hundreds of times and suffered substantial cost drag. Beating random decisions is a low bar; buy-and-hold is the more informative comparison here. Always CASH has no varying daily return, so its Sharpe ratio is undefined rather than zero.

#### Results from all five training starts

![Five individual training-start returns for each design on the final TEST; the white tick is the mean](results/figures/test-return-by-seed.svg)

Each dot is a trained agent with a different random start, evaluated on the **same** historical market path. The figure exposes how much the five outcomes vary; it is not five independent market tests.

The seed is simply the number that starts a training run. All seeds were trained on the same TRAIN period and assessed on the same later market path. The following are **total returns after modelled trading costs**, not annual returns. They show why an average must not be mistaken for a result every run achieved.

| Experiment | Seed | VALIDATION total return | Final TEST total return |
| --- | ---: | ---: | ---: |
| E1: main observation O1 + reward R1 | 42 | +38.82% | +2.24% |
| E1 | 123 | +56.15% | −12.32% |
| E1 | 2026 | +31.14% | −19.10% |
| E1 | 31415 | +219.03% | +20.56% |
| E1 | 271828 | +30.72% | −26.70% |
| E2: extended observation O2 + reward R1 | 42 | +113.16% | +68.99% |
| E2 | 123 | +122.11% | +50.22% |
| E2 | 2026 | +42.71% | +32.23% |
| E2 | 31415 | +61.38% | +71.43% |
| E2 | 271828 | +39.61% | +58.65% |
| E3: main observation O1 + reward R2 | 42 | +136.69% | −45.27% |
| E3 | 123 | +35.93% | −34.97% |
| E3 | 2026 | +12.21% | −50.36% |
| E3 | 31415 | +210.17% | −29.44% |
| E3 | 271828 | +104.98% | +19.30% |
| E4: extended observation O2 + reward R2 | 42 | +18.20% | +90.39% |
| E4 | 123 | +41.75% | +132.20% |
| E4 | 2026 | +52.60% | +81.06% |
| E4 | 31415 | +44.42% | +28.73% |
| E4 | 271828 | +110.94% | +204.14% |

All 20 VALIDATION runs had positive total return, but the final TEST was positive in only 13 of 20 runs. That illustrates why the development period cannot stand in for the final assessment. It does **not** turn the five seeds into five independent real-world trials.

#### The three research questions, in numbers

**Question 1 — Did a trained agent beat simple alternatives?** The counts below show how many of the five final TEST runs passed each comparison. A shallower largest fall means a smaller maximum drawdown than buy-and-hold. “All random” compares total return with each of the five random reference strategies.

| Experiment | Positive return | Higher return than buy-and-hold | Higher Sharpe than buy-and-hold | Shallower largest fall than buy-and-hold | Higher return than every random strategy |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1: main observation O1 + reward R1 | 2/5 | 0/5 | 0/5 | 5/5 | 2/5 |
| E2: extended observation O2 + reward R1 | 5/5 | 2/5 | 2/5 | 5/5 | 5/5 |
| E3: main observation O1 + reward R2 | 1/5 | 0/5 | 0/5 | 4/5 | 1/5 |
| E4: extended observation O2 + reward R2 | 5/5 | 4/5 | 4/5 | 5/5 | 5/5 |

**Question 2 — What changed when the agent received extended observation O2?** Each comparison pairs agents with the *same seed* and the same reward, changing only the observation. Each count is out of five pairs. The return difference is in **percentage points** of total return, not a percentage improvement over the first value.

| Period and comparison | Fewer trades | Lower cost drag | Higher return | Higher Sharpe | Shallower largest fall | Mean return difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TEST: E1 → E2, reward R1 fixed | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 | +63.37 points |
| TEST: E3 → E4, reward R2 fixed | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | +135.45 points |
| VALIDATION: E1 → E2, reward R1 fixed | 5/5 | 5/5 | 4/5 | 4/5 | 3/5 | +0.63 points |
| VALIDATION: E3 → E4, reward R2 fixed | 5/5 | 5/5 | 3/5 | 2/5 | 2/5 | −46.41 points |

The most consistent observed change was **less trading and less modelled cost exposure** with extended observation O2. The return advantage was clear within the five paired runs on TEST, but not consistent on VALIDATION. The two periods differ greatly in length (239 versus 1,618 decisions), so their raw trade counts and cumulative returns must not be compared directly as if they measured the same interval.

**Question 3 — What changed when reward R2 added trading and drawdown penalties?** Here the observation and seed are held fixed, while reward R1 changes to reward R2. The columns are again counts out of five pairs. A higher trade count is listed deliberately: despite its trading penalty, reward R2 did not reliably reduce actual trading.

| Period and comparison | More trades | Higher cost drag | Higher return | Higher Sharpe | Shallower largest fall | Mean return difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TEST: E1 → E3, main observation O1 fixed | 3/5 | 3/5 | 1/5 | 1/5 | 1/5 | −21.08 points |
| TEST: E2 → E4, extended observation O2 fixed | 3/5 | 3/5 | 4/5 | 4/5 | 4/5 | +51.00 points |
| VALIDATION: E1 → E3, main observation O1 fixed | 3/5 | 3/5 | 2/5 | 3/5 | 3/5 | +24.83 points |
| VALIDATION: E2 → E4, extended observation O2 fixed | 2/5 | 2/5 | 2/5 | 2/5 | 2/5 | −22.21 points |

The effect of reward R2 changed with the observation and the evaluation period. These comparisons are descriptive; they do not establish a general causal advantage or statistical significance for reward R2.


- **The information given to the agent mattered.** With the extended observation O2, which summarizes price movement, volatility, moving averages and RSI, the trained agents changed position less often than with the main observation O1. They therefore incurred less of the study's modelled trading cost. In the paired comparisons on TEST, O2 also had higher historical return and a higher Sharpe ratio under both reward designs. The Sharpe ratio compares return with the variability of returns; it is not a guarantee of profit.
- **Changing the learning score did not give a simple improvement.** Reward R2 adds penalties for trading and for a deepening fall below the portfolio's previous high. Its effect was different with O1 and O2. The study does not establish that reward R2 is generally better than reward R1.
- **The results varied between training runs.** Starting the same design with different random seeds produced materially different outcomes. The O2 agents showed meaningful behaviour on this historical TEST period, but also substantial drawdowns.
- **The development period did not predict the final ranking.** By average historical return, the configuration ranked first on VALIDATION was ranked last on TEST, while the one ranked last on VALIDATION was ranked first on TEST. This is why the final period was kept untouched until the design was fixed; it does not prove a particular cause or eliminate overfitting.

**Practical conclusion:** In this controlled historical study, the richer observation was more useful than the minimal one, especially because it reduced trading and modelled cost exposure. The evidence does **not** establish a reliable profitable daily investment strategy, performance in other markets or periods, or a statistically significant advantage.

The readable results above, the exact market dataset and the aggregate and per-training-start machine-readable reports are in this repository. Day-by-day agent replay, trained models, checkpoints and training logs are not included. The dashboard is currently disabled and displays no results.

For all nine static graphs and their plain-language interpretations, see the [illustrated results guide](results/README.md). It also states what the charts cannot establish.

## 8. Principal limitations

- **One asset, one history.** The study covers a single asset and a single historical TEST period that was evaluated once. It says nothing about other assets or other periods.
- **Five seeds, one market.** Material variation across training seeds limits interpretation; the five seeds share one market history and are not independent markets.
- **Descriptive only.** No statistical significance test was specified in advance. Findings describe what happened on one frozen period. They are neither a generalization nor a promise of profit.
- **A simplified market.** Daily decisions only; CASH or INVESTED only; no short selling; no leverage. Costs are fixed at a **0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale**. There is no model of changing liquidity or of the agent's own effect on prices.
- **Cost drag.** Where reported, cost drag is defined as `1 − 0.9985^legs`, the compounded effect of the cost factor on final portfolio value for the same sequence of actions. It is not fees divided by starting capital.
- **Exploratory interaction.** Any combined effect of observation and reward is exploratory; no such hypothesis was fixed in advance.
- **Different periods.** VALIDATION and TEST differ in length and in market conditions.

See [`docs/limitations.md`](docs/limitations.md).

## Setup

Install [uv](https://docs.astral.sh/uv/), clone this repository, and run these commands from its root directory:

```bash
uv sync --frozen --extra dev  # locked closure, test runner and CPU-only PyTorch wheels
uv run pytest -q              # complete tests, including dataset-integrity checks; nothing trains
uv run python -m btc_rl.baselines --split validation --no-replay
```

The included file `data/raw/btc-usdt-daily-raw-v4.csv` is the exact input expected by the code. Every load verifies its SHA-256, first date, last retained date and row count against `configs/foundation.toml`. The loader refuses a changed or incomplete file. `BTC_RL_RAW_CSV` may point to another byte-identical copy when required.

### Run a short PPO integration check

Each command below trains one new agent for 16,384 learning steps, evaluates it on VALIDATION and writes its new outputs under `artifacts/`. The published checkout intentionally omits the earlier private replay input used by the original certification gate, so public reruns first use the complete test suite above and then pass `--skip-preflight` to the training command.

```bash
uv run python -m btc_rl.ppo_e1 --seed 42 --smoke --skip-preflight
uv run python -m btc_rl.ppo_e2 --seed 42 --smoke --skip-preflight
uv run python -m btc_rl.ppo_e3 --seed 42 --smoke --skip-preflight
uv run python -m btc_rl.ppo_e4 --seed 42 --smoke --skip-preflight
```

Remove `--smoke` to use the study's full 200,000-step training budget. The original design used seeds `42`, `123`, `2026`, `31415` and `271828` for every experiment. New training is computationally expensive and may not reproduce the exact historical numbers because PPO training can vary across software and hardware environments. It creates new outputs; it does not alter the frozen historical findings reported above. See [`docs/reproduction-boundary.md`](docs/reproduction-boundary.md).

## Dashboard

```bash
cd dashboard
python -m http.server 8080   # fetch() needs HTTP, not file://
```

Without an evidence package the page shows the state *No evidence package installed* and no result. No evidence package or approved binding is installed, and the binding allowlist of this build is empty, so it rejects every package before rendering anything; placing a package under `dashboard/data/` does not enable it. Displaying evidence requires a separately reviewed enabled consumer and the approved evidence package. See `docs/dashboard.md`.

## Glossary

- **Agent**: the learning program that makes the daily decision.
- **Observation**: the numbers the agent receives before a decision. **Reward**: the score it receives after a decision.
- **CASH / INVESTED**: the two portfolio states (0 and 1). The code and the frozen evidence use the internal identifiers `FLAT` and `LONG` for the same two states; they are not renamed because frozen evidence binds them.
- **Leg**: one position change, that is, one purchase or one sale.
- **Drawdown**: how far the portfolio value is below its previous highest value.
- **Training seed**: the starting number of one independently initialised training run of the same design. Seeds share one market path; they are not independent markets.

## Not part of this repository

The author's earlier supervised-learning study, from which the cost model and data audit were adapted, and any production or advisory system are not part of this repository and are not described by it.
