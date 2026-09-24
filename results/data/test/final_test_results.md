# Final one-time TEST evaluation — E1–E4, five training seeds each

The final TEST covers 30 August 2021 to 4 February 2026, with 1,618 daily decision transitions and a last usable decision date of 2 February 2026. It evaluates the four experiment designs with five training seeds each. Each purchase or sale incurred a modelled cost of 0.15% of the traded value: a 0.10% trading fee plus a 0.05% allowance for slippage. Sharpe ratios use 365 periods per year, and actions are selected deterministically. Every reported row was reconstructed from the persisted experiment evidence, including the complete timeline, recomputed financial metrics and the separate R2 reward components.

Descriptive results for the frozen TEST period only. No statistical significance test was pre-specified and none is computed. No profitability claim. No generalisation beyond the frozen period. No post-TEST retuning. All 20 learned checkpoints are reported; none was selected, dropped or reweighted. R2 reward diagnostics are training-reward sums, never returns.

## Per-seed table (20 learned rows)

| experiment | definition | seed | policy_name | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | O1 + R1 | 42 | ppo_e1_seed42 | 1618 | +0.0224 | 1.0224 | 0.219 | -0.6873 | 330 | 660 | 0.690 | 0.6287 | 0 |
| E1 | O1 + R1 | 123 | ppo_e1_seed123 | 1618 | -0.1232 | 0.8768 | 0.108 | -0.7078 | 394 | 788 | 0.518 | 0.6936 | 0 |
| E1 | O1 + R1 | 2026 | ppo_e1_seed2026 | 1618 | -0.1910 | 0.8090 | 0.094 | -0.7028 | 213 | 425 | 0.688 | 0.4716 | 1 |
| E1 | O1 + R1 | 31415 | ppo_e1_seed31415 | 1618 | +0.2056 | 1.2056 | 0.314 | -0.6609 | 325 | 650 | 0.718 | 0.6231 | 0 |
| E1 | O1 + R1 | 271828 | ppo_e1_seed271828 | 1618 | -0.2670 | 0.7330 | -0.009 | -0.6170 | 324 | 648 | 0.533 | 0.6220 | 0 |
| E2 | O2 + R1 | 42 | ppo_e2_seed42 | 1618 | +0.6899 | 1.6899 | 0.537 | -0.4902 | 42 | 84 | 0.371 | 0.1185 | 0 |
| E2 | O2 + R1 | 123 | ppo_e2_seed123 | 1618 | +0.5022 | 1.5022 | 0.433 | -0.7541 | 71 | 142 | 0.891 | 0.1920 | 0 |
| E2 | O2 + R1 | 2026 | ppo_e2_seed2026 | 1618 | +0.3223 | 1.3223 | 0.359 | -0.6745 | 107 | 214 | 0.612 | 0.2748 | 0 |
| E2 | O2 + R1 | 31415 | ppo_e2_seed31415 | 1618 | +0.7143 | 1.7143 | 0.514 | -0.5155 | 104 | 208 | 0.543 | 0.2682 | 0 |
| E2 | O2 + R1 | 271828 | ppo_e2_seed271828 | 1618 | +0.5865 | 1.5865 | 0.462 | -0.5469 | 92 | 184 | 0.617 | 0.2413 | 0 |
| E3 | O1 + R2 | 42 | ppo_e3_seed42 | 1618 | -0.4527 | 0.5473 | -0.189 | -0.7487 | 371 | 742 | 0.536 | 0.6717 | 0 |
| E3 | O1 + R2 | 123 | ppo_e3_seed123 | 1618 | -0.3497 | 0.6503 | -0.121 | -0.7137 | 331 | 662 | 0.380 | 0.6298 | 0 |
| E3 | O1 + R2 | 2026 | ppo_e3_seed2026 | 1618 | -0.5036 | 0.4964 | -0.167 | -0.8301 | 226 | 451 | 0.701 | 0.4919 | 1 |
| E3 | O1 + R2 | 31415 | ppo_e3_seed31415 | 1618 | -0.2944 | 0.7056 | -0.003 | -0.6480 | 349 | 698 | 0.600 | 0.6493 | 0 |
| E3 | O1 + R2 | 271828 | ppo_e3_seed271828 | 1618 | +0.1930 | 1.1930 | 0.296 | -0.6832 | 212 | 423 | 0.588 | 0.4701 | 1 |
| E4 | O2 + R2 | 42 | ppo_e4_seed42 | 1618 | +0.9039 | 1.9039 | 0.605 | -0.4465 | 60 | 120 | 0.409 | 0.1648 | 0 |
| E4 | O2 + R2 | 123 | ppo_e4_seed123 | 1618 | +1.3220 | 2.3220 | 0.701 | -0.5545 | 120 | 240 | 0.485 | 0.3025 | 0 |
| E4 | O2 + R2 | 2026 | ppo_e4_seed2026 | 1618 | +0.8106 | 1.8106 | 0.569 | -0.4677 | 122 | 244 | 0.413 | 0.3067 | 0 |
| E4 | O2 + R2 | 31415 | ppo_e4_seed31415 | 1618 | +0.2873 | 1.2873 | 0.340 | -0.5335 | 90 | 180 | 0.473 | 0.2368 | 0 |
| E4 | O2 + R2 | 271828 | ppo_e4_seed271828 | 1618 | +2.0414 | 3.0414 | 0.944 | -0.3062 | 58 | 116 | 0.412 | 0.1598 | 0 |

R2 experiments — reward diagnostics (separate from financial metrics):

| experiment | seed | cumulative_financial_r1 | cumulative_turnover_penalty | cumulative_drawdown_penalty | cumulative_reward_r2 |
|---|---|---|---|---|---|
| E3 | 42 | -0.6027 | 0.3710 | 0.4620 | -1.4358 |
| E3 | 123 | -0.4304 | 0.3310 | 0.3826 | -1.1439 |
| E3 | 2026 | -0.7003 | 0.2255 | 0.5253 | -1.4511 |
| E3 | 31415 | -0.3487 | 0.3490 | 0.5895 | -1.2872 |
| E3 | 271828 | +0.1765 | 0.2115 | 0.6250 | -0.6600 |
| E4 | 42 | +0.6439 | 0.0600 | 0.5061 | +0.0778 |
| E4 | 123 | +0.8424 | 0.1200 | 0.5471 | +0.1753 |
| E4 | 2026 | +0.5936 | 0.1220 | 0.5102 | -0.0385 |
| E4 | 31415 | +0.2525 | 0.0900 | 0.5487 | -0.3862 |
| E4 | 271828 | +1.1123 | 0.0580 | 0.5078 | +0.5465 |

## Per-experiment summary (mean ± population std over the exact five seeds)

| experiment | definition | n_seeds | total_return_mean | total_return_std | final_equity_mean | final_equity_std | sharpe_ratio_mean | sharpe_ratio_std | max_drawdown_mean | max_drawdown_std | n_entries_mean | n_entries_std | n_legs_mean | n_legs_std | exposure_mean | exposure_std | total_cost_fraction_mean | total_cost_fraction_std | ending_LONG | cumulative_financial_r1_mean | cumulative_financial_r1_std | cumulative_turnover_penalty_mean | cumulative_turnover_penalty_std | cumulative_drawdown_penalty_mean | cumulative_drawdown_penalty_std | cumulative_reward_r2_mean | cumulative_reward_r2_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | O1 + R1 | 5 | -0.0707 | +0.1678 | +0.9293 | +0.1678 | +0.1452 | +0.1109 | -0.6751 | +0.0333 | +317.2000 | +58.3555 | +634.2000 | +117.0682 | +0.6294 | +0.0855 | +0.6078 | +0.0732 | 1 | - | - | - | - | - | - | - | - |
| E2 | O2 + R1 | 5 | +0.5630 | +0.1423 | +1.5630 | +0.1423 | +0.4609 | +0.0631 | -0.5963 | +0.1012 | +83.2000 | +24.1777 | +166.4000 | +48.3554 | +0.6069 | +0.1676 | +0.2189 | +0.0581 | 0 | - | - | - | - | - | - | - | - |
| E3 | O1 + R2 | 5 | -0.2815 | +0.2484 | +0.7185 | +0.2484 | -0.0367 | +0.1783 | -0.7247 | +0.0623 | +297.8000 | +65.7249 | +595.2000 | +131.9294 | +0.5612 | +0.1052 | +0.5825 | +0.0843 | 2 | -0.3811 | +0.3051 | +0.2976 | +0.0660 | +0.5169 | +0.0873 | -1.1956 | +0.2901 |
| E4 | O2 + R2 | 5 | +1.0730 | +0.5856 | +2.0730 | +0.5856 | +0.6317 | +0.1960 | -0.4617 | +0.0874 | +90.0000 | +27.7417 | +180.0000 | +55.4833 | +0.4382 | +0.0333 | +0.2341 | +0.0637 | 0 | +0.6890 | +0.2844 | +0.0900 | +0.0277 | +0.5240 | +0.0196 | +0.0750 | +0.3026 |

## Final E1–E4 matrix

| experiment | definition | n_seeds | total_return_mean | total_return_std | final_equity_mean | final_equity_std | sharpe_ratio_mean | sharpe_ratio_std | max_drawdown_mean | max_drawdown_std | n_entries_mean | n_entries_std | n_legs_mean | n_legs_std | exposure_mean | exposure_std | total_cost_fraction_mean | total_cost_fraction_std | ending_LONG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | O1 + R1 | 5 | -0.0707 | +0.1678 | +0.9293 | +0.1678 | +0.1452 | +0.1109 | -0.6751 | +0.0333 | +317.2000 | +58.3555 | +634.2000 | +117.0682 | +0.6294 | +0.0855 | +0.6078 | +0.0732 | 1 |
| E2 | O2 + R1 | 5 | +0.5630 | +0.1423 | +1.5630 | +0.1423 | +0.4609 | +0.0631 | -0.5963 | +0.1012 | +83.2000 | +24.1777 | +166.4000 | +48.3554 | +0.6069 | +0.1676 | +0.2189 | +0.0581 | 0 |
| E3 | O1 + R2 | 5 | -0.2815 | +0.2484 | +0.7185 | +0.2484 | -0.0367 | +0.1783 | -0.7247 | +0.0623 | +297.8000 | +65.7249 | +595.2000 | +131.9294 | +0.5612 | +0.1052 | +0.5825 | +0.0843 | 2 |
| E4 | O2 + R2 | 5 | +1.0730 | +0.5856 | +2.0730 | +0.5856 | +0.6317 | +0.1960 | -0.4617 | +0.0874 | +90.0000 | +27.7417 | +180.0000 | +55.4833 | +0.4382 | +0.0333 | +0.2341 | +0.0637 | 0 |

## Baselines (same TEST environment; identical through the E1, E2, E3 and E4 environments)

| policy | policy_kind | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CASH | always_flat | nan | 1618 | +0.0000 | 1.0000 | - | +0.0000 | 0 | 0 | 0.000 | 0.0000 | 0 |
| BUY_AND_HOLD | always_long | nan | 1618 | +0.6103 | 1.6103 | 0.468 | -0.7663 | 1 | 1 | 1.000 | 0.0015 | 1 |
| RANDOM_0 | random | 0.0 | 1618 | -0.0269 | 0.9731 | 0.174 | -0.6170 | 429 | 857 | 0.486 | 0.7238 | 1 |
| RANDOM_1 | random | 1.0 | 1618 | -0.7111 | 0.2889 | -0.558 | -0.7560 | 398 | 795 | 0.502 | 0.6968 | 1 |
| RANDOM_2 | random | 2.0 | 1618 | -0.7611 | 0.2389 | -0.713 | -0.8065 | 405 | 809 | 0.498 | 0.7031 | 1 |
| RANDOM_3 | random | 3.0 | 1618 | -0.2247 | 0.7753 | 0.048 | -0.5825 | 401 | 802 | 0.500 | 0.7000 | 0 |
| RANDOM_4 | random | 4.0 | 1618 | -0.6450 | 0.3550 | -0.455 | -0.6908 | 427 | 853 | 0.491 | 0.7221 | 1 |

## Seed dispersion

| experiment | metric | mean | std | min | max | seed_42 | seed_123 | seed_2026 | seed_31415 | seed_271828 |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | total_return | -0.0707 | +0.1678 | -0.2670 | +0.2056 | +0.0224 | -0.1232 | -0.1910 | +0.2056 | -0.2670 |
| E1 | sharpe_ratio | +0.1452 | +0.1109 | -0.0091 | +0.3137 | +0.2187 | +0.1085 | +0.0943 | +0.3137 | -0.0091 |
| E1 | max_drawdown | -0.6751 | +0.0333 | -0.7078 | -0.6170 | -0.6873 | -0.7078 | -0.7028 | -0.6609 | -0.6170 |
| E1 | n_legs | +634.2000 | +117.0682 | +425.0000 | +788.0000 | +660.0000 | +788.0000 | +425.0000 | +650.0000 | +648.0000 |
| E1 | total_cost_fraction | +0.6078 | +0.0732 | +0.4716 | +0.6936 | +0.6287 | +0.6936 | +0.4716 | +0.6231 | +0.6220 |
| E2 | total_return | +0.5630 | +0.1423 | +0.3223 | +0.7143 | +0.6899 | +0.5022 | +0.3223 | +0.7143 | +0.5865 |
| E2 | sharpe_ratio | +0.4609 | +0.0631 | +0.3589 | +0.5372 | +0.5372 | +0.4325 | +0.3589 | +0.5143 | +0.4615 |
| E2 | max_drawdown | -0.5963 | +0.1012 | -0.7541 | -0.4902 | -0.4902 | -0.7541 | -0.6745 | -0.5155 | -0.5469 |
| E2 | n_legs | +166.4000 | +48.3554 | +84.0000 | +214.0000 | +84.0000 | +142.0000 | +214.0000 | +208.0000 | +184.0000 |
| E2 | total_cost_fraction | +0.2189 | +0.0581 | +0.1185 | +0.2748 | +0.1185 | +0.1920 | +0.2748 | +0.2682 | +0.2413 |
| E3 | total_return | -0.2815 | +0.2484 | -0.5036 | +0.1930 | -0.4527 | -0.3497 | -0.5036 | -0.2944 | +0.1930 |
| E3 | sharpe_ratio | -0.0367 | +0.1783 | -0.1891 | +0.2959 | -0.1891 | -0.1207 | -0.1668 | -0.0027 | +0.2959 |
| E3 | max_drawdown | -0.7247 | +0.0623 | -0.8301 | -0.6480 | -0.7487 | -0.7137 | -0.8301 | -0.6480 | -0.6832 |
| E3 | n_legs | +595.2000 | +131.9294 | +423.0000 | +742.0000 | +742.0000 | +662.0000 | +451.0000 | +698.0000 | +423.0000 |
| E3 | total_cost_fraction | +0.5825 | +0.0843 | +0.4701 | +0.6717 | +0.6717 | +0.6298 | +0.4919 | +0.6493 | +0.4701 |
| E4 | total_return | +1.0730 | +0.5856 | +0.2873 | +2.0414 | +0.9039 | +1.3220 | +0.8106 | +0.2873 | +2.0414 |
| E4 | sharpe_ratio | +0.6317 | +0.1960 | +0.3396 | +0.9438 | +0.6055 | +0.7006 | +0.5689 | +0.3396 | +0.9438 |
| E4 | max_drawdown | -0.4617 | +0.0874 | -0.5545 | -0.3062 | -0.4465 | -0.5545 | -0.4677 | -0.5335 | -0.3062 |
| E4 | n_legs | +180.0000 | +55.4833 | +116.0000 | +244.0000 | +120.0000 | +240.0000 | +244.0000 | +180.0000 | +116.0000 |
| E4 | total_cost_fraction | +0.2341 | +0.0637 | +0.1598 | +0.3067 | +0.1648 | +0.3025 | +0.3067 | +0.2368 | +0.1598 |

## RQ1 — learned policies vs baselines (descriptive)

Whether learned policies show economically meaningful out-of-sample behaviour is read from the per-seed and baseline tables (return, Sharpe, drawdown, turnover and cost drag relative to always CASH, buy-and-hold and the random policy), per seed and per configuration; no claim beyond the frozen period.

## RQ2 — O2 vs O1 under both rewards (descriptive, paired by seed)

| metric | E1_mean | E2_mean | mean_diff_E2_minus_E1 | paired_min | paired_max | seeds_E2_higher |
|---|---|---|---|---|---|---|
| total_return | -0.0707 | +0.5630 | +0.6337 | +0.5087 | +0.8536 | 5 |
| final_equity | +0.9293 | +1.5630 | +0.6337 | +0.5087 | +0.8536 | 5 |
| sharpe_ratio | +0.1452 | +0.4609 | +0.3157 | +0.2007 | +0.4706 | 5 |
| max_drawdown | -0.6751 | -0.5963 | +0.0789 | -0.0463 | +0.1970 | 4 |
| n_entries | +317.2000 | +83.2000 | -234.0000 | -323.0000 | -106.0000 | 0 |
| n_legs | +634.2000 | +166.4000 | -467.8000 | -646.0000 | -211.0000 | 0 |
| exposure | +0.6294 | +0.6069 | -0.0225 | -0.3189 | +0.3733 | 2 |
| total_cost_fraction | +0.6078 | +0.2189 | -0.3889 | -0.5102 | -0.1969 | 0 |

| metric | E3_mean | E4_mean | mean_diff_E4_minus_E3 | paired_min | paired_max | seeds_E4_higher |
|---|---|---|---|---|---|---|
| total_return | -0.2815 | +1.0730 | +1.3545 | +0.5817 | +1.8484 | 5 |
| final_equity | +0.7185 | +2.0730 | +1.3545 | +0.5817 | +1.8484 | 5 |
| sharpe_ratio | -0.0367 | +0.6317 | +0.6683 | +0.3423 | +0.8213 | 5 |
| max_drawdown | -0.7247 | -0.4617 | +0.2631 | +0.1145 | +0.3770 | 5 |
| n_entries | +297.8000 | +90.0000 | -207.8000 | -311.0000 | -104.0000 | 0 |
| n_legs | +595.2000 | +180.0000 | -415.2000 | -622.0000 | -207.0000 | 0 |
| exposure | +0.5612 | +0.4382 | -0.1230 | -0.2886 | +0.1044 | 1 |
| total_cost_fraction | +0.5825 | +0.2341 | -0.3484 | -0.5069 | -0.1852 | 0 |

## RQ3 — R2 vs R1 under both observation spaces (descriptive, paired by seed)

| metric | E1_mean | E3_mean | mean_diff_E3_minus_E1 | paired_min | paired_max | seeds_E3_higher |
|---|---|---|---|---|---|---|
| total_return | -0.0707 | -0.2815 | -0.2108 | -0.4999 | +0.4600 | 1 |
| final_equity | +0.9293 | +0.7185 | -0.2108 | -0.4999 | +0.4600 | 1 |
| sharpe_ratio | +0.1452 | -0.0367 | -0.1819 | -0.4078 | +0.3050 | 1 |
| max_drawdown | -0.6751 | -0.7247 | -0.0496 | -0.1274 | +0.0129 | 1 |
| n_entries | +317.2000 | +297.8000 | -19.4000 | -112.0000 | +41.0000 | 3 |
| n_legs | +634.2000 | +595.2000 | -39.0000 | -225.0000 | +82.0000 | 3 |
| exposure | +0.6294 | +0.5612 | -0.0682 | -0.1545 | +0.0550 | 2 |
| total_cost_fraction | +0.6078 | +0.5825 | -0.0253 | -0.1519 | +0.0430 | 3 |

| metric | E2_mean | E4_mean | mean_diff_E4_minus_E2 | paired_min | paired_max | seeds_E4_higher |
|---|---|---|---|---|---|---|
| total_return | +0.5630 | +1.0730 | +0.5100 | -0.4270 | +1.4548 | 4 |
| final_equity | +1.5630 | +2.0730 | +0.5100 | -0.4270 | +1.4548 | 4 |
| sharpe_ratio | +0.4609 | +0.6317 | +0.1708 | -0.1747 | +0.4823 | 4 |
| max_drawdown | -0.5963 | -0.4617 | +0.1346 | -0.0179 | +0.2406 | 4 |
| n_entries | +83.2000 | +90.0000 | +6.8000 | -34.0000 | +49.0000 | 3 |
| n_legs | +166.4000 | +180.0000 | +13.6000 | -68.0000 | +98.0000 | 3 |
| exposure | +0.6069 | +0.4382 | -0.1687 | -0.4067 | +0.0377 | 1 |
| total_cost_fraction | +0.2189 | +0.2341 | +0.0152 | -0.0815 | +0.1105 | 3 |

## Post-TEST rules

- no feature / observation change (O1, O2, scalers frozen)
- no reward change (R1, R2 and the R2 coefficients frozen)
- no PPO / hyperparameter / architecture change
- no seed change; the cohort is exactly (42, 123, 2026, 31415, 271828) per configuration
- no retraining of any checkpoint
- no model selection followed by retuning
- no second TEST opening; one run id, one canonical root, one executor, one-shot read-only results
- interrupted IN_PROGRESS item: promoted valid evidence is reconciled (never re-evaluated); promoted invalid evidence aborts the run; otherwise exactly one deterministic RECOVERY_RETRY of the same item, then abort
- results are interpreted descriptively only, for the frozen TEST period, with no profitability or generalisation claim
