# Final five-seed cohort — VALIDATION ONLY

The report combines five independently initialized training runs for each experiment, using seeds 42, 123, 2026, 31415 and 271828. All runs used the same dataset, calendar splits, PPO configuration, [64, 64] Tanh network, CPU execution, one thread, one environment and 200,704 trained timesteps. Each purchase or sale incurred a modelled cost of 0.15% of the traded value: a 0.10% trading fee plus a 0.05% allowance for slippage. VALIDATION evaluation used deterministic actions. Financial metrics come from the portfolio-equity path; R2 reward diagnostics are reported separately and are never treated as returns. TEST was not evaluated while this VALIDATION cohort was being completed. The five-seed statistics extend the reproducibility assessment and were not used as a new model-selection stage.

## E1 = O1 + R1 — five seeds

| policy | cohort | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_e1 | accepted | 42 | 239 | +0.3882 | 1.3882 | 1.067 | -0.4680 | 55 | 110 | 0.636 | 0.364 | 0.1522 | 0 |
| ppo_e1 | accepted | 123 | 239 | +0.5615 | 1.5615 | 1.406 | -0.4036 | 64 | 128 | 0.548 | 0.452 | 0.1748 | 0 |
| ppo_e1 | accepted | 2026 | 239 | +0.3114 | 1.3114 | 0.928 | -0.4902 | 48 | 96 | 0.619 | 0.381 | 0.1342 | 0 |
| ppo_e1 | new | 31415 | 239 | +2.1903 | 3.1903 | 2.974 | -0.1869 | 51 | 102 | 0.657 | 0.343 | 0.1420 | 0 |
| ppo_e1 | new | 271828 | 239 | +0.3072 | 1.3072 | 0.941 | -0.5218 | 54 | 108 | 0.552 | 0.448 | 0.1497 | 0 |

Mean ± population std (accepted three seeds, then all five):

| policy | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_e1 three-seed (accepted) mean ± std | n=3 | 239 | 0.4203 ± 0.1046 | 1.4203 ± 0.1046 | 1.1337 ± 0.2005 | -0.4539 ± 0.0367 | 55.6667 ± 6.5490 | 111.3333 ± 13.0979 | 0.6011 ± 0.0381 | 0.3989 ± 0.0381 | 0.1537 ± 0.0166 | - |
| ppo_e1 five-seed mean ± std (population) | n=5 | 239 | 0.7517 ± 0.7252 | 1.7517 ± 0.7252 | 1.4632 ± 0.7750 | -0.4141 ± 0.1200 | 54.4000 ± 5.3889 | 108.8000 ± 10.7778 | 0.6025 ± 0.0444 | 0.3975 ± 0.0444 | 0.1506 ± 0.0137 | - |

New-seed checkpoints: 31415 `bb0a6f8a5988b044`, 271828 `e2ec8d7c640e6f06`

## E2 = O2 + R1 — five seeds

| policy | cohort | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_e2 | accepted | 42 | 239 | +1.1316 | 2.1316 | 2.130 | -0.3237 | 14 | 27 | 0.569 | 0.431 | 0.0397 | 1 |
| ppo_e2 | accepted | 123 | 239 | +1.2211 | 2.2211 | 2.065 | -0.4232 | 33 | 65 | 0.628 | 0.372 | 0.0930 | 1 |
| ppo_e2 | accepted | 2026 | 239 | +0.4271 | 1.4271 | 1.092 | -0.4563 | 32 | 63 | 0.636 | 0.364 | 0.0902 | 1 |
| ppo_e2 | new | 31415 | 239 | +0.6138 | 1.6138 | 1.333 | -0.5118 | 30 | 59 | 0.678 | 0.322 | 0.0848 | 1 |
| ppo_e2 | new | 271828 | 239 | +0.3961 | 1.3961 | 1.071 | -0.4627 | 20 | 39 | 0.594 | 0.406 | 0.0569 | 1 |

Mean ± population std (accepted three seeds, then all five):

| policy | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_e2 three-seed (accepted) mean ± std | n=3 | 239 | 0.9266 ± 0.3551 | 1.9266 ± 0.3551 | 1.7626 ± 0.4747 | -0.4010 ± 0.0563 | 26.3333 ± 8.7305 | 51.6667 ± 17.4611 | 0.6109 ± 0.0298 | 0.3891 ± 0.0298 | 0.0743 ± 0.0245 | - |
| ppo_e2 five-seed mean ± std (population) | n=5 | 239 | 0.7580 ± 0.3508 | 1.7580 ± 0.3508 | 1.5383 ± 0.4664 | -0.4355 ± 0.0627 | 25.8000 ± 7.4940 | 50.6000 ± 14.9880 | 0.6209 ± 0.0372 | 0.3791 ± 0.0372 | 0.0729 ± 0.0210 | - |

New-seed checkpoints: 31415 `b757148319053434`, 271828 `f3236f5e6e048d5b`

## E3 = O1 + R2 — five seeds

| policy | cohort | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_e3 | accepted | 42 | 239 | +1.3669 | 2.3669 | 2.341 | -0.3723 | 64 | 128 | 0.598 | 0.402 | 0.1748 | 0 |
| ppo_e3 | accepted | 123 | 239 | +0.3593 | 1.3593 | 1.068 | -0.4134 | 57 | 114 | 0.536 | 0.464 | 0.1573 | 0 |
| ppo_e3 | accepted | 2026 | 239 | +0.1221 | 1.1221 | 0.598 | -0.5068 | 57 | 114 | 0.611 | 0.389 | 0.1573 | 0 |
| ppo_e3 | new | 31415 | 239 | +2.1017 | 3.1017 | 2.995 | -0.1736 | 54 | 108 | 0.603 | 0.397 | 0.1497 | 0 |
| ppo_e3 | new | 271828 | 239 | +1.0498 | 2.0498 | 2.027 | -0.3193 | 52 | 104 | 0.573 | 0.427 | 0.1445 | 0 |

Mean ± population std (accepted three seeds, then all five):

| policy | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_e3 three-seed (accepted) mean ± std | n=3 | 239 | 0.6161 ± 0.5396 | 1.6161 ± 0.5396 | 1.3359 ± 0.7361 | -0.4308 ± 0.0563 | 59.3333 ± 3.2998 | 118.6667 ± 6.5997 | 0.5816 ± 0.0329 | 0.4184 ± 0.0329 | 0.1631 ± 0.0083 | - |
| ppo_e3 five-seed mean ± std (population) | n=5 | 239 | 1.0000 ± 0.7116 | 2.0000 ± 0.7116 | 1.8058 ± 0.8660 | -0.3571 ± 0.1104 | 56.8000 ± 4.0694 | 113.6000 ± 8.1388 | 0.5841 ± 0.0273 | 0.4159 ± 0.0273 | 0.1567 ± 0.0103 | - |

Reward diagnostics per seed (training reward, NOT return):

| policy | seed | cohort | cumulative_financial_r1 | cumulative_turnover_penalty | cumulative_drawdown_penalty | cumulative_reward_r2 | n_turnover_penalised_steps | n_drawdown_penalised_steps |
|---|---|---|---|---|---|---|---|---|
| ppo_e3 | 42 | accepted | +0.8616 | 0.0640 | 0.1701 | +0.6275 | 128 | 132 |
| ppo_e3 | 123 | accepted | +0.3070 | 0.0570 | 0.1604 | +0.0896 | 114 | 118 |
| ppo_e3 | 2026 | accepted | +0.1152 | 0.0570 | 0.1914 | -0.1332 | 114 | 130 |
| ppo_e3 | 31415 | new | +1.1319 | 0.0540 | 0.1675 | +0.9104 | 108 | 122 |
| ppo_e3 | 271828 | new | +0.7178 | 0.0520 | 0.1648 | +0.5009 | 104 | 114 |

Reward diagnostics mean ± std:

| policy | seed | cumulative_financial_r1 | cumulative_turnover_penalty | cumulative_drawdown_penalty | cumulative_reward_r2 | n_turnover_penalised_steps | n_drawdown_penalised_steps |
|---|---|---|---|---|---|---|---|
| ppo_e3 three-seed (accepted) | n=3 | 0.4279 ± 0.3165 | 0.0593 ± 0.0033 | 0.1740 ± 0.0130 | 0.1946 ± 0.3193 | 118.6667 ± 6.5997 | 126.6667 ± 6.1824 |
| ppo_e3 five-seed | n=5 | 0.6267 ± 0.3695 | 0.0568 ± 0.0041 | 0.1709 ± 0.0108 | 0.3990 ± 0.3750 | 113.6000 ± 8.1388 | 123.2000 ± 6.8819 |

New-seed checkpoints: 31415 `86caa1068db3dd60`, 271828 `4ac816a1071693f7`

## E4 = O2 + R2 — five seeds

| policy | cohort | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_e4 | accepted | 42 | 239 | +0.1820 | 1.1820 | 0.735 | -0.3524 | 18 | 35 | 0.381 | 0.619 | 0.0512 | 1 |
| ppo_e4 | accepted | 123 | 239 | +0.4175 | 1.4175 | 1.078 | -0.4732 | 30 | 60 | 0.628 | 0.372 | 0.0861 | 0 |
| ppo_e4 | accepted | 2026 | 239 | +0.5260 | 1.5260 | 1.239 | -0.4627 | 29 | 58 | 0.569 | 0.431 | 0.0834 | 0 |
| ppo_e4 | new | 31415 | 239 | +0.4442 | 1.4442 | 1.107 | -0.4426 | 24 | 47 | 0.661 | 0.339 | 0.0681 | 1 |
| ppo_e4 | new | 271828 | 239 | +1.1094 | 2.1094 | 1.935 | -0.3797 | 28 | 55 | 0.598 | 0.402 | 0.0792 | 1 |

Mean ± population std (accepted three seeds, then all five):

| policy | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_e4 three-seed (accepted) mean ± std | n=3 | 239 | 0.3752 ± 0.1436 | 1.3752 ± 0.1436 | 1.0174 ± 0.2101 | -0.4294 ± 0.0547 | 25.6667 ± 5.4365 | 51.0000 ± 11.3431 | 0.5258 ± 0.1053 | 0.4742 ± 0.1053 | 0.0736 ± 0.0159 | - |
| ppo_e4 five-seed mean ± std (population) | n=5 | 239 | 0.5358 ± 0.3087 | 1.5358 ± 0.3087 | 1.2187 ± 0.3949 | -0.4221 ± 0.0476 | 25.8000 ± 4.4000 | 51.0000 ± 9.1433 | 0.5674 ± 0.0982 | 0.4326 ± 0.0982 | 0.0736 ± 0.0128 | - |

Reward diagnostics per seed (training reward, NOT return):

| policy | seed | cohort | cumulative_financial_r1 | cumulative_turnover_penalty | cumulative_drawdown_penalty | cumulative_reward_r2 | n_turnover_penalised_steps | n_drawdown_penalised_steps |
|---|---|---|---|---|---|---|---|---|
| ppo_e4 | 42 | accepted | +0.1672 | 0.0175 | 0.1304 | +0.0193 | 35 | 65 |
| ppo_e4 | 123 | accepted | +0.3489 | 0.0300 | 0.2036 | +0.1152 | 60 | 105 |
| ppo_e4 | 2026 | accepted | +0.4227 | 0.0290 | 0.1882 | +0.2054 | 58 | 96 |
| ppo_e4 | 31415 | new | +0.3676 | 0.0235 | 0.2215 | +0.1225 | 47 | 102 |
| ppo_e4 | 271828 | new | +0.7464 | 0.0275 | 0.1887 | +0.5302 | 55 | 96 |

Reward diagnostics mean ± std:

| policy | seed | cumulative_financial_r1 | cumulative_turnover_penalty | cumulative_drawdown_penalty | cumulative_reward_r2 | n_turnover_penalised_steps | n_drawdown_penalised_steps |
|---|---|---|---|---|---|---|---|
| ppo_e4 three-seed (accepted) | n=3 | 0.3129 ± 0.1073 | 0.0255 ± 0.0057 | 0.1741 ± 0.0315 | 0.1133 ± 0.0760 | 51.0000 ± 11.3431 | 88.6667 ± 17.1335 |
| ppo_e4 five-seed | n=5 | 0.4105 ± 0.1886 | 0.0255 ± 0.0046 | 0.1865 ± 0.0306 | 0.1985 ± 0.1760 | 51.0000 ± 9.1433 | 92.8000 ± 14.3304 |

New-seed checkpoints: 31415 `6bc39bc643d76e74`, 271828 `8616ec01318edfab`

## Five-seed E1–E4 matrix (mean ± population std over 42, 123, 2026, 31415, 271828)

| experiment | definition | n_seeds | total_return_mean | total_return_std | final_equity_mean | final_equity_std | sharpe_ratio_mean | sharpe_ratio_std | max_drawdown_mean | max_drawdown_std | n_entries_mean | n_entries_std | n_legs_mean | n_legs_std | exposure_mean | exposure_std | fraction_CASH_mean | fraction_CASH_std | total_cost_fraction_mean | total_cost_fraction_std | ending_LONG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | O1 + R1 | 5 | +0.7517 | +0.7252 | +1.7517 | +0.7252 | +1.4632 | +0.7750 | -0.4141 | +0.1200 | +54.4000 | +5.3889 | +108.8000 | +10.7778 | +0.6025 | +0.0444 | +0.3975 | +0.0444 | +0.1506 | +0.0137 | 0 |
| E2 | O2 + R1 | 5 | +0.7580 | +0.3508 | +1.7580 | +0.3508 | +1.5383 | +0.4664 | -0.4355 | +0.0627 | +25.8000 | +7.4940 | +50.6000 | +14.9880 | +0.6209 | +0.0372 | +0.3791 | +0.0372 | +0.0729 | +0.0210 | 5 |
| E3 | O1 + R2 | 5 | +1.0000 | +0.7116 | +2.0000 | +0.7116 | +1.8058 | +0.8660 | -0.3571 | +0.1104 | +56.8000 | +4.0694 | +113.6000 | +8.1388 | +0.5841 | +0.0273 | +0.4159 | +0.0273 | +0.1567 | +0.0103 | 0 |
| E4 | O2 + R2 | 5 | +0.5358 | +0.3087 | +1.5358 | +0.3087 | +1.2187 | +0.3949 | -0.4221 | +0.0476 | +25.8000 | +4.4000 | +51.0000 | +9.1433 | +0.5674 | +0.0982 | +0.4326 | +0.0982 | +0.0736 | +0.0128 | 3 |

## Seed dispersion: accepted three seeds vs final five seeds

| experiment | metric | mean_3 | std_3 | mean_5 | std_5 | new_seed_31415 | new_seed_271828 |
|---|---|---|---|---|---|---|---|
| E1 | total_return | +0.4203 | +0.1046 | +0.7517 | +0.7252 | +2.1903 | +0.3072 |
| E1 | sharpe_ratio | +1.1337 | +0.2005 | +1.4632 | +0.7750 | +2.9744 | +0.9406 |
| E1 | max_drawdown | -0.4539 | +0.0367 | -0.4141 | +0.1200 | -0.1869 | -0.5218 |
| E1 | n_legs | +111.3333 | +13.0979 | +108.8000 | +10.7778 | +102.0000 | +108.0000 |
| E1 | total_cost_fraction | +0.1537 | +0.0166 | +0.1506 | +0.0137 | +0.1420 | +0.1497 |
| E2 | total_return | +0.9266 | +0.3551 | +0.7580 | +0.3508 | +0.6138 | +0.3961 |
| E2 | sharpe_ratio | +1.7626 | +0.4747 | +1.5383 | +0.4664 | +1.3325 | +1.0709 |
| E2 | max_drawdown | -0.4010 | +0.0563 | -0.4355 | +0.0627 | -0.5118 | -0.4627 |
| E2 | n_legs | +51.6667 | +17.4611 | +50.6000 | +14.9880 | +59.0000 | +39.0000 |
| E2 | total_cost_fraction | +0.0743 | +0.0245 | +0.0729 | +0.0210 | +0.0848 | +0.0569 |
| E3 | total_return | +0.6161 | +0.5396 | +1.0000 | +0.7116 | +2.1017 | +1.0498 |
| E3 | sharpe_ratio | +1.3359 | +0.7361 | +1.8058 | +0.8660 | +2.9945 | +2.0268 |
| E3 | max_drawdown | -0.4308 | +0.0563 | -0.3571 | +0.1104 | -0.1736 | -0.3193 |
| E3 | n_legs | +118.6667 | +6.5997 | +113.6000 | +8.1388 | +108.0000 | +104.0000 |
| E3 | total_cost_fraction | +0.1631 | +0.0083 | +0.1567 | +0.0103 | +0.1497 | +0.1445 |
| E4 | total_return | +0.3752 | +0.1436 | +0.5358 | +0.3087 | +0.4442 | +1.1094 |
| E4 | sharpe_ratio | +1.0174 | +0.2101 | +1.2187 | +0.3949 | +1.1067 | +1.9350 |
| E4 | max_drawdown | -0.4294 | +0.0547 | -0.4221 | +0.0476 | -0.4426 | -0.3797 |
| E4 | n_legs | +51.0000 | +11.3431 | +51.0000 | +9.1433 | +47.0000 | +55.0000 |
| E4 | total_cost_fraction | +0.0736 | +0.0159 | +0.0736 | +0.0128 | +0.0681 | +0.0792 |

## Baselines (same environment and accounting; verified identical through the E1, E2, E3 and E4 environments)

| policy | seed | n_steps | total_return | final_equity | sharpe_ratio | max_drawdown | n_entries | n_legs | exposure | fraction_CASH | total_cost_fraction | final_position |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always_CASH | - | 239 | +0.0000 | 1.0000 | - | +0.0000 | 0 | 0 | 0.000 | 1.000 | 0.0000 | 0 |
| buy_and_hold_always_LONG | - | 239 | +0.6645 | 1.6645 | 1.322 | -0.5314 | 1 | 1 | 1.000 | 0.000 | 0.0015 | 1 |
| random | 0 | 239 | -0.2435 | 0.7565 | -0.504 | -0.4554 | 67 | 133 | 0.456 | 0.544 | 0.1810 | 1 |
| random | 1 | 239 | -0.2631 | 0.7369 | -0.521 | -0.4149 | 60 | 120 | 0.481 | 0.519 | 0.1648 | 0 |
| random | 2 | 239 | +0.0082 | 1.0082 | 0.328 | -0.5320 | 58 | 116 | 0.506 | 0.494 | 0.1598 | 0 |
| random | 3 | 239 | -0.0020 | 0.9980 | 0.304 | -0.4807 | 63 | 125 | 0.477 | 0.523 | 0.1711 | 1 |
| random | 4 | 239 | +0.2359 | 1.2359 | 0.886 | -0.2254 | 63 | 125 | 0.431 | 0.569 | 0.1711 | 1 |

Baseline reward diagnostics (what R2 would have paid these fixed policies):

| policy | seed | cumulative_financial_r1 | cumulative_turnover_penalty | cumulative_drawdown_penalty | cumulative_reward_r2 | n_turnover_penalised_steps | n_drawdown_penalised_steps |
|---|---|---|---|---|---|---|---|
| always_CASH | - | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0 | 0 |
| buy_and_hold_always_LONG | - | +0.5095 | 0.0005 | 0.2866 | +0.2224 | 1 | 119 |
| random | 0 | -0.2791 | 0.0665 | 0.1565 | -0.5020 | 133 | 125 |
| random | 1 | -0.3053 | 0.0600 | 0.1595 | -0.5248 | 120 | 123 |
| random | 2 | +0.0081 | 0.0580 | 0.1598 | -0.2096 | 116 | 114 |
| random | 3 | -0.0020 | 0.0625 | 0.1473 | -0.2118 | 125 | 119 |
| random | 4 | +0.2118 | 0.0625 | 0.1325 | +0.0168 | 125 | 117 |
