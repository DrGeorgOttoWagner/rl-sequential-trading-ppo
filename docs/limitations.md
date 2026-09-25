# Limitations

These limitations apply to every statement about the study and are shown on the first screen of the dashboard whenever an evidence package is rendered.

1. One asset was studied (one spot market).
2. One historical TEST path was evaluated, exactly once.
3. Five training seeds share one market path; they are not independent markets.
4. No significance test was pre-specified, and none is reported.
5. Decisions are made at daily frequency only.
6. Two portfolio states only: CASH and INVESTED.
7. No short selling.
8. No leverage.
9. Transaction costs are fixed modelled assumptions: a 0.10% trading fee plus a 0.05% allowance for slippage on each purchase or sale, together 0.15% of the traded value per transaction leg. They are not guaranteed real-world costs.
10. No time-varying market impact or liquidity model is included.
11. Dispersion across training seeds is material.
12. Cost drag is `1 − 0.9985^legs`: the compounded cost-factor drag on terminal equity, not fees as a fraction of initial equity.
13. Any reward × observation interaction is exploratory; no interaction hypothesis was pre-specified.
14. VALIDATION and TEST differ in horizon and market regime; levels are not comparable across splits without that qualification.
15. Descriptive evidence on one frozen period is neither generalization nor guaranteed profitability.
16. PPO is the only learning algorithm tested; the study does not compare PPO with DQN, A2C or another RL method.
17. The reference strategies are CASH, buy-and-hold and random actions. No simple active rule, such as a moving-average strategy, is included.
18. O2 changes both the form of the information and the history it summarizes, so the study cannot separate the effect of representation from the effect of the longer horizon.
19. The assignment's main tables do not report complete equity curves or time in CASH versus INVESTED, and policy probabilities and critic estimates were not retained.
20. The fixed cost model omits changing liquidity and market impact; actual execution costs can be higher in stressed conditions, especially for high-turnover policies.
21. Stronger evidence would require other assets, later evaluation windows, alternative algorithms, stronger active baselines and a newly protected TEST period.

## Statements this study does not make

Universal superiority of any configuration; guaranteed or expected profitability; generalization to other markets, assets or periods; statistical significance of any difference; that the five seeds are independent markets; that TEST performance proves the presence or absence of overfitting; that the reward × observation interaction was formally tested; any trading or investment advice.
