# Architecture

```
ppo-sequential-trading-study/
  README.md                    purpose, design, glossary, limitations, status
  PUBLICATION-STATUS.md        private, unpublished candidate; gates pending
  pyproject.toml, uv.lock, .python-version   locked dependency closure (CPU-only PyTorch wheels)
  configs/foundation.toml      frozen parameters: dataset identity, splits, costs, observation, accounting
  data/raw/                    exact frozen historical dataset used by the study
  src/btc_rl/                  experiment source (see module map)
  tests/                       complete synthetic and real-data integrity tests
  dashboard/                   read-only evidence dashboard (index.html, css/, js/); disabled without an approved package
  docs/                        methodology, evidence provenance, limitations, reproduction boundary, dashboard, architecture
  results/data/                included frozen VALIDATION and TEST reports and CSV tables
  results/figures/             static explanatory figures derived from the included reports
```

Absent by design: `artifacts/` (original checkpoints, curves and logs withheld), `provenance/` and `dashboard/data/` (created only by an evidence package). `results/` contains the frozen numerical reports and static explanatory figures; `data/raw/` contains the exact study input.

## Module map (`src/btc_rl/`)

| Module | Role | Runs inside this boundary? |
|---|---|---|
| `data.py` | raw CSV loading, hard cutoff, integrity validation, pinned identity | yes; dataset included |
| `splits.py` | frozen TRAIN / VALIDATION / TEST decision windows, purge rule | yes |
| `costs.py` | cost configuration and per-leg factors | yes |
| `observations.py` | O1 minimal observation | yes |
| `observations_o2.py` | O2 engineered observation | yes |
| `scaling.py` | TRAIN-only fit of the O1 return scale | yes; dataset included |
| `env.py` | Gymnasium environment: timing contract, open-to-open accounting | yes (synthetic frames) |
| `reference.py` | independent vectorised accounting used by the tests | yes |
| `evaluation.py` | episode runner and metrics (365 periods per year) | yes |
| `policies.py` | deterministic baselines; replay helper (input not distributed) | baselines yes; replay no |
| `baselines.py` | command-line baseline run (`include_replay=False` inside this boundary) | yes; dataset included |
| `reward_r2.py` | R2 risk-aware reward mixin with frozen coefficients | yes |
| `config.py` | TOML loader with consistency checks against the frozen constants | yes |
| `preflight.py` | original environment / lock / source certification gate | partial only; private replay input withheld |
| `_pytest_certify.py` | pytest plugin used by the certification gate | yes |
| `ppo_e1.py` … `ppo_e4.py` | experiment runners: PPO hyperparameters, split enforcement, metadata | yes for new runs; use the README public-rerun procedure |
| `ppo_e1_report.py` … `ppo_e4_report.py` | VALIDATION report builders | no: private artifacts |
| `final_cohort.py`, `final_cohort_report.py` | five-seed cohort harness and reports | no: private artifacts |
| `final_test.py`, `final_test_report.py` | human-gated original TEST harness and report builder | no: required original private artifacts are withheld |

Every command-line module is guarded by `if __name__ == "__main__":`; importing a module executes no training, evaluation, file write or network access.

## Entry points

| Command | Requirement |
|---|---|
| `uv run pytest -q` | complete tests, including the included dataset |
| `uv run python -m btc_rl.baselines --split validation --no-replay` | included dataset |
| `uv run python -m btc_rl.preflight --skip-replay` | partial diagnostic only; the original replay input is withheld |
| `uv run python -m btc_rl.ppo_e1 --seed 42 --smoke --skip-preflight` and the other runners | included dataset; outputs are new outputs, not frozen evidence |
| `cd dashboard && python -m http.server 8080` | none; shows *No evidence package installed* |

## Public evidence flow

```
frozen numerical reports ──▶ results/data/
            │
            ├──────────────▶ README result tables and explanations
            └──────────────▶ results/figures/ static explanatory charts

dashboard/data/ remains empty ──▶ dashboard displays no scientific result
```

New executions write new artifacts. They do not replace or re-certify the included historical reports.
