# Architecture

```
ppo-sequential-trading-study/
  README.md                    purpose, design, glossary, limitations, status
  PUBLICATION-STATUS.md        private, unpublished candidate; gates pending
  pyproject.toml, uv.lock, .python-version   locked dependency closure (CPU-only PyTorch wheels)
  configs/foundation.toml      frozen parameters: dataset identity, splits, costs, observation, accounting
  src/btc_rl/                  experiment source (see module map)
  tests/                       tests that run without restricted data (dataset tests skip)
  dashboard/                   read-only evidence dashboard (index.html, css/, js/); data/ is empty until a package is installed
  docs/                        methodology, evidence provenance, limitations, reproduction boundary, dashboard, architecture
```

Absent by design: `data/` (dataset not distributed), `artifacts/` (checkpoints, curves, logs withheld), `results/`, `provenance/`, `figures/` and `dashboard/data/` (created only by an evidence package).

## Module map (`src/btc_rl/`)

| Module | Role | Runs inside this boundary? |
|---|---|---|
| `data.py` | raw CSV loading, hard cutoff, integrity validation, pinned identity | needs the dataset |
| `splits.py` | frozen TRAIN / VALIDATION / TEST decision windows, purge rule | yes |
| `costs.py` | cost configuration and per-leg factors | yes |
| `observations.py` | O1 minimal observation | yes |
| `observations_o2.py` | O2 engineered observation | yes |
| `scaling.py` | TRAIN-only fit of the O1 return scale | needs the dataset |
| `env.py` | Gymnasium environment: timing contract, open-to-open accounting | yes (synthetic frames) |
| `reference.py` | independent vectorised accounting used by the tests | yes |
| `evaluation.py` | episode runner and metrics (365 periods per year) | yes |
| `policies.py` | deterministic baselines; replay helper (input not distributed) | baselines yes; replay no |
| `baselines.py` | command-line baseline run (`include_replay=False` inside this boundary) | needs the dataset |
| `reward_r2.py` | R2 risk-aware reward mixin with frozen coefficients | yes |
| `config.py` | TOML loader with consistency checks against the frozen constants | yes |
| `preflight.py` | environment / lock / source certification gate | needs the dataset |
| `_pytest_certify.py` | pytest plugin used by the certification gate | yes |
| `ppo_e1.py` … `ppo_e4.py` | experiment runners: PPO hyperparameters, split enforcement, metadata | no: dataset and private artifacts |
| `ppo_e1_report.py` … `ppo_e4_report.py` | VALIDATION report builders | no: private artifacts |
| `final_cohort.py`, `final_cohort_report.py` | five-seed cohort harness and reports | no: private artifacts |
| `final_test.py`, `final_test_report.py` | human-gated one-time TEST harness and report builder | no: private artifacts; TEST is consumed |

Every command-line module is guarded by `if __name__ == "__main__":`; importing a module executes no training, evaluation, file write or network access.

## Entry points

| Command | Requirement |
|---|---|
| `uv run pytest -q` | none (dataset tests skip) |
| `uv run python -m btc_rl.baselines --split validation` | restored dataset |
| `uv run python -m btc_rl.preflight` | restored dataset |
| `uv run python -m btc_rl.ppo_e1 …` and the other runners | restored dataset; outputs are new outputs, not frozen evidence |
| `cd dashboard && python -m http.server 8080` | none; shows *No evidence package installed* |

## Evidence flow

```
private laboratory ──(deterministic producer, independent validation, review, owner decisions)──▶ evidence package
                                                                                                    │
                                        results/  provenance/  figures/  dashboard/data/  ◀────────┘
                                                                              │
                                                              dashboard verifies, then renders
```

The code in this repository never produces, edits or certifies an evidence package.
