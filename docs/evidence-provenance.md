# Evidence provenance

This repository separates the **reported historical evidence** from outputs created by later reruns.

## Included historical evidence

- The exact raw daily BTC/USDT input used by the study is included at `data/raw/btc-usdt-daily-raw-v4.csv` and is pinned by SHA-256 in `configs/foundation.toml`.
- Frozen human-readable and machine-readable VALIDATION reports are included under `results/data/validation/`.
- Frozen human-readable and machine-readable TEST reports are included under `results/data/test/`.
- The static figures under `results/figures/` are explanatory derivatives of the included numerical reports. They do not introduce new experiments or metrics.
- The main `README.md` and `results/README.md` restate the reported results in plain language and link to the underlying reports.

The historical reports identify their source CSV files. The main README gives the aggregate results, per-seed returns, baseline results and the descriptive RQ1–RQ3 comparisons used in the written assignment.

## Not included

The repository does not distribute the original trained model checkpoints, training logs, full run metadata, per-step portfolio series, policy probabilities, critic estimates or day-by-day replay payloads. Consequently, the original trained agents cannot be reconstructed byte for byte from this repository alone.

The read-only dashboard remains deliberately disabled. No dashboard evidence package or approved binding is installed, so the dashboard displays no scientific result. The included reports and static figures are the public-readable evidence surface.

## Outputs of running this code

Any model, report or other artifact created by a later execution of this repository is a **new output**. A new run can reproduce the method, but it is not one of the original frozen agents and must not be presented as the historical evidence reported in the assignment. Because the original TEST results have already been inspected, the same calendar window cannot serve as a new untouched hold-out for later model choices.
