# Reproduction boundary

What can and cannot be reproduced from this repository alone.

## Can be run from this repository

- The complete unit and integrity tests of the dataset, accounting, causality, costs, determinism, the Gymnasium API, split boundaries, temporal consistency and configuration loading (`tests/`).
- Import of every module. No module trains, evaluates, writes or fetches on import; every command-line module is guarded by `if __name__ == "__main__":`.
- The deterministic reference strategies through `python -m btc_rl.baselines --split validation --no-replay`.
- New E1–E4 PPO training and VALIDATION evaluation runs. The README provides short integration commands and the full-budget parameters.
- The dashboard, which renders nothing scientific while no evidence package is installed.

## Included dataset

- `data/raw/btc-usdt-daily-raw-v4.csv` is the exact study input. Its SHA-256 is `7ff14ebd8f2236eb733073cbea7bfd36d2e1c6977aff89e00bdfde74f3af1305`.
- `configs/foundation.toml` pins the same hash, first date, last retained date and row count. The loader refuses any file whose identity differs. `BTC_RL_RAW_CSV` may point to another byte-identical copy.

## Requires withheld private artifacts

- The original checkpoints, run metadata, detailed curves and closure records are not distributed. E1–E4 can train new models from the included data, but the original cohort and final TEST harnesses remain bound to the withheld original artifacts. A new training output is not the frozen historical evidence.
- The one-time TEST harness binds four withheld authority records by neutral relative names — `authority/cohort-corrections.md`, `authority/validation-freeze.md`, `authority/cohort-review.md` and `authority/cohort-closure-review.md` — each pinned by SHA-256. These records are not distributed. The harness therefore fails closed on a checkout of this repository and is not executable or reproducible end to end from it; no other file can satisfy the pinned digests.
- Run metadata written by this code carries the key `study_stage` with neutral stage labels (`ppo-e1` … `ppo-e4`, `final-cohort`, `final-test`). Frozen private records are not rewritten. The final-cohort sidecar reader requires `study_stage` and rejects a sidecar missing that field or carrying a different stage. These metadata changes do not establish general compatibility with historical private records. Anything written by this code is a new output, never frozen evidence.
- Model checkpoints are not distributed; nothing in this repository loads a model. `load_e1_model` and its aliases remain in the code for completeness only.

## Unavailable functionality

- The replay baseline (`policies.make_variant_a_replay`, `load_v5b_positions`, `v5b_artifact_path`) reads a position series from the author's earlier supervised-learning study. That file is not part of this repository. Inside this boundary `run_baselines` must be called with `include_replay=False`; replay-dependent baseline execution is not available and is not presented as available.
- The one-time TEST evaluation cannot recur: the TEST window is consumed.

## What reproduction would establish

A new training run reproduces the method, not necessarily the exact frozen numbers. The reported historical evidence remains identified by the original files and checkpoints, which are not part of this repository. Randomized learning and differences in supported software or hardware can also change new outputs.
