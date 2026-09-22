# Reproduction boundary

What can and cannot be reproduced from this repository alone.

## Can be run without restricted inputs

- The unit tests of accounting, causality, costs, determinism, the Gymnasium API, split boundaries, temporal consistency and configuration loading (`tests/`). They use synthetic frames. Tests that request the real dataset skip when it is absent.
- Import of every module. No module trains, evaluates, writes or fetches on import; every command-line module is guarded by `if __name__ == "__main__":`.
- The dashboard, which renders nothing scientific while no evidence package is installed.

## Requires the non-distributed dataset

- `configs/foundation.toml` pins the dataset by SHA-256, first date, last date and row count, and points to `data/raw/btc-usdt-daily-raw-v4.csv`, a path that is empty in this repository. The dataset is not distributed; redistribution rights have not been established. A locally restored copy may be used through that path or the environment variable `BTC_RL_RAW_CSV`; the loader refuses any file whose identity differs.
- Dataset integrity tests (`tests/test_data.py`), the preflight certification and every training or evaluation entry point.

## Requires withheld private artifacts

- The runner and harness modules (`ppo_e1.py` … `ppo_e4.py`, `final_cohort.py`, `final_test.py` and their report builders) reference accepted artifact digests, checkpoints, run metadata and closure records that are not distributed. They are published for methodological transparency (hyperparameters, split boundaries, one-time TEST governance) and are **not executable inside this repository's boundary**. Running them is not a reproduction of the frozen evidence; any output would be a new output, never frozen evidence.
- The one-time TEST harness binds four withheld authority records by neutral relative names — `authority/cohort-corrections.md`, `authority/validation-freeze.md`, `authority/cohort-review.md` and `authority/cohort-closure-review.md` — each pinned by SHA-256. These records are not distributed. The harness therefore fails closed on a checkout of this repository and is not executable or reproducible end to end from it; no other file can satisfy the pinned digests.
- Run metadata written by this code carries the key `study_stage` with neutral stage labels (`ppo-e1` … `ppo-e4`, `final-cohort`, `final-test`). Frozen private records are not rewritten. The final-cohort sidecar reader requires `study_stage` and rejects a sidecar missing that field or carrying a different stage. These metadata changes do not establish general compatibility with historical private records. Anything written by this code is a new output, never frozen evidence.
- Model checkpoints are not distributed; nothing in this repository loads a model. `load_e1_model` and its aliases remain in the code for completeness only.

## Unavailable functionality

- The replay baseline (`policies.make_variant_a_replay`, `load_v5b_positions`, `v5b_artifact_path`) reads a position series from the author's earlier supervised-learning study. That file is not part of this repository. Inside this boundary `run_baselines` must be called with `include_replay=False`; replay-dependent baseline execution is not available and is not presented as available.
- The one-time TEST evaluation cannot recur: the TEST window is consumed.

## What reproduction would establish

Even with the dataset restored, a new training run reproduces the method, not the frozen evidence: the frozen evidence is certified by private byte identities of the original files, which the sanitized public copies do not share. The public software identity payload of an evidence package records both hashes for every file.
