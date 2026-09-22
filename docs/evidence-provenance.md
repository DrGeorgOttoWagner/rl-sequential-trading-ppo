# Evidence provenance

This repository separates **code** from **evidence**.

- **Code** is what this repository ships: sanitized source, configuration, locked dependencies, tests, dashboard and documentation. Public source files are derivative copies; the frozen evidence of the study is bound to the private byte identities of the original files, so the public code documents the method but cannot re-certify the evidence.
- **Evidence** is the set of frozen result tables, the TEST report and their derived dashboard payloads. Evidence enters this repository only as an **evidence package** produced by a deterministic producer under the Research Evidence Contract, validated independently and reviewed before any release. **No evidence package is installed in this candidate.**

## What an evidence package will contain

| Location | Content | Kind |
|---|---|---|
| `results/test/`, `results/validation/` | byte-identical copies of the frozen result tables and the TEST report, each with its SHA-256 in the package manifest | immutable evidence |
| `dashboard/data/` | the dashboard entry point (`provenance.json`) and the derived, canonical JSON payloads the dashboard renders | deterministic derivatives |
| `provenance/` | the package manifest, design identity, withheld-artifact index, software identity (including the private → public source hash map) and the claims-and-limitations payload | deterministic derivatives |
| `figures/` | an index that is empty in contract version 1 | deterministic derivative |

Every payload lists its sources and hashes. Derivatives restate frozen values exactly and introduce no new metric. Content whose publication decision is unresolved or negative is **withheld**: it has no file, its descriptor carries null physical fields, and it never reappears in another format.

## Identities a package carries

- `evidence_set_id`: the hash of the historical baseline manifest, the dataset hash and the TEST execution commit.
- `content_gate_profile_id`: the hash of the owner's two content decisions (derived data; per-seed data).
- `dashboard_set_id`: the hash of the dashboard subset's descriptor projections, recomputed by the dashboard before rendering.
- `package_candidate_id`: the hash of the manifest identity object; it names the whole package.

None of these proves that a package was reviewed, approved or released. Validation attestations, review records and release authorizations are external records.

## Withheld and blocked material

Model checkpoints, per-step series, replay payloads, the raw dataset, run metadata, harness records, training logs and private documents are never distributed. A package names them only by logical identifier, SHA-256 and size. Per-step series and replay payloads are additionally **blocked** in contract version 1: they have no public location or schema whatever the content decisions say.

## Outputs of running this code

Any output written by a later execution of this public code is a **new output**. It is never placed at a package location, never listed in a package manifest and never presented as frozen evidence. Two persisted metadata strings of such a run (the run-scope label and the prose rationale of the observation scale) intentionally carry public wording and therefore differ from the frozen private metadata.
