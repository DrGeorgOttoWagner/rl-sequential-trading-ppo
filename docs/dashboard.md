# Dashboard

`dashboard/` is a static, dependency-free page that renders a reviewed evidence package and nothing else.

## Serving

```bash
cd dashboard
python -m http.server 8080
# open http://localhost:8080
```

`fetch()` and Web Crypto need HTTP on a secure context (`https` or `localhost`); on `file://` the page fails closed with an explanatory message.

## Access rule

The page reads only same-origin files under `dashboard/data/`. It never fetches `provenance/`, `results/`, `figures/`, replay data, any other path or any other origin. It embeds no data, loads no model, computes no new result and never falls back to cached or default values.

## Consumer sequence (fail closed)

1. Fetch `data/provenance.json`, the entry point, and nothing else first. If it is absent: state **No evidence package installed**, no result.
2. Check the contract family, a supported major version, and that the binding identity is one of the reviewed constants embedded in the build. The list of reviewed constants is empty in this candidate, so any package fails closed at this step until a compiled binding has been approved.
3. Recompute the dashboard-set identifier from the canonical bytes of `dashboard_set` and compare.
4. Validate the five descriptor projections: order, the eight required keys, status vocabulary and status-specific nullability; the claims copy must be included.
5. Fetch each included path (and no other) as bytes; check size and SHA-256.
6. Parse; check required keys and the withheld-subobject form.
7. Render. For every withheld payload or withheld subobject the page shows the fixed notice *Not published: redistribution decision pending.*

Any failure in steps 1–6 produces an integrity error naming the step and the failure code, and no scientific value, chart or table.

## What the page shows

- Before any result: the limitations and the notice *Descriptive, non-inferential — not financial advice.*
- Every claim with its strength label. All wording comes verbatim from the claims payload; the page composes no prose.
- Every panel is labelled *Frozen evidence* or *Explanatory presentation*.
- The dashboard-set identifier is shown as an integrity value only. The page makes no statement that the package is reviewed, approved, release-eligible or authentic.

## Rendering adapters

The field-level schemas of the dashboard payloads are fixed by the transformation register of the Research Evidence Contract. The adapters in `js/app.js` follow the current draft of that register and are re-reviewed together with it; a payload that lacks a required field fails closed.
