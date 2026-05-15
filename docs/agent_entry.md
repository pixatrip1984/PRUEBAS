# ESO Agent Entry

This repo is a laboratory for turning raw datasets into reproducible geometry,
features, causal targets, and supervised baselines. It is not a single trading
model and it should not hide experimental assumptions inside notebooks or ad-hoc
scripts.

## Stable Surfaces

- `eso.cli`: command entrypoint for agents.
- `eso.data`: ingestion, validation, normalization, fingerprints.
- `eso.signals`: reusable feature and target builders.
- `eso.modeling`: supervised baselines, manifests, registries.
- `eso.reporting`: report bundles and model handoffs.
- `schemas/`: versioned contracts for JSON artifacts.

Treat `experiments/` as historical research and `reports/` as generated output
unless a specific file is explicitly curated.

For merge/test handoff, use [`docs/merge_readiness.md`](merge_readiness.md).

## First Pass On A Dataset

```bash
python -m eso.cli inspect data.csv --output reports/dataset_inspect.json
python -m eso.cli explore data.csv --feature-mode compact --output reports/dataset_eso
python -m eso.cli validate-json reports/dataset_eso/report.json
python -m eso.cli validate-json reports/dataset_eso/artifacts/model_handoff.json
```

Read `artifacts/model_handoff.json` before proposing model architecture. It
contains the recommended targets, known nulls, causal guardrails, and suggested
commands.

## Supervised Baseline Rule

Every model proposal needs a baseline manifest:

```bash
python -m eso.cli model-eval data.csv \
  --task volatility \
  --feature-mode cycle \
  --horizon 12 \
  --output reports/model_volatility
```

Use `feature-mode cycle` for BTC-style ring features and `feature-mode compact`
for cheaper raw financial features. The resulting `model_run_manifest.json` is
the unit of comparison. Do not claim improvement unless the primary metric beats
its baseline in that manifest.

Useful registry commands:

```bash
python -m eso.cli model-runs list
python -m eso.cli model-runs compare <run_id_a> <run_id_b>
```

## Asset Dissection

Before proposing strategy logic for an asset, generate a dissection bundle:

```bash
python -m eso.cli dissect data/BTCUSDT_1h.csv \
  --asset-id BTCUSDT_1h \
  --feature-mode cycle \
  --horizons 3 12 \
  --train-size 15000 \
  --output reports/BTCUSDT_1h_dissect
```

Key outputs:

- `asset_dissect.json`: target summaries and top exploratory associations.
- `feature_target_correlations.csv`: train/test correlations per horizon.
- `quantile_regimes.csv`: feature-bin behavior for volatility and direction.
- `phase_sectors.csv`: ring phase sector behavior when cycle features exist.

Use this bundle to decide what to test next. Do not treat correlations or sector
tables as trading proof.

For multiple assets, use a universe file:

```bash
python -m eso.cli dissect-many universes/btc_single.json --output reports/universe_dissect
python -m eso.cli validate-json reports/universe_dissect/universe_dissect.json
```

The aggregate `asset_summary.csv` ranks assets by test-set feature/target
structure per horizon. This is the first table to open when deciding which
asset deserves deeper modeling.

For alts, include BTC as a reference in the universe spec:

```json
{
  "defaults": {
    "feature_mode": "alt_funding",
    "reference_path": "data/BTCUSDT_4h.csv",
    "relative_windows": [12, 42]
  }
}
```

This adds relative-return, rolling correlation, rolling beta, and price-ratio
features to the dissection tables.

## Benchmark Suites

Use benchmark suites when you want a repeatable battery of probes:

```bash
python -m eso.cli benchmark benchmarks/btc_core.json --output reports/btc_core_benchmark
python -m eso.cli validate-json reports/btc_core_benchmark/benchmark_result.json
```

The suite writes one `model_run_manifest.json` per run plus an aggregate
`benchmark_result.json`. Add a new run to the suite before changing model code
when the goal is to compare a hypothesis against the existing probes.

## Causal Guardrails

- Use temporal splits only; no shuffled train/test for market data.
- Feature rows are indexed at decision time `t`.
- Targets may use future prices only as labels.
- Use `rolling(..., center=False)` for prediction-time features.
- After `dropna()`, align targets by original dataframe index, not reset row
  positions.
- Compare direction classifiers to majority class.
- Compare volatility regressors to `vol_20` persistence when available.

## BTC-Specific State

The current BTC work supports volatility/regime modeling more than direction.
The 12h direction signal from ring/order-flow features is a documented null.
Short-horizon direction remains a low-priority probe, not a confirmed edge.
