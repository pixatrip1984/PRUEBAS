# ESO — Explorador de Espacios Originales

ESO es una reestructuración del proyecto BTC-SHSE. El objetivo ya no es asumir una geometría concreta para predecir una serie temporal, sino explorar qué espacio geométrico explica mejor la estructura interna de los datos.

## Motivación

La experiencia con SHSE mostró que una esfera de Fibonacci puede ser una intuición útil, pero también una hipótesis prematura: sus vecinos k-NN angulares no preservan vecindad temporal. ESO generaliza esa intuición en un método:

1. Diagnosticar los datos sin asumir geometría.
2. Probar varias variedades candidatas.
3. Medir reconstrucción, suavidad y uso latente.
4. Registrar resultados.
5. Aprender a recomendar nuevas geometrías.

## Flujo agent-ready

Guía operativa para agentes: [`docs/agent_entry.md`](docs/agent_entry.md).

Entrada mínima:

```bash
python -m eso.cli explore my_data.csv --output reports/my_data
```

Salida esperada:

```text
reports/my_data/
  report.html              # informe visual para humanos
  report.md                # informe textual
  report.json              # salida estructurada para agentes
  metrics.csv              # ranking topológico tabular
  diagnosis.json           # diagnóstico ciego completo
  artifacts/model_handoff.json # contrato compacto para agentes de modelado
  figures/                 # visualizaciones PNG
  artifacts/               # espacio para artefactos auxiliares
```

Comandos disponibles:

```bash
python -m eso.cli inspect examples/harmonic.csv --columns x y
python -m eso.cli diagnose examples/harmonic.csv --columns x y --output reports/diagnosis.json
python -m eso.cli explore examples/harmonic.csv --columns x y --output reports/harmonic_eso
```

Ejemplo para datos BTC/mercado:

```bash
python -m eso.cli explore data/btc_3m/splits/train.parquet \
  --columns open high low close volume quote_asset_volume number_of_trades \
  --max-rows 10000 \
  --normalize robust \
  --manifolds circle sphere2 torus2 cylinder \
  --k 8 \
  --n-masks 5 \
  --output reports/btc_train_eso
```

El reporte responde:

- dimensión intrínseca aproximada;
- estacionariedad/tendencia;
- periodicidad/FFT/autocorrelación;
- disponibilidad de Lyapunov;
- ranking de variedades candidatas;
- estabilidad por múltiples máscaras;
- visualizaciones legibles.

## Salida para agentes de modelos

Cada `explore` genera `artifacts/model_handoff.json`, un contrato compacto con:

- fingerprint SHA-256 de la matriz numérica procesada;
- geometría candidata y ranking topológico;
- columnas/features disponibles y descripciones conocidas;
- targets recomendados para modelado;
- null results y guardrails causales para no repetir bugs de lookahead.

Para construir labels supervisadas sin reimplementar offsets:

```python
from eso.signals.targets import build_market_target_frame, regime_from_training_median

targets = build_market_target_frame(df, horizons=(1, 3, 12), close_col="close")
vol_regime, threshold = regime_from_training_median(targets["future_rv_12"], train_size=15000)
```

Para correr un baseline supervisado reproducible con split temporal:

```bash
python -m eso.cli model-eval data/BTCUSDT_1h.csv \
  --task volatility \
  --feature-mode cycle \
  --horizon 12 \
  --train-size 15000 \
  --output reports/btc_vol_model
```

Salida:

```text
reports/btc_vol_model/model_run_manifest.json
```

Ese manifest incluye features, target, split, baselines, métricas, fingerprint
del feature frame y guardrails causales.

Las corridas se registran por defecto en `reports/model_runs.csv`:

```bash
python -m eso.cli model-runs list --registry reports/model_runs.csv
python -m eso.cli model-runs compare <run_id_a> <run_id_b> --registry reports/model_runs.csv
python -m eso.cli validate-json reports/btc_vol_model/model_run_manifest.json
```

Para ejecutar una batería declarativa de probes:

```bash
python -m eso.cli benchmark benchmarks/btc_core.json --output reports/btc_core_benchmark
```

Para diseccionar un activo antes de proponer nueva lógica:

```bash
python -m eso.cli dissect data/BTCUSDT_1h.csv \
  --asset-id BTCUSDT_1h \
  --feature-mode cycle \
  --horizons 3 12 \
  --train-size 15000 \
  --output reports/BTCUSDT_1h_dissect
```

Para diseccionar un universo:

```bash
python -m eso.cli dissect-many universes/btc_single.json --output reports/universe_dissect
```

Los universos pueden incluir `reference_path` para añadir contexto relativo
contra BTC: retornos relativos, correlación rolling, beta rolling y z-score del
ratio de precio. Para alts con derivados, usa `feature_mode=alt_funding`.

## Estructura

```text
eso/
  cli.py            CLI agent-ready: inspect, diagnose, explore, model-eval,
                    benchmark, dissect, dissect-many.
  data/             Carga CSV/Parquet/NPY, validación y preprocesamiento.
  diagnostics/      Diagnóstico ciego: dimensión, estacionariedad, caos, simetrías.
  topology/         Variedades, reconstrucción relacional, validación y evaluación.
  signals/          Features estructurales y targets causales reutilizables.
  modeling/         Baselines supervisados y manifests reproducibles.
  meta/             Firmas, registro experimental y recomendador heurístico.
  reporting/        report.html, report.md, report.json, metrics.csv y figuras.
  pipeline.py       Orquestador principal ESOExplorer.
experiments/        Scripts ejecutables heredados/compatibles.
examples/           Datos mínimos para smoke tests y demos.
legacy/PRUEBAS/     Documentación del experimento SHSE original.
```

## Instalación

```bash
pip install -r requirements.txt
```

Para desarrollo:

```bash
pip install pytest
pytest -q
```

## Estado actual

Esta rama inicia la transición desde SHSE hacia ESO:

- SHSE queda como legado histórico y fuente de lecciones.
- ESO implementa un pipeline modular para diagnóstico y pruebas topológicas.
- El objetivo primario es comprensión geométrica, no predicción directa.
- El flujo ya está orientado a agentes: CSV/Parquet/NPY → reporte completo HTML/JSON/CSV.
