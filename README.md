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

## Estructura

```text
eso/
  cli.py            CLI agent-ready: inspect, diagnose, explore.
  data/             Carga CSV/Parquet/NPY, validación y preprocesamiento.
  diagnostics/      Diagnóstico ciego: dimensión, estacionariedad, caos, simetrías.
  topology/         Variedades, reconstrucción relacional, validación y evaluación.
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
