# ESO — Explorador de Espacios Originales

ESO es una reestructuración del proyecto BTC-SHSE. El objetivo ya no es asumir una geometría concreta para predecir una serie temporal, sino explorar qué espacio geométrico explica mejor la estructura interna de los datos.

## Motivación

La experiencia con SHSE mostró que una esfera de Fibonacci puede ser una intuición útil, pero también una hipótesis prematura: sus vecinos k-NN angulares no preservan vecindad temporal. ESO generaliza esa intuición en un método:

1. Diagnosticar los datos sin asumir geometría.
2. Probar varias variedades candidatas.
3. Medir reconstrucción, suavidad y uso latente.
4. Registrar resultados.
5. Aprender a recomendar nuevas geometrías.

## Estructura

```text
eso/
  diagnostics/      Diagnóstico ciego: dimensión, estacionariedad, caos, simetrías.
  topology/         Variedades, reconstrucción relacional y evaluación.
  meta/             Firmas, registro experimental y recomendador.
  data/             Datos sintéticos y cargadores.
  pipeline.py       Orquestador principal ESOExplorer.
experiments/        Scripts ejecutables.
legacy/PRUEBAS/     Documentación del experimento SHSE original.
```

## Instalación

```bash
pip install -r requirements.txt
```

PyTorch es opcional para algunas partes topológicas. La Fase 1 funciona con NumPy/SciPy/scikit-learn/statsmodels/nolds.

## Uso rápido

Diagnóstico de un oscilador armónico sintético:

```bash
python experiments/01_diagnose_btc.py --synthetic harmonic
```

Prueba comparativa de variedades sobre Lorenz:

```bash
python experiments/02_test_manifolds.py --dataset lorenz --manifolds circle sphere2 torus2
```

## Estado actual

Esta rama inicia la transición desde SHSE hacia ESO:

- SHSE queda como legado histórico y fuente de lecciones.
- ESO implementa un primer pipeline modular para diagnóstico y pruebas topológicas.
- El objetivo primario es comprensión geométrica, no predicción directa.
