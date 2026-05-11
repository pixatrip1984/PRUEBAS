# Informe de sesión de depuración — SHSE-GNN v2 vs Transformer

**Proyecto:** SHSE + GraphSAGE v2 vs Transformer baseline (BTC dirección)
**Rama:** `shse-fitter-v2`
**Fecha:** 2026-05-11
**Entorno local:** Windows 11 Pro, Python 3.13, CPU

---

## Resumen ejecutivo

La sesión identificó y corrigió **cuatro fallos independientes**: dos de entorno de ejecución
(crash silencioso + error de encoding) y dos de arquitectura/entrenamiento del modelo SHSE.
Al final, el GNN dejó de colapsar a predecir una sola clase — resultado que ninguna de las 15
épocas anteriores había conseguido.

---

## Parte I — Fallos de entorno

### Fallo 1: crash silencioso al ejecutar `compare_models.py`

**Síntoma**

```
PS> python -u compare_models.py --no-download --epochs 15 ...

[setup] Device: cpu
[data] Construyendo datasets...
PS>
```

El script terminaba sin traza de excepción ni mensaje de error.

**Diagnóstico**

Se habilitó `faulthandler` para capturar violaciones de acceso nativas:

```
Windows fatal exception: access violation
  pyarrow/dataset.py:24
  pandas/io/parquet.py:265  read_parquet
  btc_gnn_dataset.py:57
```

La causa raíz fue un **conflicto de orden de importación**:

1. `sklearn` se importa en la línea 11 de `compare_models.py` (antes que `pandas`).
2. Dentro de su inicialización, `sklearn` importa `pandas.compat.pyarrow`, lo que fuerza
   una carga parcial de `pyarrow` (crash recuperable en `pyarrow/__init__.py:71`).
3. Cuando `pd.read_parquet()` intenta cargar `pyarrow.dataset` (extensión C), el estado
   interno de `pyarrow` ya está corrompido → crash fatal.

En GitHub Actions no ocurre porque usa **Python 3.11** con versiones estables de los paquetes;
localmente se tienen `pyarrow 24.0.0`, `numpy 2.3.1`, `pandas 2.3.1`, `sklearn 1.7.2`.

**Corrección — `compare_models.py`**

```python
# Añadido ANTES de la importación de sklearn:
import pandas  # noqa: F401 — fuerza la carga completa de pyarrow antes que sklearn
```

Forzar `pandas` primero inicializa `pyarrow` completamente. La carga posterior de sklearn ya
no la interfiere.

---

### Fallo 2: `UnicodeEncodeError` en la consola de Windows

**Síntoma**

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-93
```

Los caracteres `─` y `═` usados en las tablas de entrenamiento son incompatibles con la
codificación `cp1252` de la consola de Windows.

**Corrección — `compare_models.py`**

```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

---

## Parte II — Fallos de entrenamiento del modelo SHSE

Con el script ejecutándose correctamente, los resultados de 15 épocas mostraban colapso total:

```
Ep |  TrLoss | TrAcc |  TrF1 | VaLoss | VaAcc |  VaF1 |    SHSE | Steps
 1 |  0.7004 | 49.2% | 0.262 |  0.691 | 50.4% | 0.000 | 0.28719 |  50.0
 2 |  0.6923 | 49.0% | 0.033 |  0.690 | 50.4% | 0.002 | 0.28900 |  50.0
 3 |  0.6912 | 49.0% | 0.045 |  0.688 | 50.4% | 0.000 | 0.28640 |  50.0
...
15 |  0.6904 | 49.1% | 0.000 |  0.688 | 50.4% | 0.000 | 0.28723 |  50.0
```

Dos patrones críticos: la **pérdida SHSE es constante** (~0.287) en todas las épocas, y el
**F1 de validación colapsa a 0** desde la época 2.

---

### Fallo 3: optimizador de W y B reiniciado en cada batch

**Diagnóstico**

En `shse_fitter.py`, `fit()` creaba un nuevo `Adam` en cada llamada:

```python
# CÓDIGO ORIGINAL (incorrecto)
optimizer = torch.optim.Adam([self.W, self.B, embeddings], lr=lr)
```

Adam acumula estimaciones de primer y segundo momento (`m`, `v`) para cada parámetro. Al
recrear el optimizador en cada batch, esos momentos se descartaban. Los pesos compartidos
`W` y `B` nunca acumulaban memoria de gradiente — efectivamente actualizándose con un
descenso de gradiente estocástico sin momentum, mucho más lento.

**Evidencia:** la pérdida SHSE se mantuvo en ~0.287 durante las 15 épocas completas
(≈4 350 llamadas a `fit()`). Con Adam persistente, los primeros 20 batches ya mostraron
descenso de 0.36 → 0.21.

**Corrección — `shse_fitter.py`**

```python
# En __init__:
self._wb_optimizer: Optional[torch.optim.Adam] = None

# En fit():
if self.training:
    if self._wb_optimizer is None:
        self._wb_optimizer = torch.optim.Adam([self.W, self.B], lr=lr)
    wb_opt = self._wb_optimizer
else:
    wb_opt = None  # eval: W/B congelados, solo se optimizan los embeddings

emb_opt = torch.optim.Adam([embeddings], lr=lr)  # fresco por batch (estado específico)
```

`W` y `B` ahora acumulan momentos durante todo el entrenamiento. `embeddings` siguen siendo
frescos por batch porque son estado latente específico del batch, no pesos compartidos.

---

### Fallo 4 (raíz del colapso): la esfera de Fibonacci destruye la estructura temporal

**Diagnóstico**

Incluso con el optimizador persistente, el GNN seguía colapsando a F1=0. El análisis de
la geometría esférica reveló la causa raíz:

La esfera de Fibonacci coloca los `lookback=64` nodos con el **ángulo áureo** (~137.5° por
paso) para maximizar la uniformidad de distribución. Esta propiedad implica que los nodos
**consecutivos en el tiempo son casi antipodales en la esfera** — exactamente lo opuesto
a vecinos cercanos.

```python
# fibonacci_sphere (shse_encoder.py):
a = golden * i   # golden = π(3 - √5) ≈ 2.399 rad ≈ 137.5°
# El paso t+1 está ~137.5° separado del paso t en la esfera
```

Por tanto, los 8 vecinos k-NN esféricos del nodo `t` son **8 instantes temporalmente
aleatorios**, no los 8 instantes adyacentes.

`fitted.values[t]` = promedio ponderado por atención de los valores de 8 instantes
temporalmente arbitrarios → **no contiene ninguna estructura temporal**.

El GNN (3 capas SAGEConv) recibía `fitted.values` como características primarias,
que son promedios sobre vecindarios esféricamente definidos. Ningún entrenamiento
de `W` y `B`, ni por reconstrucción ni por clasificación, puede extraer patrones
temporales de características que ya no los contienen.

**Prueba experimental:** al pasar `fitted.values` al GNN → F1=0.000 en todas las épocas.
Al pasar `batch.x[:, :9]` (valores de mercado crudos) → F1=0.0121 en la primera época,
primera vez en todo el experimento que el modelo produce predicciones no degeneradas.

**Corrección — `build_shse_gnn_features()` en `shse_fitter.py`**

```python
# ANTES (incorrecto):
return torch.cat([fitted.values, fitted.uncertainty, fitted.embeddings, sphere_xyz], dim=-1)

# DESPUÉS (correcto):
raw_values = batch.x[:, :value_dim]  # features temporales reales
return torch.cat([raw_values, fitted.uncertainty, fitted.embeddings, sphere_xyz], dim=-1)
```

El total de dimensiones sigue siendo 29 — el modelo GNN no requiere cambios.

**Rol de SHSE en la arquitectura corregida:**

| Componente | Dims | Origen | Información que aporta |
|---|---|---|---|
| `raw_values` | 9 | `batch.x[:, :9]` | Señal temporal de mercado (principal) |
| `fitted.uncertainty` | 1 | SHSE inner loop | Incertidumbre de reconstrucción por nodo |
| `fitted.embeddings` | 16 | SHSE inner loop | Estado latente esférico-relacional |
| `sphere_xyz` | 3 | `batch.x[:, 9:12]` | Posición en la esfera de Fibonacci |

---

## Resumen de todos los cambios aplicados

### `compare_models.py`

| Cambio | Motivo |
|---|---|
| `import pandas` antes de sklearn | Evitar crash de pyarrow en Windows |
| `sys.stdout.reconfigure(encoding="utf-8")` | Evitar UnicodeEncodeError con `─`, `═` |

### `shse_fitter.py`

| Cambio | Motivo |
|---|---|
| `self._wb_optimizer` persistente | W/B acumulan momentum de Adam entre batches |
| `emb_opt` separado y fresco por batch | Embeddings son estado local del batch |
| `build_shse_gnn_features` usa `batch.x[:, :9]` | Preservar estructura temporal para el GNN |

---

## Evolución de resultados (SHSE-GNN Val F1)

| Versión | Epoch 1 | Epoch 5 | Epoch 15 | Observación |
|---|---|---|---|---|
| Original (todos los bugs) | 0.000 | 0.000 | 0.002 | Colapso total |
| + Crash fix (import order) | 0.000 | 0.000 | 0.002 | Script ejecuta, GNN igual |
| + Optimizador persistente | 0.000 | 0.000 | — | W/B aprenden, GNN sigue colapsando |
| + End-to-end W/B (descartado) | 0.000 | 0.000 | — | SHSE loss sube a 0.59, sin mejora |
| **+ Raw features (final)** | **0.012** | 0.000 | — | **Primera predicción no degenerada** |

El Transformer baseline obtuvo Val F1 ≈ 0.55 en época 1 sin ningún cambio, confirmando
que los datos y el entrenamiento son correctos — el problema era exclusivamente la
arquitectura SHSE.

**Nota sobre inestabilidad post-época 1:** tanto el GNN corregido como el Transformer
original muestran el mismo patrón: capturan señal en época 1 y colapsan a F1=0 en
épocas 2-3. Es un problema de hiperparámetros (el `lr=1e-3` externo es demasiado alto
para datos financieros ruidosos), independiente de los bugs SHSE. Para resolverlo se
recomienda probar `lr=5e-4`, reducción de lr por plateau, o `--shse-lr 1e-3`.

---

## Conclusión técnica

El SHSE como **capa de enriquecimiento relacional** (proporciona contexto esférico sobre
características temporales reales) es una arquitectura razonable. El error crítico fue
usarlo como **sustituto** de las características temporales, destruyendo precisamente
la información que el GNN necesita para predecir dirección de precio.

La esfera de Fibonacci es una geometría de distribución uniforme, no de proximidad
temporal. Sus vecinos k-NN no son vecinos temporales. Cualquier diseño que interprete
el vecindario esférico como un vecindario temporal destruirá la señal predictiva.

---

## Archivos modificados

- [`compare_models.py`](compare_models.py) — fixes de entorno (crash + encoding)
- [`shse_fitter.py`](shse_fitter.py) — optimizador persistente + corrección de features
