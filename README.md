# BTC-SHSE — Spherical Holographic Surface Embedding para predicción de dirección BTC

## Qué es la primitiva SHSE

**SHSE (Spherical Holographic Surface Embedding)** es un preprocesador geométrico que convierte una ventana de datos de mercado en un grafo sobre la superficie de una esfera unitaria S².

La idea central:

```
ventana de tiempo (lookback, n_features)
    ↓
cada timestep → un nodo en la esfera de Fibonacci
    ↓
grafo k-NN por distancia angular entre nodos
    ↓
aprendizaje relacional mascarado:
  ocultar algunos nodos → predecir sus valores
  desde los vecinos visibles → ajustar pesos de atención
    ↓
estado latente entrenado: valores reconstruidos
+ incertidumbre por nodo + embeddings aprendidos
    ↓
tensor exportable para red neuronal externa
```

La primitiva tiene dos capas:

### Capa 1 — Geometría fija (siempre igual para un `lookback` dado)

| Elemento | Descripción | Código |
|---|---|---|
| `fibonacciSphere(n)` | N puntos equidistantes en S² | [test.js:86](test.js#L86) · [shse_encoder.py:14](shse_encoder.py#L14) |
| `buildKnn(points, k)` | Grafo k-NN por distancia angular | [test.js:98](test.js#L98) · [shse_encoder.py:25](shse_encoder.py#L25) |

Cada nodo `i` corresponde al timestep `i` del lookback. Los vecinos k-NN son los timesteps temporalmente más cercanos en la espiral de Fibonacci.

### Capa 2 — Estado latente entrenado (el corazón de la primitiva)

Aquí es donde SHSE se diferencia de un encoder fijo:

| Elemento | Descripción | Código JS |
|---|---|---|
| `W[valueDim][featureDim]`, `B[valueDim]` | Pesos de atención relacional | [test.js:186](test.js#L186) |
| `node.embedding[latentDim]` | Embedding aprendido por nodo | [test.js:136](test.js#L136) |
| `node.uncertainty` | Incertidumbre por nodo (actualizada en training) | [test.js:133](test.js#L133) |
| `predictNode(id)` | Predice valores de un nodo desde sus vecinos via atención softmax | [test.js:343](test.js#L343) |
| `trainStep(options)` | Un paso de masked relational learning (actualiza W, B, embeddings) | [test.js:378](test.js#L378) |
| `fit(steps)` | N pasos de entrenamiento hasta convergencia | [test.js:451](test.js#L451) |
| `exportTrainingTensors()` | Exporta el estado entrenado completo para red externa | [test.js:618](test.js#L618) |

**El error de entrenamiento interno** es la pérdida de reconstrucción mascarada:

```
maskedLoss = MSE(pred[nodos_ocultos], target[nodos_ocultos])
```

Esto se minimiza en cada llamada a `fit()`. El resultado es una superficie con:
- **values**: campo reconstruido sobre S² (no los valores crudos)
- **uncertainty**: alta donde la reconstrucción fue difícil
- **embeddings**: representaciones aprendidas por posición esférica

---

## El gap actual (v1 — estado del repo)

`shse_encoder.py` solo porta la **Capa 1** (geometría). La **Capa 2** (loop de entrenamiento) existe únicamente en JavaScript y **nunca se llama** en el pipeline Python:

```python
# shse_encoder.py:69 — encode() actual
# Solo asigna valores normalizados a los nodos.
# NO llama a fit(). El estado latente nunca se entrena.
x = np.concatenate([normed, self._positions.numpy()], axis=1)
```

Consecuencia: el GNN recibe valores crudos normalizados en los nodos, no representaciones aprendidas. La diferencia entre SHSE y el baseline es mínima porque ambos ven los mismos datos sin ninguna estructura latente entrenada.

---

## Estructura del proyecto

```
.
├── test.js                      # La primitiva SHSE completa (JS, referencia)
├── shse_encoder.py              # Puerto Python PARCIAL (solo geometría, sin training loop)
├── data.py                      # Descarga klines Binance + splits temporales
├── btc_gnn_dataset.py           # Dataset PyG — ventanas → grafos SHSE
├── btc_gnn_model.py             # GraphSAGE para clasificación binaria
├── btc_transformer_dataset.py   # Dataset plano — ventanas → tensores (baseline)
├── btc_transformer_model.py     # Transformer temporal (baseline sin SHSE)
├── compare_models.py            # Compara ambos modelos en 3 meses de datos
└── train_btc.py                 # Entrenamiento completo (descarga + train + eval)
```

---

## Pipeline completo

```
BTCUSDT 1m klines (Binance)
    ↓  data.py → BinanceBTCDatasetBuilder
./data/btc_Xm/splits/{train,val,test}.parquet
    ↓  btc_gnn_dataset.py → BTCDirectionDataset
Por cada ventana (lookback=64, 9 features):
    ↓  shse_encoder.py → SHSEEncoder.encode()
    PyG Data: 64 nodos × 12 features, 512 aristas
    ↓  btc_gnn_model.py → BTCDirectionGNN
    3 × SAGEConv(64) + mean/max pooling + MLP
    ↓
    label: sube(1) o baja(0) en horizon=5 minutos
```

---

## Instalación

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric pyarrow pandas numpy scikit-learn requests tqdm
```

---

## Uso

### Comparación rápida (3 meses, ~2 min descarga)

```bash
python compare_models.py
```

Opciones:
```bash
python compare_models.py --epochs 20 --step 5     # más épocas, más muestras
python compare_models.py --no-download            # si los parquets ya existen
```

### Entrenamiento completo

```bash
python train_btc.py                               # descarga 2020–2024, entrena
python train_btc.py --no-download --epochs 30     # si los datos ya existen
python train_btc.py --max-train 20000             # dataset reducido en CPU
```

---

## Resultados actuales (v1)

Con 3 meses de datos BTCUSDT 1m (Oct–Dic 2024), 15 épocas, step=10, lookback=64:

| Modelo | Params | Test Acc | Test F1 | Val F1 | s/epoch |
|---|---|---|---|---|---|
| SHSE + GraphSAGE | 26,882 | 51.3% | 0.024 | 0.022 | 15s |
| Transformer (sin SHSE) | 30,146 | 48.7% | 0.617 | 0.629 | 29s |

**Interpretación**: Ambos modelos colapsan a predecir una sola clase:
- SHSE-GNN predice casi siempre "Baja" (recall Baja=98%, recall Sube=1%)
- Transformer predice casi siempre "Sube" (recall Sube=85%)

Ambos son soluciones degeneradas. La causa raíz: el SHSE no entrena su estado latente antes de pasarlo al GNN. La primitiva JS tiene el loop de convergencia; la versión Python no lo usa.

---

## Próxima iteración (v2 — objetivo)

> Ver prompt de agente al final de este documento.

El cambio central: **hacer que en cada época el estado latente de SHSE converja sobre cada ventana antes de que el GNN la vea**.

```
Actual (v1):
  window → IQR_normalize → nodos_crudos_en_esfera → GNN

Propuesto (v2):
  window → IQR_normalize → nodos_en_esfera
               ↓
          SHSE.fit()  [loop interno hasta convergencia]
          Loss: MSE(pred_mascarados, target_mascarados)
               ↓
          estado_latente_optimo: values_reconstruidos
                                 + uncertainty + embeddings
               ↓
          GNN usa este estado, no los valores crudos
```

El GNN ve una **representación aprendida de la estructura interna del mercado**, no datos crudos. La hipótesis es que la incertidumbre por nodo y los embeddings entrenados codifican información estructural (zonas de alta varianza, clusters temporales) que el GNN puede explotar para predecir dirección.

---

## Diagrama conceptual SHSE

```
Fibonacci sphere (64 nodos = 64 minutos de historia)

  nodo_0  = minuto hace 64        nodo_63 = minuto más reciente
  │                                   │
  └── k-NN k=8: conectado a          └── k-NN k=8: vecinos
      los 8 timesteps más cercanos        en espiral temporal
      en distancia angular

Estado por nodo (post-fit):
  value[9]       = campo reconstruido (no el valor crudo)
  embedding[16]  = representación aprendida de esa posición temporal
  uncertainty    = qué tan difícil fue reconstruir este timestep
  latent[3]      = proyección volumétrica (visualización)
```
