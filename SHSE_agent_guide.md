# SHSE — Guía para usar la primitiva con datasets multidimensionales y entrenar distintas redes

## 1. Qué es SHSE

**SHSE** significa **Spherical Holographic Surface Embedding**.

La idea es convertir datos crudos en una representación estructurada sobre la superficie de una esfera. La esfera no cambia de dimensión cuando tus datos tienen más columnas. Lo que cambia es el tamaño del vector almacenado en cada nodo superficial.

```text
dato tabular / vectorial
        ↓
nodos sobre S²
        ↓
cada nodo contiene un vector de features
        ↓
grafo superficial k-NN
        ↓
masked modeling / imputación / embeddings
        ↓
tensores exportables para redes neuronales
```

La distinción principal es:

```text
nodeCount  = resolución de la superficie
valueDim   = número de columnas/features por nodo
latentDim  = tamaño del embedding aprendido por nodo
neighbors  = conectividad local del grafo superficial
```

Para un CSV de 10 columnas:

```js
const shse = new SphericalHolographicSurface({
  nodeCount: 1024,
  valueDim: 10,
  latentDim: 32,
  neighbors: 12
});
```

No necesitas una esfera de 10 dimensiones. Necesitas una superficie 2D con valores vectoriales de dimensión 10 en cada celda.

---

## 2. ¿Sirve la primitiva actual para CSV multidimensional?

Sí.

La primitiva actual acepta matrices:

```js
[
  [x11, x12, x13, ..., x1D],
  [x21, x22, x23, ..., x2D],
  [x31, x32, x33, ..., x3D],
  ...
]
```

Si el CSV tiene 10 columnas numéricas, entonces:

```js
valueDim = 10
```

y cada nodo contiene:

```js
node.value = [col1, col2, col3, ..., col10]
```

La función `encodeVector(rows)` ya puede recibir `Array<Array<number>>`.

---

## 3. Cómo obtener datos con `btc_dataset_builder`

El script `data.py` expone dos clases principales: `BinanceBTCDatasetBuilder` para descargar y construir el dataset histórico de klines de Binance, y `NeuralMarketDataset` para convertir ese dataset en ventanas listas para entrenar redes neuronales.

### 3.1 Construir el dataset raw

```python
from btc_dataset_builder import BinanceKlineConfig, BinanceBTCDatasetBuilder

config = BinanceKlineConfig(
    symbol="BTCUSDT",
    interval="1m",
    start_year=2017,
    start_month=8,
    end_year=2026,
    end_month=4,
    output_dir="./data/btc",
    save_csv=False,
    save_parquet=True,
    delete_zips_after_reading=True,
)

builder = BinanceBTCDatasetBuilder(config)
df = builder.build()

print(df.head())
print(df.tail())
print(df.columns)
```

El resultado es un `DataFrame` con las columnas de klines de Binance. El proceso descarga los archivos zip mensuales, los concatena y genera la estructura de directorios siguiente:

```text
./data/btc/
  raw/
    BTCUSDT_1m_raw_klines.parquet

  splits/
    train.parquet
    val.parquet
    test.parquet
```

Los splits son **cronológicos**, no aleatorios. Esto evita data leakage temporal, que es el error más común al entrenar modelos sobre series de tiempo financieras.

### 3.2 Preparar ventanas para una red neuronal

```python
import pandas as pd
from btc_dataset_builder import NeuralMarketDataset

train_df = pd.read_parquet("./data/btc/splits/train.parquet")
val_df   = pd.read_parquet("./data/btc/splits/val.parquet")
test_df  = pd.read_parquet("./data/btc/splits/test.parquet")

train_dataset = NeuralMarketDataset(train_df, lookback=256, horizon=10, target_mode="future_return")
val_dataset   = NeuralMarketDataset(val_df,   lookback=256, horizon=10, target_mode="future_return")
test_dataset  = NeuralMarketDataset(test_df,  lookback=256, horizon=10, target_mode="future_return")

X_train, y_train = train_dataset.to_numpy()
X_val,   y_val   = val_dataset.to_numpy()
X_test,  y_test  = test_dataset.to_numpy()

print(X_train.shape, y_train.shape)
# Ejemplo de salida:
# (3200000, 256, 9)  (3200000,)
```

La forma resultante es:

```text
X.shape = (n_samples, lookback, n_features)
y.shape = (n_samples,)
```

Parámetros clave de `NeuralMarketDataset`:

```text
lookback      = cuántos pasos de tiempo hacia atrás ve cada muestra
horizon       = cuántos pasos hacia adelante define el target
target_mode   = "future_return" → y es el retorno futuro normalizado
```

### 3.3 Conexión con SHSE

Cada muestra `X[i]` tiene forma `(lookback, n_features)`. Para alimentarla a SHSE hay dos estrategias:

**Opción A — SHSE sobre el snapshot de features (sin dimensión temporal)**

Usa directamente las 9 columnas de un instante de tiempo como `valueDim`:

```js
const shse = new SphericalHolographicSurface({
  nodeCount: 1024,
  valueDim: 9,   // n_features del dataset BTC
  latentDim: 32,
  neighbors: 12
});
```

Pasa cada fila del parquet como vector:

```python
import json, numpy as np, pandas as pd

df = pd.read_parquet("./data/btc/splits/train.parquet")
rows = df.select_dtypes("number").values.tolist()
# rows[i] = [open, high, low, close, volume, ...]  longitud 9
```

**Opción B — SHSE sobre la ventana temporal completa**

Aplana la ventana `(lookback, n_features)` y trata los `lookback * n_features` valores como el vector del nodo:

```js
valueDim = lookback * n_features   // 256 * 9 = 2304
```

Esto codifica toda la historia reciente en el valor de cada nodo superficial. Útil para modelos donde la topología temporal importa, pero más costoso en memoria.

### 3.4 Resumen de dimensiones BTC → SHSE

```text
Dataset              n_features = 9
                     n_samples  ≈ 3.2 millones (train, 1m desde 2017)

SHSE (opción A)      valueDim = 9
                     nodeCount = 1024 o 2048
                     latentDim = 32

SHSE (opción B)      valueDim = lookback × 9
                     nodeCount = 512 o 1024
                     latentDim = 64
```

---

## 4. Ejemplo completo con CSV de 10 columnas en Node.js

### 3.1 Instalar dependencias

```bash
npm init -y
npm install csv-parse
```

### 3.2 Estructura del proyecto

```text
project/
  spherical_holographic_surface_primitive.js
  datos.csv
  train_csv_shse.js
```

### 3.3 Código de entrenamiento

```js
const fs = require("fs");
const { parse } = require("csv-parse/sync");
const { SphericalHolographicSurface } = require("./spherical_holographic_surface_primitive.js");

const raw = fs.readFileSync("./datos.csv", "utf8");

const records = parse(raw, {
  columns: true,
  skip_empty_lines: true
});

// Cambia estos nombres por las columnas reales del CSV.
const columns = [
  "col1", "col2", "col3", "col4", "col5",
  "col6", "col7", "col8", "col9", "col10"
];

const rows = records.map(row =>
  columns.map(c => Number(row[c] ?? 0))
);

const shse = new SphericalHolographicSurface({
  nodeCount: 1024,
  valueDim: 10,
  latentDim: 32,
  neighbors: 12,
  learningRate: 0.025,
  maskRate: 0.35,
  seed: 123
});

shse.encodeVector(rows, {
  knownRatio: 0.9,
  normalize: true
});

const history = shse.fit(600, {
  maskRate: 0.4,
  batchSize: 160,
  remaskEachStep: true
});

console.log("Resumen SHSE:");
console.log(shse.summary());

console.log("Últimos pasos:");
console.table(history.slice(-5));

const tensors = shse.exportTrainingTensors();

console.log("Tensores exportados:");
console.log({
  positions: [tensors.positions.length, tensors.positions[0].length],
  values: [tensors.values.length, tensors.values[0].length],
  mask: tensors.mask.length,
  uncertainty: tensors.uncertainty.length,
  edges: [tensors.edges.length, tensors.edges[0].length],
  edgeAttr: [tensors.edgeAttr.length, tensors.edgeAttr[0].length],
  latent: [tensors.latent.length, tensors.latent[0].length],
  embeddings: [tensors.embeddings.length, tensors.embeddings[0].length]
});
```

---

## 5. Qué exporta `exportTrainingTensors()`

La salida principal es:

```js
const tensors = shse.exportTrainingTensors();
```

Estructura:

```js
{
  positions,     // [N, 3] coordenadas de cada nodo sobre S²
  values,        // [N, valueDim] valores codificados en superficie
  mask,          // [N] 1 si el valor es conocido, 0 si está oculto/desconocido
  uncertainty,   // [N] incertidumbre por nodo
  edges,         // [E, 2] aristas del grafo superficial
  edgeAttr,      // [E, 1] distancia angular u otro atributo de arista
  latent,        // [N, 3] reconstrucción interna superficie → volumen
  embeddings     // [N, latentDim] embedding aprendido por nodo
}
```

Para un CSV de 10 columnas y `nodeCount = 1024`:

```text
positions   = [1024, 3]
values      = [1024, 10]
mask        = [1024]
uncertainty = [1024]
edges       = [1024 * neighbors, 2]
edgeAttr    = [1024 * neighbors, 1]
latent      = [1024, 3]
embeddings  = [1024, latentDim]
```

---

## 6. Qué redes puedes entrenar con estos tensores

### 5.1 GNN / Graph Neural Network

Usa:

```text
x        = concat(values, positions, uncertainty, embeddings)
edge_idx = edges
edge_attr = edgeAttr
mask     = mask
target   = values originales
```

Objetivo recomendado:

```text
predecir valores enmascarados
```

Pérdida:

```text
MSE(pred[mask == 0], target[mask == 0])
```

Es la opción más natural, porque SHSE ya produce un grafo superficial.

---

### 5.2 Graph Autoencoder

Entrada:

```text
values + positions + mask + uncertainty
```

Encoder:

```text
GNN → z_node
```

Decoder:

```text
z_node → reconstructed values
```

Objetivos:

```text
1. reconstrucción de values
2. reconstrucción de valores ocultos
3. regularización de embeddings
```

Útil para:

```text
imputación
compresión
detección de anomalías
generación
```

---

### 5.3 Masked Autoencoder sobre superficie

SHSE ya genera máscaras. Puedes entrenar una red tipo masked modeling:

```text
ocultar parte de values
predecir values ocultos usando contexto superficial
```

Entrada:

```text
positions
masked values
mask
uncertainty
```

Salida:

```text
values reconstruidos
```

Pérdida:

```text
loss = MSE(predicted_values_on_masked_nodes, true_values_on_masked_nodes)
```

---

### 5.4 Transformer sobre nodos superficiales

Trata cada nodo como token:

```text
token_i = [position_i, value_i, mask_i, uncertainty_i, embedding_i]
```

Puedes usar atención global o atención local restringida por `edges`.

Ventajas:

```text
captura relaciones largas
puede aprender dependencias globales entre regiones de la esfera
sirve para generación
```

Desventajas:

```text
costo O(N²) si usas atención global
```

Recomendación:

```text
N <= 1024 para pruebas
N > 1024 usar atención local, sparse attention o graph transformer
```

---

### 5.5 MLP baseline

También puedes aplanar la representación:

```js
const flat = tensors.values.flat();
```

Pero esto pierde parte del valor de SHSE, porque destruye la estructura superficial.

Úsalo sólo como baseline.

---

## 7. Ejemplo conceptual en PyTorch Geometric

La primitiva exporta datos en un formato fácil de convertir a PyTorch.

```python
import torch
from torch_geometric.data import Data

positions = torch.tensor(tensors["positions"], dtype=torch.float32)
values = torch.tensor(tensors["values"], dtype=torch.float32)
uncertainty = torch.tensor(tensors["uncertainty"], dtype=torch.float32).unsqueeze(-1)
embeddings = torch.tensor(tensors["embeddings"], dtype=torch.float32)
mask = torch.tensor(tensors["mask"], dtype=torch.bool)

edge_index = torch.tensor(tensors["edges"], dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(tensors["edgeAttr"], dtype=torch.float32)

x = torch.cat([positions, values, uncertainty, embeddings], dim=-1)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=values,
    mask=mask
)
```

Pérdida de masked reconstruction:

```python
pred = model(data.x, data.edge_index, data.edge_attr)

masked_nodes = ~data.mask
loss = torch.nn.functional.mse_loss(
    pred[masked_nodes],
    data.y[masked_nodes]
)
```

---

## 8. Ejemplo de arquitectura GNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SHSEGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, value_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h)
```

Uso:

```python
input_dim = 3 + value_dim + 1 + latent_dim
model = SHSEGNN(input_dim=input_dim, hidden_dim=128, value_dim=value_dim)
```

---

## 9. Entrenar distintas redes con la misma primitiva

La primitiva SHSE debe verse como un **preprocesador geométrico**.

```text
CSV / vector / tabla
        ↓
SHSE
        ↓
tensores estructurados
        ↓
red A: GNN
red B: Graph Autoencoder
red C: Transformer
red D: Masked Autoencoder
red E: modelo generativo
```

La misma salida `exportTrainingTensors()` puede alimentar distintas redes.

---

## 10. Recomendaciones de configuración

### CSV pequeño

```js
{
  nodeCount: 256,
  valueDim: numberOfColumns,
  latentDim: 16,
  neighbors: 8
}
```

### CSV medio

```js
{
  nodeCount: 1024,
  valueDim: numberOfColumns,
  latentDim: 32,
  neighbors: 12
}
```

### CSV grande

```js
{
  nodeCount: 2048,
  valueDim: numberOfColumns,
  latentDim: 64,
  neighbors: 16
}
```

### Datos muy ruidosos

```js
{
  maskRate: 0.45,
  neighbors: 16,
  learningRate: 0.015
}
```

### Datos muy suaves / continuos

```js
{
  maskRate: 0.25,
  neighbors: 8,
  learningRate: 0.03
}
```

---

## 11. Qué conviene actualizar en la primitiva

La primitiva actual funciona, pero para entrenamiento serio conviene añadir métodos explícitos:

```js
shse.encodeTable(rows, {
  columnNames,
  normalize: "robust",
  knownRatio: 0.9
});

shse.decodeTable({
  length: originalRowCount
});

shse.exportTorchGeometricJSON();
shse.exportGraphDataset();
```

La actualización importante no es cambiar la dimensión de la esfera. La actualización importante es mejorar la interfaz para tablas multidimensionales.

---

## 12. Diseño recomendado para `encodeTable()`

Interfaz propuesta:

```js
shse.encodeTable(rows, {
  columns: ["a", "b", "c"],
  normalize: "robust",
  knownRatio: 0.9,
  missingValue: null
});
```

Debe guardar estadísticas por columna:

```js
shse.columnStats = [
  { name: "a", median, q1, q3, mean, std },
  { name: "b", median, q1, q3, mean, std },
  ...
];
```

Así luego se puede decodificar:

```js
const reconstructed = shse.decodeTable({
  denormalize: true
});
```

---

## 13. Resumen para el agente experto

### Pipeline completo de datos → SHSE → red neuronal

```text
Binance (klines históricas)
        ↓
BinanceBTCDatasetBuilder   →  raw/BTCUSDT_1m_raw_klines.parquet
        ↓
splits cronológicos        →  splits/{train,val,test}.parquet
        ↓
NeuralMarketDataset        →  X: (n_samples, lookback, n_features)
        ↓
SHSE (valueDim = n_features)
        ↓
exportTrainingTensors()
        ↓
red neuronal (GNN / Transformer / Autoencoder)
```

### Dimensiones clave para datos BTC 1m

```text
n_features  = 9     (columnas numéricas del kline)
valueDim    = 9     (opción A: snapshot)
           = 2304   (opción B: ventana lookback=256 aplanada)
nodeCount   = 1024–2048
latentDim   = 32–64
```

### Adquisición de datos

```python
# Descarga completa (puede tardar varios minutos la primera vez)
from btc_dataset_builder import BinanceKlineConfig, BinanceBTCDatasetBuilder

config = BinanceKlineConfig(
    symbol="BTCUSDT", interval="1m",
    start_year=2017, start_month=8,
    end_year=2026,   end_month=4,
    output_dir="./data/btc",
    save_parquet=True,
    delete_zips_after_reading=True,
)
df = BinanceBTCDatasetBuilder(config).build()
```

### Preparar tensores de entrenamiento

```python
from btc_dataset_builder import NeuralMarketDataset
import pandas as pd

X, y = NeuralMarketDataset(
    pd.read_parquet("./data/btc/splits/train.parquet"),
    lookback=256, horizon=10, target_mode="future_return"
).to_numpy()
# X.shape → (n_samples, 256, 9)
# y.shape → (n_samples,)
```

### Primitiva SHSE

```js
valueDim = 9   // n_features del dataset BTC
```

La esfera mantiene posiciones `[x, y, z]`, pero cada nodo almacena un vector `[9]`.

La salida de SHSE es un dataset gráfico sobre superficie esférica:

```text
nodes = valores + posiciones + incertidumbre + embeddings
edges = vecindad superficial
targets = valores originales
mask = nodos conocidos/desconocidos
```

Esto permite entrenar varias familias de redes:

```text
GNN
Graph Autoencoder
Masked Autoencoder
Graph Transformer
Transformer sobre nodos
modelo generativo condicional
```

La siguiente mejora de la librería debería ser una interfaz explícita para tablas:

```js
encodeTable()
decodeTable()
exportGraphDataset()
exportTorchGeometricJSON()
```
