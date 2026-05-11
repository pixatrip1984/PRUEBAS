/*
  Spherical Holographic Surface Embedding Primitive
  -------------------------------------------------
  Una primitiva sin gráficos para convertir datos en una memoria superficial
  entrenable sobre S², aprender relaciones superficie-superficie y generar datos.

  Idea:
    data vector -> superficie esférica estructurada -> masked relational learning
    -> imputación / sampling / embeddings / reconstrucción generativa

  No depende de librerías externas. Está pensada como núcleo experimental:
  después puede portarse a PyTorch/JAX/TF para entrenamiento real con autograd.
*/

class RNG {
  constructor(seed = 123456789) {
    this.seed = seed >>> 0;
  }

  next() {
    // Mulberry32
    let t = this.seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(a, b) {
    return a + (b - a) * this.next();
  }

  normal() {
    const u = Math.max(1e-12, this.next());
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function norm3(a) {
  return Math.sqrt(dot3(a, a));
}

function normalize3(a) {
  const n = norm3(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}

function dist3(a, b) {
  const x = a[0] - b[0];
  const y = a[1] - b[1];
  const z = a[2] - b[2];
  return Math.sqrt(x * x + y * y + z * z);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-clamp(x, -50, 50)));
}

function tanh(x) {
  return Math.tanh(clamp(x, -30, 30));
}

function softmax(logits) {
  let m = -Infinity;
  for (const x of logits) if (x > m) m = x;
  let s = 0;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - m);
    out[i] = e;
    s += e;
  }
  const inv = 1 / Math.max(1e-12, s);
  for (let i = 0; i < out.length; i++) out[i] *= inv;
  return out;
}

function fibonacciSphere(n) {
  const pts = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / Math.max(1, n - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const a = golden * i;
    pts.push([r * Math.cos(a), y, r * Math.sin(a)]);
  }
  return pts;
}

function buildKnn(points, k) {
  const n = points.length;
  const neighbors = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    const list = [];
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      // Spherical affinity: angular distance via dot product.
      const cos = clamp(dot3(points[i], points[j]), -1, 1);
      const d = Math.acos(cos);
      list.push([j, d]);
    }
    list.sort((a, b) => a[1] - b[1]);
    neighbors[i] = list.slice(0, k).map(([id, d]) => ({ id, angularDistance: d }));
  }
  return neighbors;
}

function percentile(values, p) {
  const arr = Array.from(values).sort((a, b) => a - b);
  if (!arr.length) return 0;
  const idx = clamp(Math.floor((arr.length - 1) * p), 0, arr.length - 1);
  return arr[idx];
}

class SHSENode {
  constructor(id, pos, valueDim, latentDim, rng) {
    this.id = id;
    this.pos = pos;
    this.value = new Float32Array(valueDim);
    this.target = new Float32Array(valueDim);
    this.pred = new Float32Array(valueDim);
    this.known = false;
    this.masked = false;
    this.confidence = 0;
    this.uncertainty = 1;
    this.visits = 0;
    this.neighbors = [];
    this.embedding = new Float32Array(latentDim);
    this.latent = [0, 0, 0];

    for (let i = 0; i < latentDim; i++) {
      this.embedding[i] = rng.normal() * 0.03;
    }
  }
}

class SphericalHolographicSurface {
  constructor(options = {}) {
    this.nodeCount = options.nodeCount ?? 512;
    this.valueDim = options.valueDim ?? 1;
    this.latentDim = options.latentDim ?? 12;
    this.k = options.neighbors ?? 10;
    this.seed = options.seed ?? 777;
    this.rng = new RNG(this.seed);

    this.learningRate = options.learningRate ?? 0.035;
    this.embeddingRate = options.embeddingRate ?? 0.012;
    this.weightDecay = options.weightDecay ?? 0.0001;
    this.temperature = options.temperature ?? 0.65;
    this.maskRate = options.maskRate ?? 0.35;
    this.noise = options.noise ?? 0.015;
    this.uncertaintyMomentum = options.uncertaintyMomentum ?? 0.94;

    this.nodes = [];
    this.W = null;
    this.B = null;
    this.stats = {
      step: 0,
      loss: 0,
      maskedLoss: 0,
      coverage: 0,
      meanUncertainty: 1,
      knownRatio: 0,
    };

    this._init();
  }

  _init() {
    const pts = fibonacciSphere(this.nodeCount);
    this.nodes = pts.map((p, i) => new SHSENode(i, p, this.valueDim, this.latentDim, this.rng));
    const knn = buildKnn(pts, this.k);
    for (let i = 0; i < this.nodeCount; i++) this.nodes[i].neighbors = knn[i];

    // Relational attention weights: one vector per output dimension.
    // Input features per neighbor: [neighbor value dims, relative pos 3, dot, dist, neighbor uncertainty, bias]
    this.featureDim = this.valueDim + 3 + 1 + 1 + 1 + this.latentDim + 1;
    this.W = Array.from({ length: this.valueDim }, () => new Float32Array(this.featureDim));
    this.B = new Float32Array(this.valueDim);

    for (let d = 0; d < this.valueDim; d++) {
      for (let f = 0; f < this.featureDim; f++) this.W[d][f] = this.rng.normal() * 0.02;
      this.B[d] = 0;
    }

    this.updateLatentGeometry();
  }

  resetState() {
    for (const node of this.nodes) {
      node.value.fill(0);
      node.target.fill(0);
      node.pred.fill(0);
      node.known = false;
      node.masked = false;
      node.confidence = 0;
      node.uncertainty = 1;
      node.visits = 0;
    }
    this.stats.step = 0;
    this.updateLatentGeometry();
  }

  encodeVector(vector, options = {}) {
    /*
      Codifica un vector 1D sobre la superficie.
      - vector puede ser Array<number> o Array<Array<number>>.
      - Si vector.length < nodeCount, se distribuye por interpolación circular.
      - Si vector.length > nodeCount, se comprime por muestreo/interpolación.
    */
    const knownRatio = options.knownRatio ?? 1.0;
    const normalize = options.normalize ?? true;
    const arr = Array.from(vector);
    const isMulti = Array.isArray(arr[0]) || ArrayBuffer.isView(arr[0]);

    let flat = [];
    if (isMulti) {
      flat = arr.map(v => Array.from(v));
    } else {
      flat = arr.map(v => [Number(v)]);
    }

    const dims = Math.min(this.valueDim, flat[0]?.length ?? 1);
    const vals = [];
    for (let d = 0; d < dims; d++) {
      vals[d] = flat.map(v => Number(v[d] ?? 0));
      if (normalize) {
        const med = percentile(vals[d], 0.5);
        const q1 = percentile(vals[d], 0.25);
        const q3 = percentile(vals[d], 0.75);
        const scale = Math.max(1e-6, q3 - q1);
        vals[d] = vals[d].map(x => clamp((x - med) / scale, -4, 4));
      }
    }

    for (let i = 0; i < this.nodeCount; i++) {
      const t = i / Math.max(1, this.nodeCount - 1);
      const x = t * Math.max(1, flat.length - 1);
      const a = Math.floor(x);
      const b = Math.min(flat.length - 1, a + 1);
      const u = x - a;
      const node = this.nodes[i];
      for (let d = 0; d < this.valueDim; d++) {
        const va = vals[d]?.[a] ?? 0;
        const vb = vals[d]?.[b] ?? va;
        const y = va * (1 - u) + vb * u;
        node.value[d] = y;
        node.target[d] = y;
      }
      node.known = this.rng.next() < knownRatio;
      node.confidence = node.known ? 1 : 0;
      node.uncertainty = node.known ? 0.05 : 1.0;
      node.masked = false;
    }

    this.updateLatentGeometry();
    return this;
  }

  encodeSamples(samples, mapper) {
    /*
      Codificación flexible:
      mapper(sample, node, index) -> Array<number> | number | null
      Si devuelve null, el nodo queda desconocido.
    */
    for (let i = 0; i < this.nodeCount; i++) {
      const node = this.nodes[i];
      const sample = samples[i % samples.length];
      const out = mapper(sample, node, i);
      if (out == null) {
        node.known = false;
        node.confidence = 0;
        node.uncertainty = 1;
        node.value.fill(0);
        node.target.fill(0);
        continue;
      }
      const vals = Array.isArray(out) || ArrayBuffer.isView(out) ? out : [out];
      for (let d = 0; d < this.valueDim; d++) {
        const y = Number(vals[d] ?? 0);
        node.value[d] = y;
        node.target[d] = y;
      }
      node.known = true;
      node.confidence = 1;
      node.uncertainty = 0.05;
    }
    this.updateLatentGeometry();
    return this;
  }

  applyRandomMask(rate = this.maskRate) {
    for (const node of this.nodes) {
      node.masked = node.known && this.rng.next() < rate;
      if (node.masked) {
        node.confidence = Math.min(node.confidence, 0.1);
        node.uncertainty = Math.max(node.uncertainty, 0.85);
      }
    }
  }

  _neighborFeature(center, neighbor) {
    const f = new Float32Array(this.featureDim);
    let idx = 0;

    for (let d = 0; d < this.valueDim; d++) {
      f[idx++] = neighbor.masked ? neighbor.pred[d] : neighbor.value[d];
    }

    f[idx++] = neighbor.pos[0] - center.pos[0];
    f[idx++] = neighbor.pos[1] - center.pos[1];
    f[idx++] = neighbor.pos[2] - center.pos[2];

    const cos = clamp(dot3(center.pos, neighbor.pos), -1, 1);
    const ang = Math.acos(cos);
    f[idx++] = cos;
    f[idx++] = ang / Math.PI;
    f[idx++] = neighbor.uncertainty;

    for (let e = 0; e < this.latentDim; e++) {
      f[idx++] = neighbor.embedding[e];
    }

    f[idx++] = 1;
    return f;
  }

  _scoreFeature(feature, dim) {
    const w = this.W[dim];
    let s = this.B[dim];
    for (let i = 0; i < feature.length; i++) s += w[i] * feature[i];
    return s;
  }

  predictNode(nodeId) {
    const node = this.nodes[nodeId];
    const pred = new Float32Array(this.valueDim);
    const attentionByDim = [];

    for (let d = 0; d < this.valueDim; d++) {
      const logits = [];
      const features = [];

      for (const nb of node.neighbors) {
        const other = this.nodes[nb.id];
        const f = this._neighborFeature(node, other);
        features.push({ id: nb.id, f });

        // Atención relacional: compatibilidad + penalización por incertidumbre.
        const raw = this._scoreFeature(f, d);
        const locality = -nb.angularDistance / Math.max(1e-6, this.temperature);
        const reliability = -1.25 * other.uncertainty;
        logits.push(raw + locality + reliability);
      }

      const att = softmax(logits);
      let y = 0;
      for (let j = 0; j < features.length; j++) {
        const other = this.nodes[features[j].id];
        const sourceValue = other.masked ? other.pred[d] : other.value[d];
        y += att[j] * sourceValue;
      }
      pred[d] = y;
      attentionByDim[d] = { att, features };
    }

    return { pred, attentionByDim };
  }

  trainStep(options = {}) {
    const lr = options.learningRate ?? this.learningRate;
    const embLr = options.embeddingRate ?? this.embeddingRate;
    const batchSize = options.batchSize ?? Math.min(96, this.nodeCount);
    const trainMaskedOnly = options.trainMaskedOnly ?? true;

    let loss = 0;
    let maskedLoss = 0;
    let count = 0;
    let maskedCount = 0;

    for (let b = 0; b < batchSize; b++) {
      const node = this.selectTrainingNode(trainMaskedOnly);
      const { pred, attentionByDim } = this.predictNode(node.id);

      for (let d = 0; d < this.valueDim; d++) {
        node.pred[d] = pred[d];
        const target = node.target[d];
        const err = pred[d] - target;
        const e2 = err * err;
        loss += e2;
        count++;
        if (node.masked) {
          maskedLoss += e2;
          maskedCount++;
        }

        // Actualización local tipo delta sobre las características relacionales.
        // No es autograd completo, pero da una primitiva diferenciable aproximada.
        const pack = attentionByDim[d];
        for (let j = 0; j < pack.features.length; j++) {
          const { id, f } = pack.features[j];
          const att = pack.att[j];
          const other = this.nodes[id];
          const sourceValue = other.masked ? other.pred[d] : other.value[d];
          const gradScale = err * att * sourceValue;

          for (let q = 0; q < this.featureDim; q++) {
            this.W[d][q] -= lr * (gradScale * f[q] + this.weightDecay * this.W[d][q]);
          }
          this.B[d] -= lr * err * att;

          // Empujar embeddings de vecinos hacia valores útiles para explicar el nodo.
          for (let e = 0; e < this.latentDim; e++) {
            const sign = Math.sign(sourceValue || 1);
            other.embedding[e] -= embLr * err * att * sign * 0.01;
            other.embedding[e] = clamp(other.embedding[e], -2, 2);
          }
        }
      }

      const nodeErr = Math.sqrt(Math.max(0, Array.from(node.pred).reduce((s, v, d) => {
        const e = v - node.target[d];
        return s + e * e;
      }, 0) / this.valueDim));

      node.uncertainty = this.uncertaintyMomentum * node.uncertainty +
        (1 - this.uncertaintyMomentum) * clamp(nodeErr, 0, 1.5);
      node.confidence = clamp(1 - node.uncertainty, 0, 1);
      node.visits++;
    }

    // Imputar suavemente nodos enmascarados o desconocidos.
    this.impute({ alpha: 0.15, onlyUncertain: true });
    this.updateLatentGeometry();

    this.stats.step++;
    this.stats.loss = loss / Math.max(1, count);
    this.stats.maskedLoss = maskedLoss / Math.max(1, maskedCount);
    this._updateStats();
    return { ...this.stats };
  }

  fit(steps = 100, options = {}) {
    const history = [];
    if (options.remaskEachStep ?? true) this.applyRandomMask(options.maskRate ?? this.maskRate);
    for (let i = 0; i < steps; i++) {
      if ((options.remaskEachStep ?? true) && i > 0) this.applyRandomMask(options.maskRate ?? this.maskRate);
      history.push(this.trainStep(options));
    }
    return history;
  }

  selectTrainingNode(maskedOnly = true) {
    // Muestreo activo: incertidumbre alta + pocas visitas + preferencia por nodos enmascarados.
    let best = null;
    let bestScore = -Infinity;
    const tries = Math.min(48, this.nodeCount);

    for (let t = 0; t < tries; t++) {
      const node = this.nodes[Math.floor(this.rng.next() * this.nodeCount)];
      if (maskedOnly && !node.masked && node.known) continue;
      const novelty = 1 / Math.sqrt(1 + node.visits);
      const maskedBonus = node.masked ? 0.55 : 0;
      const unknownBonus = !node.known ? 0.35 : 0;
      const score = node.uncertainty + novelty + maskedBonus + unknownBonus + this.rng.range(0, 0.05);
      if (score > bestScore) {
        best = node;
        bestScore = score;
      }
    }

    if (best) return best;

    for (const node of this.nodes) {
      const score = node.uncertainty + 1 / Math.sqrt(1 + node.visits);
      if (score > bestScore) {
        best = node;
        bestScore = score;
      }
    }
    return best ?? this.nodes[0];
  }

  impute(options = {}) {
    const alpha = options.alpha ?? 0.25;
    const onlyUncertain = options.onlyUncertain ?? false;

    for (const node of this.nodes) {
      if (onlyUncertain && node.known && !node.masked && node.uncertainty < 0.25) continue;
      const { pred } = this.predictNode(node.id);
      for (let d = 0; d < this.valueDim; d++) {
        node.pred[d] = pred[d];
        if (!node.known || node.masked || node.uncertainty > 0.35) {
          node.value[d] = node.value[d] * (1 - alpha) + pred[d] * alpha;
        }
      }
      node.confidence = clamp(node.confidence + alpha * 0.05, 0, 1);
      node.uncertainty = clamp(1 - node.confidence, 0.02, 1);
    }
    this.updateLatentGeometry();
    return this;
  }

  updateLatentGeometry() {
    /*
      Reconstrucción holográfica visual/latente:
      cada nodo de superficie se proyecta al volumen.
      No se usa para aprender como campo físico; sirve como resumen estructural.
    */
    for (const node of this.nodes) {
      let energy = 0;
      for (let d = 0; d < this.valueDim; d++) energy += Math.abs(node.value[d]);
      energy /= Math.max(1, this.valueDim);

      let embEnergy = 0;
      for (let e = 0; e < this.latentDim; e++) embEnergy += node.embedding[e] * node.embedding[e];
      embEnergy = Math.sqrt(embEnergy / Math.max(1, this.latentDim));

      const depth = clamp(
        0.08 + 0.78 * sigmoid(energy + embEnergy - node.uncertainty),
        0.05,
        0.95
      );

      node.latent = [
        node.pos[0] * depth,
        node.pos[1] * depth,
        node.pos[2] * depth,
      ];
    }
  }

  decodeVector(length = this.nodeCount, dim = 0) {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      const t = i / Math.max(1, length - 1);
      const x = t * Math.max(1, this.nodeCount - 1);
      const a = Math.floor(x);
      const b = Math.min(this.nodeCount - 1, a + 1);
      const u = x - a;
      out[i] = this.nodes[a].value[dim] * (1 - u) + this.nodes[b].value[dim] * u;
    }
    return out;
  }

  sample(options = {}) {
    /*
      Generación:
      - toma nodos semilla de baja incertidumbre;
      - propaga a nodos inciertos;
      - añade temperatura como variación.
    */
    const steps = options.steps ?? 48;
    const temperature = options.temperature ?? 0.08;
    const preserveKnown = options.preserveKnown ?? false;

    const snapshot = this.nodes.map(n => Array.from(n.value));

    for (const node of this.nodes) {
      if (!preserveKnown || !node.known) {
        node.value.fill(0);
        node.confidence = node.known ? 0.3 : 0.05;
        node.uncertainty = 1 - node.confidence;
      }
    }

    for (let s = 0; s < steps; s++) {
      const node = this.selectTrainingNode(false);
      const { pred } = this.predictNode(node.id);
      for (let d = 0; d < this.valueDim; d++) {
        const noise = this.rng.normal() * temperature * node.uncertainty;
        node.value[d] = 0.7 * node.value[d] + 0.3 * (pred[d] + noise);
      }
      node.confidence = clamp(node.confidence + 0.04, 0, 1);
      node.uncertainty = clamp(1 - node.confidence, 0.03, 1);
      node.visits++;
    }

    this.updateLatentGeometry();
    const generated = this.decodeVector(options.length ?? this.nodeCount, options.dim ?? 0);

    if (options.restore ?? false) {
      for (let i = 0; i < this.nodes.length; i++) {
        for (let d = 0; d < this.valueDim; d++) this.nodes[i].value[d] = snapshot[i][d];
      }
      this.updateLatentGeometry();
    }

    return generated;
  }

  getPatch(nodeId, radius = 2) {
    const seen = new Set([nodeId]);
    let frontier = [nodeId];
    for (let r = 0; r < radius; r++) {
      const next = [];
      for (const id of frontier) {
        for (const nb of this.nodes[id].neighbors) {
          if (!seen.has(nb.id)) {
            seen.add(nb.id);
            next.push(nb.id);
          }
        }
      }
      frontier = next;
    }
    return Array.from(seen).map(id => this.nodes[id]);
  }

  exportTrainingTensors() {
    /*
      Salida pensada para alimentar redes externas.
      positions: [N, 3]
      values: [N, valueDim]
      mask: [N]
      uncertainty: [N]
      edges: [E, 2]
      edgeAttr: [E, 1]
      latent: [N, 3]
      embeddings: [N, latentDim]
    */
    const positions = [];
    const values = [];
    const mask = [];
    const uncertainty = [];
    const latent = [];
    const embeddings = [];
    const edges = [];
    const edgeAttr = [];

    for (const node of this.nodes) {
      positions.push([...node.pos]);
      values.push(Array.from(node.value));
      mask.push(node.known && !node.masked ? 1 : 0);
      uncertainty.push(node.uncertainty);
      latent.push([...node.latent]);
      embeddings.push(Array.from(node.embedding));
      for (const nb of node.neighbors) {
        edges.push([node.id, nb.id]);
        edgeAttr.push([nb.angularDistance]);
      }
    }

    return { positions, values, mask, uncertainty, edges, edgeAttr, latent, embeddings };
  }

  importTrainingTensors(tensors) {
    if (tensors.values) {
      for (let i = 0; i < Math.min(this.nodeCount, tensors.values.length); i++) {
        const node = this.nodes[i];
        for (let d = 0; d < this.valueDim; d++) node.value[d] = Number(tensors.values[i][d] ?? 0);
      }
    }
    if (tensors.mask) {
      for (let i = 0; i < Math.min(this.nodeCount, tensors.mask.length); i++) {
        this.nodes[i].known = Boolean(tensors.mask[i]);
        this.nodes[i].masked = !this.nodes[i].known;
      }
    }
    if (tensors.uncertainty) {
      for (let i = 0; i < Math.min(this.nodeCount, tensors.uncertainty.length); i++) {
        this.nodes[i].uncertainty = clamp(Number(tensors.uncertainty[i]), 0, 1);
        this.nodes[i].confidence = 1 - this.nodes[i].uncertainty;
      }
    }
    if (tensors.embeddings) {
      for (let i = 0; i < Math.min(this.nodeCount, tensors.embeddings.length); i++) {
        for (let e = 0; e < this.latentDim; e++) this.nodes[i].embedding[e] = Number(tensors.embeddings[i][e] ?? 0);
      }
    }
    this.updateLatentGeometry();
    return this;
  }

  _updateStats() {
    let known = 0;
    let covered = 0;
    let unc = 0;
    for (const node of this.nodes) {
      if (node.known) known++;
      if (node.confidence > 0.5) covered++;
      unc += node.uncertainty;
    }
    this.stats.knownRatio = known / this.nodeCount;
    this.stats.coverage = covered / this.nodeCount;
    this.stats.meanUncertainty = unc / this.nodeCount;
  }

  summary() {
    this._updateStats();
    return {
      nodeCount: this.nodeCount,
      valueDim: this.valueDim,
      latentDim: this.latentDim,
      neighbors: this.k,
      ...this.stats,
    };
  }
}

// Example usage:
// const shse = new SphericalHolographicSurface({ nodeCount: 768, valueDim: 1, latentDim: 16 });
// shse.encodeVector(myArray, { knownRatio: 0.8 });
// shse.fit(300, { maskRate: 0.4, batchSize: 128 });
// const tensors = shse.exportTrainingTensors();
// const generated = shse.sample({ steps: 200, length: myArray.length });

if (typeof module !== 'undefined') {
  module.exports = { SphericalHolographicSurface, RNG, fibonacciSphere };
}
