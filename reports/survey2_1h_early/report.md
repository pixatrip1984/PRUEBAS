# ESO Report — BTC_1h_early_v2

## 1. Resumen ejecutivo
- Mejor geometría candidata: **plane4d**
- Score: `0.012464291137480724`
- Error reconstrucción: `0.012464241137480726`
- Dimensión intrínseca estimada: `4.0`
- Resumen diagnóstico: intrinsic_dimension≈4.000; stationarity=stationary; lyapunov=0.0510; chaotic_hint=True; periodic_hint=False; dominant_frequency=0.3168

## 2. Datos de entrada
- Fuente: `<dataframe>`
- Shape: `[9999, 4]`
- Columnas usadas: `['log_return', 'vol_20', 'vwap_dev', 'volume_imbalance']`

## 3. Ranking topológico
| rank | manifold | score | recon_error | std | smoothness | utilization |
|---:|---|---:|---:|---:|---:|---:|
| 1 | plane4d | 0.0124643 | 0.0124642 | 0.00085751 | 0.175897 | 0.0806081 |
| 2 | klein_bottle | 0.0601242 | 0.0601241 | 0.00255081 | 0.467244 | 0.20081 |
| 3 | plane3d | 0.0616888 | 0.0616888 | 0.00203866 | 0.467244 | 0.20081 |
| 4 | torus2 | 0.181779 | 0.181779 | 0.00876638 | 1.2398 | 0.385995 |
| 5 | plane2d | 0.204496 | 0.204496 | 0.00657176 | 1.41868 | 0.652778 |
| 6 | cone | 0.214459 | 0.214459 | 0.00655568 | 1.42817 | 0.0972222 |
| 7 | sphere3 | 0.225055 | 0.225055 | 0.0134658 | 1.6666 | 0.306731 |
| 8 | s2_r | 0.229336 | 0.229336 | 0.0180548 | 1.66933 | 0.159916 |
| 9 | s1_r2 | 0.243117 | 0.243117 | 0.0160147 | 1.76828 | 0.0948095 |
| 10 | h2_r | 0.244438 | 0.244438 | 0.0124746 | 1.70534 | 0.446181 |
| 11 | s2_s1 | 0.279699 | 0.279699 | 0.014114 | 1.9318 | 0.110711 |
| 12 | sphere2 | 0.314432 | 0.314432 | 0.0118726 | 2.13966 | 0.340856 |
| 13 | cylinder | 0.329271 | 0.329271 | 0.0122254 | 2.30612 | 0.189236 |
| 14 | torus3 | 0.347394 | 0.347394 | 0.0178202 | 2.41116 | 0.176318 |
| 15 | t2_r | 0.361108 | 0.361108 | 0.0152581 | 2.41116 | 0.176318 |
| 16 | mobius | 0.401433 | 0.401433 | 0.0254086 | 2.52957 | 0.207755 |
| 17 | hyperbolic | 0.445443 | 0.445443 | 0.0184793 | 2.8158 | 0.916667 |
| 18 | circle | 0.594631 | 0.594631 | 0.0309603 | 3.56253 | 0.305556 |

## 4. Visualizaciones
### time_series_overview
![time_series_overview](figures/time_series_overview.png)

### recurrence_plot
![recurrence_plot](figures/recurrence_plot.png)

### fft_spectrum
![fft_spectrum](figures/fft_spectrum.png)

### autocorrelation
![autocorrelation](figures/autocorrelation.png)

### manifold_ranking
![manifold_ranking](figures/manifold_ranking.png)

### latent_projection_plane4d
![latent_projection_plane4d](figures/latent_projection_plane4d.png)

### latent_projection_klein_bottle
![latent_projection_klein_bottle](figures/latent_projection_klein_bottle.png)

### latent_projection_plane3d
![latent_projection_plane3d](figures/latent_projection_plane3d.png)

### latent_projection_torus2
![latent_projection_torus2](figures/latent_projection_torus2.png)

## 5. Interpretación
Best current candidate is plane4d with intrinsic dimension estimate 4.0.

Acción recomendada: Inspect report.html and run a second explore pass with more rows/masks if confidence is not high.