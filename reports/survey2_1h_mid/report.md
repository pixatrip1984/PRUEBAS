# ESO Report — BTC_1h_mid_v2

## 1. Resumen ejecutivo
- Mejor geometría candidata: **plane4d**
- Score: `0.013434770997874013`
- Error reconstrucción: `0.013434720997874012`
- Dimensión intrínseca estimada: `3.941986960388802`
- Resumen diagnóstico: intrinsic_dimension≈3.942; stationarity=stationary; lyapunov=0.0538; chaotic_hint=True; periodic_hint=False; dominant_frequency=0.0739

## 2. Datos de entrada
- Fuente: `<dataframe>`
- Shape: `[9999, 4]`
- Columnas usadas: `['log_return', 'vol_20', 'vwap_dev', 'volume_imbalance']`

## 3. Ranking topológico
| rank | manifold | score | recon_error | std | smoothness | utilization |
|---:|---|---:|---:|---:|---:|---:|
| 1 | plane4d | 0.0134348 | 0.0134347 | 0.00151716 | 0.174946 | 0.0684068 |
| 2 | plane3d | 0.0661911 | 0.066191 | 0.00319224 | 0.513486 | 0.206597 |
| 3 | klein_bottle | 0.0677008 | 0.0677008 | 0.00413134 | 0.513486 | 0.206597 |
| 4 | cone | 0.199645 | 0.199645 | 0.00354829 | 1.38119 | 0.10706 |
| 5 | torus2 | 0.20506 | 0.20506 | 0.00960206 | 1.49788 | 0.372685 |
| 6 | plane2d | 0.21346 | 0.21346 | 0.0086847 | 1.43744 | 0.75 |
| 7 | s1_r2 | 0.244101 | 0.244101 | 0.00989358 | 1.8389 | 0.0723072 |
| 8 | s2_r | 0.257246 | 0.257246 | 0.0125847 | 1.99594 | 0.127413 |
| 9 | h2_r | 0.276214 | 0.276214 | 0.0115661 | 1.9767 | 0.429977 |
| 10 | sphere3 | 0.290709 | 0.290708 | 0.0172167 | 2.10218 | 0.30393 |
| 11 | s2_s1 | 0.339232 | 0.339232 | 0.0166614 | 2.4219 | 0.107411 |
| 12 | cylinder | 0.361935 | 0.361935 | 0.0201277 | 2.56217 | 0.185185 |
| 13 | sphere2 | 0.393826 | 0.393826 | 0.0225775 | 2.72008 | 0.335069 |
| 14 | torus3 | 0.419352 | 0.419352 | 0.0157836 | 2.8771 | 0.173417 |
| 15 | t2_r | 0.420849 | 0.420849 | 0.0228323 | 2.8771 | 0.173417 |
| 16 | mobius | 0.453641 | 0.453641 | 0.012173 | 2.98708 | 0.211806 |
| 17 | hyperbolic | 0.556807 | 0.556807 | 0.0204923 | 3.44346 | 0.916667 |
| 18 | circle | 0.719429 | 0.719429 | 0.038205 | 4.24703 | 0.305556 |

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

### latent_projection_plane3d
![latent_projection_plane3d](figures/latent_projection_plane3d.png)

### latent_projection_klein_bottle
![latent_projection_klein_bottle](figures/latent_projection_klein_bottle.png)

### latent_projection_cone
![latent_projection_cone](figures/latent_projection_cone.png)

## 5. Interpretación
Best current candidate is plane4d with intrinsic dimension estimate 3.941986960388802.

Acción recomendada: Inspect report.html and run a second explore pass with more rows/masks if confidence is not high.