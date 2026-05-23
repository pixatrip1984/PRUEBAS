# BTCUSDT_1h — stability h=12 w=1000

```
Rolling stability: window=1000, horizon=12, n_bars=19486

Per-feature rolling correlation with future log-return:
  sin_theta_6h       mean=-0.022   sign_flips=81
  cos_theta_6h       mean=+0.018   sign_flips=83
  sin_theta_24h      mean=-0.058   sign_flips=65
  cos_theta_24h      mean=+0.057   sign_flips=47
  sin_theta_72h      mean=-0.060   sign_flips=81
  cos_theta_72h      mean=+0.056   sign_flips=85

Top regime-indicator cross-correlations (|r|):
  sin_theta_6h__ring_radius                  r=+0.379
  cos_theta_6h__ring_radius                  r=-0.350
  sin_theta_24h__vol_20                      r=-0.301
  sin_theta_24h__lr_z20                      r=+0.285
  sin_theta_6h__vol_20                       r=-0.283
  sin_theta_24h__ring_radius                 r=+0.277
  cos_theta_6h__vol_20                       r=+0.267
  sin_theta_72h__lr_z20                      r=+0.238

Most unstable: cos_theta_72h (85 flips). Most stable: cos_theta_24h (47 flips).
```
