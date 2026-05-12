"""Experiment 09: Ring sector decode — what does each phase position mean?

Key findings (2026-05-12):

1. MECHANISM: the UMAP ring primarily encodes ORDER FLOW + MOMENTUM
   r(cos(theta_raw), volume_imbalance) = -0.822  ← ring = order flow direction!
   r(sin(theta_raw), log_return)       = +0.680
   r(sin(theta_raw), vwap_dev)         = +0.621
   r(theta_raw,      vol_20)           = -0.068  ← vol is SECONDARY

   The ring is NOT a 'price cycle' indicator. It maps current market microstate:
   one side = heavy selling + below VWAP, other side = heavy buying + above VWAP.

2. SECTOR ANALYSIS (12 sectors, raw phase):
   Sector  15 deg: heavy SELLING zone (vol_imb=-0.115, mean_lr=-98bps, rv12=0.0063)
   Sector  45 deg: quiet zone (vol_imb=-0.128, low vol, low future rv, 44% of data)
   Sector 105 deg: mild buying zone (vol_imb=+0.145, slightly positive returns)
   Sector 135 deg: heavy BUYING zone (vol_imb=+0.156, mean_lr=+170bps, rv12=0.0062)

   BOTH extremes (15 and 135 deg) have HIGH future volatility.
   The quiet zone (45-105 deg) has LOW future volatility.
   This is consistent with |order flow| predicting volatility.

3. NON-LINEAR DIRECTION SIGNAL: ABSENT
   Non-linear models (RF, logistic+poly) on raw UMAP coords all fail (<baseline).
   The sector P_up asymmetry doesn't generalize to a reliable direction signal.

4. UMAP ADDS VALUE OVER RAW SMOOTH:
   Raw smoothed features: MAE=0.001731 (WORSE than vol_20 only!)
   UMAP sin/cos_24h:      MAE=0.001706 (BETTER than vol_20 only)
   The ring captures non-linear 4D interactions that raw smoothing misses.

Summary: ring = polar representation of (order flow, momentum) in 4D compact space.
UMAP finds the primary manifold of this space (the ring). Phase = current microstate.
The ring's predictive power for volatility comes from encoding |current activity|,
not from any cycle-based forecasting.
"""

# This file serves as documentation of the exp09 findings.
# The actual analysis was run interactively; see reports/ring_sector_decode/.

FINDINGS = {
    "mechanism": {
        "r_cos_theta_volume_imbalance": -0.822,
        "r_sin_theta_log_return": 0.680,
        "r_sin_theta_vwap_dev": 0.621,
        "r_theta_vol_20": -0.068,
        "interpretation": "Ring = polar encoding of (order flow direction, momentum). NOT a price cycle.",
    },
    "sector_analysis": {
        "15_deg": {"vol_imb": -0.115, "log_ret_bps": -98, "rv12_future": 0.00628, "P_up": 0.571, "n": 1418},
        "45_deg": {"vol_imb": -0.128, "log_ret_bps": -16, "rv12_future": 0.00425, "P_up": 0.519, "n": 8683},
        "75_deg": {"vol_imb": -0.007, "log_ret_bps": +17, "rv12_future": 0.00434, "P_up": 0.509, "n": 4065},
        "105_deg": {"vol_imb": +0.145, "log_ret_bps": +34, "rv12_future": 0.00428, "P_up": 0.510, "n": 5074},
        "135_deg": {"vol_imb": +0.156, "log_ret_bps": +170, "rv12_future": 0.00620, "P_up": 0.543, "n": 269},
    },
    "non_linear_direction": {
        "logistic_poly": "FAIL (<baseline 0.513)",
        "rf_depth5": "FAIL (<baseline 0.513)",
        "conclusion": "No non-linear direction signal found in raw ring coordinates",
    },
    "umap_value_add": {
        "vol_20_only_MAE": 0.001711,
        "raw_smooth_s24_MAE": 0.001731,
        "umap_sin_cos_24h_MAE": 0.001706,
        "conclusion": "UMAP captures non-linear 4D interactions not captured by raw smoothing",
    },
}
