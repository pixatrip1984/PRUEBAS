# Validation: STGUSDT_4h

**Verdict: REGIME-DEPENDENT — parameter-stable but not temporally stable**

Bars: 3000    Feature-vector rows: 1423

## 1. Half-split (temporal robustness)

| Half | Bars | Trades | Gross Sharpe | Net Sharpe | Net Return |
|------|-----:|-------:|-------------:|-----------:|-----------:|
| first_half | 711 | 7 | -0.72 | -0.74 | -18.1% |
| second_half | 712 | 15 | +1.84 | +1.81 | +90.4% |

**Passed half-split:** NO

## 2. Parameter sweep (robustness)

Tested 48 combinations of (gate × signal × enter_threshold).
- Positive net return: 33 / 48
- Best net Sharpe: 1.02
- Robust (>=50% positive): YES

Top 5 by net Sharpe:

| Gate | Signal | Enter | Trades | GrossSh | NetSh | NetRet | Edge |
|-----:|--------|------:|-------:|--------:|------:|-------:|-----:|
| 0.30 | cos_theta_24h | 0.40 | 19 | +1.05 | +1.02 | +91.9% | +498bps |
| 0.50 | cos_theta_24h | 0.20 | 27 | +1.01 | +0.98 | +87.5% | +338bps |
| 0.30 | cos_theta_24h | 0.20 | 31 | +0.94 | +0.90 | +82.7% | +281bps |
| 0.50 | cos_theta_24h | 0.40 | 19 | +0.86 | +0.84 | +68.9% | +375bps |
| 0.30 | cos_theta_24h | 0.30 | 23 | +0.79 | +0.77 | +64.7% | +294bps |

## 3. Signal stability

| Feature | mean rolling corr | sign flips |
|---------|------------------:|-----------:|
| sin_theta_6h | -0.022 | 8 |
| cos_theta_6h | +0.091 | 6 |
| sin_theta_24h | -0.120 | 3 |
| cos_theta_24h | +0.149 | 9 |
| sin_theta_72h | -0.177 | 0 |
| cos_theta_72h | +0.163 | 3 |