# Validation: GNOUSDT_4h

**Verdict: REGIME-DEPENDENT — parameter-stable but not temporally stable**

Bars: 3000    Feature-vector rows: 1423

## 1. Half-split (temporal robustness)

| Half | Bars | Trades | Gross Sharpe | Net Sharpe | Net Return |
|------|-----:|-------:|-------------:|-----------:|-----------:|
| first_half | 711 | 19 | +0.86 | +0.78 | +15.8% |
| second_half | 712 | 27 | -0.03 | -0.11 | -2.6% |

**Passed half-split:** NO

## 2. Parameter sweep (robustness)

Tested 48 combinations of (gate × signal × enter_threshold).
- Positive net return: 35 / 48
- Best net Sharpe: 1.73
- Robust (>=50% positive): YES

Top 5 by net Sharpe:

| Gate | Signal | Enter | Trades | GrossSh | NetSh | NetRet | Edge |
|-----:|--------|------:|-------:|--------:|------:|-------:|-----:|
| 0.70 | cos_theta_72h | 0.30 | 15 | +1.76 | +1.73 | +115.4% | +786bps |
| 0.85 | cos_theta_72h | 0.30 | 15 | +1.71 | +1.69 | +108.0% | +736bps |
| 0.85 | cos_theta_72h | 0.40 | 16 | +1.57 | +1.55 | +93.9% | +602bps |
| 0.70 | cos_theta_72h | 0.20 | 15 | +1.52 | +1.50 | +95.9% | +654bps |
| 0.85 | cos_theta_72h | 0.20 | 15 | +1.48 | +1.46 | +91.0% | +621bps |

## 3. Signal stability

| Feature | mean rolling corr | sign flips |
|---------|------------------:|-----------:|
| sin_theta_6h | -0.076 | 9 |
| cos_theta_6h | -0.005 | 5 |
| sin_theta_24h | -0.056 | 4 |
| cos_theta_24h | +0.008 | 9 |
| sin_theta_72h | +0.061 | 3 |
| cos_theta_72h | +0.083 | 6 |