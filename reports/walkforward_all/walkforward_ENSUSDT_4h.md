# Walk-forward validation: ENSUSDT_4h

**Verdict: REJECTED — negative aggregate net return**

Folds traded: 3 / 3
Aggregate net return: -55.1%
Aggregate net Sharpe: -2.45
Aggregate trades: 23

## Per-fold breakdown

| Fold | Bars | Selected (gate/sig/enter) | Trades | NetSh | NetRet |
|-----:|------|----------------------------|-------:|------:|-------:|
| 1 | 355–710 | p30/sin_theta_24h/0.30 | 13 | -5.31 | -41.8% |
| 2 | 710–1065 | p85/cos_theta_6h/0.30 | 9 | -3.05 | -31.2% |
| 3 | 1065–1423 | p30/cos_theta_72h/0.40 | 1 | +1.14 | +12.2% |