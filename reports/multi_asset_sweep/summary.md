# Multi-asset sweep — gated_phase_threshold strategy

Run config: gate=ring_radius@p50, signal=cos_theta_24h, enter±0.3, exit±0.1, fee=5.5bps, slippage=2bps.

Ranked by edge-per-trade (gross). >= 20 bps is the tradeable threshold.

| Asset | Trades | Gross Sh | Net Sh | Net Return | Edge/trade | Verdict |
|-------|-------:|---------:|-------:|-----------:|-----------:|---------|
| VVVUSDT_4h | 3 | +2.47 | +2.47 | +775.0% | +25898.2 bps | PROMISING — net Sharpe >= 1.0 after costs |
| STGUSDT_4h | 21 | +0.75 | +0.72 | +58.2% | +289.0 bps | MARGINAL — signal exists but costs eat most edge |
| GNOUSDT_4h | 45 | +0.34 | +0.26 | +12.2% | +35.7 bps | WEAK — barely positive, not tradeable as-is |
| MORPHOUSDT_4h | 7 | -0.12 | -0.13 | -7.2% | -95.4 bps | NEGATIVE — signal does not survive costs |
| GRASSUSDT_4h | 19 | -0.50 | -0.52 | -36.8% | -189.1 bps | NEGATIVE — signal does not survive costs |
| SIGNUSDT_4h | 16 | -1.64 | -1.66 | -60.0% | -372.2 bps | NEGATIVE — signal does not survive costs |
| ENSUSDT_4h | 11 | -2.31 | -2.32 | -72.0% | -652.5 bps | NEGATIVE — signal does not survive costs |
| POPCATUSDT_4h | 5 | -1.51 | -1.51 | -66.0% | -1318.0 bps | NEGATIVE — signal does not survive costs |
| DRIFTUSDT_4h | 7 | -3.89 | -3.90 | -95.7% | -1366.8 bps | NEGATIVE — signal does not survive costs |
| NILUSDT_4h | 3 | -1.59 | -1.59 | -71.7% | -2387.4 bps | NEGATIVE — signal does not survive costs |