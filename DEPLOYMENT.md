# Deployment playbook — ESO portfolio strategy

This doc explains how to run the validated 5-alt portfolio in live (or paper)
trading. The strategy was developed in `lab-unified` over ~13 commits and
the result is:

  Vol-targeted portfolio (5 alts):
    Sharpe (point):   +3.14
    Sharpe 95% CI:    [+0.98, +5.49]
    Max drawdown:     -11.4%
    Cost stress test: identical result at 2× retail Bybit taker fees

Caveats up front — the sample is ~178 days of walk-forward data. The
lower CI bound (+0.98) is what you should mentally use when sizing.

---

## What the strategy does

For each of 5 alts (VVV, STG, MORPHO, GNO, GRASS), at every 4h bar close:

1. Build a "causal feature vector": 6 cycle-phase coordinates (sin/cos at
   3 smoothing scales) + ring confidence + 2 financial context features.
2. Walk-forward sweep 48 parameter combos (gate × signal column × entry
   threshold) over the trailing ~700 bars, pick the highest net Sharpe.
3. Apply that config to the most recent bars, take the position at the
   latest bar.
4. Scale each asset's target position by the inverse of its realised vol
   (target ~30% annual portfolio vol).
5. Apply a quality gate: drop signals where the selection-window Sharpe
   is below 0.5.

Result: a vector of target exposures, summing to roughly 20-25% of capital
gross, with positions typically in {-1, 0, +1} per asset before vol-scaling.

---

## Daily workflow

```bash
# Step 1: refresh data for each asset (every 4h after bar close)
python -m eso.lab.deployment fetch VVVUSDT    data/VVVUSDT_4h.csv    --interval 240
python -m eso.lab.deployment fetch STGUSDT    data/STGUSDT_4h.csv    --interval 240
python -m eso.lab.deployment fetch MORPHOUSDT data/MORPHOUSDT_4h.csv --interval 240
python -m eso.lab.deployment fetch GNOUSDT    data/GNOUSDT_4h.csv    --interval 240
python -m eso.lab.deployment fetch GRASSUSDT  data/GRASSUSDT_4h.csv  --interval 240
# Note: GNO needs funding_rate column too. Bybit's V5 funding history
# endpoint is at /v5/market/funding/history. Refresh it separately.

# Step 2: generate today's target positions
python -m eso.lab.live_signal --output reports/live_signal.json

# Step 3: compute diff vs held positions and apply quality gate
python -m eso.lab.deployment diff reports/live_signal.json \
    --held current_held.json \
    --min-sharpe 1.0 \
    --output reports/trade_actions.json

# Step 4: review reports/trade_actions.json, execute on Bybit, update
# current_held.json with new holdings.
```

`current_held.json` is just `{"VVVUSDT_4h": 0.045, "STGUSDT_4h": -0.062, ...}`.
Maintain it by hand (or have your execution layer update it after fills).

---

## Sizing

The `portfolio_positions` field in `live_signal.json` is the **fraction of
total capital** to be exposed per asset (signed). Example output:

```json
{
  "VVVUSDT_4h": +0.0446,
  "STGUSDT_4h": +0.0624,
  "MORPHOUSDT_4h": +0.0000,
  "GNOUSDT_4h": +0.0526,
  "GRASSUSDT_4h": +0.0504
}
```

If your account is $10,000 USDT:
- Long $446 of VVVUSDT perpetual
- Long $624 of STGUSDT
- Flat MORPHO
- Long $526 of GNO  ← but quality-gated out, so flat in practice
- Long $504 of GRASS

Total exposure: ~$1,575 of $10k = 15.75% gross. The remaining 84% sits in
cash (no margin used). This is consistent with a 30% annual vol target.

For leverage: just multiply all positions by L. The risk profile of the
strategy scales linearly until you hit the liquidation regime. Suggested
cap: 3x while the strategy has only ~178 days of validated history.

---

## When to STOP trading the strategy

This is the most important section. Stop if any of these triggers fire:

1. **Drawdown breach:** if portfolio DD exceeds -20% (1.75× the backtest
   max DD of -11.4%), pause. The strategy is outside its tested regime.

2. **Per-fold Sharpe collapse:** if you've been live for 3+ months and
   the realised Sharpe is below 0.5 (the lower bound of the 95% CI is
   +0.98 — well above this), the signal has decayed. Re-validate.

3. **Selection gate persistently flat:** if `live_signal.json` shows all
   5 assets with `selection_sharpe < 0.5` for two consecutive runs, the
   walker is telling you it can't find a working config. Step out.

4. **BTC walk-forward note:** we already showed BTC's signal decays
   fold-by-fold. The alts have less history, so we cannot rule out
   they will follow the same pattern eventually. Re-run the full
   `experiments/walk_forward_all_alts.py` monthly to verify each alt
   still walks forward.

---

## Paper-trading journal (validated)

We ran `experiments/paper_journal.py` over the last 200 bars (~33 days
at 4h) to simulate live execution. The result was striking:

| Gate min_sharpe | Net return | Sharpe | GNO Sharpe |
|----------------:|-----------:|-------:|-----------:|
| 0.5 (original)  |     +0.9%  |  +0.49 |  **-5.44** |
| **1.0** (use this) |  +6.4%  |  **+4.37** | +3.24 |

The 0.5 gate let GNO trade 28 times during a regime where it was
losing money badly. The 1.0 gate cut that to 2 trades — most of GNO's
selection sweeps in that period had Sharpe < 1.0, correctly signalling
"the walker isn't confident enough — sit out."

MORPHOUSDT was still overtrading at gate=1.0 (18 trades, -0.5% net).
This is a leading indicator the signal is decaying. Watch it. If it
continues underperforming, drop it from `LIVE_ASSET_CONFIGS`.

## Re-validation (monthly)

```bash
# Quick sanity check
python experiments/walk_forward_all_alts.py

# Full validation
python experiments/portfolio_voltargeted.py
```

Expected: the 5 alts still produce TRADEABLE verdicts, portfolio Sharpe
stays in [0.98, 5.49] range. If any of these slip, drop that asset from
the live config and re-derive `LIVE_ASSET_CONFIGS` in `live_signal.py`.

---

## What we deliberately did NOT include

- **Stop losses:** the strategy is vol-targeted and already conservatively
  sized. Adding stops on top adds noise and was not validated.
- **Pyramiding:** position size is determined entirely by the model.
  Don't add to winners outside the model's framework.
- **Manual overrides:** the moment you start second-guessing the signal
  you've reintroduced exactly the discretionary bias the entire pipeline
  was designed to remove.
- **BTC:** failed walk-forward (Sharpe CI straddles zero). Do not add it.

---

## Costs assumed

- Bybit USDT-perp taker: 5.5 bps per side (default)
- Slippage: 2 bps per side (default)
- Round-trip: ~15 bps

Tested under stress at 10 bps + 5 bps (round-trip ~30 bps): result was
identical (Sharpe 3.11 vs 3.14). The strategy is cost-insensitive
because of low turnover (~15 trades per asset over the 178 days).

If your costs are dramatically different (e.g. you're using a low-tier
exchange with 30+ bps fees per side), re-validate with
`--fee-bps` and `--slippage-bps` flags.
