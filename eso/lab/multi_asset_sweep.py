"""Multi-asset sweep: run the ESO causal pipeline + backtest on each asset.

For each asset:
  1. Build OHLCV-derived causal features (no microstructure required).
  2. Fit CausalCyclePhase on first 50% of data, transform the rest.
  3. Assemble the 9-feature vector.
  4. Run gated_phase_threshold backtest (best-performing BTC strategy).
  5. Collect metrics for cross-asset ranking.

The goal is to find any asset whose ring-phase signal has gross edge per
trade >= 20 bps (clearly above Bybit's 7.5 bps cost floor).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from eso.backtest import BacktestConfig, gated_phase_threshold_strategy, run_backtest
from eso.data.features import build_financial_features
from eso.signals.causal_cycle import CausalCyclePhase


# Features available for OHLCV-only assets (no microstructure)
OHLCV_FEATURES = ["log_return", "log_return_2", "hl_range", "body",
                  "vol_5", "vol_20", "lr_z20"]

# OHLCV + funding rate (perp positioning signal — genuinely orthogonal to price)
OHLCV_FUNDING_FEATURES = OHLCV_FEATURES + ["funding_rate", "funding_z20"]


@dataclass
class AssetResult:
    asset: str
    n_bars: int
    n_train: int
    n_test: int
    trades: int
    gross_return: float
    gross_sharpe: float
    net_return: float
    net_sharpe: float
    max_drawdown: float
    hit_rate: float
    cost_drag: float
    edge_bps_per_trade: float  # gross_return / n_trades, in bps
    verdict: str


def _smooth_phase(theta: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(np.cos(theta)).rolling(window, min_periods=max(1, window // 4)).mean()
    c = pd.Series(np.sin(theta)).rolling(window, min_periods=max(1, window // 4)).mean()
    theta_s = np.angle(s.to_numpy() + 1j * c.to_numpy())
    return np.sin(theta_s), np.cos(theta_s)


def build_alt_feature_vector(
    df: pd.DataFrame,
    train_frac: float = 0.5,
    feature_cols: list[str] | None = None,
    smooth_windows: tuple[int, ...] = (6, 24, 72),
    seed: int = 42,
    include_funding: bool = False,
) -> pd.DataFrame:
    """Build a 9-feature vector for an OHLCV-only asset.

    Returns DataFrame with index aligned to test slice (post-train).

    Args:
        include_funding: If True and 'funding_rate' is in df, append
                         funding_rate and its 20-bar z-score to the
                         features used by the UMAP fit.
    """
    feat = build_financial_features(df)
    # Optionally enrich with funding rate features (perp positioning signal)
    if include_funding and "funding_rate" in df.columns:
        fr = df["funding_rate"].astype(float)
        # Align to feat's index by position (feat dropped initial NaN rows)
        # We append directly assuming positional alignment, then dropna later.
        feat = feat.copy()
        feat["funding_rate"] = fr.iloc[-len(feat):].values
        rolling_mean = feat["funding_rate"].rolling(20, min_periods=5).mean()
        rolling_std = feat["funding_rate"].rolling(20, min_periods=5).std().replace(0, np.nan)
        feat["funding_z20"] = (feat["funding_rate"] - rolling_mean) / rolling_std

    default_set = OHLCV_FUNDING_FEATURES if include_funding else OHLCV_FEATURES
    cols = [c for c in (feature_cols or default_set) if c in feat.columns]
    if len(cols) < 3:
        raise ValueError(f"Need at least 3 features for UMAP; got {cols}")

    feat_clean = feat.dropna(subset=cols)
    orig_idx = feat_clean.index.to_numpy()
    n = len(feat_clean)
    train_n = int(n * train_frac)

    causal = CausalCyclePhase(train_size=train_n, seed=seed)
    feat_train = feat_clean[cols].iloc[:train_n]
    feat_test = feat_clean[cols].iloc[train_n:]
    if len(feat_test) < 100:
        raise ValueError(f"Test slice too small ({len(feat_test)} rows)")

    causal.fit(feat_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_test = causal.transform(feat_test)

    theta = np.unwrap(np.arctan2(emb_test[:, 1], emb_test[:, 0]))
    radius = np.sqrt(emb_test[:, 0] ** 2 + emb_test[:, 1] ** 2)

    out = pd.DataFrame(index=feat_test.index)
    for win in smooth_windows:
        s, c = _smooth_phase(theta, win)
        out[f"sin_theta_{win}h"] = s
        out[f"cos_theta_{win}h"] = c
    out["ring_radius"] = radius
    for col in ["vol_20", "lr_z20"]:
        if col in feat_clean.columns:
            out[col] = feat_clean[col].iloc[train_n:].values

    test_idx = orig_idx[train_n:]
    out["close"] = df["close"].to_numpy(dtype=float)[test_idx]
    if "timestamp" in df.columns:
        out.insert(0, "timestamp", df["timestamp"].iloc[test_idx].values)

    warmup = max(smooth_windows)
    return out.iloc[warmup:].copy()


def evaluate_asset(
    csv_path: str | Path,
    bars_per_year: int = 2190,  # 4h bars
    gate_percentile: float = 0.50,
    enter_threshold: float = 0.3,
    exit_threshold: float = 0.1,
    fee_bps: float = 5.5,
    slippage_bps: float = 2.0,
    seed: int = 42,
) -> AssetResult | None:
    """Run the full pipeline on one asset. Returns None if data too short."""
    df = pd.read_csv(csv_path)
    if len(df) < 600:
        return None
    asset = Path(csv_path).stem

    try:
        fv = build_alt_feature_vector(df, seed=seed)
    except Exception as exc:
        print(f"  {asset}: feature build failed — {exc}")
        return None

    if fv.empty or len(fv) < 200:
        return None

    pos = gated_phase_threshold_strategy(
        fv,
        signal_col="cos_theta_24h",
        gate_col="ring_radius",
        gate_percentile=gate_percentile,
        gate_window=min(500, len(fv) // 3),
        enter_threshold=enter_threshold,
        exit_threshold=exit_threshold,
        allow_short=True,
    )

    cfg = BacktestConfig(
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        bars_per_year=bars_per_year,
        allow_short=True,
    )
    bt = run_backtest(fv["close"], pos, cfg)

    n_trades = bt.metrics["n_trades"]
    edge_bps = (
        (bt.metrics["gross_total_return"] * 1e4) / n_trades
        if n_trades > 0 else 0.0
    )

    return AssetResult(
        asset=asset,
        n_bars=len(df),
        n_train=len(df) // 2,
        n_test=len(fv),
        trades=n_trades,
        gross_return=bt.metrics["gross_total_return"],
        gross_sharpe=bt.metrics["gross_sharpe"],
        net_return=bt.metrics["net_total_return"],
        net_sharpe=bt.metrics["net_sharpe"],
        max_drawdown=bt.metrics["max_drawdown"],
        hit_rate=bt.metrics["hit_rate"],
        cost_drag=bt.metrics["cost_drag"],
        edge_bps_per_trade=edge_bps,
        verdict=bt.metrics["verdict"],
    )


def sweep_assets(
    csv_paths: list[str | Path],
    output_dir: str | Path = "reports/multi_asset_sweep",
    **kwargs,
) -> pd.DataFrame:
    """Run the pipeline on each asset, return ranked results."""
    rows = []
    for path in csv_paths:
        asset = Path(path).stem
        print(f"-> {asset} ...", flush=True)
        res = evaluate_asset(path, **kwargs)
        if res is None:
            print(f"  {asset}: skipped (insufficient data)")
            continue
        print(f"  trades={res.trades:4d}  gross_Sharpe={res.gross_sharpe:+.2f}  "
              f"net={res.net_return*100:+.1f}%  edge={res.edge_bps_per_trade:+.1f}bps")
        rows.append(asdict(res))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("edge_bps_per_trade", ascending=False)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ranking.csv", index=False)

    # Markdown summary
    lines = [
        "# Multi-asset sweep — gated_phase_threshold strategy",
        "",
        f"Run config: gate=ring_radius@p50, signal=cos_theta_24h, "
        f"enter±0.3, exit±0.1, fee=5.5bps, slippage=2bps.",
        "",
        "Ranked by edge-per-trade (gross). >= 20 bps is the tradeable threshold.",
        "",
        "| Asset | Trades | Gross Sh | Net Sh | Net Return | Edge/trade | Verdict |",
        "|-------|-------:|---------:|-------:|-----------:|-----------:|---------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['asset']} | {int(r['trades'])} | {r['gross_sharpe']:+.2f} | "
            f"{r['net_sharpe']:+.2f} | {r['net_return']*100:+.1f}% | "
            f"{r['edge_bps_per_trade']:+.1f} bps | {r['verdict']} |"
        )
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    return df
