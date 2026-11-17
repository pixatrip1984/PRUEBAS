import numpy as np
import pandas as pd

from eso.fingerprints import (
    CapitalProfileSegmenter,
    CascadeAnalyzer,
    compute_flow_toxicity_vpin,
    compute_microstructure_fingerprints,
    compute_order_splitting_fingerprint,
    compute_stealth_accumulation_index,
)


def _frame(seed=7, n=180):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.002, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    spread = rng.uniform(0.001, 0.004, n)
    volume = rng.lognormal(6.0, 0.35, n)
    n_trades = rng.integers(30, 90, n).astype(float)
    imbalance = rng.uniform(-0.35, 0.35, n)
    buy = volume * (1.0 + imbalance) / 2.0
    sell = volume - buy
    return pd.DataFrame({
        "open": close * np.exp(-ret),
        "high": close * (1.0 + spread),
        "low": close * (1.0 - spread),
        "close": close,
        "volume": volume,
        "n_trades": n_trades,
        "taker_buy_volume": buy,
        "taker_sell_volume": sell,
        "delta": buy - sell,
        "vwap": close * (1.0 + rng.normal(0, 0.0007, n)),
    })


def test_stealth_order_splitting_scores_exceed_noise():
    noise = _frame()
    stealth = noise.copy()
    rows = stealth.index[100:150]
    stealth.loc[rows, "volume"] *= 8.0
    stealth.loc[rows, "n_trades"] *= 14.0
    stealth.loc[rows, "taker_buy_volume"] = (
        stealth.loc[rows, "volume"] * 0.92
    )
    stealth.loc[rows, "taker_sell_volume"] = (
        stealth.loc[rows, "volume"] * 0.08
    )
    stealth.loc[rows, "delta"] = (
        stealth.loc[rows, "taker_buy_volume"]
        - stealth.loc[rows, "taker_sell_volume"]
    )
    close = stealth.loc[rows, "close"]
    stealth.loc[rows, "high"] = close * 1.00035
    stealth.loc[rows, "low"] = close * 0.99965

    s0 = compute_stealth_accumulation_index(noise, 20).iloc[110:145].mean()
    s1 = compute_stealth_accumulation_index(stealth, 20).iloc[110:145].mean()
    f0 = compute_order_splitting_fingerprint(noise, 20).iloc[110:145].mean()
    f1 = compute_order_splitting_fingerprint(stealth, 20).iloc[110:145].mean()
    assert s1 > 1.25 * max(s0, 1e-12)
    assert f1 > 2.0 * max(f0, 1e-12)


def test_vpin_bounds_and_finite_kyle_lambda():
    out = compute_flow_toxicity_vpin(_frame(), window=20)
    assert out["vpin"].between(0.0, 1.0).all()
    assert np.isfinite(out["kyle_lambda"]).all()
    assert (out["kyle_lambda"] >= 0.0).all()


def _shock_frame(clustered):
    n = 240
    ret = np.zeros(n)
    vol = np.full(n, 100.0)
    if clustered:
        positions = [60, 61, 62, 63, 120, 121, 122, 180, 181, 182, 183, 184]
    else:
        positions = [40, 70, 100, 130, 160, 190, 220]
    for i, pos in enumerate(positions):
        ret[pos] = (
            (1 if i % 2 == 0 else -1) * (0.05 + 0.004 * (i % 3))
        )
        vol[pos] = 1500.0 + 100.0 * i
    close = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "close": close,
        "high": close * 1.002,
        "low": close * 0.998,
        "volume": vol,
    })


def test_cascade_clustered_shocks_have_more_self_excitation():
    analyzer = CascadeAnalyzer(window=25, shock_z=1.5, max_gap=1)
    clustered = analyzer.analyze(_shock_frame(True))
    isolated = analyzer.analyze(_shock_frame(False))
    assert clustered["branching_ratio_eta"] > isolated["branching_ratio_eta"]
    assert clustered["mean_avalanche_size"] > isolated["mean_avalanche_size"]
    assert (
        clustered["mean_avalanche_duration"]
        >= isolated["mean_avalanche_duration"]
    )
    assert np.isfinite(
        clustered["multiscale_snowball_amplification"]
    ).all()
    assert clustered["rolling_cascade_risk"].between(0.0, 1.0).all()


def test_capital_profile_segmenter_distinguishes_regimes():
    df = _frame(n=270)
    retail = df.index[:90]
    institutional = df.index[90:180]
    aggressive = df.index[180:270]

    df.loc[retail, "volume"] = 120.0
    df.loc[retail, "n_trades"] = 140.0
    df.loc[retail, "vwap"] = df.loc[retail, "close"] * 1.004

    inst_close = float(df.loc[institutional[0], "close"]) * np.exp(
        np.cumsum(np.full(90, 0.00005))
    )
    df.loc[institutional, "close"] = inst_close
    df.loc[institutional, "volume"] = 8000.0
    df.loc[institutional, "n_trades"] = 35.0
    df.loc[institutional, "vwap"] = (
        df.loc[institutional, "close"] * 1.00005
    )

    momentum = np.where(np.arange(90) % 2 == 0, 0.018, 0.012)
    start = float(df.loc[179, "close"])
    df.loc[aggressive, "close"] = start * np.exp(np.cumsum(momentum))
    df.loc[aggressive, "volume"] = 12000.0
    df.loc[aggressive, "n_trades"] = 22.0
    df.loc[aggressive, "vwap"] = df.loc[aggressive, "close"] * 0.985

    prof = CapitalProfileSegmenter(window=30).transform(df)
    assert (
        prof.iloc[40:85]["capital_regime"].mode().iat[0]
        == "fragmented_retail"
    )
    assert (
        prof.iloc[125:175]["capital_regime"].mode().iat[0]
        == "patient_institutional"
    )
    assert (
        prof.iloc[215:265]["capital_regime"].mode().iat[0]
        == "aggressive_momentum"
    )
    assert prof["confidence"].between(0.0, 1.0).all()

    summary = CapitalProfileSegmenter(window=30).summarize(df)
    assert set(summary.index) == {
        "fragmented_retail",
        "patient_institutional",
        "aggressive_momentum",
    }


def test_minimal_and_zero_volume_are_finite_and_causal():
    minimal = pd.DataFrame({
        "open": [10.0, 10.0, 10.1, 10.1, 10.2],
        "high": [10.0, 10.1, 10.2, 10.2, 10.3],
        "low": [10.0, 9.9, 10.0, 10.0, 10.1],
        "close": [10.0, 10.0, 10.1, 10.1, 10.2],
        "volume": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    fp = compute_microstructure_fingerprints(minimal, window=3)
    assert np.isfinite(fp.to_numpy()).all()

    cascade = CascadeAnalyzer(window=3, shock_z=2.0).analyze(minimal)
    assert np.isfinite(cascade["rolling_cascade_risk"]).all()
    profiles = CapitalProfileSegmenter(window=3).transform(minimal)
    numeric = profiles.drop(columns=["capital_regime"])
    assert np.isfinite(numeric.to_numpy()).all()

    source = _frame(n=80)
    full = compute_microstructure_fingerprints(source, window=10)
    prefix = compute_microstructure_fingerprints(source.iloc[:50], window=10)
    pd.testing.assert_frame_equal(full.iloc[:50], prefix)

    analyzer = CascadeAnalyzer(window=10, shock_z=1.5)
    pd.testing.assert_frame_equal(
        analyzer.transform(source).iloc[:50],
        analyzer.transform(source.iloc[:50]),
    )
    segmenter = CapitalProfileSegmenter(window=10)
    pd.testing.assert_frame_equal(
        segmenter.transform(source).iloc[:50],
        segmenter.transform(source.iloc[:50]),
    )
