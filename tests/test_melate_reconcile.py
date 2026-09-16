import pandas as pd

from eso.melate.reconcile import reconcile_prize_tiers


def test_reconcile_requires_exact_combo():
    draws = pd.DataFrame([{
        "contest": 4265, "date": "2026-09-13", "n1": 20, "n2": 21, "n3": 27,
        "n4": 28, "n5": 37, "n6": 42, "additional": 16, "jackpot": 51600000,
    }])
    meta = pd.DataFrame([{"contest": 4265, "main": (20, 21, 27, 28, 37, 42), "additional": 16}])
    tiers = pd.DataFrame([{
        "contest": 4265, "tier": 1, "match_natural": 6, "match_additional": False,
        "winners": 0, "prize_individual": 0.0, "source": "melate_e", "source_url": "x",
        "source_hash": "h", "retrieved_at": "now", "source_quality": 1,
    }])
    accepted, quarantine = reconcile_prize_tiers(draws, meta, tiers)
    assert len(accepted) == 1 and quarantine.empty
    assert int(accepted.iloc[0]["source_quality"]) == 2
