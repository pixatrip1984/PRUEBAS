import pandas as pd

from eso.melate.schema import canonicalize_draws, validate_draws


def test_canonical_official_shape():
    df = pd.DataFrame([{
        "CONCURSO": 4265, "FECHA": "2026-09-13", "R1": 20, "R2": 21, "R3": 27,
        "R4": 28, "R5": 37, "R6": 42, "R7": 16, "BOLSA": 51600000,
    }])
    out = canonicalize_draws(df)
    assert tuple(out.loc[0, ["n1", "n2", "n3", "n4", "n5", "n6"]]) == (20, 21, 27, 28, 37, 42)
    assert int(out.loc[0, "additional"]) == 16
    assert validate_draws(out).invalid_rows == 0
