"""Walk-forward validation on STG and GNO."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.walk_forward import walk_forward_many


def main() -> int:
    paths = ["data/STGUSDT_4h.csv", "data/GNOUSDT_4h.csv"]
    df = walk_forward_many(paths, output_dir="reports/walkforward", n_folds=4)
    if df.empty:
        print("No results.")
        return 1
    print("\n=== WALK-FORWARD RANKING ===")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
