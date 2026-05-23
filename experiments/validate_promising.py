"""Validate the promising alts from the multi-asset sweep with rigor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eso.lab.asset_validation import validate_asset


PROMISING_ASSETS = [
    "data/STGUSDT_4h.csv",
    "data/GNOUSDT_4h.csv",
]


def main() -> int:
    out_dir = "reports/validation"
    for path in PROMISING_ASSETS:
        asset = Path(path).stem
        print(f"\n=== Validating {asset} ===", flush=True)
        try:
            report = validate_asset(path, output_dir=out_dir)
        except Exception as exc:
            print(f"  Failed: {exc}")
            continue
        print(f"  Verdict: {report.verdict}")
        print(f"  Half-split passed: {report.half_split_passed}")
        print(f"  Parameter robust:  {report.sweep_robust}")
        print(f"  Best net Sharpe in sweep: {report.sweep_best_net_sharpe:.2f}")
        if "first_half" in report.half_split:
            h1 = report.half_split["first_half"]
            h2 = report.half_split["second_half"]
            print(f"  Half 1 net: {h1['net_return']*100:+.1f}%   "
                  f"Sharpe={h1['net_sharpe']:+.2f}")
            print(f"  Half 2 net: {h2['net_return']*100:+.1f}%   "
                  f"Sharpe={h2['net_sharpe']:+.2f}")
    print(f"\nReports written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
