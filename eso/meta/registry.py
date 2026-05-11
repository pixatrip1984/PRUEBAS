"""Local experiment registry for ESO."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class ExperimentRegistry:
    """Append-only CSV registry for ESO experiments."""

    def __init__(self, path: str = "experiments/eso_registry.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, dataset_id: str, diagnosis: dict, evaluation: dict, signature: dict | None = None) -> dict:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "manifold": evaluation.get("manifold"),
            "reconstruction_error": evaluation.get("reconstruction_error"),
            "smoothness": evaluation.get("smoothness"),
            "latent_utilization": evaluation.get("latent_utilization"),
            "diagnosis_json": json.dumps(diagnosis, sort_keys=True),
            "evaluation_json": json.dumps(evaluation, sort_keys=True),
        }
        if signature:
            for key, value in signature.items():
                if key not in row:
                    row[key] = value
        df = pd.DataFrame([row])
        if self.path.exists():
            df.to_csv(self.path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.path, index=False)
        return row

    def load_experiments(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.path)
