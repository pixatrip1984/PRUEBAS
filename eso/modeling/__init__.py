"""Model-evaluation helpers for ESO downstream agents."""

from .supervised import (
    DEFAULT_TRAIN_FRACTION,
    build_feature_frame,
    evaluate_supervised_task,
    write_model_run,
)
from .registry import compare_model_runs, list_model_runs, model_run_id, save_model_run

__all__ = [
    "DEFAULT_TRAIN_FRACTION",
    "build_feature_frame",
    "compare_model_runs",
    "evaluate_supervised_task",
    "list_model_runs",
    "model_run_id",
    "save_model_run",
    "write_model_run",
]
