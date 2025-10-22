"""Utilities for writing the final result report required by the assignment."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ensure_output_directory


def write_result_report(
    predictions: Iterable[int],
    accuracy: float,
    f1: float,
    output_path: Path,
) -> None:
    """Write predictions and evaluation metrics to the submission file."""

    ensure_output_directory(output_path)
    rounded_accuracy = round(float(accuracy), 3)
    rounded_f1 = round(float(f1), 3)
    with output_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(f"{int(prediction)},\n")
        file.write(f"{rounded_accuracy:.3f},{rounded_f1:.3f},\n")


def validate_predictions(predictions: Iterable[int]) -> np.ndarray:
    """Validate that predictions are binary integers."""

    preds = np.asarray(list(predictions), dtype=int)
    if not np.isin(preds, [0, 1]).all():
        raise ValueError("Predictions must be binary labels (0 or 1).")
    return preds
