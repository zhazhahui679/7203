"""Utilities for loading the INFS4203/7203 training and test datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import CATEGORICAL_PREFIX, NUMERIC_PREFIX, TARGET_COLUMN


class MissingColumnError(RuntimeError):
    """Raised when the expected dataset columns are not present."""


def load_datasets(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and test datasets from CSV files."""

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    expected_columns = [col for col in train_df.columns if col != TARGET_COLUMN]
    missing = [col for col in expected_columns if col not in test_df.columns]
    if missing:
        raise MissingColumnError(
            f"The test dataset is missing the following columns: {', '.join(missing)}"
        )

    return train_df, test_df


def split_features(train_df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Derive lists of numeric and categorical columns from the header."""

    numeric_columns = [
        column for column in train_df.columns if column.startswith(NUMERIC_PREFIX)
    ]
    categorical_columns = [
        column for column in train_df.columns if column.startswith(CATEGORICAL_PREFIX)
    ]

    if not numeric_columns or not categorical_columns:
        raise MissingColumnError(
            "Unable to infer numeric and categorical feature columns from the headers."
        )

    return numeric_columns, categorical_columns
