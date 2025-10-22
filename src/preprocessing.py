"""Preprocessing utilities for the INFS4203/7203 classifiers."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str], *, dense: bool = False
) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features."""

    numeric_pipeline: List[Tuple[str, Pipeline]] = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
        )
    ]
    categorical_pipeline: List[Tuple[str, Pipeline]] = [
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse=not dense),
                    ),
                ]
            ),
        )
    ]

    transformers = []
    transformers.extend([("numeric", numeric_pipeline[0][1], numeric_features)])
    transformers.extend([("categorical", categorical_pipeline[0][1], categorical_features)])

    return ColumnTransformer(transformers=transformers)


def to_dense(array) -> np.ndarray:
    """Ensure the provided array is converted to a dense ``numpy.ndarray``."""

    if hasattr(array, "toarray"):
        return array.toarray()
    return np.asarray(array)
