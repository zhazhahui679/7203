"""Model evaluation utilities for cross-validation and reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

from .config import CV_FOLDS, RANDOM_SEED
from .model_factory import ModelSpec, build_voting_ensemble


@dataclass
class EvaluationResult:
    """Hold the evaluation metrics for a fitted estimator."""

    name: str
    mean_accuracy: float
    mean_f1: float
    std_accuracy: float
    std_f1: float
    best_estimator


class ModelEvaluator:
    """Perform cross-validation and hyperparameter tuning for model candidates."""

    def __init__(self) -> None:
        self.results: Dict[str, EvaluationResult] = {}

    def evaluate_models(
        self, X: pd.DataFrame, y: pd.Series, model_specs: Iterable[ModelSpec]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate each model spec using stratified cross-validation."""

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        for spec in model_specs:
            grid_search = GridSearchCV(
                estimator=spec.pipeline,
                param_grid=spec.param_grid,
                cv=cv,
                scoring={"accuracy": "accuracy", "f1": "f1"},
                refit="f1",
                n_jobs=-1,
            )
            grid_search.fit(X, y)
            best_index = grid_search.best_index_
            accuracy_scores = grid_search.cv_results_["mean_test_accuracy"]
            accuracy_stds = grid_search.cv_results_["std_test_accuracy"]
            f1_scores = grid_search.cv_results_["mean_test_f1"]
            f1_stds = grid_search.cv_results_["std_test_f1"]

            result = EvaluationResult(
                name=spec.name,
                mean_accuracy=float(accuracy_scores[best_index]),
                mean_f1=float(f1_scores[best_index]),
                std_accuracy=float(accuracy_stds[best_index]),
                std_f1=float(f1_stds[best_index]),
                best_estimator=grid_search.best_estimator_,
            )
            self.results[spec.name] = result

        return self.results

    def evaluate_ensemble(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[str, EvaluationResult] | None:
        """Construct and cross-validate a voting ensemble from the best estimators."""

        if not self.results:
            return None

        top_estimators = {
            name: result.best_estimator for name, result in self.results.items()
        }
        name, ensemble = build_voting_ensemble(top_estimators)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        accuracy_scores = cross_val_score(
            ensemble, X, y, scoring="accuracy", cv=cv, n_jobs=-1
        )
        f1_scores = cross_val_score(ensemble, X, y, scoring="f1", cv=cv, n_jobs=-1)
        return name, EvaluationResult(
            name=name,
            mean_accuracy=float(np.mean(accuracy_scores)),
            mean_f1=float(np.mean(f1_scores)),
            std_accuracy=float(np.std(accuracy_scores)),
            std_f1=float(np.std(f1_scores)),
            best_estimator=ensemble,
        )

    @staticmethod
    def fit_on_full_data(estimator, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the provided estimator on the entire training dataset."""

        estimator.fit(X, y)

    @staticmethod
    def predict(estimator, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted estimator."""

        return estimator.predict(X)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute accuracy and F1 score for predictions."""

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, f1
