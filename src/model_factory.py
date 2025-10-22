"""Model factory for classifiers used in the INFS4203/7203 assignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_preprocessor


@dataclass
class ModelSpec:
    """Container describing an estimator pipeline and its parameter grid."""

    name: str
    pipeline: Pipeline
    param_grid: Dict[str, Iterable]


def build_model_specs(numeric_features, categorical_features) -> list[ModelSpec]:
    """Construct the list of candidate models and their hyperparameter grids."""

    preprocessor = build_preprocessor(numeric_features, categorical_features, dense=False)
    dense_preprocessor = build_preprocessor(
        numeric_features, categorical_features, dense=True
    )

    models: list[ModelSpec] = []

    decision_tree = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(random_state=0, class_weight="balanced"),
            ),
        ]
    )
    decision_tree_grid = {
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    }
    models.append(ModelSpec("decision_tree", decision_tree, decision_tree_grid))

    random_forest = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=0,
                    class_weight="balanced",
                    n_estimators=200,
                ),
            ),
        ]
    )
    random_forest_grid = {
        "classifier__max_depth": [None, 15, 25],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
    }
    models.append(ModelSpec("random_forest", random_forest, random_forest_grid))

    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier()),
        ]
    )
    knn_grid = {
        "classifier__n_neighbors": [5, 15, 25],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    }
    models.append(ModelSpec("knn", knn_pipeline, knn_grid))

    nb_pipeline = Pipeline(
        steps=[
            ("preprocessor", dense_preprocessor),
            ("classifier", GaussianNB()),
        ]
    )
    nb_grid = {
        "classifier__var_smoothing": [1e-9, 1e-8, 1e-7],
    }
    models.append(ModelSpec("naive_bayes", nb_pipeline, nb_grid))

    return models


def build_voting_ensemble(best_estimators: Dict[str, ClassifierMixin]) -> Tuple[str, VotingClassifier]:
    """Build a simple majority voting ensemble from the top estimators."""

    ensemble = VotingClassifier(
        estimators=[(name, estimator) for name, estimator in best_estimators.items()],
        voting="soft",
    )
    return "voting_ensemble", ensemble
