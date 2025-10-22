"""Entrypoint for the INFS4203/7203 data mining assignment solution."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from src.config import DataPaths, TARGET_COLUMN
from src.data_loading import load_datasets, split_features
from src.evaluation import EvaluationResult, ModelEvaluator
from src.model_factory import build_model_specs
from src.reporting import validate_predictions, write_result_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=DataPaths.train_path,
        help="Path to train.csv",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DataPaths.test_path,
        help="Path to test_data.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DataPaths.report_path,
        help="Where to write the submission report",
    )
    return parser.parse_args()


def select_best_model(results: Dict[str, EvaluationResult]) -> EvaluationResult:
    """Select the model with the highest mean F1 score."""

    if not results:
        raise RuntimeError("No evaluation results available. Did you run evaluation?")
    return max(results.values(), key=lambda result: result.mean_f1)


def main() -> None:
    args = parse_args()
    train_df, test_df = load_datasets(args.train, args.test)
    numeric_features, categorical_features = split_features(train_df)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)

    evaluator = ModelEvaluator()
    model_specs = build_model_specs(numeric_features, categorical_features)
    evaluation_results = evaluator.evaluate_models(X_train, y_train, model_specs)

    ensemble_info = evaluator.evaluate_ensemble(X_train, y_train)
    if ensemble_info:
        name, ensemble_result = ensemble_info
        evaluation_results[name] = ensemble_result

    best_result = select_best_model(evaluation_results)
    best_estimator = best_result.best_estimator

    evaluator.fit_on_full_data(best_estimator, X_train, y_train)
    predictions = evaluator.predict(best_estimator, test_df)
    predictions = validate_predictions(predictions)

    # Evaluate the model using cross-validation predictions for accuracy and F1
    accuracy = best_result.mean_accuracy
    f1 = best_result.mean_f1

    write_result_report(predictions, accuracy, f1, args.report)

    print("Best model:", best_result.name)
    print("Cross-validated accuracy:", round(accuracy, 3))
    print("Cross-validated F1:", round(f1, 3))
    print("Result report written to:", args.report)


if __name__ == "__main__":
    main()
