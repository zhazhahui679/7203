"""Configuration constants for the INFS4203/7203 data mining pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Container for the key data file locations.

    Attributes
    ----------
    train_path: Path
        Location of the training data CSV file.
    test_path: Path
        Location of the test data CSV file.
    report_path: Path
        Location where the final submission report will be written.
    """

    train_path: Path = Path("train.csv")
    test_path: Path = Path("test_data.csv")
    report_path: Path = Path("result_report.infs4203")


RANDOM_SEED: int = 42
CV_FOLDS: int = 5
TARGET_COLUMN: str = "Target"
NUMERIC_PREFIX: str = "Num_Col"
CATEGORICAL_PREFIX: str = "Nom_Col"


def ensure_output_directory(path: Path) -> None:
    """Ensure the parent directory for ``path`` exists.

    Parameters
    ----------
    path:
        The path whose parent directory needs to be created.
    """

    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
