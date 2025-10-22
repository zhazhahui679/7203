# INFS4203/7203 Data Mining Pipeline

This repository provides a fully reproducible command line pipeline for the INFS4203/7203
assignment. It trains the classifiers covered in Weeks 2–8, evaluates them via stratified
cross-validation, and produces the submission file required by the assignment brief.

## Project structure

```
.
├── main.py                # Orchestrates training, evaluation, and report generation
├── src
│   ├── config.py          # Configuration constants and default paths
│   ├── data_loading.py    # CSV loading and feature inference utilities
│   ├── evaluation.py      # Cross-validation, model selection, and metric helpers
│   ├── model_factory.py   # Definitions of the supported classifiers and grids
│   ├── preprocessing.py   # Shared preprocessing pipelines
│   └── reporting.py       # Submission report writer
└── requirements.txt       # Python dependencies
```

Place `train.csv` and `test_data.csv` in the repository root before executing the pipeline.

## Environment

- Python 3.11+
- [pip-tools](https://pip.pypa.io/) or `pip`
- Operating system: Linux, macOS, or Windows

Install dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage

1. Ensure `train.csv` and `test_data.csv` are in the repository root.
2. Run the pipeline:

   ```bash
   python main.py --report s1234567.infs4203
   ```

   Replace `s1234567` with your student number. The script prints the selected model and
   cross-validation metrics, and writes the submission file to the specified path.

3. The generated result file already follows the required CSV-style format:

   - Rows 1–2,713: predicted labels for the test set
   - Row 2,714: cross-validated accuracy and F1 score rounded to three decimals

## Reproducibility

- Random seeds are fixed at module level in `src/config.py`.
- All preprocessing, model selection, and hyperparameter tuning steps are executed inside
  `main.py`, ensuring a single command can reproduce the reported results.

## Extending or debugging

- Modify parameter grids in `src/model_factory.py` to explore additional hyperparameter
  values.
- Adjust the number of cross-validation folds in `src/config.py` (`CV_FOLDS`).
- The evaluation logic stores detailed metrics in memory; add custom logging inside
  `ModelEvaluator.evaluate_models` if you need richer diagnostics.
