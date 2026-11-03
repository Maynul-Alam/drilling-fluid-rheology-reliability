"""
src/models/baseline_pv.py

Stage 6 -- First leakage-free baseline for PV prediction.

Design
------
Target          : PV_cP
Features        : 13 primary formulation + test-temperature features
Split           : 70% train / 15% calibration / 15% final test
Stratification  : Formulation_Family
Random seed     : 42
Models          : DummyRegressor, RidgeCV, RandomForestRegressor
Metrics         : MAE (primary), RMSE, R2, bias, negative predictions

Rules enforced
--------------
- Preprocessing fitted ONLY on training data.
- Calibration set untouched here (reserved for later uncertainty work).
- Final test set used ONCE for reporting.
- No target leakage: other rheological outputs are NOT used as predictors.
- No XGBoost yet. No distribution shift yet.

Paths are resolved relative to the project root (two levels above this file),
not the current working directory, so you can run this from anywhere.

Run from project root (PowerShell):
    python src/models/baseline_pv.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
RANDOM_STATE = 42

# Resolve project root: this file is at <root>/src/models/baseline_pv.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
SHEET = "Full_Dataset"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

for d in (METRICS_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "PV_cP"
STRATUM = "Formulation_Family"

FEATURES = [
    "Bentonite_ppb",
    "Barite_ppb",
    "CMC_HV_ppb",
    "CMC_LV_ppb",
    "PAC_R_ppb",
    "PAC_L_ppb",
    "XCD_Polymer_ppb",
    "KCl_ppb",
    "NaCl_ppb",
    "KOH_ppb",
    "Lubricant_ppb",
    "Aging_Hours",
    "Test_Temperature_C",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def make_splits(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    strata = df[STRATUM].copy()

    X_train, X_temp, y_train, y_temp, strata_train, strata_temp = train_test_split(
        X, y, strata,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=strata_temp,
    )

    return X_train, X_cal, X_test, y_train, y_cal, y_test, strata_train, strata_temp


def build_models():
    return {
        "Dummy_mean": DummyRegressor(strategy="mean"),
        "RidgeCV": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    bias = float(np.mean(y_pred - y_test))
    negative_preds = int(np.sum(y_pred < 0))

    return {
        "model": name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
        "bias": bias,
        "min_pred": float(np.min(y_pred)),
        "max_pred": float(np.max(y_pred)),
        "negative_preds": negative_preds,
    }, y_pred


def plot_predictions(predictions, y_test, out_path: Path) -> None:
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolor="k")
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel(f"Observed {TARGET}")
        ax.set_ylabel(f"Predicted {TARGET}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(predictions, y_test, out_path: Path) -> None:
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"{name} residuals")
        ax.set_xlabel(f"Predicted {TARGET}")
        ax.set_ylabel("Observed - Predicted")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Dataset path:", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found at:\n  {DATA_PATH}\n\n"
            f"Place the Excel file at that exact location and re-run.\n"
            f"Expected folder: {DATA_PATH.parent}"
        )

    section("Load dataset")
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    print("Rows:", len(df), "Columns:", len(df.columns))

    required = FEATURES + [TARGET, STRATUM]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    section("Create 70 / 15 / 15 stratified split")
    (X_train, X_cal, X_test,
     y_train, y_cal, y_test,
     strata_train, strata_temp) = make_splits(df)

    print("Train:", len(X_train))
    print("Calibration:", len(X_cal))
    print("Test:", len(X_test))

    print("\nTrain family distribution:")
    print(strata_train.value_counts())
    print("\nCalibration family distribution:")
    print(strata_temp.loc[X_cal.index].value_counts())
    print("\nTest family distribution:")
    print(strata_temp.loc[X_test.index].value_counts())

    section("Fit and evaluate models on the final test set")
    models = build_models()
    rows = []
    predictions = {}

    for name, model in models.items():
        metrics, y_pred = evaluate(
            name, model, X_train, y_train, X_test, y_test
        )
        rows.append(metrics)
        predictions[name] = y_pred

    results_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    print("\nBaseline PV results (sorted by MAE):")
    print(results_df.to_string(index=False))

    section("Save results")
    metrics_path = METRICS_DIR / "baseline_pv_metrics.csv"
    results_df.to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)

    tables_path = TABLES_DIR / "baseline_pv_metrics.csv"
    results_df.to_csv(tables_path, index=False)
    print("Saved:", tables_path)

    json_path = METRICS_DIR / "baseline_pv_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_df.to_dict(orient="records"), f, indent=2)
    print("Saved:", json_path)

    section("Save plots")
    pred_path = FIGURES_DIR / "baseline_pv_predictions.png"
    plot_predictions(predictions, y_test, pred_path)
    print("Saved:", pred_path)

    res_path = FIGURES_DIR / "baseline_pv_residuals.png"
    plot_residuals(predictions, y_test, res_path)
    print("Saved:", res_path)

    section("DONE")
    print("Baseline pipeline for PV is complete and leakage-free.")
    print("Report the metrics table and both plots before proceeding.")


if __name__ == "__main__":
    main()