"""
src/models/baseline_multitarget.py

Stage 6 -- Baseline modeling across all 5 primary independent rheological targets.

Targets
-------
PV_cP, YP_lb100ft2, GS_10s_lb100ft2, GS_10min_lb100ft2, n_FlowIndex

Design
------
Split           : 70% train / 15% calibration / 15% final test
Stratification  : Formulation_Family
Random seed     : 42
Models
  Dummy_mean      : lower bound
  RidgeCV         : linear / regularized baseline (internal CV on alpha)
  RandomForest_CV : non-linear baseline (internal CV over a small grid)

Rules enforced
--------------
- Preprocessing fitted ONLY on training data.
- Hyperparameter tuning via internal CV on TRAIN ONLY.
- Calibration and final test sets never seen during training or tuning.
- No target leakage: other rheological outputs are NEVER used as predictors.
- Final test set used once per target for reporting.
- No XGBoost yet. No distribution shift yet.

Windows note
------------
joblib parallel backends can crash on Windows in some environments.
To keep this script reliable, all models and CV use n_jobs=1 and
joblib multiprocessing is disabled globally. Slower but stable.

Outputs
-------
results/metrics/baseline_multitarget_metrics.csv
results/tables/baseline_multitarget_metrics.csv
results/metrics/baseline_multitarget_metrics.json
results/figures/baseline_<target>_predictions.png
results/figures/baseline_<target>_residuals.png

Run from project root:
    python src/models/baseline_multitarget.py
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["JOBLIB_BACKEND"] = "sequential"

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
SHEET = "Full_Dataset"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
for d in (METRICS_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

STRATUM = "Formulation_Family"

TARGETS = [
    "PV_cP",
    "YP_lb100ft2",
    "GS_10s_lb100ft2",
    "GS_10min_lb100ft2",
    "n_FlowIndex",
]

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


def make_splits(df: pd.DataFrame, target: str):
    X = df[FEATURES].copy()
    y = df[target].copy()
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
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])

    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf_param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    }
    rf_grid = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )

    return {
        "Dummy_mean": DummyRegressor(strategy="mean"),
        "RidgeCV": ridge,
        "RandomForest_CV": rf_grid,
    }


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    bias = float(np.mean(y_pred - y_test))

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
        "negative_preds": int(np.sum(y_pred < 0)),
    }, y_pred


def plot_predictions(predictions, y_test, target, out_path: Path) -> None:
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
        ax.set_xlabel(f"Observed {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(predictions, y_test, target, out_path: Path) -> None:
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="k")
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"{name} residuals")
        ax.set_xlabel(f"Predicted {target}")
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
            f"Dataset not found at {DATA_PATH}. Place it at data/raw/ and re-run."
        )

    section("Load dataset")
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    print("Rows:", len(df), "Columns:", len(df.columns))

    required = FEATURES + TARGETS + [STRATUM]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_results = []
    best_model_choice = {}

    for target in TARGETS:
        section(f"TARGET: {target}")

        (X_train, X_cal, X_test,
         y_train, y_cal, y_test,
         strata_train, strata_temp) = make_splits(df, target)

        print(f"Train: {len(X_train)}  Calibration: {len(X_cal)}  Test: {len(X_test)}")

        models = build_models()
        predictions = {}
        rows = []

        for name, model in models.items():
            metrics, y_pred = evaluate(
                name, model, X_train, y_train, X_test, y_test
            )
            metrics["target"] = target
            rows.append(metrics)
            predictions[name] = y_pred

        target_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
        print(target_df.to_string(index=False))

        pred_path = FIGURES_DIR / f"baseline_{target}_predictions.png"
        plot_predictions(predictions, y_test, target, pred_path)
        res_path = FIGURES_DIR / f"baseline_{target}_residuals.png"
        plot_residuals(predictions, y_test, target, res_path)

        all_results.append(target_df)
        best_model_choice[target] = target_df.iloc[0]["model"]

    section("Combined baseline results across all targets")
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined[["target", "model", "n_train", "n_test",
                         "MAE", "RMSE", "R2", "bias",
                         "min_pred", "max_pred", "negative_preds"]]
    print(combined.to_string(index=False))

    section("Best model per target (by MAE)")
    print(pd.DataFrame(
        [{"target": k, "best_model": v} for k, v in best_model_choice.items()]
    ).to_string(index=False))

    section("Save results")
    csv_path = METRICS_DIR / "baseline_multitarget_metrics.csv"
    combined.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    tables_path = TABLES_DIR / "baseline_multitarget_metrics.csv"
    combined.to_csv(tables_path, index=False)
    print("Saved:", tables_path)

    json_path = METRICS_DIR / "baseline_multitarget_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined.to_dict(orient="records"), f, indent=2)
    print("Saved:", json_path)

    section("DONE")
    print("Multi-target baseline complete for:", ", ".join(TARGETS))
    print("Report the combined table and best-model summary.")


if __name__ == "__main__":
    main()