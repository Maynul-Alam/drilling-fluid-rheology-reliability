"""
src/validation/run_shift_experiments.py

Stage 7 -- Distribution-shift experiments.

Purpose
-------
Test H1: ML prediction error increases when the test distribution differs
from the training distribution. Compare each shift experiment to the
random-split reference using identical features, models, and metrics.

Experiments
-----------
  random                 : 70/15/15 stratified reference (same as Stage 6)
  temp_<T>               : hold out one temperature
  temp_150_stress        : extreme-temperature stress test (n=8)
  family_<name>          : hold out one formulation family
  temporal_2023_to_2024  : train on 2023, test on 2024
  combo_<family>_<temp>  : double extrapolation (family AND temp absent)

Models
------
  RidgeCV (internal alpha CV)
  RandomForest (n_estimators=500, fixed config, no grid search)

Rules
-----
  - Preprocessing fit on TRAIN only.
  - Calibration carved from pool, never used for training.
  - Test set used once per experiment.
  - No target leakage.
  - 150C experiment flagged as "stress_test".

Outputs
-------
results/metrics/shift_<experiment>.csv
results/metrics/shift_all.csv
results/metrics/shift_summary.csv   (delta vs random per target/model)
results/tables/shift_all.csv

Run examples
------------
  python src/validation/run_shift_experiments.py --experiment random
  python src/validation/run_shift_experiments.py --experiment random temp_90 family_KCl_Polymer
  python src/validation/run_shift_experiments.py --experiment all
"""

import argparse
import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
SHEET = "Full_Dataset"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"
for d in (METRICS_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CAL_FRACTION_OF_POOL = 0.15

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

EXPERIMENTS = {
    "random": {"type": "random"},
    "temp_25": {"type": "temp", "temp": 25},
    "temp_40": {"type": "temp", "temp": 40},
    "temp_50": {"type": "temp", "temp": 50},
    "temp_60": {"type": "temp", "temp": 60},
    "temp_75": {"type": "temp", "temp": 75},
    "temp_90": {"type": "temp", "temp": 90},
    "temp_120": {"type": "temp", "temp": 120},
    "temp_150_stress": {"type": "temp", "temp": 150, "stress_test": True},
    "family_Base_Bentonite": {"type": "family", "family": "Base_Bentonite"},
    "family_High_Density": {"type": "family", "family": "High_Density"},
    "family_KCl_Polymer": {"type": "family", "family": "KCl_Polymer"},
    "family_Low_Solid": {"type": "family", "family": "Low_Solid"},
    "family_Polymer_Enhanced": {"type": "family", "family": "Polymer_Enhanced"},
    "family_Salt_Tolerant": {"type": "family", "family": "Salt_Tolerant"},
    "temporal_2023_to_2024": {"type": "temporal"},
    "combo_Base_Bentonite_25": {"type": "combo", "family": "Base_Bentonite", "temp": 25},
    "combo_Base_Bentonite_50": {"type": "combo", "family": "Base_Bentonite", "temp": 50},
    "combo_Polymer_Enhanced_75": {"type": "combo", "family": "Polymer_Enhanced", "temp": 75},
    "combo_KCl_Polymer_90": {"type": "combo", "family": "KCl_Polymer", "temp": 90},
    "combo_High_Density_60": {"type": "combo", "family": "High_Density", "temp": 60},
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def split_pool(pool_idx: np.ndarray, strata: pd.Series) -> tuple:
    """Split a training pool into 85% train / 15% calibration, stratified."""
    strata_pool = strata.loc[pool_idx]
    train_idx, cal_idx = train_test_split(
        pool_idx,
        test_size=CAL_FRACTION_OF_POOL,
        random_state=RANDOM_STATE,
        stratify=strata_pool,
    )
    return np.array(train_idx), np.array(cal_idx)


def get_split_indices(df: pd.DataFrame, spec: dict) -> dict:
    kind = spec["type"]
    all_idx = df.index.to_numpy()

    if kind == "random":
        strata = df["Formulation_Family"]
        train_pool, test_idx = train_test_split(
            all_idx, test_size=0.15, random_state=RANDOM_STATE, stratify=strata,
        )
        train_idx, cal_idx = split_pool(np.array(train_pool), strata)
        return {"train": train_idx, "cal": cal_idx, "test": np.array(test_idx)}

    if kind == "temp":
        temp = spec["temp"]
        test_mask = df["Test_Temperature_C"] == temp
        test_idx = df.index[test_mask].to_numpy()
        pool_idx = df.index[~test_mask].to_numpy()
        train_idx, cal_idx = split_pool(pool_idx, df["Formulation_Family"])
        return {"train": train_idx, "cal": cal_idx, "test": test_idx}

    if kind == "family":
        family = spec["family"]
        test_mask = df["Formulation_Family"] == family
        test_idx = df.index[test_mask].to_numpy()
        pool_idx = df.index[~test_mask].to_numpy()
        train_idx, cal_idx = split_pool(pool_idx, df["Formulation_Family"])
        return {"train": train_idx, "cal": cal_idx, "test": test_idx}

    if kind == "temporal":
        dates = pd.to_datetime(df["Test_Date"], errors="coerce")
        test_mask = dates.dt.year == 2024
        pool_mask = dates.dt.year == 2023
        test_idx = df.index[test_mask].to_numpy()
        pool_idx = df.index[pool_mask].to_numpy()
        train_idx, cal_idx = split_pool(pool_idx, df["Formulation_Family"])
        return {"train": train_idx, "cal": cal_idx, "test": test_idx}

    if kind == "combo":
        family = spec["family"]
        temp = spec["temp"]
        test_mask = (df["Formulation_Family"] == family) & (df["Test_Temperature_C"] == temp)
        pool_mask = (df["Formulation_Family"] != family) & (df["Test_Temperature_C"] != temp)
        test_idx = df.index[test_mask].to_numpy()
        pool_idx = df.index[pool_mask].to_numpy()
        train_idx, cal_idx = split_pool(pool_idx, df["Formulation_Family"])
        return {"train": train_idx, "cal": cal_idx, "test": test_idx}

    raise ValueError(f"Unknown split type: {kind}")


def build_models() -> dict:
    return {
        "RidgeCV": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "min_pred": float(np.min(y_pred)),
        "max_pred": float(np.max(y_pred)),
        "negative_preds": int(np.sum(y_pred < 0)),
    }


def run_experiment(df: pd.DataFrame, exp_name: str, spec: dict) -> list:
    splits = get_split_indices(df, spec)
    train_idx = splits["train"]
    cal_idx = splits["cal"]
    test_idx = splits["test"]

    n_test = len(test_idx)
    stress = bool(spec.get("stress_test", False))

    section(f"Experiment: {exp_name}  (train={len(train_idx)}, cal={len(cal_idx)}, test={n_test}, stress={stress})")

    test_families = df.loc[test_idx, "Formulation_Family"].value_counts().to_dict()
    test_temps = df.loc[test_idx, "Test_Temperature_C"].value_counts().to_dict()
    print("  Test family mix :", test_families)
    print("  Test temp mix   :", test_temps)

    rows = []
    models = build_models()

    for target in TARGETS:
        X_train = df.loc[train_idx, FEATURES]
        y_train = df.loc[train_idx, target]
        X_test = df.loc[test_idx, FEATURES]
        y_test = df.loc[test_idx, target]

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            m = compute_metrics(y_test, y_pred)
            m.update({
                "experiment": exp_name,
                "target": target,
                "model": model_name,
                "n_train": int(len(train_idx)),
                "n_cal": int(len(cal_idx)),
                "n_test": int(n_test),
                "stress_test": stress,
            })
            rows.append(m)

    out_df = pd.DataFrame(rows)
    out_df = out_df[["experiment", "target", "model", "n_train", "n_cal", "n_test",
                     "MAE", "RMSE", "R2", "bias", "min_pred", "max_pred",
                     "negative_preds", "stress_test"]]
    out_df = out_df.sort_values(["target", "model"]).reset_index(drop=True)
    print(out_df.to_string(index=False))

    csv_path = METRICS_DIR / f"shift_{exp_name}.csv"
    out_df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    return rows


def build_summary(all_results: list) -> pd.DataFrame:
    df = pd.DataFrame(all_results)
    random_df = df[df["experiment"] == "random"]
    if len(random_df) == 0:
        print("WARNING: 'random' experiment was not run; cannot compute shift deltas.")
        return pd.DataFrame()

    summary_rows = []
    for _, row in df.iterrows():
        base = random_df[
            (random_df["target"] == row["target"]) & (random_df["model"] == row["model"])
        ]
        if len(base) == 0:
            continue
        base = base.iloc[0]
        mae_ratio = row["MAE"] / base["MAE"] if base["MAE"] > 0 else np.nan
        summary_rows.append({
            "experiment": row["experiment"],
            "target": row["target"],
            "model": row["model"],
            "n_test": row["n_test"],
            "MAE": row["MAE"],
            "MAE_random": base["MAE"],
            "MAE_delta": row["MAE"] - base["MAE"],
            "MAE_ratio": mae_ratio,
            "R2": row["R2"],
            "R2_random": base["R2"],
            "R2_delta": row["R2"] - base["R2"],
            "negative_preds": row["negative_preds"],
            "stress_test": row["stress_test"],
        })
    return pd.DataFrame(summary_rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        nargs="+",
        required=True,
        help='One or more experiment names, or "all".',
    )
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    df = df.reset_index(drop=True)

    required = FEATURES + TARGETS + ["Formulation_Family", "Test_Temperature_C", "Test_Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.experiment == ["all"]:
        experiment_names = list(EXPERIMENTS.keys())
    else:
        experiment_names = args.experiment

    unknown = [e for e in experiment_names if e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Choose from {list(EXPERIMENTS.keys())}")

    print("Running experiments:", experiment_names)
    print("Dataset size:", len(df))

    all_results = []
    for name in experiment_names:
        rows = run_experiment(df, name, EXPERIMENTS[name])
        all_results.extend(rows)

    section("Save combined metrics")
    combined = pd.DataFrame(all_results)
    all_csv = METRICS_DIR / "shift_all.csv"
    combined.to_csv(all_csv, index=False)
    tables_csv = TABLES_DIR / "shift_all.csv"
    combined.to_csv(tables_csv, index=False)
    print("Saved:", all_csv)
    print("Saved:", tables_csv)

    section("Shift summary (delta vs random)")
    summary = build_summary(all_results)
    if len(summary) > 0:
        summary_csv = METRICS_DIR / "shift_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print("Saved:", summary_csv)
        # Print a compact view
        compact = summary[[
            "experiment", "target", "model", "n_test",
            "MAE", "MAE_random", "MAE_ratio", "R2", "R2_random", "negative_preds",
        ]]
        print("\nCompact summary:")
        print(compact.to_string(index=False))

    section("DONE")
    print("Stage 7 experiments complete.")


if __name__ == "__main__":
    main()