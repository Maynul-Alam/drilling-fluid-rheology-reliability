"""
src/evaluation/physics_consistency.py

Stage 10 -- Physics / rheology consistency analysis.

Purpose
-------
Check whether the models' predictions obey basic rheological/physical
constraints, and whether conformal intervals respect physical bounds.

Four checks
-----------
1. Non-negativity
   Count negative predictions for non-negative rheological targets.

2. AV consistency invariant
   AV_cP should approximately equal PV_cP + YP_lb100ft2 / 2 (Stage 6 audit
   found this on the raw data). Compare:
     - composed AV from PV and YP predictions:  ÂV = P̂V + ŶP / 2
     - directly-trained AV model
   on the same test points.

3. Temperature monotonicity
   For each test point, perturb Test_Temperature_C to each grid value
   and check the sign of pred(T=150) - pred(T=25).
   Rheology generally expects decreasing PV/YP/GS with increasing T,
   so a positive slope is "wrong sign".

4. Conformal interval physical validity
   Basic split conformal at 90% nominal (RidgeCV).
   Report fraction of test points whose interval lower bound is negative
   (physically impossible for PV, YP, GS targets).

Outputs
-------
  results/metrics/physics_target_summary.csv
  results/metrics/physics_av_summary.csv
  results/tables/physics_target_summary.csv
  results/tables/physics_av_summary.csv

Run
---
  python src/evaluation/physics_consistency.py --experiment all
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
from sklearn.metrics import mean_absolute_error


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
NOMINAL_ALPHA = 0.10

# Targets where a negative prediction is physically impossible
NON_NEGATIVE_TARGETS = [
    "PV_cP",
    "YP_lb100ft2",
    "GS_10s_lb100ft2",
    "GS_10min_lb100ft2",
]

# Targets where increasing T should generally decrease the prediction
TEMP_MONO_TARGETS = [
    "PV_cP",
    "YP_lb100ft2",
    "GS_10s_lb100ft2",
    "GS_10min_lb100ft2",
]

TEMPERATURE_GRID = [25, 40, 50, 60, 75, 90, 120, 150]

# All targets we report on (n included for negatives, excluded from monotonicity)
REPORT_TARGETS = [
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
def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def get_split_indices(df, spec):
    kind = spec["type"]
    all_idx = df.index.to_numpy()

    def _split_pool(pool_idx):
        strata = df.loc[pool_idx, "Formulation_Family"]
        tr, ca = train_test_split(
            pool_idx,
            test_size=CAL_FRACTION_OF_POOL,
            random_state=RANDOM_STATE,
            stratify=strata,
        )
        return np.array(tr), np.array(ca)

    if kind == "random":
        train_pool, test_idx = train_test_split(
            all_idx, test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=df["Formulation_Family"],
        )
        tr, ca = _split_pool(np.array(train_pool))
        return tr, ca, np.array(test_idx)

    if kind == "temp":
        m = df["Test_Temperature_C"] == spec["temp"]
        test_idx = df.index[m].to_numpy()
        pool = df.index[~m].to_numpy()
        tr, ca = _split_pool(pool)
        return tr, ca, test_idx

    if kind == "family":
        m = df["Formulation_Family"] == spec["family"]
        test_idx = df.index[m].to_numpy()
        pool = df.index[~m].to_numpy()
        tr, ca = _split_pool(pool)
        return tr, ca, test_idx

    if kind == "temporal":
        dates = pd.to_datetime(df["Test_Date"], errors="coerce")
        test_idx = df.index[dates.dt.year == 2024].to_numpy()
        pool = df.index[dates.dt.year == 2023].to_numpy()
        tr, ca = _split_pool(pool)
        return tr, ca, test_idx

    if kind == "combo":
        fam = spec["family"]; temp = spec["temp"]
        m_test = (df["Formulation_Family"] == fam) & (df["Test_Temperature_C"] == temp)
        m_pool = (df["Formulation_Family"] != fam) & (df["Test_Temperature_C"] != temp)
        test_idx = df.index[m_test].to_numpy()
        pool = df.index[m_pool].to_numpy()
        tr, ca = _split_pool(pool)
        return tr, ca, test_idx

    raise ValueError(f"Unknown split type: {kind}")


def build_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])


def build_rf():
    return RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=1,
    )


def conformal_q(y_cal, pred_cal, alpha):
    n = len(y_cal)
    resid = np.abs(y_cal - pred_cal)
    level = min(1.0, (n + 1) * (1 - alpha) / n)
    return float(np.quantile(resid, level, method="higher"))


def predict_at_grid(model, X_test, temperature_grid):
    preds = []
    for T in temperature_grid:
        X_mod = X_test.copy()
        X_mod["Test_Temperature_C"] = T
        preds.append(model.predict(X_mod))
    return np.stack(preds, axis=1)


# ----------------------------------------------------------------------
# Main experiment runner
# ----------------------------------------------------------------------
def run_experiment(df, exp_name, spec):
    train_idx, cal_idx, test_idx = get_split_indices(df, spec)
    stress = bool(spec.get("stress_test", False))
    n_test = len(test_idx)

    section(f"Experiment: {exp_name}  train={len(train_idx)} cal={len(cal_idx)} test={n_test} stress={stress}")

    X_train = df.loc[train_idx, FEATURES]
    X_cal = df.loc[cal_idx, FEATURES]
    X_test = df.loc[test_idx, FEATURES]

    target_rows = []
    av_rows = []

    model_builders = [
        ("RidgeCV", build_ridge),
        ("RandomForest", build_rf),
    ]

    for model_name, build_fn in model_builders:
        # Fit and evaluate each target once; cache predictions for reuse.
        preds = {}
        for target in REPORT_TARGETS:
            y_train = df.loc[train_idx, target].values
            y_cal = df.loc[cal_idx, target].values
            y_test = df.loc[test_idx, target].values

            model = build_fn().fit(X_train, y_train)
            pred_test = model.predict(X_test)
            pred_cal = model.predict(X_cal)
            preds[target] = {
                "model": model,
                "pred_test": pred_test,
                "pred_cal": pred_cal,
                "y_test": y_test,
                "y_cal": y_cal,
            }

            # Check 1: negatives
            neg_count = int(np.sum(pred_test < 0)) if target in NON_NEGATIVE_TARGETS else 0
            neg_fraction = neg_count / n_test if target in NON_NEGATIVE_TARGETS else 0.0

            # Check 4: conformal physical validity
            if model_name == "RidgeCV":
                q = conformal_q(y_cal, pred_cal, NOMINAL_ALPHA)
                lb = pred_test - q
                ub = pred_test + q
                lb_neg_frac = float(np.mean(lb < 0)) if target in NON_NEGATIVE_TARGETS else float("nan")
                coverage = float(np.mean((y_test >= lb) & (y_test <= ub)))
                width = float(2 * q)
            else:
                q = float("nan")
                lb_neg_frac = float("nan")
                coverage = float("nan")
                width = float("nan")

            # Check 3: temperature monotonicity
            if target in TEMP_MONO_TARGETS:
                grid_preds = predict_at_grid(model, X_test, TEMPERATURE_GRID)
                slope = grid_preds[:, -1] - grid_preds[:, 0]
                wrong_sign = float(np.mean(slope > 0))
                median_slope = float(np.median(slope))
            else:
                wrong_sign = float("nan")
                median_slope = float("nan")

            target_rows.append({
                "experiment": exp_name,
                "model": model_name,
                "target": target,
                "n_test": n_test,
                "stress": stress,
                "neg_count": neg_count,
                "neg_fraction": float(neg_fraction),
                "temp_wrong_sign_fraction": wrong_sign,
                "temp_median_slope_150_minus_25": median_slope,
                "conformal_q": q,
                "conformal_mean_width": width,
                "conformal_coverage": coverage,
                "conformal_lb_negative_fraction": lb_neg_frac,
            })

        # Check 2: AV consistency
        y_av_train = df.loc[train_idx, "AV_cP"].values
        y_av_test = df.loc[test_idx, "AV_cP"].values

        pred_pv = preds["PV_cP"]["pred_test"]
        pred_yp = preds["YP_lb100ft2"]["pred_test"]
        av_composed = pred_pv + pred_yp / 2.0

        m_av = build_fn().fit(X_train, y_av_train)
        av_direct = m_av.predict(X_test)

        mae_composed = float(mean_absolute_error(y_av_test, av_composed))
        mae_direct = float(mean_absolute_error(y_av_test, av_direct))
        resid = y_av_test - av_composed

        av_rows.append({
            "experiment": exp_name,
            "model": model_name,
            "n_test": n_test,
            "stress": stress,
            "av_composed_mae": mae_composed,
            "av_direct_mae": mae_direct,
            "av_mae_ratio_composed_over_direct": mae_composed / mae_direct if mae_direct > 0 else float("nan"),
            "av_composed_mean_residual": float(np.mean(resid)),
            "av_composed_median_abs_residual": float(np.median(np.abs(resid))),
            "av_composed_max_abs_residual": float(np.max(np.abs(resid))),
            "av_composed_negative_count": int(np.sum(av_composed < 0)),
        })

    target_df = pd.DataFrame(target_rows)
    av_df = pd.DataFrame(av_rows)

    print("--- target physics summary ---")
    print(target_df[[
        "target", "model", "neg_count", "neg_fraction",
        "temp_wrong_sign_fraction",
        "conformal_lb_negative_fraction", "conformal_coverage",
    ]].to_string(index=False))

    print("\n--- AV consistency ---")
    print(av_df[[
        "model", "av_composed_mae", "av_direct_mae",
        "av_mae_ratio_composed_over_direct",
        "av_composed_max_abs_residual", "av_composed_negative_count",
    ]].to_string(index=False))

    return target_rows, av_rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs="+", required=True)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET).reset_index(drop=True)

    if args.experiment == ["all"]:
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = args.experiment

    unknown = [e for e in exp_names if e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")

    all_targets = []
    all_av = []
    for name in exp_names:
        t, a = run_experiment(df, name, EXPERIMENTS[name])
        all_targets.extend(t)
        all_av.extend(a)

    section("Save results")
    targets_df = pd.DataFrame(all_targets)
    targets_csv = METRICS_DIR / "physics_target_summary.csv"
    targets_df.to_csv(targets_csv, index=False)
    print("Saved:", targets_csv)

    targets_tbl = TABLES_DIR / "physics_target_summary.csv"
    targets_df.to_csv(targets_tbl, index=False)
    print("Saved:", targets_tbl)

    av_df = pd.DataFrame(all_av)
    av_csv = METRICS_DIR / "physics_av_summary.csv"
    av_df.to_csv(av_csv, index=False)
    print("Saved:", av_csv)

    av_tbl = TABLES_DIR / "physics_av_summary.csv"
    av_df.to_csv(av_tbl, index=False)
    print("Saved:", av_tbl)

    section("Aggregate checks across all experiments")

    # Negative predictions (worst offenders)
    neg = targets_df[targets_df["neg_count"] > 0].copy()
    if len(neg) > 0:
        neg_top = neg.sort_values("neg_count", ascending=False).head(20)
        print("\nTop negative-prediction cases:")
        print(neg_top[[
            "experiment", "target", "model", "n_test",
            "neg_count", "neg_fraction",
        ]].to_string(index=False))
    else:
        print("\nNo negative predictions found across any experiment.")

    # Temperature monotonicity (worst offenders)
    mono = targets_df.dropna(subset=["temp_wrong_sign_fraction"]).copy()
    mono_top = mono.sort_values("temp_wrong_sign_fraction", ascending=False).head(20)
    print("\nTop wrong-sign temperature cases (pred(T=150) > pred(T=25)):")
    print(mono_top[[
        "experiment", "target", "model",
        "temp_wrong_sign_fraction", "temp_median_slope_150_minus_25",
    ]].to_string(index=False))

    # Conformal LB validity
    conf = targets_df.dropna(subset=["conformal_lb_negative_fraction"]).copy()
    conf_top = conf.sort_values("conformal_lb_negative_fraction", ascending=False).head(20)
    print("\nTop conformal lower-bound-negative cases:")
    print(conf_top[[
        "experiment", "target", "model",
        "conformal_lb_negative_fraction",
        "conformal_mean_width", "conformal_coverage",
    ]].to_string(index=False))

    # AV consistency worst ratios
    av_sorted = av_df.sort_values("av_mae_ratio_composed_over_direct", ascending=False).head(20)
    print("\nAV consistency: worst composed-vs-direct ratios:")
    print(av_sorted[[
        "experiment", "model",
        "av_composed_mae", "av_direct_mae",
        "av_mae_ratio_composed_over_direct",
        "av_composed_max_abs_residual",
    ]].to_string(index=False))

    section("DONE")
    print("Stage 10 physics-consistency analysis complete.")


if __name__ == "__main__":
    main()