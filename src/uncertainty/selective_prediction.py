"""
src/uncertainty/selective_prediction.py

Stage 9 -- Selective prediction (risk-coverage analysis).

Purpose
-------
Test whether per-point uncertainty can identify a subset of predictions
we can trust. For each Stage 7 experiment, each target, and each
uncertainty signal, we:

  1. Fit the model on train, predict on test.
  2. Compute per-point sigma(x) (kNN for RidgeCV, tree-variance for RF).
  3. Sort test points by ascending sigma  (most confident first).
  4. Sweep retention fractions and compute MAE on the retained subset.
  5. Compute AURC (area under risk-coverage curve).
  6. Compare AURC to the unconditional MAE.
  7. Report Spearman(sigma, |error|).

Key metrics
-----------
  baseline_MAE      : MAE over all test points (equivalent to random retention)
  AURC              : mean of risk@k for k=1..n (Geifman & El-Yaniv 2017)
  AURC_normalized   : AURC / baseline_MAE  (< 1 means uncertainty helps)
  MAE_at_50 / _75 / _90 : MAE when retaining 50 / 75 / 90 percent of points
  Spearman_sigma_err : rank correlation between sigma and |error|

Outputs
-------
  results/metrics/selective_summary.csv   one row per (experiment, target, model)
  results/metrics/selective_curves.csv    per retention fraction
  results/tables/selective_summary.csv

Run examples
------------
  python src/uncertainty/selective_prediction.py --experiment random temp_120 family_Salt_Tolerant
  python src/uncertainty/selective_prediction.py --experiment all
"""

import argparse
import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
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
KNN_K = 10
EPS = 1e-6

RETENTION_FRACTIONS = np.round(np.arange(0.10, 1.0001, 0.05), 2)

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


def compute_knn_sigma(X_train, X_target):
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xtg = scaler.transform(X_target)
    k = min(KNN_K, len(Xtr))
    knn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    dist, _ = knn.kneighbors(Xtg)
    return dist.mean(axis=1)


def compute_rf_per_tree(rf, X):
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    tree_preds = np.stack([t.predict(X_np) for t in rf.estimators_], axis=1)
    return tree_preds.mean(axis=1), tree_preds.std(axis=1)


def risk_coverage(sigma, errors, fractions):
    """Sort by sigma ascending; return per-fraction MAE and AURC."""
    n = len(sigma)
    order = np.argsort(sigma, kind="stable")
    sorted_err = errors[order]
    cum = np.cumsum(sorted_err)

    curve_rows = []
    for f in fractions:
        k = max(1, min(n, int(np.ceil(f * n))))
        mae_k = float(cum[k - 1] / k)
        curve_rows.append({
            "retention": float(f),
            "n_retained": int(k),
            "mae_retained": mae_k,
        })

    ks = np.arange(1, n + 1)
    risks = cum / ks
    aurc = float(risks.mean())
    baseline = float(errors.mean())
    return curve_rows, aurc, baseline


def mae_at_retention(curve_rows, target_fraction):
    best = min(curve_rows, key=lambda r: abs(r["retention"] - target_fraction))
    return float(best["mae_retained"])


def run_experiment(df, exp_name, spec):
    train_idx, cal_idx, test_idx = get_split_indices(df, spec)
    stress = bool(spec.get("stress_test", False))
    n_test = len(test_idx)

    section(f"Experiment: {exp_name}  train={len(train_idx)} cal={len(cal_idx)} test={n_test} stress={stress}")

    X_train = df.loc[train_idx, FEATURES]
    X_test = df.loc[test_idx, FEATURES]

    sigma_knn = compute_knn_sigma(X_train, X_test)

    summary_rows = []
    curve_rows_all = []

    for target in TARGETS:
        y_train = df.loc[train_idx, target].values
        y_test = df.loc[test_idx, target].values

        # ---- RidgeCV ----
        ridge = build_ridge().fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)
        err_ridge = np.abs(y_test - pred_ridge)

        rows, aurc, base = risk_coverage(sigma_knn, err_ridge, RETENTION_FRACTIONS)
        try:
            sp_r, _ = spearmanr(sigma_knn, err_ridge)
        except Exception:
            sp_r = float("nan")

        summary_rows.append({
            "experiment": exp_name,
            "target": target,
            "model": "RidgeCV",
            "n_test": n_test,
            "stress": stress,
            "baseline_MAE": base,
            "AURC": aurc,
            "AURC_normalized": aurc / base if base > 0 else float("nan"),
            "MAE_at_50": mae_at_retention(rows, 0.50),
            "MAE_at_75": mae_at_retention(rows, 0.75),
            "MAE_at_90": mae_at_retention(rows, 0.90),
            "Spearman_sigma_err": float(sp_r),
        })
        for r in rows:
            r.update({
                "experiment": exp_name,
                "target": target,
                "model": "RidgeCV",
                "stress": stress,
            })
        curve_rows_all.extend(rows)

        # ---- RandomForest ----
        rf = build_rf().fit(X_train, y_train)
        pred_rf, sigma_rf = compute_rf_per_tree(rf, X_test)
        err_rf = np.abs(y_test - pred_rf)

        rows_rf, aurc_rf, base_rf = risk_coverage(sigma_rf, err_rf, RETENTION_FRACTIONS)
        try:
            sp_rf, _ = spearmanr(sigma_rf, err_rf)
        except Exception:
            sp_rf = float("nan")

        summary_rows.append({
            "experiment": exp_name,
            "target": target,
            "model": "RandomForest",
            "n_test": n_test,
            "stress": stress,
            "baseline_MAE": base_rf,
            "AURC": aurc_rf,
            "AURC_normalized": aurc_rf / base_rf if base_rf > 0 else float("nan"),
            "MAE_at_50": mae_at_retention(rows_rf, 0.50),
            "MAE_at_75": mae_at_retention(rows_rf, 0.75),
            "MAE_at_90": mae_at_retention(rows_rf, 0.90),
            "Spearman_sigma_err": float(sp_rf),
        })
        for r in rows_rf:
            r.update({
                "experiment": exp_name,
                "target": target,
                "model": "RandomForest",
                "stress": stress,
            })
        curve_rows_all.extend(rows_rf)

    summary_df = pd.DataFrame(summary_rows)
    cols = [
        "experiment", "target", "model", "n_test", "stress",
        "baseline_MAE", "AURC", "AURC_normalized",
        "MAE_at_50", "MAE_at_75", "MAE_at_90",
        "Spearman_sigma_err",
    ]
    print(summary_df[cols].to_string(index=False))
    return summary_rows, curve_rows_all


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

    all_summary = []
    all_curves = []
    for name in exp_names:
        s, c = run_experiment(df, name, EXPERIMENTS[name])
        all_summary.extend(s)
        all_curves.extend(c)

    section("Save results")
    summary_df = pd.DataFrame(all_summary)
    summary_cols = [
        "experiment", "target", "model", "n_test", "stress",
        "baseline_MAE", "AURC", "AURC_normalized",
        "MAE_at_50", "MAE_at_75", "MAE_at_90",
        "Spearman_sigma_err",
    ]
    summary_df = summary_df[summary_cols]
    summary_csv = METRICS_DIR / "selective_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)

    tables_csv = TABLES_DIR / "selective_summary.csv"
    summary_df.to_csv(tables_csv, index=False)
    print("Saved:", tables_csv)

    curves_df = pd.DataFrame(all_curves)
    curve_cols = [
        "experiment", "target", "model", "stress",
        "retention", "n_retained", "mae_retained",
    ]
    curves_df = curves_df[curve_cols]
    curves_csv = METRICS_DIR / "selective_curves.csv"
    curves_df.to_csv(curves_csv, index=False)
    print("Saved:", curves_csv)

    section("Top AURC-normalized improvements (uncertainty is useful)")
    df_sorted = summary_df.sort_values("AURC_normalized").reset_index(drop=True)
    print(df_sorted.head(20).to_string(index=False))

    section("Worst AURC-normalized (uncertainty is not useful)")
    print(df_sorted.tail(20).to_string(index=False))

    section("DONE")
    print("Stage 9 selective-prediction analysis complete.")


if __name__ == "__main__":
    main()