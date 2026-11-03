"""
src/uncertainty/conformal_shift.py

Stage 8 -- Conformal prediction under distribution shift.

Testing H2 (uncertainty vs error) and H3 (calibration degradation).

Design
------
For each Stage 7 experiment:
  RidgeCV      : split conformal (basic + locally-adaptive kNN)
  RandomForest : tree variance as alternative uncertainty signal

Conformal settings
------------------
  alpha        : 0.10  (nominal 90% coverage)
  kNN k        : 10    (for locally-adaptive sigma(x))
  Finite-sample correction : (n+1)(1-alpha)/n

Outputs
-------
  results/metrics/conformal_summary.csv    one row per (experiment, target)
  results/metrics/conformal_bins_all.csv   per-bin analysis of H2

Run examples
------------
  python src/uncertainty/conformal_shift.py --experiment random temp_90
  python src/uncertainty/conformal_shift.py --experiment all
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
NOMINAL_ALPHA = 0.10
KNN_K = 10
N_BINS = 5
EPS = 1e-6

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
        strata_pool = df.loc[pool_idx, "Formulation_Family"]
        tr, ca = train_test_split(
            pool_idx,
            test_size=CAL_FRACTION_OF_POOL,
            random_state=RANDOM_STATE,
            stratify=strata_pool,
        )
        return np.array(tr), np.array(ca)

    if kind == "random":
        strata = df["Formulation_Family"]
        train_pool, test_idx = train_test_split(
            all_idx, test_size=0.15, random_state=RANDOM_STATE, stratify=strata,
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
    Xtr_s = scaler.transform(X_train)
    Xtg_s = scaler.transform(X_target)
    k = min(KNN_K, len(Xtr_s))
    knn = NearestNeighbors(n_neighbors=k).fit(Xtr_s)
    dist, _ = knn.kneighbors(Xtg_s)
    return dist.mean(axis=1)


def compute_rf_per_tree(rf, X):
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    tree_preds = np.stack([t.predict(X_np) for t in rf.estimators_], axis=1)
    return tree_preds.mean(axis=1), tree_preds.std(axis=1)


def conformal(y_cal, pred_cal, y_test, pred_test, sigma_cal, sigma_test, alpha):
    """Split conformal with sigma-normalization. Returns q, hits, widths."""
    norm_resid = np.abs(y_cal - pred_cal) / (sigma_cal + EPS)
    n = len(norm_resid)
    q_level = min(1.0, (n + 1) * (1 - alpha) / n)
    q = float(np.quantile(norm_resid, q_level, method="higher"))
    widths = 2.0 * q * (sigma_test + EPS)
    hits = np.abs(y_test - pred_test) <= q * (sigma_test + EPS)
    return q, hits, widths


def bin_report(values, hits, errors, widths, n_bins=N_BINS):
    qs = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    bin_ids = np.digitize(values, qs[1:-1])
    rows = []
    for b in range(n_bins):
        m = bin_ids == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "n": int(m.sum()),
            "mean_sigma": float(values[m].mean()),
            "coverage": float(hits[m].mean()),
            "mean_abs_error": float(errors[m].mean()),
            "mean_width": float(widths[m].mean()),
        })
    return rows


def run_experiment(df, exp_name, spec):
    train_idx, cal_idx, test_idx = get_split_indices(df, spec)
    stress = bool(spec.get("stress_test", False))
    n_test = len(test_idx)

    section(f"Experiment: {exp_name}  train={len(train_idx)} cal={len(cal_idx)} test={n_test} stress={stress}")

    X_train_all = df.loc[train_idx, FEATURES]
    X_cal_all = df.loc[cal_idx, FEATURES]
    X_test_all = df.loc[test_idx, FEATURES]

    sigma_cal_knn = compute_knn_sigma(X_train_all, X_cal_all)
    sigma_test_knn = compute_knn_sigma(X_train_all, X_test_all)

    summary_rows = []
    bin_rows_all = []

    for target in TARGETS:
        y_train = df.loc[train_idx, target].values
        y_cal = df.loc[cal_idx, target].values
        y_test = df.loc[test_idx, target].values

        # ---- RidgeCV + conformal ----
        ridge = build_ridge().fit(X_train_all, y_train)
        pred_cal_ridge = ridge.predict(X_cal_all)
        pred_test_ridge = ridge.predict(X_test_all)

        sigma_ones_cal = np.ones_like(y_cal)
        sigma_ones_test = np.ones_like(y_test)

        q_basic, hits_basic, widths_basic = conformal(
            y_cal, pred_cal_ridge, y_test, pred_test_ridge,
            sigma_ones_cal, sigma_ones_test, NOMINAL_ALPHA,
        )
        q_la, hits_la, widths_la = conformal(
            y_cal, pred_cal_ridge, y_test, pred_test_ridge,
            sigma_cal_knn, sigma_test_knn, NOMINAL_ALPHA,
        )

        errors_ridge = np.abs(y_test - pred_test_ridge)
        try:
            sp_width, _ = spearmanr(widths_la, errors_ridge)
        except Exception:
            sp_width = float("nan")
        try:
            sp_knn, _ = spearmanr(sigma_test_knn, errors_ridge)
        except Exception:
            sp_knn = float("nan")

        bins_la = bin_report(sigma_test_knn, hits_la, errors_ridge, widths_la)

        # ---- RandomForest tree variance ----
        rf = build_rf().fit(X_train_all, y_train)
        pred_test_rf, sigma_rf_test = compute_rf_per_tree(rf, X_test_all)
        errors_rf = np.abs(y_test - pred_test_rf)
        try:
            sp_rf, _ = spearmanr(sigma_rf_test, errors_rf)
        except Exception:
            sp_rf = float("nan")

        summary_rows.append({
            "experiment": exp_name,
            "target": target,
            "n_test": n_test,
            "stress": stress,
            "ridge_coverage_basic": float(hits_basic.mean()),
            "ridge_MPIW_basic": float(widths_basic.mean()),
            "ridge_coverage_LA": float(hits_la.mean()),
            "ridge_MPIW_LA": float(widths_la.mean()),
            "spearman_width_vs_error_LA": float(sp_width),
            "spearman_knn_vs_error": float(sp_knn),
            "spearman_rf_var_vs_error": float(sp_rf),
            "ridge_MAE": float(mean_absolute_error(y_test, pred_test_ridge)),
            "rf_MAE": float(mean_absolute_error(y_test, pred_test_rf)),
        })

        for b in bins_la:
            b.update({"experiment": exp_name, "target": target, "method": "ridge_LA"})
            bin_rows_all.append(b)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    return summary_rows, bin_rows_all


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
    all_bins = []
    for name in exp_names:
        s, b = run_experiment(df, name, EXPERIMENTS[name])
        all_summary.extend(s)
        all_bins.extend(b)

    section("Save combined results")
    summary_df = pd.DataFrame(all_summary)
    summary_csv = METRICS_DIR / "conformal_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)

    bins_df = pd.DataFrame(all_bins)
    bins_csv = METRICS_DIR / "conformal_bins_all.csv"
    bins_df.to_csv(bins_csv, index=False)
    print("Saved:", bins_csv)

    section("DONE")
    print("Stage 8 conformal experiments complete.")


if __name__ == "__main__":
    main()