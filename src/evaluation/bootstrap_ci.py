"""
src/evaluation/bootstrap_ci.py

Stage 13 -- Bootstrap confidence intervals for all reported metrics.

Purpose
-------
Attach 95% confidence intervals to every performance claim in Stages 6-12
so that each number in the paper is defensible and each ablation effect
has an explicit significance verdict.

What is bootstrapped
--------------------
1. Shift MAE CIs
   For the reference (random) and 7 shift experiments, compute the
   bootstrap distribution of test-set MAE per target per model.

2. Shift-delta-vs-random CIs
   For each shift experiment, compute delta_MAE = MAE(shift) - MAE(random)
   with an independently bootstrapped 95% CI. Report the fraction of
   bootstrap deltas > 0 (i.e., probability that shift is worse).

3. Ablation-pair delta CIs (paired bootstrap)
   For each ablation pair (base, new), the SAME test set is used, so we
   use paired bootstrap: resample test indices, compute both MAEs on
   the same resampled set, take the difference. Report the fraction of
   bootstrap deltas < 0 (i.e., probability that the new feature set helps).

4. Conformal coverage CIs
   For 4 experiments and 5 targets, compute the bootstrap distribution
   of empirical coverage at 90% nominal. Report whether the CI contains
   0.90 (calibrated) or excludes it (miscalibrated).

Design
------
- B = 1000 bootstrap resamples (standard)
- alpha = 0.05 -> 95% CIs by percentile method
- Predictions are cached so each (experiment, ablation, target, model)
  is fit and predicted exactly once
- Random seed = 42 for reproducibility

Runtime
-------
~3-5 minutes on the target laptop (420 model fits + 350 delta bootstraps)

Outputs
-------
  results/metrics/ci_shift_mae.csv
  results/metrics/ci_shift_deltas.csv
  results/metrics/ci_ablation_deltas.csv
  results/metrics/ci_conformal_coverage.csv

Run
---
  python src/evaluation/bootstrap_ci.py
"""

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
for d in [METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CAL_FRACTION_OF_POOL = 0.15
N_BOOT = 1000
ALPHA = 0.05
NOMINAL_ALPHA_CONF = 0.10  # 90% prediction interval

TARGETS = [
    "PV_cP",
    "YP_lb100ft2",
    "GS_10s_lb100ft2",
    "GS_10min_lb100ft2",
    "n_FlowIndex",
]

# Stage 6 baseline feature set (13 features)
FEATURES_STAGE6 = [
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

# Ablation sets used in the pairs
COMPOSITION_FEATURES = FEATURES_STAGE6[:11]  # first 11

ABLATIONS = {
    "A1_comp_only": COMPOSITION_FEATURES,
    "A2_comp_temp": COMPOSITION_FEATURES + ["Test_Temperature_C"],
    "A3_comp_temp_aging": COMPOSITION_FEATURES + [
        "Test_Temperature_C", "Aging_Hours", "Aging_Temperature_C"
    ],
    "A4_comp_temp_mw": COMPOSITION_FEATURES + ["Test_Temperature_C", "MW_ppg"],
    "A5_full": COMPOSITION_FEATURES + [
        "Test_Temperature_C", "Aging_Hours", "Aging_Temperature_C", "MW_ppg"
    ],
    "A6_stage6_baseline": COMPOSITION_FEATURES + [
        "Test_Temperature_C", "Aging_Hours"
    ],
}

ABLATION_PAIRS = [
    ("temp_effect",        "A1_comp_only",       "A2_comp_temp"),
    ("aging_hours_effect", "A2_comp_temp",       "A6_stage6_baseline"),
    ("aging_temp_effect",  "A6_stage6_baseline", "A3_comp_temp_aging"),
    ("mw_effect",          "A2_comp_temp",       "A4_comp_temp_mw"),
    ("full_vs_stage6",     "A6_stage6_baseline", "A5_full"),
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

# Experiments selected for CI computation
FOCUS_SHIFT = [
    "random",
    "temp_25",
    "temp_120",
    "family_Base_Bentonite",
    "family_Salt_Tolerant",
    "combo_Base_Bentonite_25",
    "combo_KCl_Polymer_90",
    "temporal_2023_to_2024",
]

FOCUS_CONFORMAL = [
    "random",
    "temp_120",
    "family_Salt_Tolerant",
    "combo_KCl_Polymer_90",
]


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
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=1,
    )


MODELS = [("RidgeCV", build_ridge), ("RandomForest", build_rf)]


def mae_fn(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ----------------------------------------------------------------------
# Bootstrap primitives
# ----------------------------------------------------------------------
def bootstrap_ci_metric(y_true, y_pred, metric_fn, B=N_BOOT, alpha=ALPHA, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = metric_fn(y_true[idx], y_pred[idx])
    return {
        "point": float(metric_fn(y_true, y_pred)),
        "ci_low": float(np.quantile(vals, alpha / 2)),
        "ci_high": float(np.quantile(vals, 1 - alpha / 2)),
        "std": float(vals.std()),
    }


def bootstrap_ci_independent_delta(
    y_true_a, pred_a, y_true_b, pred_b, metric_fn,
    B=N_BOOT, alpha=ALPHA, seed=RANDOM_STATE,
):
    """Compare MAE(b) - MAE(a) with independently bootstrapped samples."""
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1)
    na = len(y_true_a); nb = len(y_true_b)
    mae_a_b = np.empty(B)
    mae_b_b = np.empty(B)
    for b in range(B):
        ia = rng_a.integers(0, na, na)
        ib = rng_b.integers(0, nb, nb)
        mae_a_b[b] = metric_fn(y_true_a[ia], pred_a[ia])
        mae_b_b[b] = metric_fn(y_true_b[ib], pred_b[ib])
    delta_b = mae_b_b - mae_a_b
    return {
        "point_a": float(metric_fn(y_true_a, pred_a)),
        "point_b": float(metric_fn(y_true_b, pred_b)),
        "delta_point": float(metric_fn(y_true_b, pred_b) - metric_fn(y_true_a, pred_a)),
        "delta_ci_low": float(np.quantile(delta_b, alpha / 2)),
        "delta_ci_high": float(np.quantile(delta_b, 1 - alpha / 2)),
        "delta_std": float(delta_b.std()),
        "p_delta_gt_zero": float((delta_b > 0).mean()),
    }


def bootstrap_ci_paired_delta(
    y_true, pred_base, pred_new, metric_fn,
    B=N_BOOT, alpha=ALPHA, seed=RANDOM_STATE,
):
    """Compare MAE(new) - MAE(base) on the SAME test set (paired bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mae_base_b = np.empty(B)
    mae_new_b = np.empty(B)
    delta_b = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        base_b = metric_fn(y_true[idx], pred_base[idx])
        new_b = metric_fn(y_true[idx], pred_new[idx])
        mae_base_b[b] = base_b
        mae_new_b[b] = new_b
        delta_b[b] = new_b - base_b
    return {
        "point_base": float(metric_fn(y_true, pred_base)),
        "point_new": float(metric_fn(y_true, pred_new)),
        "delta_point": float(metric_fn(y_true, pred_new) - metric_fn(y_true, pred_base)),
        "base_ci_low": float(np.quantile(mae_base_b, alpha / 2)),
        "base_ci_high": float(np.quantile(mae_base_b, 1 - alpha / 2)),
        "new_ci_low": float(np.quantile(mae_new_b, alpha / 2)),
        "new_ci_high": float(np.quantile(mae_new_b, 1 - alpha / 2)),
        "delta_ci_low": float(np.quantile(delta_b, alpha / 2)),
        "delta_ci_high": float(np.quantile(delta_b, 1 - alpha / 2)),
        "delta_std": float(delta_b.std()),
        "p_delta_lt_zero": float((delta_b < 0).mean()),
    }


def bootstrap_ci_coverage(y_true, lb, ub, B=N_BOOT, alpha=ALPHA, seed=RANDOM_STATE):
    hits = ((y_true >= lb) & (y_true <= ub)).astype(float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    cov_b = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        cov_b[b] = hits[idx].mean()
    return {
        "point": float(hits.mean()),
        "ci_low": float(np.quantile(cov_b, alpha / 2)),
        "ci_high": float(np.quantile(cov_b, 1 - alpha / 2)),
    }


def conformal_q(y_cal, pred_cal, alpha):
    n = len(y_cal)
    resid = np.abs(y_cal - pred_cal)
    level = min(1.0, (n + 1) * (1 - alpha) / n)
    return float(np.quantile(resid, level, method="higher"))


# ----------------------------------------------------------------------
# Section 1 + 2: shift MAE CIs and delta-vs-random
# ----------------------------------------------------------------------
def run_shift_section(df):
    print("\n>>> Section 1+2: shift MAE CIs and shift-delta CIs")
    # Cache predictions for all focus shift experiments
    cache = {}  # (exp, target, model) -> (y_test, pred, n_test)
    for exp in FOCUS_SHIFT:
        spec = EXPERIMENTS[exp]
        train_idx, _, test_idx = get_split_indices(df, spec)
        X_train = df.loc[train_idx, FEATURES_STAGE6]
        X_test = df.loc[test_idx, FEATURES_STAGE6]
        for target in TARGETS:
            y_train = df.loc[train_idx, target].values
            y_test = df.loc[test_idx, target].values
            for model_name, build_fn in MODELS:
                model = build_fn().fit(X_train, y_train)
                pred = model.predict(X_test)
                cache[(exp, target, model_name)] = (y_test, pred, len(test_idx))
        print(f"  cached: {exp}")

    # Section 1: MAE with CI
    rows_mae = []
    for (exp, target, model_name), (y_test, pred, n_test) in cache.items():
        ci = bootstrap_ci_metric(y_test, pred, mae_fn)
        rows_mae.append({
            "experiment": exp,
            "target": target,
            "model": model_name,
            "n_test": n_test,
            "mae_point": ci["point"],
            "mae_ci_low": ci["ci_low"],
            "mae_ci_high": ci["ci_high"],
            "mae_std": ci["std"],
        })
    mae_df = pd.DataFrame(rows_mae)

    # Section 2: delta vs random
    rows_delta = []
    for exp in FOCUS_SHIFT:
        if exp == "random":
            continue
        for target in TARGETS:
            for model_name, _ in MODELS:
                key_ref = ("random", target, model_name)
                key_exp = (exp, target, model_name)
                if key_ref not in cache or key_exp not in cache:
                    continue
                y_ref, pred_ref, n_ref = cache[key_ref]
                y_exp, pred_exp, n_exp = cache[key_exp]
                d = bootstrap_ci_independent_delta(
                    y_ref, pred_ref, y_exp, pred_exp, mae_fn,
                )
                rows_delta.append({
                    "experiment": exp,
                    "target": target,
                    "model": model_name,
                    "n_test_shift": n_exp,
                    "n_test_random": n_ref,
                    "mae_random_point": d["point_a"],
                    "mae_shift_point": d["point_b"],
                    "delta_point": d["delta_point"],
                    "delta_ci_low": d["delta_ci_low"],
                    "delta_ci_high": d["delta_ci_high"],
                    "delta_std": d["delta_std"],
                    "p_delta_gt_zero": d["p_delta_gt_zero"],
                })
    delta_df = pd.DataFrame(rows_delta)
    return mae_df, delta_df


# ----------------------------------------------------------------------
# Section 3: ablation-pair delta CIs
# ----------------------------------------------------------------------
def run_ablation_section(df):
    print("\n>>> Section 3: ablation-pair delta CIs (paired bootstrap)")

    # Determine all (exp, ablation, target, model) needed for the pairs
    needed = set()
    for _, base_set, new_set in ABLATION_PAIRS:
        for exp in EXPERIMENTS:
            for target in TARGETS:
                for model_name, _ in MODELS:
                    needed.add((exp, base_set, target, model_name))
                    needed.add((exp, new_set, target, model_name))

    # Fit and cache predictions once per needed combination
    pred_cache = {}  # (exp, ablation, target, model) -> (y_test, pred)
    split_cache = {}  # exp -> (train_idx, test_idx)
    for exp in EXPERIMENTS:
        if not any(k[0] == exp for k in needed):
            continue
        spec = EXPERIMENTS[exp]
        train_idx, _, test_idx = get_split_indices(df, spec)
        split_cache[exp] = (train_idx, test_idx)

    fit_count = 0
    for (exp, ablation, target, model_name) in sorted(needed):
        features = ABLATIONS[ablation]
        train_idx, test_idx = split_cache[exp]
        X_train = df.loc[train_idx, features]
        X_test = df.loc[test_idx, features]
        y_train = df.loc[train_idx, target].values
        y_test = df.loc[test_idx, target].values
        build_fn = dict(MODELS)[model_name]
        model = build_fn().fit(X_train, y_train)
        pred = model.predict(X_test)
        pred_cache[(exp, ablation, target, model_name)] = (y_test, pred)
        fit_count += 1
        if fit_count % 60 == 0:
            print(f"  fitted {fit_count} / {len(needed)}")

    print(f"  fitted total: {fit_count}")

    # Now compute paired delta CIs for every pair x exp x target x model
    rows = []
    total_cells = len(ABLATION_PAIRS) * len(EXPERIMENTS) * len(TARGETS) * len(MODELS)
    done = 0
    for pair_label, base_set, new_set in ABLATION_PAIRS:
        for exp in EXPERIMENTS:
            for target in TARGETS:
                for model_name, _ in MODELS:
                    key_base = (exp, base_set, target, model_name)
                    key_new = (exp, new_set, target, model_name)
                    if key_base not in pred_cache or key_new not in pred_cache:
                        continue
                    y_test, pred_base = pred_cache[key_base]
                    _, pred_new = pred_cache[key_new]
                    d = bootstrap_ci_paired_delta(
                        y_test, pred_base, pred_new, mae_fn,
                    )
                    rows.append({
                        "pair": pair_label,
                        "base_ablation": base_set,
                        "new_ablation": new_set,
                        "experiment": exp,
                        "target": target,
                        "model": model_name,
                        "n_test": len(y_test),
                        "mae_base_point": d["point_base"],
                        "mae_base_ci_low": d["base_ci_low"],
                        "mae_base_ci_high": d["base_ci_high"],
                        "mae_new_point": d["point_new"],
                        "mae_new_ci_low": d["new_ci_low"],
                        "mae_new_ci_high": d["new_ci_high"],
                        "delta_point": d["delta_point"],
                        "delta_ci_low": d["delta_ci_low"],
                        "delta_ci_high": d["delta_ci_high"],
                        "delta_std": d["delta_std"],
                        "p_delta_lt_zero": d["p_delta_lt_zero"],
                    })
                    done += 1
                    if done % 70 == 0:
                        print(f"  delta CIs computed: {done} / {total_cells}")

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Section 4: conformal coverage CIs
# ----------------------------------------------------------------------
def run_conformal_section(df):
    print("\n>>> Section 4: conformal coverage CIs")
    rows = []
    for exp in FOCUS_CONFORMAL:
        spec = EXPERIMENTS[exp]
        train_idx, cal_idx, test_idx = get_split_indices(df, spec)
        X_train = df.loc[train_idx, FEATURES_STAGE6]
        X_cal = df.loc[cal_idx, FEATURES_STAGE6]
        X_test = df.loc[test_idx, FEATURES_STAGE6]
        for target in TARGETS:
            y_train = df.loc[train_idx, target].values
            y_cal = df.loc[cal_idx, target].values
            y_test = df.loc[test_idx, target].values
            model = build_ridge().fit(X_train, y_train)
            pred_cal = model.predict(X_cal)
            pred_test = model.predict(X_test)
            q = conformal_q(y_cal, pred_cal, NOMINAL_ALPHA_CONF)
            lb = pred_test - q
            ub = pred_test + q
            cov = bootstrap_ci_coverage(y_test, lb, ub)
            rows.append({
                "experiment": exp,
                "target": target,
                "model": "RidgeCV",
                "n_test": len(y_test),
                "nominal_coverage": 1 - NOMINAL_ALPHA_CONF,
                "coverage_point": cov["point"],
                "coverage_ci_low": cov["ci_low"],
                "coverage_ci_high": cov["ci_high"],
                "ci_excludes_nominal": (
                    (cov["ci_high"] < (1 - NOMINAL_ALPHA_CONF)) or
                    (cov["ci_low"] > (1 - NOMINAL_ALPHA_CONF))
                ),
            })
        print(f"  computed: {exp}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET).reset_index(drop=True)
    print(f"Rows: {len(df)}")
    print(f"B = {N_BOOT} bootstrap resamples; alpha = {ALPHA} (95% CI)")

    # Section 1+2: shift MAE and delta CIs
    mae_df, delta_df = run_shift_section(df)

    section("Shift MAE with 95% CI")
    print(mae_df.to_string(index=False))

    section("Shift delta vs random with 95% CI")
    print(delta_df.to_string(index=False))

    # Section 3: ablation CIs
    abl_df = run_ablation_section(df)

    section("Ablation pair summary with 95% CI")
    # Aggregate by pair x model
    agg = abl_df.groupby(["pair", "model"]).agg(
        n_cells=("delta_point", "count"),
        delta_mean=("delta_point", "mean"),
        delta_median=("delta_point", "median"),
        frac_ci_excludes_zero=("delta_ci_high", lambda s: float((s < 0).mean())),
    )
    print(agg.round(4).to_string())

    # Section 4: conformal coverage CIs
    conf_df = run_conformal_section(df)

    section("Conformal coverage with 95% CI")
    print(conf_df.to_string(index=False))

    # Save CSVs
    section("Save results")
    p1 = METRICS_DIR / "ci_shift_mae.csv"; mae_df.to_csv(p1, index=False); print("Saved:", p1)
    p2 = METRICS_DIR / "ci_shift_deltas.csv"; delta_df.to_csv(p2, index=False); print("Saved:", p2)
    p3 = METRICS_DIR / "ci_ablation_deltas.csv"; abl_df.to_csv(p3, index=False); print("Saved:", p3)
    p4 = METRICS_DIR / "ci_conformal_coverage.csv"; conf_df.to_csv(p4, index=False); print("Saved:", p4)

    # Interpretive summary
    section("Interpretive summary")

    # How many shift delta CIs exclude zero (on the positive side)?
    sig_delta = (delta_df["delta_ci_low"] > 0).mean()
    print(f"Fraction of shift-delta cells with delta_ci_low > 0 "
          f"(shift is significantly worse than random): {sig_delta:.1%}")

    # How many ablation deltas exclude zero on the negative side?
    sig_abl = (abl_df["delta_ci_high"] < 0).mean()
    print(f"Fraction of ablation cells where new_set is significantly better: "
          f"{sig_abl:.1%}")

    # How many conformal coverage CIs exclude 0.90?
    conf_excl = conf_df["ci_excludes_nominal"].mean()
    print(f"Fraction of conformal cells where 95% CI excludes 0.90 nominal: "
          f"{conf_excl:.1%}")

    section("DONE")
    print("Stage 13 bootstrap confidence intervals complete.")


if __name__ == "__main__":
    main()