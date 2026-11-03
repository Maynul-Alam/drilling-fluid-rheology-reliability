"""
src/validation/ablation.py

Stage 12 -- Feature-set ablation under distribution shift.

Purpose
-------
Test whether specific feature groups genuinely improve generalization, or
merely add parameters. Questions addressed:

  Q1 (temperature). Does adding Test_Temperature_C to composition
     improve accuracy, especially on unseen temperatures?

  Q2 (aging hours). Does Aging_Hours help beyond composition + temperature?

  Q3 (aging temperature). Does Aging_Temperature_C help beyond
     Aging_Hours, especially on temperature shifts?

  Q4 (MW vs Barite). Does MW_ppg add anything beyond Barite_ppb?
     (Barite and MW have correlation 0.998 from the Stage 6 audit.)

  Q5 (full feature set). Does the full set beat the Stage 6 baseline?

Ablation sets
-------------
  A1_comp_only            11 composition only
  A2_comp_temp            11 comp + Test_Temperature_C
  A3_comp_temp_aging      11 comp + Test_Temperature_C
                          + Aging_Hours + Aging_Temperature_C
  A4_comp_temp_mw         11 comp + Test_Temperature_C + MW_ppg
  A5_full                 11 comp + Test_Temperature_C
                          + Aging_Hours + Aging_Temperature_C + MW_ppg
  A6_stage6_baseline      11 comp + Test_Temperature_C + Aging_Hours
                          (identical to Stage 6 feature set)

Experiments
-----------
A representative set covering random, temperature shift, family shift,
temporal shift, and combined shift.

Outputs
-------
  results/metrics/ablation_all.csv
  results/metrics/ablation_deltas.csv
  results/tables/ablation_all.csv

Run
---
  python src/validation/ablation.py
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

COMPOSITION_FEATURES = [
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
]

ABLATIONS = {
    "A1_comp_only": (
        COMPOSITION_FEATURES
    ),
    "A2_comp_temp": (
        COMPOSITION_FEATURES + ["Test_Temperature_C"]
    ),
    "A3_comp_temp_aging": (
        COMPOSITION_FEATURES + ["Test_Temperature_C",
                                "Aging_Hours", "Aging_Temperature_C"]
    ),
    "A4_comp_temp_mw": (
        COMPOSITION_FEATURES + ["Test_Temperature_C", "MW_ppg"]
    ),
    "A5_full": (
        COMPOSITION_FEATURES + ["Test_Temperature_C",
                                "Aging_Hours", "Aging_Temperature_C",
                                "MW_ppg"]
    ),
    "A6_stage6_baseline": (
        COMPOSITION_FEATURES + ["Test_Temperature_C", "Aging_Hours"]
    ),
}

# Representative set covering each shift type
EXPERIMENTS = {
    "random": {"type": "random"},
    "temp_25": {"type": "temp", "temp": 25},
    "temp_120": {"type": "temp", "temp": 120},
    "family_Base_Bentonite": {"type": "family", "family": "Base_Bentonite"},
    "family_Salt_Tolerant": {"type": "family", "family": "Salt_Tolerant"},
    "combo_Base_Bentonite_25": {"type": "combo", "family": "Base_Bentonite", "temp": 25},
    "temporal_2023_to_2024": {"type": "temporal"},
}

# Pairwise comparisons for the summary
# Each entry: (label, base_set, new_set)
# ΔMAE = MAE(new) - MAE(base).  Negative ΔMAE means the added features helped.
PAIRS = [
    ("temp_effect",         "A1_comp_only",       "A2_comp_temp"),
    ("aging_hours_effect",  "A2_comp_temp",       "A6_stage6_baseline"),
    ("aging_temp_effect",   "A6_stage6_baseline", "A3_comp_temp_aging"),
    ("mw_effect",           "A2_comp_temp",       "A4_comp_temp_mw"),
    ("full_vs_stage6",      "A6_stage6_baseline", "A5_full"),
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


def compute_metrics(y_true, y_pred, target):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    negatives = int(np.sum(y_pred < 0))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "bias": bias,
        "negatives": negatives,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_all_ablations(df):
    """Run every (experiment, ablation, target, model) combination."""
    rows = []

    for exp_name, spec in EXPERIMENTS.items():
        train_idx, cal_idx, test_idx = get_split_indices(df, spec)
        n_test = len(test_idx)
        section(f"Experiment: {exp_name}  train={len(train_idx)} "
                f"cal={len(cal_idx)} test={n_test}")

        for ablation_name, features in ABLATIONS.items():
            X_train = df.loc[train_idx, features]
            X_test = df.loc[test_idx, features]

            for target in TARGETS:
                y_train = df.loc[train_idx, target].values
                y_test = df.loc[test_idx, target].values

                for model_name, build_fn in [
                    ("RidgeCV", build_ridge),
                    ("RandomForest", build_rf),
                ]:
                    model = build_fn().fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    m = compute_metrics(y_test, y_pred, target)
                    rows.append({
                        "experiment": exp_name,
                        "ablation": ablation_name,
                        "n_features": len(features),
                        "target": target,
                        "model": model_name,
                        "n_train": int(len(train_idx)),
                        "n_test": int(n_test),
                        "MAE": m["MAE"],
                        "RMSE": m["RMSE"],
                        "R2": m["R2"],
                        "bias": m["bias"],
                        "negatives": m["negatives"],
                    })

        print(f"  {len(ABLATIONS) * len(TARGETS) * 2} model fits complete")

    return pd.DataFrame(rows)


def compute_deltas(df_all):
    """For each PAIR, compute delta_MAE = MAE(new) - MAE(base)."""
    rows = []
    for label, base_set, new_set in PAIRS:
        base = df_all[df_all["ablation"] == base_set]
        new = df_all[df_all["ablation"] == new_set]
        merged = base.merge(
            new,
            on=["experiment", "target", "model"],
            suffixes=("_base", "_new"),
        )
        for _, r in merged.iterrows():
            rows.append({
                "pair": label,
                "base_ablation": base_set,
                "new_ablation": new_set,
                "experiment": r["experiment"],
                "target": r["target"],
                "model": r["model"],
                "MAE_base": r["MAE_base"],
                "MAE_new": r["MAE_new"],
                "delta_MAE": r["MAE_new"] - r["MAE_base"],
                "delta_R2": r["R2_new"] - r["R2_base"],
                "delta_negatives": r["negatives_new"] - r["negatives_base"],
            })
    return pd.DataFrame(rows)


def print_pair_summary(deltas_df):
    section("Ablation pair summaries")
    for pair_name, group in deltas_df.groupby("pair"):
        base_set = group["base_ablation"].iloc[0]
        new_set = group["new_ablation"].iloc[0]
        print(f"\n--- {pair_name} ---")
        print(f"    base = {base_set}")
        print(f"    new  = {new_set}")
        print(f"    ΔMAE = MAE(new) - MAE(base)   "
              f"(negative = new set better)")

        # Focus on RidgeCV; RF summary separately
        for model in ["RidgeCV", "RandomForest"]:
            g = group[group["model"] == model]
            if len(g) == 0:
                continue
            print(f"\n  {model}: mean ΔMAE across experiment×target = "
                  f"{g['delta_MAE'].mean():+.4f}")
            print(f"          median ΔMAE = {g['delta_MAE'].median():+.4f}")
            frac_improved = float((g['delta_MAE'] < 0).mean())
            print(f"          fraction improved = {frac_improved:.2%}")

            # Breakdown by experiment
            by_exp = g.groupby("experiment")["delta_MAE"].agg(["mean", "median", "count"])
            print(by_exp.to_string())


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET).reset_index(drop=True)

    print(f"Rows: {len(df)}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Ablation sets: {len(ABLATIONS)}")
    print(f"Targets: {len(TARGETS)}")
    print(f"Models: 2")
    total_fits = len(EXPERIMENTS) * len(ABLATIONS) * len(TARGETS) * 2
    print(f"Total model fits: {total_fits}")

    df_all = run_all_ablations(df)

    section("Save full results")
    all_csv = METRICS_DIR / "ablation_all.csv"
    df_all.to_csv(all_csv, index=False)
    print("Saved:", all_csv)

    tables_csv = TABLES_DIR / "ablation_all.csv"
    df_all.to_csv(tables_csv, index=False)
    print("Saved:", tables_csv)

    section("Compute ablation deltas")
    deltas_df = compute_deltas(df_all)
    deltas_csv = METRICS_DIR / "ablation_deltas.csv"
    deltas_df.to_csv(deltas_csv, index=False)
    print("Saved:", deltas_csv)

    print_pair_summary(deltas_df)

    section("Top-5 largest MAE improvements across any (pair, exp, target)")
    top_improved = deltas_df.sort_values("delta_MAE").head(5)
    print(top_improved[[
        "pair", "experiment", "target", "model",
        "MAE_base", "MAE_new", "delta_MAE",
    ]].to_string(index=False))

    section("Top-5 largest MAE degradations")
    top_worse = deltas_df.sort_values("delta_MAE", ascending=False).head(5)
    print(top_worse[[
        "pair", "experiment", "target", "model",
        "MAE_base", "MAE_new", "delta_MAE",
    ]].to_string(index=False))

    section("DONE")
    print("Stage 12 feature-set ablation complete.")


if __name__ == "__main__":
    main()