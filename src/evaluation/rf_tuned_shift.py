"""
src/evaluation/rf_tuned_shift.py

Stage 24 -- Does tuning RandomForest rescue its extrapolation behaviour?

Tunes RF by 5-fold grid search on the random-split training set, then
evaluates the tuned model on the same shift cells where default RF failed.

Runtime: ~3-5 minutes.

Output: results/metrics/rf_tuned_shift.csv
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "rf_tuned_shift.csv"

RANDOM_STATE = 42
CAL_FRACTION = 0.15

TARGETS = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2",
           "GS_10min_lb100ft2", "n_FlowIndex"]
FEATURES = [
    "Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb",
    "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb",
    "NaCl_ppb", "KOH_ppb", "Lubricant_ppb",
    "Aging_Hours", "Test_Temperature_C",
]

PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}

EXPERIMENTS = {
    "random": {"type": "random"},
    "temp_120": {"type": "temp", "temp": 120},
    "combo_Base_Bentonite_25": {"type": "combo", "family": "Base_Bentonite", "temp": 25},
}


def get_split(df, spec):
    if spec["type"] == "random":
        train_pool, te = train_test_split(
            df.index.to_numpy(), test_size=0.15,
            random_state=RANDOM_STATE, stratify=df["Formulation_Family"],
        )
        strata = df.loc[train_pool, "Formulation_Family"]
        tr, ca = train_test_split(train_pool, test_size=CAL_FRACTION,
                                  random_state=RANDOM_STATE, stratify=strata)
        return np.array(tr), np.array(ca), np.array(te)
    
    # FIX APPLIED HERE
    if spec["type"] == "temp":
        m = df["Test_Temperature_C"] == spec["temp"]
    elif spec["type"] == "family":
        m = df["Formulation_Family"] == spec["family"]
    else:
        m = (df["Formulation_Family"] == spec["family"]) & \
            (df["Test_Temperature_C"] == spec["temp"])
        
    te = df.index[m].to_numpy()
    pool = df.index[~m].to_numpy()
    strata = df.loc[pool, "Formulation_Family"]
    tr, ca = train_test_split(pool, test_size=CAL_FRACTION,
                              random_state=RANDOM_STATE, stratify=strata)
    return np.array(tr), np.array(ca), te


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for exp, spec in EXPERIMENTS.items():
        tr, ca, te = get_split(df, spec)
        X_tr = df.loc[tr, FEATURES]
        X_te = df.loc[te, FEATURES]
        print(f"\n{exp}: train={len(tr)} test={len(te)}")

        for tgt in TARGETS:
            y_tr = df.loc[tr, tgt].values
            y_te = df.loc[te, tgt].values

            rf_def = RandomForestRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
            ).fit(X_tr, y_tr)
            mae_def = float(mean_absolute_error(y_te, rf_def.predict(X_te)))

            grid = GridSearchCV(
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
                PARAM_GRID, cv=inner_cv,
                scoring="neg_mean_absolute_error",
                n_jobs=1, refit=True,
            ).fit(X_tr, y_tr)
            mae_tuned = float(mean_absolute_error(y_te, grid.predict(X_te)))

            rows.append({
                "experiment": exp,
                "target": tgt,
                "n_test": len(te),
                "mae_default": mae_def,
                "mae_tuned": mae_tuned,
                "delta": mae_tuned - mae_def,
                "best_params": str(grid.best_params_),
            })
            print(f"  {tgt:20s} default={mae_def:.4f}  tuned={mae_tuned:.4f}  "
                  f"delta={mae_tuned-mae_def:+.4f}  best={grid.best_params_}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT}")
    print("\nMean delta (tuned - default):")
    print(out.groupby("experiment")["delta"].agg(["mean", "median", "count"]))


if __name__ == "__main__":
    main()