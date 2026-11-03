"""
src/validation/xgboost_shift.py

Stage 24 -- XGBoost under the full shift grid.

Adds a third model family (gradient-boosted trees) to address reviewer
critique M1.

Runtime: ~5-10 minutes depending on single-threaded speed.

Output: results/metrics/xgboost_shift.csv
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "xgboost_shift.csv"

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

EXPERIMENTS = {
    "random": {"type": "random"},
    "temp_25": {"type": "temp", "temp": 25},
    "temp_40": {"type": "temp", "temp": 40},
    "temp_50": {"type": "temp", "temp": 50},
    "temp_60": {"type": "temp", "temp": 60},
    "temp_75": {"type": "temp", "temp": 75},
    "temp_90": {"type": "temp", "temp": 90},
    "temp_120": {"type": "temp", "temp": 120},
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
    if spec["type"] == "temp":
        m = df["Test_Temperature_C"] == spec["temp"]
    elif spec["type"] == "family":
        m = df["Formulation_Family"] == spec["family"]
    elif spec["type"] == "temporal":
        dates = pd.to_datetime(df["Test_Date"], errors="coerce")
        te = df.index[dates.dt.year == 2024].to_numpy()
        pool = df.index[dates.dt.year == 2023].to_numpy()
        strata = df.loc[pool, "Formulation_Family"]
        tr, ca = train_test_split(pool, test_size=CAL_FRACTION,
                                  random_state=RANDOM_STATE, stratify=strata)
        return np.array(tr), np.array(ca), te
    else:
        m = (df["Formulation_Family"] == spec["family"]) & \
            (df["Test_Temperature_C"] == spec["temp"])
    te = df.index[m].to_numpy()
    pool = df.index[~m].to_numpy()
    strata = df.loc[pool, "Formulation_Family"]
    tr, ca = train_test_split(pool, test_size=CAL_FRACTION,
                              random_state=RANDOM_STATE, stratify=strata)
    return np.array(tr), np.array(ca), te


def xgb():
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        random_state=RANDOM_STATE, n_jobs=1,
        tree_method="exact", verbosity=0,
    )


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)

    rows = []
    for exp, spec in EXPERIMENTS.items():
        tr, ca, te = get_split(df, spec)
        X_tr = df.loc[tr, FEATURES]
        X_te = df.loc[te, FEATURES]
        print(f"\n{exp}: train={len(tr)} test={len(te)}")

        for tgt in TARGETS:
            y_tr = df.loc[tr, tgt].values
            y_te = df.loc[te, tgt].values
            model = xgb().fit(X_tr, y_tr)
            p = model.predict(X_te)
            rows.append({
                "experiment": exp,
                "target": tgt,
                "n_test": len(te),
                "model": "XGBoost",
                "mae": float(mean_absolute_error(y_te, p)),
                "negative_preds": int((p < 0).sum()),
            })
            print(f"  {tgt:20s} MAE={rows[-1]['mae']:.4f}  neg={rows[-1]['negative_preds']}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT}")

    # Compare to RidgeCV / RandomForest on the same experiments
    print("\n=== PV MAE: Ridge / RF / XGB across shift experiments ===")
    pv = out[(out["target"] == "PV_cP")]
    print(pv[["experiment", "mae"]].to_string(index=False))


if __name__ == "__main__":
    main()