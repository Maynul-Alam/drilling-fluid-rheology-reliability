"""
src/validation/combo_grid_extension.py

Stage 24 -- Extend combined-shift grid to all (family, temperature) cells
with >= 15 observations.

Runtime: ~3-4 minutes.

Output: results/metrics/combo_grid_extended.csv
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

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "combo_grid_extended.csv"

RANDOM_STATE = 42
CAL_FRACTION = 0.15
MIN_CELL_N = 15

TARGETS = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2",
           "GS_10min_lb100ft2", "n_FlowIndex"]
FEATURES = [
    "Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb",
    "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb",
    "NaCl_ppb", "KOH_ppb", "Lubricant_ppb",
    "Aging_Hours", "Test_Temperature_C",
]


def ridge():
    return Pipeline([
        ("sc", StandardScaler()),
        ("r", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])


def rf():
    return RandomForestRegressor(n_estimators=300,
                                 random_state=RANDOM_STATE, n_jobs=1)


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)

    # Identify all (family, temperature) cells with >= MIN_CELL_N
    ct = pd.crosstab(df["Formulation_Family"], df["Test_Temperature_C"])
    cells = []
    for fam in ct.index:
        for t in ct.columns:
            if ct.loc[fam, t] >= MIN_CELL_N:
                cells.append((fam, int(t), int(ct.loc[fam, t])))

    print(f"Combined-shift cells with >= {MIN_CELL_N} observations:")
    for c in cells:
        print(f"  {c[0]:25s} @ {c[1]:>3} C   n={c[2]}")

    rows = []
    for fam, t, n in cells:
        m = (df["Formulation_Family"] == fam) & (df["Test_Temperature_C"] == t)
        te = df.index[m].to_numpy()
        pool = df.index[~m].to_numpy()
        strata = df.loc[pool, "Formulation_Family"]
        tr, ca = train_test_split(pool, test_size=CAL_FRACTION,
                                  random_state=RANDOM_STATE, stratify=strata)
        tr = np.array(tr)

        X_tr = df.loc[tr, FEATURES]
        X_te = df.loc[te, FEATURES]

        for tgt in TARGETS:
            y_tr = df.loc[tr, tgt].values
            y_te = df.loc[te, tgt].values
            for name, build in [("RidgeCV", ridge), ("RandomForest", rf)]:
                model = build().fit(X_tr, y_tr)
                p = model.predict(X_te)
                rows.append({
                    "family": fam,
                    "temperature": t,
                    "n_test": n,
                    "target": tgt,
                    "model": name,
                    "mae": float(mean_absolute_error(y_te, p)),
                })
        print(f"  {fam}@{t}: done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT}")

    print("\n=== PV MAE by combined cell (RidgeCV) ===")
    pv = out[(out["target"] == "PV_cP") & (out["model"] == "RidgeCV")]
    print(pv[["family", "temperature", "n_test", "mae"]]
          .sort_values("mae", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()