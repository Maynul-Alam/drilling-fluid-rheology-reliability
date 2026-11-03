"""
src/evaluation/av_invariant_direct.py

Stage 24 -- Direct test of the AV invariant: |pred_AV_composed - pred_AV_direct|.

Tests whether independently-trained PV and YP predictions, composed as
PV + YP/2, agree with an independently-trained AV model's predictions
on the SAME test points.

Runtime: ~30 seconds.

Output: results/metrics/av_invariant_direct.csv
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

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "av_invariant_direct.csv"

RANDOM_STATE = 42
CAL_FRACTION = 0.15

FEATURES = [
    "Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb",
    "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb",
    "NaCl_ppb", "KOH_ppb", "Lubricant_ppb",
    "Aging_Hours", "Test_Temperature_C",
]

EXPERIMENTS = {
    "random": {"type": "random"},
    "temp_120": {"type": "temp", "temp": 120},
    "family_Salt_Tolerant": {"type": "family", "family": "Salt_Tolerant"},
    "combo_KCl_Polymer_90": {"type": "combo", "family": "KCl_Polymer", "temp": 90},
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
    else:
        m = (df["Formulation_Family"] == spec["family"]) & \
            (df["Test_Temperature_C"] == spec["temp"])
    te = df.index[m].to_numpy()
    pool = df.index[~m].to_numpy()
    strata = df.loc[pool, "Formulation_Family"]
    tr, ca = train_test_split(pool, test_size=CAL_FRACTION,
                              random_state=RANDOM_STATE, stratify=strata)
    return np.array(tr), np.array(ca), te


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

    rows = []
    for exp, spec in EXPERIMENTS.items():
        tr, ca, te = get_split(df, spec)
        X_tr = df.loc[tr, FEATURES]
        X_te = df.loc[te, FEATURES]
        print(f"\n{exp}: test={len(te)}")

        for model_name, build in [("RidgeCV", ridge), ("RandomForest", rf)]:
            m_pv = build().fit(X_tr, df.loc[tr, "PV_cP"].values)
            m_yp = build().fit(X_tr, df.loc[tr, "YP_lb100ft2"].values)
            m_av = build().fit(X_tr, df.loc[tr, "AV_cP"].values)

            av_composed = m_pv.predict(X_te) + 0.5 * m_yp.predict(X_te)
            av_direct = m_av.predict(X_te)

            diff = av_composed - av_direct
            rows.append({
                "experiment": exp,
                "model": model_name,
                "n_test": len(te),
                "diff_median": float(np.median(np.abs(diff))),
                "diff_p95": float(np.percentile(np.abs(diff), 95)),
                "diff_max": float(np.max(np.abs(diff))),
                "corr_pearson": float(np.corrcoef(av_composed, av_direct)[0, 1]),
            })
            print(f"  {model_name:15s} |composed - direct| : "
                  f"median={rows[-1]['diff_median']:.3f}  "
                  f"p95={rows[-1]['diff_p95']:.3f}  "
                  f"max={rows[-1]['diff_max']:.3f}  "
                  f"corr={rows[-1]['corr_pearson']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()