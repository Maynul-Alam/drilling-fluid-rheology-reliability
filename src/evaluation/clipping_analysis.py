"""
src/evaluation/clipping_analysis.py

Stage 24 -- Sensitivity of MAE to clipping negative predictions to zero.

For the five worst-negatives experiments, fit RidgeCV, predict, and
compare MAE(raw) vs MAE(clipped at 0) on the same test set.

Runtime: ~2-3 minutes.

Output: results/metrics/clipping_analysis.csv
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
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "clipping_analysis.csv"

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
    "temp_120": {"type": "temp", "temp": 120},
    "family_Salt_Tolerant": {"type": "family", "family": "Salt_Tolerant"},
    "combo_KCl_Polymer_90": {"type": "combo", "family": "KCl_Polymer", "temp": 90},
    "combo_Base_Bentonite_25": {"type": "combo", "family": "Base_Bentonite", "temp": 25},
    "family_KCl_Polymer": {"type": "family", "family": "KCl_Polymer"},
}


def get_split(df, spec):
    if spec["type"] == "temp":
        m = df["Test_Temperature_C"] == spec["temp"]
    elif spec["type"] == "family":
        m = df["Formulation_Family"] == spec["family"]
    else:
        m = (df["Formulation_Family"] == spec["family"]) & \
            (df["Test_Temperature_C"] == spec["temp"])
    test_idx = df.index[m].to_numpy()
    pool = df.index[~m].to_numpy()
    strata = df.loc[pool, "Formulation_Family"]
    tr, ca = train_test_split(pool, test_size=CAL_FRACTION,
                              random_state=RANDOM_STATE, stratify=strata)
    return np.array(tr), np.array(ca), test_idx


def ridge():
    return Pipeline([
        ("sc", StandardScaler()),
        ("r", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])


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
            model = ridge().fit(X_tr, y_tr)
            p = model.predict(X_te)
            p_clipped = np.maximum(p, 0.0)

            mae_raw = float(mean_absolute_error(y_te, p))
            mae_clip = float(mean_absolute_error(y_te, p_clipped))
            n_neg = int((p < 0).sum())

            rows.append({
                "experiment": exp,
                "target": tgt,
                "n_test": len(te),
                "n_negative": n_neg,
                "mae_raw": mae_raw,
                "mae_clipped": mae_clip,
                "delta_mae": mae_clip - mae_raw,
                "rel_improvement": (mae_raw - mae_clip) / mae_raw if mae_raw > 0 else float("nan"),
            })
            print(f"  {tgt:20s} n_neg={n_neg:>3}  MAE raw={mae_raw:8.4f}  "
                  f"clipped={mae_clip:8.4f}  delta={mae_clip - mae_raw:+.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT}")

    print("\n--- Summary across cells with n_negative > 0 ---")
    sub = out[out["n_negative"] > 0]
    if len(sub) > 0:
        print(f"  cells with negatives: {len(sub)}")
        print(f"  mean relative MAE improvement from clipping: "
              f"{sub['rel_improvement'].mean():.3%}")
        print(f"  max relative improvement: {sub['rel_improvement'].max():.3%}")
        print(f"  cells where clipping helps: "
              f"{(sub['delta_mae'] < 0).sum()} / {len(sub)}")


if __name__ == "__main__":
    main()