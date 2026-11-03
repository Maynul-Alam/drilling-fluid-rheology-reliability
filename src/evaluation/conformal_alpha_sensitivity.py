"""
src/uncertainty/conformal_alpha_sensitivity.py

Stage 24 -- Does conformal coverage collapse depend on the choice of alpha?

For 3 experiments x 5 targets, compute empirical coverage at
alpha = 0.05, 0.10, 0.20 (nominal 95%, 90%, 80%).

Runtime: ~1-2 minutes.

Output: results/metrics/conformal_alpha_sensitivity.csv
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

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "conformal_alpha_sensitivity.csv"

RANDOM_STATE = 42
CAL_FRACTION = 0.15

TARGETS = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2",
           "GS_10min_lb100ft2", "n_FlowIndex"]
ALPHAS = [0.05, 0.10, 0.20]

FEATURES = [
    "Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb",
    "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb",
    "NaCl_ppb", "KOH_ppb", "Lubricant_ppb",
    "Aging_Hours", "Test_Temperature_C",
]

EXPERIMENTS = {
    "random": {"type": "random"},
    "temp_120": {"type": "temp", "temp": 120},
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


def conformal_q(y_cal, pred_cal, alpha):
    n = len(y_cal)
    resid = np.abs(y_cal - pred_cal)
    level = min(1.0, (n + 1) * (1 - alpha) / n)
    return float(np.quantile(resid, level, method="higher"))


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)

    rows = []
    for exp, spec in EXPERIMENTS.items():
        tr, ca, te = get_split(df, spec)
        X_tr = df.loc[tr, FEATURES]
        X_ca = df.loc[ca, FEATURES]
        X_te = df.loc[te, FEATURES]
        print(f"\n{exp}: test={len(te)}")
        for tgt in TARGETS:
            y_tr = df.loc[tr, tgt].values
            y_ca = df.loc[ca, tgt].values
            y_te = df.loc[te, tgt].values
            model = ridge().fit(X_tr, y_tr)
            p_ca = model.predict(X_ca)
            p_te = model.predict(X_te)
            for alpha in ALPHAS:
                q = conformal_q(y_ca, p_ca, alpha)
                lb = p_te - q
                ub = p_te + q
                cov = float(np.mean((y_te >= lb) & (y_te <= ub)))
                rows.append({
                    "experiment": exp,
                    "target": tgt,
                    "alpha": alpha,
                    "nominal_coverage": 1 - alpha,
                    "coverage": cov,
                    "mpiw": float(2 * q),
                })
                print(f"  {tgt:20s} alpha={alpha:.2f} "
                      f"nominal={1-alpha:.2f}  cov={cov:.3f}  mpiw={2*q:.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT}")

    # Compact summary
    print("\n=== Coverage by (experiment, alpha) averaged over targets ===")
    print(out.groupby(["experiment", "alpha"])[["nominal_coverage", "coverage"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()