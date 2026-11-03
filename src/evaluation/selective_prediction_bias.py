"""
src/evaluation/selective_prediction_bias.py

Stage 24 -- What gets discarded when we filter by uncertainty?

For each experiment x model, retain the 50% of test points with smallest
kNN sigma. Report the composition of the retained vs discarded subsets:
mean temperature, mean bentonite, mean barite, mean XCD, mean aging_hours.

Answers: does filtering preferentially discard the extreme cases (bad for
deployment) or a representative sample (good)?

Runtime: ~1-2 minutes.

Output: results/metrics/selective_bias.csv
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "selective_bias.csv"

RANDOM_STATE = 42
CAL_FRACTION = 0.15
KNN_K = 10

FEATURES = [
    "Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb",
    "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb",
    "NaCl_ppb", "KOH_ppb", "Lubricant_ppb",
    "Aging_Hours", "Test_Temperature_C",
]

COMPARE_COLS = ["Test_Temperature_C", "Bentonite_ppb",
                "Barite_ppb", "XCD_Polymer_ppb", "Aging_Hours"]

EXPERIMENTS = {
    "random": {"type": "random"},
    "combo_Base_Bentonite_25": {"type": "combo", "family": "Base_Bentonite", "temp": 25},
    "family_Salt_Tolerant": {"type": "family", "family": "Salt_Tolerant"},
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
    if spec["type"] == "family":
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


def knn_sigma(X_train, X_target, k=KNN_K):
    sc = StandardScaler().fit(X_train)
    Xtr = sc.transform(X_train)
    Xtg = sc.transform(X_target)
    k = min(k, len(Xtr))
    knn = NearestNeighbors(n_neighbors=k).fit(Xtr)
    dist, _ = knn.kneighbors(Xtg)
    return dist.mean(axis=1)


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)

    rows = []
    for exp, spec in EXPERIMENTS.items():
        tr, ca, te = get_split(df, spec)
        X_tr = df.loc[tr, FEATURES]
        X_te = df.loc[te, FEATURES]
        sig = knn_sigma(X_tr, X_te)

        n = len(te)
        k_retain = int(np.ceil(0.5 * n))
        order = np.argsort(sig, kind="stable")
        retain_idx = order[:k_retain]
        discard_idx = order[k_retain:]

        print(f"\n{exp}: test={n}, retained={len(retain_idx)}, discarded={len(discard_idx)}")

        for col in COMPARE_COLS:
            v = df.loc[te, col].values
            retained = v[retain_idx]
            discarded = v[discard_idx]
            rows.append({
                "experiment": exp,
                "column": col,
                "retained_mean": float(np.mean(retained)),
                "discarded_mean": float(np.mean(discarded)),
                "retained_std": float(np.std(retained)),
                "discarded_std": float(np.std(discarded)),
                "retained_min": float(np.min(retained)),
                "retained_max": float(np.max(retained)),
                "discarded_min": float(np.min(discarded)),
                "discarded_max": float(np.max(discarded)),
            })

        # family distribution in retained vs discarded (only meaningful for random)
        if spec["type"] == "random":
            for fam in df["Formulation_Family"].unique():
                fam_mask = (df.loc[te, "Formulation_Family"].values == fam)
                ret_count = int(fam_mask[retain_idx].sum())
                dis_count = int(fam_mask[discard_idx].sum())
                if ret_count + dis_count == 0:
                    continue
                rows.append({
                    "experiment": exp,
                    "column": f"family_{fam}",
                    "retained_mean": float(ret_count),
                    "discarded_mean": float(dis_count),
                    "retained_std": float("nan"),
                    "discarded_std": float("nan"),
                    "retained_min": float("nan"),
                    "retained_max": float("nan"),
                    "discarded_min": float("nan"),
                    "discarded_max": float("nan"),
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nSaved: {OUT}")

    print("\n=== Retained vs discarded summary (per experiment x column) ===")
    print(out[["experiment", "column", "retained_mean", "discarded_mean"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()