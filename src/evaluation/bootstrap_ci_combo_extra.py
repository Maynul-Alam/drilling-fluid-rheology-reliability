"""
src/evaluation/bootstrap_ci_combo_extra.py

Supplementary Stage 13 -- bootstrap CIs for the three combo experiments
that were not in FOCUS_SHIFT of the main bootstrap_ci.py run:
    combo_Base_Bentonite_50
    combo_Polymer_Enhanced_75
    combo_High_Density_60

Reads nothing, computes fresh. Writes:
    results/metrics/ci_shift_deltas_extra.csv
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
SHEET = "Full_Dataset"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CAL_FRACTION_OF_POOL = 0.15
N_BOOT = 1000
ALPHA = 0.05

TARGETS = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2",
           "GS_10min_lb100ft2", "n_FlowIndex"]

FEATURES_STAGE6 = [
    "Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb",
    "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb",
    "NaCl_ppb", "KOH_ppb", "Lubricant_ppb",
    "Aging_Hours", "Test_Temperature_C",
]

COMBOS = {
    "combo_Base_Bentonite_50":     {"family": "Base_Bentonite", "temp": 50},
    "combo_Polymer_Enhanced_75":   {"family": "Polymer_Enhanced", "temp": 75},
    "combo_High_Density_60":       {"family": "High_Density", "temp": 60},
}


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def build_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])


def build_rf():
    return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1)


MODELS = [("RidgeCV", build_ridge), ("RandomForest", build_rf)]


def random_split_indices(df):
    """The exact random-split used in bootstrap_ci.py FOCUS_SHIFT reference."""
    all_idx = df.index.to_numpy()
    train_pool, test_idx = train_test_split(
        all_idx, test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df["Formulation_Family"],
    )
    return np.array(train_pool), np.array(test_idx)


def combo_split_indices(df, family, temp):
    m_test = (df["Formulation_Family"] == family) & (df["Test_Temperature_C"] == temp)
    m_pool = (df["Formulation_Family"] != family) & (df["Test_Temperature_C"] != temp)
    test_idx = df.index[m_test].to_numpy()
    pool = df.index[m_pool].to_numpy()
    strata = df.loc[pool, "Formulation_Family"]
    tr, ca = train_test_split(
        pool, test_size=CAL_FRACTION_OF_POOL,
        random_state=RANDOM_STATE, stratify=strata,
    )
    return np.array(tr), np.array(ca), test_idx


def mae_fn(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def bootstrap_delta(y_ref, pred_ref, y_shift, pred_shift, B=N_BOOT, alpha=ALPHA):
    rng_ref = np.random.default_rng(RANDOM_STATE)
    rng_sh = np.random.default_rng(RANDOM_STATE + 1)
    n_ref, n_sh = len(y_ref), len(y_shift)
    mae_ref_b = np.empty(B)
    mae_sh_b = np.empty(B)
    for b in range(B):
        i_ref = rng_ref.integers(0, n_ref, n_ref)
        i_sh = rng_sh.integers(0, n_sh, n_sh)
        mae_ref_b[b] = mae_fn(y_ref[i_ref], pred_ref[i_ref])
        mae_sh_b[b] = mae_fn(y_shift[i_sh], pred_shift[i_sh])
    d = mae_sh_b - mae_ref_b
    return {
        "mae_ref_point": mae_fn(y_ref, pred_ref),
        "mae_shift_point": mae_fn(y_shift, pred_shift),
        "delta_point": mae_fn(y_shift, pred_shift) - mae_fn(y_ref, pred_ref),
        "delta_ci_low": float(np.quantile(d, alpha / 2)),
        "delta_ci_high": float(np.quantile(d, 1 - alpha / 2)),
        "p_delta_gt_zero": float((d > 0).mean()),
        "n_ref": n_ref,
        "n_shift": n_sh,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET).reset_index(drop=True)

    # Fit random reference once per (target, model)
    section("Fitting random reference predictions")
    ref_train, ref_test = random_split_indices(df)
    ref_preds = {}
    X_ref_tr = df.loc[ref_train, FEATURES_STAGE6]
    X_ref_te = df.loc[ref_test, FEATURES_STAGE6]
    for tgt in TARGETS:
        y_tr = df.loc[ref_train, tgt].values
        y_te = df.loc[ref_test, tgt].values
        for name, build in MODELS:
            model = build().fit(X_ref_tr, y_tr)
            ref_preds[(tgt, name)] = (y_te, model.predict(X_ref_te))
        print(f"  random reference: {tgt} done")

    rows = []
    for exp_name, cfg in COMBOS.items():
        section(f"Combo: {exp_name}")
        tr, ca, te = combo_split_indices(df, cfg["family"], cfg["temp"])
        X_tr = df.loc[tr, FEATURES_STAGE6]
        X_te = df.loc[te, FEATURES_STAGE6]
        print(f"  train={len(tr)}  cal={len(ca)}  test={len(te)}")

        for tgt in TARGETS:
            y_tr = df.loc[tr, tgt].values
            y_te = df.loc[te, tgt].values
            y_ref, p_ref = ref_preds[(tgt, "RidgeCV")]
            for name, build in MODELS:
                model = build().fit(X_tr, y_tr)
                p_te = model.predict(X_te)
                y_ref, p_ref = ref_preds[(tgt, name)]
                d = bootstrap_delta(y_ref, p_ref, y_te, p_te)
                rows.append({
                    "experiment": exp_name,
                    "target": tgt,
                    "model": name,
                    "n_test_shift": d["n_shift"],
                    "n_test_random": d["n_ref"],
                    "mae_random_point": d["mae_ref_point"],
                    "mae_shift_point": d["mae_shift_point"],
                    "delta_point": d["delta_point"],
                    "delta_ci_low": d["delta_ci_low"],
                    "delta_ci_high": d["delta_ci_high"],
                    "p_delta_gt_zero": d["p_delta_gt_zero"],
                })
            print(f"    {tgt}: done")

    out = pd.DataFrame(rows)
    out_csv = METRICS_DIR / "ci_shift_deltas_extra.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8")
    section(f"Saved: {out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()