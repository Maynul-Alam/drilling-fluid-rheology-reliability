"""
src/validation/repeated_splits.py
Runs 30 random splits to establish a baseline MAE distribution.
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
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "repeated_splits.csv"

FEATURES = ["Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb", "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb", "NaCl_ppb", "KOH_ppb", "Lubricant_ppb", "Aging_Hours", "Test_Temperature_C"]
TARGETS = ["PV_cP", "YP_lb100ft2"]

def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)
    rows = []
    
    for seed in range(30):
        X_train, X_test, y_train_dict, y_test_dict = train_test_split(
            df[FEATURES], df[TARGETS], test_size=0.15, random_state=seed, stratify=df["Formulation_Family"]
        )
        
        for tgt in TARGETS:
            y_tr, y_te = y_train_dict[tgt], y_test_dict[tgt]
            
            # Ridge
            r = Pipeline([("sc", StandardScaler()), ("r", RidgeCV(alphas=np.logspace(-3,3,13)))]).fit(X_train, y_tr)
            rows.append({"seed": seed, "target": tgt, "model": "RidgeCV", "mae": mean_absolute_error(y_te, r.predict(X_test))})
            
            # RF
            rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1).fit(X_train, y_tr)
            rows.append({"seed": seed, "target": tgt, "model": "RandomForest", "mae": mean_absolute_error(y_te, rf.predict(X_test))})
            
            # XGB
            xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1).fit(X_train, y_tr)
            rows.append({"seed": seed, "target": tgt, "model": "XGBoost", "mae": mean_absolute_error(y_te, xgb.predict(X_test))})
            
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    
    # Report distribution
    print("\n=== Baseline MAE Distribution (30 splits) ===")
    summary = out.groupby(["target", "model"])["mae"].agg(["mean", "std", lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)]).reset_index()
    summary.columns = ["Target", "Model", "Mean_MAE", "Std", "P05", "P95"]
    print(summary.round(4).to_string(index=False))

if __name__ == "__main__":
    main()