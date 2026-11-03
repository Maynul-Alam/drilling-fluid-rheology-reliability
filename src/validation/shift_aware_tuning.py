"""
src/validation/shift_aware_tuning.py
Tests RandomForest tuning with random CV vs group-aware CV.
"""
import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "shift_aware_tuning.csv"

FEATURES = ["Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb", "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb", "NaCl_ppb", "KOH_ppb", "Lubricant_ppb", "Aging_Hours", "Test_Temperature_C"]
TARGET = "PV_cP"
PARAM_GRID = {"n_estimators": [300], "max_depth": [None, 10], "min_samples_leaf": [1, 2]}

def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)
    
    test = df[df["Test_Temperature_C"] == 25].index
    train_pool = df[df["Test_Temperature_C"] != 25].index
    train = train_test_split(train_pool, test_size=0.15, random_state=42)[0]
    
    X_tr, y_tr = df.loc[train, FEATURES], df.loc[train, TARGET]
    X_te, y_te = df.loc[test, FEATURES], df.loc[test, TARGET]
    
    rows = []
    
    # 1. Random CV tuning
    grid_rand = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), 
                             PARAM_GRID, cv=KFold(5), scoring="neg_mean_absolute_error", n_jobs=1)
    grid_rand.fit(X_tr, y_tr)
    mae_rand = mean_absolute_error(y_te, grid_rand.predict(X_te))
    rows.append({"strategy": "RandomCV", "test_mae": mae_rand, "params": str(grid_rand.best_params_)})
    
    # 2. Group-aware CV tuning (grouped by Formulation_Family)
    groups = df.loc[train, "Formulation_Family"]
    grid_group = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), 
                              PARAM_GRID, cv=GroupKFold(5), scoring="neg_mean_absolute_error", n_jobs=1)
    grid_group.fit(X_tr, y_tr, groups=groups)
    mae_group = mean_absolute_error(y_te, grid_group.predict(X_te))
    rows.append({"strategy": "GroupCV", "test_mae": mae_group, "params": str(grid_group.best_params_)})
    
    print("\n=== Shift-Aware Tuning on temp_25 ===")
    out = pd.DataFrame(rows)
    print(out[["strategy", "test_mae"]].to_string(index=False))
    out.to_csv(OUT, index=False)

if __name__ == "__main__":
    main()