"""
src/validation/extrapolation_controls.py
Tests Physics Baseline, GP, and LightGBM linear-tree under shift.
"""
import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
OUT = ROOT / "results" / "metrics" / "extrapolation_controls.csv"

FEATURES = ["Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb", "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb", "NaCl_ppb", "KOH_ppb", "Lubricant_ppb", "Aging_Hours", "Test_Temperature_C"]
EXPERIMENTS = {"random": None, "temp_25": 25, "temp_120": 120}
TARGET = "PV_cP"

def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)
    
    # Add physics feature: Arrhenius term (1/T)
    df["Inv_Temp"] = 1.0 / (df["Test_Temperature_C"] + 273.15)
    
    rows = []
    
    for exp_name, temp in EXPERIMENTS.items():
        if temp is None:
            pool, test = train_test_split(df.index, test_size=0.15, random_state=42, stratify=df["Formulation_Family"])
        else:
            test = df[df["Test_Temperature_C"] == temp].index
            pool = df[df["Test_Temperature_C"] != temp].index
            
        train = pool # We skip cal set for point predictions here to save time
        
        for model_name in ["Physics_Linear", "GaussianProcess", "LightGBM_LinearTree"]:
            if model_name == "Physics_Linear":
                # Linear composition + Arrhenius temperature
                feat_cols = FEATURES[:-1] + ["Inv_Temp"]
                model = LinearRegression().fit(df.loc[train, feat_cols], df.loc[train, TARGET])
                preds = model.predict(df.loc[test, feat_cols])
                
            elif model_name == "GaussianProcess":
                # GP with RBF kernel (standardized inputs)
                sc = StandardScaler().fit(df.loc[train, FEATURES])
                kernel = ConstantKernel() * RBF(length_scale=np.ones(len(FEATURES))) + WhiteKernel()
                model = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=2)
                model.fit(sc.transform(df.loc[train, FEATURES]), df.loc[train, TARGET])
                preds = model.predict(sc.transform(df.loc[test, FEATURES]))
                
            elif model_name == "LightGBM_LinearTree":
                # Linear tree
                model = lgb.LGBMRegressor(n_estimators=100, linear_tree=True, random_state=42, verbose=-1)
                model.fit(df.loc[train, FEATURES], df.loc[train, TARGET])
                preds = model.predict(df.loc[test, FEATURES])
                
            mae = mean_absolute_error(df.loc[test, TARGET], preds)
            rows.append({"experiment": exp_name, "model": model_name, "mae": mae})
            print(f"{exp_name} | {model_name:20s} MAE: {mae:.4f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()