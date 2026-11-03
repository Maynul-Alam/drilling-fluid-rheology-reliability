import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)

FEATURES = ["Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb", "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb", "NaCl_ppb", "KOH_ppb", "Lubricant_ppb", "Aging_Hours", "Test_Temperature_C"]
TARGET = "PV_cP"

# 1. Fix the Tuning Group (Leave-one-temperature-out)
print("=== 1. Temperature-Grouped Tuning ===")
train_pool = df[df["Test_Temperature_C"] != 25].index
test = df[df["Test_Temperature_C"] == 25].index
X_tr, y_tr = df.loc[train_pool, FEATURES], df.loc[train_pool, TARGET]
X_te, y_te = df.loc[test, FEATURES], df.loc[test, TARGET]
groups = df.loc[train_pool, "Test_Temperature_C"]

grid_temp = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=1), 
                         {"n_estimators": [300], "max_depth": [None, 10], "min_samples_leaf": [1, 2]}, 
                         cv=LeaveOneGroupOut(), scoring="neg_mean_absolute_error", n_jobs=1)
grid_temp.fit(X_tr, y_tr, groups=groups)
print(f"Temp-Grouped Tuned MAE at 25C: {mean_absolute_error(y_te, grid_temp.predict(X_te)):.4f}")
print(f"Best params: {grid_temp.best_params_}")

# 2. Fixed GP (Standardized X and y, ARD Matern)
print("\n=== 2. Fixed Gaussian Process ===")
# Use random split for baseline
tr_idx, te_idx = train_test_split(df.index, test_size=0.15, random_state=42)
X_tr, y_tr = df.loc[tr_idx, FEATURES], df.loc[tr_idx, TARGET]
X_te, y_te = df.loc[te_idx, FEATURES], df.loc[te_idx, TARGET]

sc_X = StandardScaler().fit(X_tr)
sc_y = StandardScaler().fit(np.array(y_tr).reshape(-1, 1))
kernel = ConstantKernel() * Matern(length_scale=np.ones(len(FEATURES)), nu=2.5) + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=2)
gp.fit(sc_X.transform(X_tr), sc_y.transform(np.array(y_tr).reshape(-1, 1)).ravel())
gp_preds = sc_y.inverse_transform(gp.predict(sc_X.transform(X_te)).reshape(-1, 1)).ravel()
print(f"Fixed GP Random Split MAE: {mean_absolute_error(y_te, gp_preds):.4f}")

# GP at temp 25
gp_train_idx = df[df["Test_Temperature_C"] != 25].index
gp_test_idx = df[df["Test_Temperature_C"] == 25].index
X_gtr, y_gtr = df.loc[gp_train_idx, FEATURES], df.loc[gp_train_idx, TARGET]
X_gte, y_gte = df.loc[gp_test_idx, FEATURES], df.loc[gp_test_idx, TARGET]
sc_X_g = StandardScaler().fit(X_gtr)
sc_y_g = StandardScaler().fit(np.array(y_gtr).reshape(-1, 1))
gp2 = GaussianProcessRegressor(kernel=kernel, random_state=42)
gp2.fit(sc_X_g.transform(X_gtr), sc_y_g.transform(np.array(y_gtr).reshape(-1, 1)).ravel())
gp_preds_25 = sc_y_g.inverse_transform(gp2.predict(sc_X_g.transform(X_gte)).reshape(-1, 1)).ravel()
print(f"Fixed GP 25C MAE: {mean_absolute_error(y_gte, gp_preds_25):.4f}")

# 3. LightGBM Ablation (Standard vs Linear leaves)
print("\n=== 3. LightGBM Ablation ===")
lgb_std = lgb.LGBMRegressor(n_estimators=100, linear_tree=False, random_state=42, verbose=-1).fit(X_tr, y_tr)
lgb_lin = lgb.LGBMRegressor(n_estimators=100, linear_tree=True, random_state=42, verbose=-1).fit(X_tr, y_tr)
print(f"LGBM Standard Random MAE: {mean_absolute_error(y_te, lgb_std.predict(X_te)):.4f}")
print(f"LGBM Linear Random MAE: {mean_absolute_error(y_te, lgb_lin.predict(X_te)):.4f}")

lgb_std_25 = lgb.LGBMRegressor(n_estimators=100, linear_tree=False, random_state=42, verbose=-1).fit(X_gtr, y_gtr)
lgb_lin_25 = lgb.LGBMRegressor(n_estimators=100, linear_tree=True, random_state=42, verbose=-1).fit(X_gtr, y_gtr)
print(f"LGBM Standard 25C MAE: {mean_absolute_error(y_gte, lgb_std_25.predict(X_gte)):.4f}")
print(f"LGBM Linear 25C MAE: {mean_absolute_error(y_gte, lgb_lin_25.predict(X_gte)):.4f}")

# 4. Selective Prediction Exact Counts
print("\n=== 4. Selective Prediction Bias ===")
from sklearn.neighbors import NearestNeighbors
sc = StandardScaler().fit(X_tr)
knn = NearestNeighbors(n_neighbors=10).fit(sc.transform(X_tr))
dist, _ = knn.kneighbors(sc.transform(X_te))
sig = dist.mean(axis=1)
retain_idx = np.argsort(sig)[:46] # 50% of 93
discard_idx = np.argsort(sig)[46:]

ret_fams = df.loc[te_idx[retain_idx], "Formulation_Family"].value_counts()
dis_fams = df.loc[te_idx[discard_idx], "Formulation_Family"].value_counts()
print("Retained families:\n", ret_fams.to_string())
print("\nDiscarded families:\n", dis_fams.to_string())