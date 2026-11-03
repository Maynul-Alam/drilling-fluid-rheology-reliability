import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="Full_Dataset").reset_index(drop=True)

FEATURES = ["Bentonite_ppb", "Barite_ppb", "CMC_HV_ppb", "CMC_LV_ppb", "PAC_R_ppb", "PAC_L_ppb", "XCD_Polymer_ppb", "KCl_ppb", "NaCl_ppb", "KOH_ppb", "Lubricant_ppb", "Aging_Hours", "Test_Temperature_C"]
TARGET = "PV_cP"
df["Inv_Temp"] = 1.0 / (df["Test_Temperature_C"] + 273.15)

print("=== Missing Table 2 Cells & CIs ===")

# 1. Temp 120 (Interior Gap)
train_120 = df[df["Test_Temperature_C"] != 120].index
test_120 = df[df["Test_Temperature_C"] == 120].index
X_tr, y_tr = df.loc[train_120, FEATURES], df.loc[train_120, TARGET]
X_te, y_te = df.loc[test_120, FEATURES], df.loc[test_120, TARGET]

# LGBM Standard
lgb_std = lgb.LGBMRegressor(n_estimators=100, linear_tree=False, random_state=42, verbose=-1).fit(X_tr, y_tr)
print(f"LGBM Standard 120C MAE: {mean_absolute_error(y_te, lgb_std.predict(X_te)):.4f}")

# GP
sc_X = StandardScaler().fit(X_tr)
sc_y = StandardScaler().fit(np.array(y_tr).reshape(-1, 1))
kernel = ConstantKernel() * Matern(length_scale=np.ones(len(FEATURES)), nu=2.5) + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(sc_X.transform(X_tr), sc_y.transform(np.array(y_tr).reshape(-1, 1)).ravel())
gp_preds = sc_y.inverse_transform(gp.predict(sc_X.transform(X_te)).reshape(-1, 1)).ravel()
print(f"GP 120C MAE: {mean_absolute_error(y_te, gp_preds):.4f}")

# 2. Bootstrap CIs for 25C Extrapolation (to prove significance)
train_25 = df[df["Test_Temperature_C"] != 25].index
test_25 = df[df["Test_Temperature_C"] == 25].index
X_25_tr, y_25_tr = df.loc[train_25, FEATURES], df.loc[train_25, TARGET]
X_25_te, y_25_te = df.loc[test_25, FEATURES], df.loc[test_25, TARGET]

# Physics Baseline
phys = LinearRegression().fit(df.loc[train_25, FEATURES[:-1] + ["Inv_Temp"]], y_25_tr)
pred_phys = phys.predict(df.loc[test_25, FEATURES[:-1] + ["Inv_Temp"]])

# GP
sc_X_25 = StandardScaler().fit(X_25_tr)
sc_y_25 = StandardScaler().fit(np.array(y_25_tr).reshape(-1, 1))
gp_25 = GaussianProcessRegressor(kernel=kernel, random_state=42).fit(sc_X_25.transform(X_25_tr), sc_y_25.transform(np.array(y_25_tr).reshape(-1, 1)).ravel())
pred_gp = sc_y_25.inverse_transform(gp_25.predict(sc_X_25.transform(X_25_te)).reshape(-1, 1)).ravel()

# Bootstrap
rng = np.random.default_rng(42)
def get_ci(y_true, y_pred, B=1000):
    mae_b = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        mae_b.append(mean_absolute_error(y_true.values[idx], y_pred[idx]))
    return np.percentile(mae_b, 2.5), np.percentile(mae_b, 97.5)

ci_phys = get_ci(y_25_te, pred_phys)
ci_gp = get_ci(y_25_te, pred_gp)
print(f"Physics Baseline 25C MAE CI: [{ci_phys[0]:.2f}, {ci_phys[1]:.2f}]")
print(f"GP 25C MAE CI: [{ci_gp[0]:.2f}, {ci_gp[1]:.2f}]")

# GP Length Scale
print("\nGP Learned Length Scales:")
for f, l in zip(FEATURES, gp_25.kernel_.k1.k2.length_scale):
    print(f"  {f}: {l:.2f}")