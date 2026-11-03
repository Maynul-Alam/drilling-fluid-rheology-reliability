import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures" / "paper"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight"
})

path = METRICS_DIR / "shift_summary.csv"
if not path.exists():
    print(f"Error: {path} not found.")
else:
    df = pd.read_csv(path)
    df = df[df["model"] == "RidgeCV"]

    order_exp = ["temp_25", "temp_40", "temp_50", "temp_60", "temp_75", "temp_90", "temp_120",
                 "family_Base_Bentonite", "family_High_Density", "family_KCl_Polymer", 
                 "family_Low_Solid", "family_Polymer_Enhanced", "family_Salt_Tolerant",
                 "temporal_2023_to_2024", "combo_Base_Bentonite_25", "combo_Base_Bentonite_50",
                 "combo_KCl_Polymer_90", "combo_High_Density_60", "combo_Polymer_Enhanced_75"]
    targets = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2", "GS_10min_lb100ft2", "n_FlowIndex"]
    
    present = df["experiment"].unique()
    order_exp = [e for e in order_exp if e in present]

    matrix = np.full((len(targets), len(order_exp)), np.nan)
    for i, t in enumerate(targets):
        for j, e in enumerate(order_exp):
            row = df[(df["target"] == t) & (df["experiment"] == e)]
            if len(row) > 0:
                matrix[i, j] = row["MAE_ratio"].values[0]

    # Make the figure slightly taller to accommodate the rotated labels
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0.5, vmax=2.5)
    
    short_names = [e.replace("family_", "").replace("combo_", "").replace("temp_", "T=").replace("_lb100ft2", "").replace("_cP", "") for e in order_exp]
    ax.set_xticks(range(len(order_exp)))
    
    # FIX: Rotate 45 degrees, align to the right, and anchor properly
    ax.set_xticklabels(short_names, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(["PV", "YP", "GS 10s", "GS 10min", "n"])

    for i in range(len(targets)):
        for j in range(len(order_exp)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if (v < 0.85 or v > 2.0) else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("MAE ratio (experiment / random)")
    ax.set_title("RidgeCV: MAE degradation under distribution shift")
    
    # Add extra padding at the bottom for the labels
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(FIGURES_DIR / "fig1_shift_heatmap.png")
    print(f"Saved: {FIGURES_DIR / 'fig1_shift_heatmap.png'}")