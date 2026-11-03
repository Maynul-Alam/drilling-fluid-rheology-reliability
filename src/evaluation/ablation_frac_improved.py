"""
src/evaluation/ablation_frac_improved.py

Recovers frac_improved (fraction of cells where new set beats base set)
for the 21-experiment scope from ci_ablation_deltas.csv.
"""
import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "results" / "metrics" / "ci_ablation_deltas.csv")

grp = df.groupby(["pair", "model"]).agg(
    n_cells=("delta_point", "count"),
    delta_mean=("delta_point", "mean"),
    delta_median=("delta_point", "median"),
    frac_improved=("delta_point", lambda s: float((s < 0).mean())),
    frac_ci_sig=("delta_ci_high", lambda s: float((s < 0).mean())),
)
print(grp.round(4).to_string())