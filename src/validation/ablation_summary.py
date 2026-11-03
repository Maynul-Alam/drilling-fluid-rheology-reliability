"""
src/validation/ablation_summary.py

Stage 12 -- Supplementary summaries of ablation deltas.

Reads results/metrics/ablation_deltas.csv and prints:
  1. Per-pair per-model summary (mean, median, count, frac improved)
  2. Per-pair per-experiment mean dMAE (RidgeCV)
  3. Per-pair per-experiment mean dMAE (RandomForest)
  4. Per-pair per-target mean dMAE (both models)
  5. Top-5 MAE improvements and degradations

Run from project root:
    python src/validation/ablation_summary.py
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELTAS_PATH = PROJECT_ROOT / "results" / "metrics" / "ablation_deltas.csv"


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    if not DELTAS_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {DELTAS_PATH}\n"
            f"Run src/validation/ablation.py first."
        )

    deltas = pd.read_csv(DELTAS_PATH)

    section("1. Per-pair per-model summary (all targets)")
    grp = deltas.groupby(["pair", "model"])["delta_MAE"].agg(
        mean="mean",
        median="median",
        count="count",
        frac_improved=lambda s: float((s < 0).mean()),
    )
    print(grp.round(4).to_string())

    section("2. Per-pair per-experiment mean delta_MAE (RidgeCV)")
    r = deltas[deltas["model"] == "RidgeCV"]
    by = r.groupby(["pair", "experiment"])["delta_MAE"].agg(
        mean="mean", median="median", count="count",
    )
    print(by.round(4).to_string())

    section("3. Per-pair per-experiment mean delta_MAE (RandomForest)")
    r = deltas[deltas["model"] == "RandomForest"]
    by = r.groupby(["pair", "experiment"])["delta_MAE"].agg(
        mean="mean", median="median", count="count",
    )
    print(by.round(4).to_string())

    section("4a. Per-pair per-target mean delta_MAE (RidgeCV)")
    r = deltas[deltas["model"] == "RidgeCV"]
    by = r.groupby(["pair", "target"])["delta_MAE"].agg(
        mean="mean", median="median", count="count",
    )
    print(by.round(4).to_string())

    section("4b. Per-pair per-target mean delta_MAE (RandomForest)")
    r = deltas[deltas["model"] == "RandomForest"]
    by = r.groupby(["pair", "target"])["delta_MAE"].agg(
        mean="mean", median="median", count="count",
    )
    print(by.round(4).to_string())

    section("5a. Top-5 MAE improvements (delta_MAE most negative)")
    cols = ["pair", "experiment", "target", "model",
            "MAE_base", "MAE_new", "delta_MAE"]
    print(deltas.sort_values("delta_MAE").head(5)[cols].to_string(index=False))

    section("5b. Top-5 MAE degradations (delta_MAE most positive)")
    print(deltas.sort_values("delta_MAE", ascending=False).head(5)[cols].to_string(index=False))

    section("DONE")
    print("Ablation summary complete.")


if __name__ == "__main__":
    main()