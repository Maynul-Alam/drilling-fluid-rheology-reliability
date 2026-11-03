"""
src/evaluation/interpretability.py

Stage 11 -- Interpretability and stability of feature importance
under distribution shift.

Purpose
-------
Permutation importance on the test set of each experiment, so we can ask
two questions that matter for the paper:

  1. Which features drive predictions of PV, YP, GS 10s?
  2. Does that ranking change under distribution shift?

The second question is the scientifically novel one. If a model relies on
Test_Temperature_C on exchangeable data but on Barite_ppb under temperature
shift, that tells us something about what the model is actually learning.

Experiments
-----------
  random                    reference
  temp_25, temp_50, temp_120    temperature shift
  family_Base_Bentonite, family_KCl_Polymer,
  family_Polymer_Enhanced, family_Salt_Tolerant   family shift
  temporal_2023_to_2024     temporal shift

Targets
-------
  PV_cP, YP_lb100ft2, GS_10s_lb100ft2

Models
------
  RidgeCV
  RandomForest (200 trees, fixed config)

Method
------
  sklearn.inspection.permutation_importance on the held-out test set of
  each experiment, scoring = neg_mean_absolute_error, n_repeats = 15.
  Higher importance = larger MAE increase when the feature is permuted.

Outputs
-------
  results/metrics/interpretability_importance.csv
  results/metrics/interpretability_ranks.csv
  results/metrics/interpretability_stability.csv
  results/figures/interpretability_<target>_<model>_heatmap.png
  results/figures/interpretability_top_features_<target>.png

Run
---
  python src/evaluation/interpretability.py
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WBM_Rheological_Dataset_2023_2024.xlsx"
SHEET = "Full_Dataset"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
for d in [METRICS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CAL_FRACTION_OF_POOL = 0.15
N_REPEATS = 15

TARGETS = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2"]

FEATURES = [
    "Bentonite_ppb",
    "Barite_ppb",
    "CMC_HV_ppb",
    "CMC_LV_ppb",
    "PAC_R_ppb",
    "PAC_L_ppb",
    "XCD_Polymer_ppb",
    "KCl_ppb",
    "NaCl_ppb",
    "KOH_ppb",
    "Lubricant_ppb",
    "Aging_Hours",
    "Test_Temperature_C",
]

EXPERIMENTS = {
    "random": {"type": "random"},
    "temp_25": {"type": "temp", "temp": 25},
    "temp_50": {"type": "temp", "temp": 50},
    "temp_120": {"type": "temp", "temp": 120},
    "family_Base_Bentonite": {"type": "family", "family": "Base_Bentonite"},
    "family_KCl_Polymer": {"type": "family", "family": "KCl_Polymer"},
    "family_Polymer_Enhanced": {"type": "family", "family": "Polymer_Enhanced"},
    "family_Salt_Tolerant": {"type": "family", "family": "Salt_Tolerant"},
    "temporal_2023_to_2024": {"type": "temporal"},
}

MODELS = ["RidgeCV", "RandomForest"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def get_split_indices(df, spec):
    kind = spec["type"]
    all_idx = df.index.to_numpy()

    def _split_pool(pool_idx):
        strata = df.loc[pool_idx, "Formulation_Family"]
        tr, ca = train_test_split(
            pool_idx,
            test_size=CAL_FRACTION_OF_POOL,
            random_state=RANDOM_STATE,
            stratify=strata,
        )
        return np.array(tr), np.array(ca)

    if kind == "random":
        train_pool, test_idx = train_test_split(
            all_idx, test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=df["Formulation_Family"],
        )
        tr, ca = _split_pool(np.array(train_pool))
        return tr, ca, np.array(test_idx)

    if kind == "temp":
        m = df["Test_Temperature_C"] == spec["temp"]
        test_idx = df.index[m].to_numpy()
        pool = df.index[~m].to_numpy()
        tr, ca = _split_pool(pool)
        return tr, ca, test_idx

    if kind == "family":
        m = df["Formulation_Family"] == spec["family"]
        test_idx = df.index[m].to_numpy()
        pool = df.index[~m].to_numpy()
        tr, ca = _split_pool(pool)
        return tr, ca, test_idx

    if kind == "temporal":
        dates = pd.to_datetime(df["Test_Date"], errors="coerce")
        test_idx = df.index[dates.dt.year == 2024].to_numpy()
        pool = df.index[dates.dt.year == 2023].to_numpy()
        tr, ca = _split_pool(pool)
        return tr, ca, test_idx

    raise ValueError(f"Unknown split type: {kind}")


def build_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
    ])


def build_rf():
    return RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=1,
    )


MODEL_BUILDERS = {
    "RidgeCV": build_ridge,
    "RandomForest": build_rf,
}


# ----------------------------------------------------------------------
# Main computation
# ----------------------------------------------------------------------
def compute_importances(df):
    rows = []
    for exp_name, spec in EXPERIMENTS.items():
        train_idx, _, test_idx = get_split_indices(df, spec)
        X_train = df.loc[train_idx, FEATURES]
        X_test = df.loc[test_idx, FEATURES]
        n_test = len(test_idx)

        section(f"Experiment: {exp_name}  train={len(train_idx)} test={n_test}")

        for target in TARGETS:
            y_train = df.loc[train_idx, target].values
            y_test = df.loc[test_idx, target].values

            for model_name in MODELS:
                build_fn = MODEL_BUILDERS[model_name]
                model = build_fn().fit(X_train, y_train)

                result = permutation_importance(
                    model, X_test, y_test,
                    scoring="neg_mean_absolute_error",
                    n_repeats=N_REPEATS,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )

                for i, feat in enumerate(FEATURES):
                    rows.append({
                        "experiment": exp_name,
                        "target": target,
                        "model": model_name,
                        "n_test": n_test,
                        "feature": feat,
                        "importance_mean": float(result.importances_mean[i]),
                        "importance_std": float(result.importances_std[i]),
                    })
                print(f"  {target:20s} {model_name:15s} done")

    return pd.DataFrame(rows)


def compute_ranks(imp_df):
    """Rank features by importance within each (experiment, target, model)."""
    ranks = imp_df.copy()
    ranks["rank"] = ranks.groupby(
        ["experiment", "target", "model"]
    )["importance_mean"].rank(ascending=False, method="min").astype(int)
    return ranks


def compute_stability(imp_df):
    """Per feature, per target/model: how stable is its rank across experiments?"""
    ranks = compute_ranks(imp_df)
    rows = []
    for (target, model, feature), g in ranks.groupby(
        ["target", "model", "feature"]
    ):
        exp_ranks = g["rank"].values
        rows.append({
            "target": target,
            "model": model,
            "feature": feature,
            "rank_mean": float(exp_ranks.mean()),
            "rank_std": float(exp_ranks.std()),
            "rank_min": int(exp_ranks.min()),
            "rank_max": int(exp_ranks.max()),
            "rank_range": int(exp_ranks.max() - exp_ranks.min()),
            "importance_mean_across_exps": float(g["importance_mean"].mean()),
            "importance_std_across_exps": float(g["importance_mean"].std()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_heatmap(imp_df, target, model, out_path):
    sub = imp_df[(imp_df["target"] == target) & (imp_df["model"] == model)]
    if len(sub) == 0:
        return
    experiments = list(EXPERIMENTS.keys())
    features = FEATURES

    matrix = np.zeros((len(features), len(experiments)))
    for j, exp in enumerate(experiments):
        g = sub[sub["experiment"] == exp].set_index("feature")
        vals = g.reindex(features)["importance_mean"].fillna(0.0).values
        vmax = vals.max() if vals.max() > 0 else 1.0
        matrix[:, j] = vals / vmax  # normalize per experiment

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(
        [e.replace("family_", "").replace("_2023_to_2024", "") for e in experiments],
        rotation=45, ha="right",
    )
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)

    for i in range(len(features)):
        for j in range(len(experiments)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontsize=7)

    ax.set_title(
        f"Normalized permutation importance — {target} / {model}\n"
        f"(each column normalized so 1.00 = top feature within that experiment)"
    )
    fig.colorbar(im, ax=ax, label="Normalized importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_features(imp_df, target, out_path):
    """Bar chart of top-6 features on random vs three shift experiments."""
    focus = ["random", "temp_120", "family_Salt_Tolerant", "combo_never_used"]
    # Use only experiments that exist
    focus = [e for e in ["random", "temp_120", "family_Salt_Tolerant"]
             if e in EXPERIMENTS]

    sub = imp_df[(imp_df["target"] == target) & (imp_df["model"] == "RidgeCV")]
    if len(sub) == 0:
        return

    # Determine global top-6 features from the random experiment
    ref = sub[sub["experiment"] == "random"].nlargest(6, "importance_mean")
    top_feats = list(ref["feature"])

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / len(focus)
    xs = np.arange(len(top_feats))

    for k, exp in enumerate(focus):
        sub_e = sub[sub["experiment"] == exp].set_index("feature")
        vals = [float(sub_e.loc[f, "importance_mean"]) if f in sub_e.index else 0.0
                for f in top_feats]
        ax.bar(xs + k * width, vals, width=width, label=exp)

    ax.set_xticks(xs + width * (len(focus) - 1) / 2)
    ax.set_xticklabels(top_feats, rotation=25, ha="right")
    ax.set_ylabel("Permutation importance (ΔMAE)")
    ax.set_title(f"Top-6 features on random, and their importance under shift — {target} / RidgeCV")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET).reset_index(drop=True)
    print(f"Rows: {len(df)}")
    print(f"Experiments: {len(EXPERIMENTS)}  Targets: {len(TARGETS)}  Models: 2")
    print(f"n_repeats per feature permutation: {N_REPEATS}")

    imp_df = compute_importances(df)

    section("Save importance table")
    imp_csv = METRICS_DIR / "interpretability_importance.csv"
    imp_df.to_csv(imp_csv, index=False)
    print("Saved:", imp_csv)

    section("Compute ranks and stability")
    ranks_df = compute_ranks(imp_df)
    ranks_csv = METRICS_DIR / "interpretability_ranks.csv"
    ranks_df.to_csv(ranks_csv, index=False)
    print("Saved:", ranks_csv)

    stability_df = compute_stability(imp_df)
    stab_csv = METRICS_DIR / "interpretability_stability.csv"
    stability_df.to_csv(stab_csv, index=False)
    print("Saved:", stab_csv)

    section("Top-5 features per (target, model) on the random experiment")
    for target in TARGETS:
        for model in MODELS:
            sub = imp_df[
                (imp_df["target"] == target)
                & (imp_df["model"] == model)
                & (imp_df["experiment"] == "random")
            ].sort_values("importance_mean", ascending=False).head(5)
            print(f"\n  {target}  /  {model}")
            print(sub[["feature", "importance_mean", "importance_std"]].to_string(index=False))

    section("Rank stability (target=PV_cP, model=RidgeCV)")
    stab_pv = stability_df[
        (stability_df["target"] == "PV_cP")
        & (stability_df["model"] == "RidgeCV")
    ].sort_values("rank_mean")
    print(stab_pv[[
        "feature", "rank_mean", "rank_std", "rank_min", "rank_max", "rank_range",
        "importance_mean_across_exps",
    ]].to_string(index=False))

    section("Rank shift: features whose importance is unstable across experiments (rank_range >= 5)")
    unstable = stability_df[stability_df["rank_range"] >= 5].sort_values(
        ["target", "model", "rank_range"], ascending=[True, True, False]
    )
    print(unstable[[
        "target", "model", "feature", "rank_mean",
        "rank_min", "rank_max", "rank_range",
    ]].to_string(index=False))

    section("Save figures")
    for target in TARGETS:
        for model in MODELS:
            out = FIGURES_DIR / f"interpretability_{target}_{model}_heatmap.png"
            plot_heatmap(imp_df, target, model, out)
            print("Saved:", out)
    for target in TARGETS:
        out = FIGURES_DIR / f"interpretability_top_features_{target}.png"
        plot_top_features(imp_df, target, out)
        print("Saved:", out)

    section("DONE")
    print("Stage 11 interpretability complete.")


if __name__ == "__main__":
    main()