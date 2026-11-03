"""
src/evaluation/paper_figures_tables.py

Stage 16 -- Generate paper-quality figures and LaTeX-ready tables.

Reads only from CSVs already saved by Stages 6-13. Does not fit or refit
any model. Produces final publication elements for the manuscript.

All output is ASCII (no Greek letters, no Unicode arrows) so that the
files open correctly on any system, including Windows with cp1252 locale.

Run
---
  python src/evaluation/paper_figures_tables.py
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PAPER_FIGS = FIGURES_DIR / "paper"
PAPER_TABLES = TABLES_DIR / "paper"
for d in [PAPER_FIGS, PAPER_TABLES]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TARGET_SHORT = {
    "PV_cP": "PV",
    "YP_lb100ft2": "YP",
    "GS_10s_lb100ft2": "GS 10s",
    "GS_10min_lb100ft2": "GS 10min",
    "n_FlowIndex": "n",
}

EXP_SHORT = {
    "random": "random",
    "temp_25": "T=25",
    "temp_40": "T=40",
    "temp_50": "T=50",
    "temp_60": "T=60",
    "temp_75": "T=75",
    "temp_90": "T=90",
    "temp_120": "T=120",
    "temp_150_stress": "T=150*",
    "family_Base_Bentonite": "BaseBent",
    "family_High_Density": "HighDens",
    "family_KCl_Polymer": "KCl",
    "family_Low_Solid": "LowSolid",
    "family_Polymer_Enhanced": "PolyEnh",
    "family_Salt_Tolerant": "SaltTol",
    "temporal_2023_to_2024": "2023-to-24",  # ASCII-safe
    "combo_Base_Bentonite_25": "BaseBent@25",
    "combo_Base_Bentonite_50": "BaseBent@50",
    "combo_Polymer_Enhanced_75": "PolyEnh@75",
    "combo_KCl_Polymer_90": "KCl@90",
    "combo_High_Density_60": "HighDens@60",
}

FEATURE_SHORT = {
    "Bentonite_ppb": "Bentonite",
    "Barite_ppb": "Barite",
    "CMC_HV_ppb": "CMC-HV",
    "CMC_LV_ppb": "CMC-LV",
    "PAC_R_ppb": "PAC-R",
    "PAC_L_ppb": "PAC-L",
    "XCD_Polymer_ppb": "XCD",
    "KCl_ppb": "KCl",
    "NaCl_ppb": "NaCl",
    "KOH_ppb": "KOH",
    "Lubricant_ppb": "Lubricant",
    "Aging_Hours": "Aging h",
    "Test_Temperature_C": "Temp",
    "Aging_Temperature_C": "Aging T",
    "MW_ppg": "MW",
}


def short_exp(s):
    return EXP_SHORT.get(s, s)


def short_target(s):
    return TARGET_SHORT.get(s, s)


def short_feature(s):
    return FEATURE_SHORT.get(s, s)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe_write(path, text):
    """Write text as ASCII-safe UTF-8 (works on Windows cp1252 locale)."""
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------
# Figure 1 -- Shift MAE ratio heatmap
# ----------------------------------------------------------------------
def fig1_shift_heatmap():
    path = METRICS_DIR / "shift_summary.csv"
    if not path.exists():
        print(f"  Skipping fig1: {path} not found")
        return
    df = pd.read_csv(path)
    df = df[df["model"] == "RidgeCV"]

    order_exp = [
        "temp_25", "temp_40", "temp_50", "temp_60",
        "temp_75", "temp_90", "temp_120", "temp_150_stress",
        "family_Base_Bentonite", "family_High_Density",
        "family_KCl_Polymer", "family_Low_Solid",
        "family_Polymer_Enhanced", "family_Salt_Tolerant",
        "temporal_2023_to_2024",
        "combo_Base_Bentonite_25", "combo_Base_Bentonite_50",
        "combo_KCl_Polymer_90", "combo_High_Density_60",
        "combo_Polymer_Enhanced_75",
    ]
    targets = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2",
               "GS_10min_lb100ft2", "n_FlowIndex"]

    present = df["experiment"].unique()
    order_exp = [e for e in order_exp if e in present]

    matrix = np.full((len(targets), len(order_exp)), np.nan)
    for i, t in enumerate(targets):
        for j, e in enumerate(order_exp):
            row = df[(df["target"] == t) & (df["experiment"] == e)]
            if len(row) > 0:
                matrix[i, j] = row["MAE_ratio"].values[0]

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r",
                   vmin=0.5, vmax=2.5, interpolation="nearest")
    ax.set_xticks(range(len(order_exp)))
    ax.set_xticklabels([short_exp(e) for e in order_exp],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([short_target(t) for t in targets])

    for i in range(len(targets)):
        for j in range(len(order_exp)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            color = "white" if (v < 0.85 or v > 2.0) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    boundaries = []
    prev_group = None
    for k, e in enumerate(order_exp):
        if e.startswith("temp"): g = "temp"
        elif e.startswith("family"): g = "family"
        elif e.startswith("combo"): g = "combo"
        elif e.startswith("temporal"): g = "temporal"
        else: g = "other"
        if prev_group is not None and g != prev_group:
            boundaries.append(k - 0.5)
        prev_group = g
    for b in boundaries:
        ax.axvline(b, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("MAE ratio (experiment / random)")
    ax.set_title("RidgeCV: MAE degradation under distribution shift\n"
                 "(1.00 = same as random; >1 = worse; <1 = better)")
    fig.tight_layout()
    out = PAPER_FIGS / "fig1_shift_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ----------------------------------------------------------------------
# Figure 2 -- Conformal coverage with CI
# ----------------------------------------------------------------------
def fig2_coverage_ci():
    path = METRICS_DIR / "ci_conformal_coverage.csv"
    if not path.exists():
        print(f"  Skipping fig2: {path} not found")
        return
    df = pd.read_csv(path)

    experiments = df["experiment"].unique()
    targets = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2",
               "GS_10min_lb100ft2", "n_FlowIndex"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(targets)))
    x_positions = np.arange(len(experiments))
    offset = (np.arange(len(targets)) - (len(targets) - 1) / 2) * 0.13

    for k, t in enumerate(targets):
        sub = df[df["target"] == t].set_index("experiment").reindex(experiments)
        xs = x_positions + offset[k]
        ys = sub["coverage_point"].values
        y_lo = sub["coverage_ci_low"].values
        y_hi = sub["coverage_ci_high"].values
        ax.errorbar(xs, ys, yerr=[ys - y_lo, y_hi - ys],
                    fmt="o", color=colors[k], capsize=3, markersize=5,
                    label=short_target(t))

    ax.axhline(0.90, color="black", linestyle="--", linewidth=1,
               label="Nominal 90%")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([short_exp(e) for e in experiments],
                       rotation=20, ha="right")
    ax.set_ylabel("Empirical coverage (95% CI)")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("Conformal prediction coverage degrades under distribution shift\n"
                 "(RidgeCV, 90% nominal; 95% bootstrap CIs)")
    ax.legend(ncol=3, loc="lower left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGS / "fig2_coverage_ci.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ----------------------------------------------------------------------
# Figure 3 -- Risk-coverage curves
# ----------------------------------------------------------------------
def fig3_risk_coverage():
    path = METRICS_DIR / "selective_curves.csv"
    if not path.exists():
        print(f"  Skipping fig3: {path} not found")
        return
    df = pd.read_csv(path)

    cases = [
        ("random", "PV_cP", "RidgeCV"),
        ("temp_120", "YP_lb100ft2", "RidgeCV"),
        ("family_Salt_Tolerant", "GS_10min_lb100ft2", "RidgeCV"),
        ("combo_Base_Bentonite_25", "PV_cP", "RidgeCV"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharey=False)
    for ax, (exp, tgt, model) in zip(axes, cases):
        sub = df[(df["experiment"] == exp)
                 & (df["target"] == tgt)
                 & (df["model"] == model)].sort_values("retention")
        if len(sub) == 0:
            ax.set_title(f"{short_exp(exp)} / {short_target(tgt)}\n(no data)")
            continue
        x = sub["retention"].values
        y = sub["mae_retained"].values
        baseline_mae = y[-1]
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, color="tab:blue")
        ax.axhline(baseline_mae, color="black", linestyle="--",
                   linewidth=1, label="Unconditional MAE")
        ax.set_xlabel("Retention fraction")
        ax.set_ylabel("MAE (retained subset)")
        ax.set_title(f"{short_exp(exp)}\n{short_target(tgt)} / {model}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Selective prediction: risk-coverage curves (sorted by uncertainty)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    out = PAPER_FIGS / "fig3_risk_coverage.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ----------------------------------------------------------------------
# Figure 4 -- Ablation summary with CI
# ----------------------------------------------------------------------
def fig4_ablation_summary():
    path = METRICS_DIR / "ci_ablation_deltas.csv"
    if not path.exists():
        print(f"  Skipping fig4: {path} not found")
        return
    df = pd.read_csv(path)

    pairs = ["temp_effect", "aging_hours_effect", "aging_temp_effect",
             "mw_effect", "full_vs_stage6"]
    models = ["RidgeCV", "RandomForest"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    width = 0.35
    x = np.arange(len(pairs))

    for k, model in enumerate(models):
        means, los, his = [], [], []
        for p in pairs:
            sub = df[(df["pair"] == p) & (df["model"] == model)]
            if len(sub) == 0:
                means.append(0); los.append(0); his.append(0); continue
            mean = sub["delta_point"].mean()
            lo = sub["delta_ci_low"].min()
            hi = sub["delta_ci_high"].max()
            means.append(mean); los.append(lo); his.append(hi)
        means = np.array(means); los = np.array(los); his = np.array(his)
        yerr = np.vstack([means - los, his - means])
        ax.bar(x + (k - 0.5) * width, means, width=width,
               yerr=yerr, capsize=3, label=model, alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in pairs])
    ax.set_ylabel("Delta MAE (new set - base set)")
    ax.set_title("Feature-set ablation: mean Delta MAE with bootstrap CI envelope\n"
                 "(negative = new features help; positive = hurt)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = PAPER_FIGS / "fig4_ablation_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ----------------------------------------------------------------------
# Figure 5 -- Physics summary
# ----------------------------------------------------------------------
def fig5_physics_summary():
    path = METRICS_DIR / "physics_target_summary.csv"
    if not path.exists():
        print(f"  Skipping fig5: {path} not found")
        return
    df = pd.read_csv(path)

    targets = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2", "GS_10min_lb100ft2"]
    df = df[(df["model"] == "RidgeCV")
            & (df["target"].isin(targets))]

    experiments = sorted(df["experiment"].unique())
    matrix_neg = np.zeros((len(targets), len(experiments)))
    matrix_lb = np.zeros((len(targets), len(experiments)))
    for i, t in enumerate(targets):
        for j, e in enumerate(experiments):
            row = df[(df["target"] == t) & (df["experiment"] == e)]
            if len(row) > 0:
                matrix_neg[i, j] = row["neg_fraction"].values[0]
                matrix_lb[i, j] = row["conformal_lb_negative_fraction"].values[0]

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax, mat, title in [
        (axes[0], matrix_neg, "(a) Fraction of negative point predictions (physically impossible)"),
        (axes[1], matrix_lb, "(b) Fraction of 90% conformal intervals with negative lower bound"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=0.25,
                       interpolation="nearest")
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels([short_target(t) for t in targets])
        for i in range(len(targets)):
            for j in range(len(experiments)):
                v = mat[i, j]
                if v > 0.005:
                    color = "white" if v > 0.15 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color=color)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    axes[1].set_xticks(range(len(experiments)))
    axes[1].set_xticklabels([short_exp(e) for e in experiments],
                            rotation=45, ha="right", fontsize=8)
    fig.suptitle("Physics consistency of RidgeCV predictions under distribution shift",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    out = PAPER_FIGS / "fig5_physics_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ----------------------------------------------------------------------
# Figure 6 -- Interpretability rank stability
# ----------------------------------------------------------------------
def fig6_interpretability_stability():
    path = METRICS_DIR / "interpretability_stability.csv"
    if not path.exists():
        print(f"  Skipping fig6: {path} not found")
        return
    df = pd.read_csv(path)

    targets = ["PV_cP", "YP_lb100ft2", "GS_10s_lb100ft2"]
    df = df[(df["model"] == "RidgeCV") & (df["target"].isin(targets))]

    features_sorted = (df.groupby("feature")["importance_mean_across_exps"]
                       .mean().sort_values(ascending=False).head(8).index.tolist())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, t in zip(axes, targets):
        sub = df[df["target"] == t].set_index("feature").reindex(features_sorted)
        ranks = sub["rank_range"].values
        colors = ["tab:red" if r >= 5 else "tab:blue" for r in ranks]
        ax.barh(range(len(features_sorted)), ranks, color=colors, alpha=0.85)
        ax.set_yticks(range(len(features_sorted)))
        ax.set_yticklabels([short_feature(f) for f in features_sorted])
        ax.set_xlabel("Rank range across experiments")
        ax.set_title(short_target(t))
        ax.grid(True, axis="x", alpha=0.3)
        ax.axvline(5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.suptitle("Feature importance stability under distribution shift\n"
                 "(red = unstable, rank range >= 5; dashed line = threshold)",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    out = PAPER_FIGS / "fig6_interpretability_stability.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ----------------------------------------------------------------------
# LaTeX table builder (ASCII-only)
# ----------------------------------------------------------------------
def to_latex_table(df, caption, label, float_format="%.3f"):
    cols = list(df.columns)
    header = " & ".join(cols) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v}")
            elif isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    cells.append("--")
                else:
                    cells.append(float_format % v)
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def tab1_baseline():
    path = METRICS_DIR / "ci_shift_mae.csv"
    if not path.exists():
        print(f"  Skipping tab1: {path} not found")
        return
    df = pd.read_csv(path)
    df = df[df["experiment"] == "random"]
    out = df[["target", "model", "n_test",
              "mae_point", "mae_ci_low", "mae_ci_high"]].copy()
    out.columns = ["Target", "Model", "n", "MAE", "CI low", "CI high"]
    out = out.sort_values(["Target", "Model"]).reset_index(drop=True)
    out["Target"] = out["Target"].map(short_target)
    out.to_csv(PAPER_TABLES / "tab1_baseline_mae.csv", index=False, encoding="utf-8")
    tex = to_latex_table(
        out,
        caption=("Baseline performance on the random 70/15/15 split "
                 "(RidgeCV and RandomForest; 95\\% bootstrap CIs, B=1000)."),
        label="tab:baseline",
    )
    safe_write(PAPER_TABLES / "tab1_baseline_mae.tex", tex)
    print(f"  Saved: {PAPER_TABLES}/tab1_baseline_mae.{{csv,tex}}")


def tab2_shift_delta():
    path = METRICS_DIR / "ci_shift_deltas.csv"
    if not path.exists():
        print(f"  Skipping tab2: {path} not found")
        return
    df = pd.read_csv(path)
    df = df[(df["model"] == "RidgeCV")
            & (df["target"].isin(["PV_cP", "YP_lb100ft2"]))]
    out = df[["experiment", "target",
              "mae_random_point", "mae_shift_point",
              "delta_point", "delta_ci_low", "delta_ci_high",
              "p_delta_gt_zero"]].copy()
    # ASCII-safe column labels (no Greek letters)
    out.columns = ["Experiment", "Target", "MAE(random)", "MAE(shift)",
                   "Delta MAE", "CI low", "CI high", "P(Delta>0)"]
    out = out.sort_values(["Experiment", "Target"]).reset_index(drop=True)
    out["Experiment"] = out["Experiment"].map(short_exp)
    out["Target"] = out["Target"].map(short_target)
    out.to_csv(PAPER_TABLES / "tab2_shift_delta.csv", index=False, encoding="utf-8")
    tex = to_latex_table(
        out,
        caption=("MAE degradation on shift experiments relative to the "
                 "random-split reference (RidgeCV; 95\\% bootstrap CIs)."),
        label="tab:shift_delta",
    )
    safe_write(PAPER_TABLES / "tab2_shift_delta.tex", tex)
    print(f"  Saved: {PAPER_TABLES}/tab2_shift_delta.{{csv,tex}}")


def tab3_conformal_coverage():
    path = METRICS_DIR / "ci_conformal_coverage.csv"
    if not path.exists():
        print(f"  Skipping tab3: {path} not found")
        return
    df = pd.read_csv(path)
    out = df[["experiment", "target", "n_test",
              "coverage_point", "coverage_ci_low", "coverage_ci_high",
              "ci_excludes_nominal"]].copy()
    out.columns = ["Experiment", "Target", "n",
                   "Coverage", "CI low", "CI high", "CI excludes 0.90"]
    out = out.sort_values(["Experiment", "Target"]).reset_index(drop=True)
    out["Experiment"] = out["Experiment"].map(short_exp)
    out["Target"] = out["Target"].map(short_target)
    out.to_csv(PAPER_TABLES / "tab3_conformal_coverage.csv", index=False, encoding="utf-8")
    tex = to_latex_table(
        out,
        caption=("Empirical coverage of 90\\% conformal prediction intervals "
                 "under distribution shift (RidgeCV; 95\\% bootstrap CIs)."),
        label="tab:coverage",
    )
    safe_write(PAPER_TABLES / "tab3_conformal_coverage.tex", tex)
    print(f"  Saved: {PAPER_TABLES}/tab3_conformal_coverage.{{csv,tex}}")


def tab4_ablation_summary():
    path = METRICS_DIR / "ci_ablation_deltas.csv"
    if not path.exists():
        print(f"  Skipping tab4: {path} not found")
        return
    df = pd.read_csv(path)
    pairs = ["temp_effect", "aging_hours_effect", "aging_temp_effect",
             "mw_effect", "full_vs_stage6"]
    rows = []
    for p in pairs:
        for model in ["RidgeCV", "RandomForest"]:
            sub = df[(df["pair"] == p) & (df["model"] == model)]
            if len(sub) == 0:
                continue
            rows.append({
                "Pair": p.replace("_", " "),
                "Model": model,
                "n cells": len(sub),
                "Mean Delta MAE": sub["delta_point"].mean(),
                "Median Delta MAE": sub["delta_point"].median(),
                "Fraction significant": float((sub["delta_ci_high"] < 0).mean()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(PAPER_TABLES / "tab4_ablation_summary.csv", index=False, encoding="utf-8")
    tex = to_latex_table(
        out,
        caption=("Feature-set ablation summary. Mean/median Delta MAE across "
                 "all experiment-target cells. Fraction significant = "
                 "fraction where paired-bootstrap 95\\% CI excludes zero on "
                 "the negative side (new set helps)."),
        label="tab:ablation",
    )
    safe_write(PAPER_TABLES / "tab4_ablation_summary.tex", tex)
    print(f"  Saved: {PAPER_TABLES}/tab4_ablation_summary.{{csv,tex}}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reading from: {METRICS_DIR}")
    print(f"Writing figures to: {PAPER_FIGS}")
    print(f"Writing tables to:  {PAPER_TABLES}")

    section("Figures")
    fig1_shift_heatmap()
    fig2_coverage_ci()
    fig3_risk_coverage()
    fig4_ablation_summary()
    fig5_physics_summary()
    fig6_interpretability_stability()

    section("Tables")
    tab1_baseline()
    tab2_shift_delta()
    tab3_conformal_coverage()
    tab4_ablation_summary()

    section("DONE")
    print("Stage 16 paper figures and tables complete.")
    print(f"Figures: {PAPER_FIGS}")
    print(f"Tables:  {PAPER_TABLES}")


if __name__ == "__main__":
    main()