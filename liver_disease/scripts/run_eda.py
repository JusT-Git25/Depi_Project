"""CLI script for Phase 3: Comprehensive Exploratory Data Analysis (EDA) and Visualizations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from liver_disease.config import (
    FIGURES_DIR,
    NUMERIC_FEATURE_COLUMNS,
    REPORTS_DIR,
    TARGET_COLUMN,
    ensure_project_directories,
)
from liver_disease.data.loader import load_raw_dataset
from liver_disease.utils.io import save_json
from liver_disease.utils.logging_config import setup_logging

sns.set_theme(style="whitegrid", context="notebook", palette="muted")


def generate_eda_artifacts() -> Dict[str, Any]:
    """Generate publication-quality EDA plots and statistical association summaries."""
    ensure_project_directories()
    df_raw = load_raw_dataset()
    # Deduplicated view for unbiased statistical testing (exact duplicates documented in audit)
    df = df_raw.drop_duplicates(keep="first").reset_index(drop=True)
    df["TargetLabel"] = df[TARGET_COLUMN].map({1: "Liver Disease (1)", 2: "No Liver Disease (2)"})

    # 1. Class distribution & Gender breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    order = ["Liver Disease (1)", "No Liver Disease (2)"]
    palette = {"Liver Disease (1)": "#D9534F", "No Liver Disease (2)": "#4A90D9"}

    ax = axes[0]
    counts = df["TargetLabel"].value_counts().reindex(order)
    bars = ax.bar(counts.index, counts.values, color=[palette[k] for k in order], edgecolor="black", alpha=0.88)
    ax.set_title("Class Distribution (Deduplicated ILPD, n={})".format(len(df)), fontweight="bold")
    ax.set_ylabel("Patient Count")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 6,
            f"{val} ({100.0 * val / len(df):.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylim(0, counts.max() * 1.18)

    ax = axes[1]
    ct = pd.crosstab(df["Gender"], df["TargetLabel"])[order]
    ct.plot(kind="bar", ax=ax, color=[palette[k] for k in order], edgecolor="black", alpha=0.88)
    ax.set_title("Liver Disease Distribution by Gender", fontweight="bold")
    ax.set_ylabel("Patient Count")
    ax.set_xlabel("Gender")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Class")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_target_and_gender_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 2. Numerical feature distributions (histograms + KDE, log1p noted for skewed markers)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, col in zip(axes.flat, NUMERIC_FEATURE_COLUMNS):
        for label in order:
            subset = df.loc[df["TargetLabel"] == label, col].dropna()
            sns.kdeplot(subset, ax=ax, label=label, fill=True, alpha=0.35, color=palette[label], linewidth=1.8)
        skew = df[col].skew()
        ax.set_title(f"{col} (skew={skew:.2f})", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        if col == "Age":
            ax.legend(fontsize=8)
    plt.suptitle("Numerical Feature Distributions by Target Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_numeric_distributions_by_class.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 3. Log-scale / Boxplots to visualize outliers & class separation clearly
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    skewed_cols = {"TB", "DB", "Alkphos", "Sgpt", "Sgot"}
    for ax, col in zip(axes.flat, NUMERIC_FEATURE_COLUMNS):
        plot_df = df[["TargetLabel", col]].dropna().copy()
        y_col = col
        title = col
        if col in skewed_cols:
            plot_df["val"] = np.log1p(plot_df[col])
            y_col = "val"
            title = f"log1p({col})"
        sns.boxplot(
            data=plot_df,
            x="TargetLabel",
            y=y_col,
            hue="TargetLabel",
            palette=palette,
            ax=ax,
            legend=False,
            width=0.5,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", labelsize=9)
    plt.suptitle("Class Separation & Outlier Inspection (Log-Transformed Skewed Enzymes/Bilirubin)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_boxplots_and_outliers.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 4. Spearman correlation heatmap (robust to strong skewness and outliers)
    corr_df = df[NUMERIC_FEATURE_COLUMNS].copy()
    corr_df["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    corr_df["LiverDisease"] = (df[TARGET_COLUMN] == 1).astype(int)
    spearman_corr = corr_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
    sns.heatmap(
        spearman_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Spearman Rank Correlation Matrix (Deduplicated ILPD)", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_spearman_correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 5. Statistical comparison between Disease vs No Disease (Mann-Whitney U test)
    feature_tests: Dict[str, Any] = {}
    for col in NUMERIC_FEATURE_COLUMNS:
        g1 = df.loc[df[TARGET_COLUMN] == 1, col].dropna()
        g2 = df.loc[df[TARGET_COLUMN] == 2, col].dropna()
        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        # Rank-biserial correlation as effect size
        r_rb = 1.0 - (2.0 * u_stat) / (len(g1) * len(g2))
        feature_tests[col] = {
            "disease_median": round(float(g1.median()), 4),
            "disease_iqr": round(float(g1.quantile(0.75) - g1.quantile(0.25)), 4),
            "no_disease_median": round(float(g2.median()), 4),
            "no_disease_iqr": round(float(g2.quantile(0.75) - g2.quantile(0.25)), 4),
            "mann_whitney_u": round(float(u_stat), 2),
            "p_value": float(p_val),
            "spearman_r_with_disease": round(float(spearman_corr.loc[col, "LiverDisease"]), 4),
            "rank_biserial_effect": round(float(r_rb), 4),
        }

    # Chi-square test for Gender vs Target
    gender_ct = pd.crosstab(df["Gender"], df[TARGET_COLUMN])
    chi2, chi2_p, dof, _ = stats.chi2_contingency(gender_ct)

    eda_summary = {
        "raw_rows": int(len(df_raw)),
        "deduplicated_rows": int(len(df)),
        "feature_statistical_tests_mann_whitney": feature_tests,
        "gender_chi2_test": {
            "chi2_statistic": round(float(chi2), 4),
            "p_value": float(chi2_p),
            "dof": int(dof),
            "disease_rate_male_pct": round(100.0 * float((df.loc[df["Gender"] == "Male", TARGET_COLUMN] == 1).mean()), 2),
            "disease_rate_female_pct": round(100.0 * float((df.loc[df["Gender"] == "Female", TARGET_COLUMN] == 1).mean()), 2),
        },
        "high_collinearity_pairs_spearman_gt_0_7": [
            {
                "feature_1": c1,
                "feature_2": c2,
                "spearman_r": round(float(spearman_corr.loc[c1, c2]), 4),
            }
            for i, c1 in enumerate(NUMERIC_FEATURE_COLUMNS)
            for c2 in NUMERIC_FEATURE_COLUMNS[i + 1 :]
            if abs(spearman_corr.loc[c1, c2]) >= 0.70
        ],
        "generated_figures": [
            "artifacts/reports/figures/01_target_and_gender_distribution.png",
            "artifacts/reports/figures/02_numeric_distributions_by_class.png",
            "artifacts/reports/figures/03_boxplots_and_outliers.png",
            "artifacts/reports/figures/04_spearman_correlation_heatmap.png",
        ],
    }

    save_json(eda_summary, REPORTS_DIR / "eda_summary.json")
    return eda_summary


def main() -> None:
    logger = setup_logging("INFO")
    summary = generate_eda_artifacts()
    logger.info("EDA complete. Saved eda_summary.json and 4 figures.")
    print("\n=== EDA STATISTICAL SUMMARY ===")
    for feat, stats_d in summary["feature_statistical_tests_mann_whitney"].items():
        print(
            f"{feat:10s} | Disease med={stats_d['disease_median']:7.2f} vs "
            f"No-Dis med={stats_d['no_disease_median']:7.2f} | "
            f"p={stats_d['p_value']:.3e} | rho={stats_d['spearman_r_with_disease']:+.3f}"
        )
    print(f"\nHigh collinearity pairs (|rho| >= 0.70): {summary['high_collinearity_pairs_spearman_gt_0_7']}")
    print(f"Gender chi2 p-value: {summary['gender_chi2_test']['p_value']:.4f} "
          f"(Male disease rate={summary['gender_chi2_test']['disease_rate_male_pct']}%, "
          f"Female={summary['gender_chi2_test']['disease_rate_female_pct']}%)")


if __name__ == "__main__":
    main()
