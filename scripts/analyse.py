"""Stage 4: statistical analysis of LLM stability vs human disagreement.

Reads all scored CSVs, runs:
  1. Descriptive stats  (mean ± std flip_rate per model x tier x dataset)
  2. Spearman ρ         (llm_flip_rate vs human_entropy, per model x dataset)
  3. Kruskal-Wallis     (flip_rate across tiers, per model x dataset)
  4. Mann-Whitney U     (pairwise tier comparisons, per model x dataset)
  5. Plots              (box plots per tier, scatter flip_rate vs human_entropy)

Outputs written to data/outputs/analysis/

Run:
    python scripts/analyse.py
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from itertools import combinations

# ── config ────────────────────────────────────────────────────────────────────

SCORED_FILES = {
    ("gemini",     "mhs"):       "data/outputs/gemini/mhs_scored_full.csv",
    ("gemini",     "crehate"):   "data/outputs/gemini/crehate_scored_full.csv",
    ("mistral",    "mhs"):       "data/outputs/mistral/scored_mhs.csv",
    ("mistral",    "crehate"):   "data/outputs/mistral/scored_crehate.csv",
    ("hatexplain", "mhs"):       "data/outputs/hatexplain/scored_mhs.csv",
    ("hatexplain", "crehate"):   "data/outputs/hatexplain/scored_crehate.csv",
    ("llama",      "mhs"):       "data/outputs/llama/scored_mhs.csv",
    ("llama",      "crehate"):   "data/outputs/llama/scored_crehate.csv",
}

MODEL_LABELS = {
    "gemini-2.5-flash":                           "Gemini 2.5 Flash",
    "mistralai/Mistral-7B-Instruct-v0.3":         "Mistral 7B",
    "Hate-speech-CNERG/bert-base-uncased-hatexplain": "HateXplain",
    "llama-3.1-8b-instant":                       "Llama 3.1 8B",
}

TIER_ORDER     = ["low", "medium", "high"]
OUT_DIR        = Path("data/outputs/analysis")
EXCLUDE_MODELS = {"HateXplain"}  # fully deterministic — excluded from plots

MODEL_ORDER  = ["Gemini 2.5 Flash", "Mistral 7B", "Llama 3.1 8B"]
# Wong (2011) colorblind-safe palette
MODEL_COLORS = {
    "Gemini 2.5 Flash": "#0072B2",   # dark blue
    "Mistral 7B":        "#D55E00",   # vermillion
    "Llama 3.1 8B":      "#009E73",   # bluish-green
}
TIER_COLORS = {
    "low":    "#56B4E9",   # sky blue
    "medium": "#E69F00",   # amber
    "high":   "#CC79A7",   # rose
}

sns.set_theme(
    style="ticks",
    font_scale=1.1,
    rc={
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    ".25",
        "axes.linewidth":    0.8,
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    },
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    frames = []
    for (model_key, dataset), path in SCORED_FILES.items():
        try:
            df = pd.read_csv(path)
            df["model_key"] = model_key
            df["model_label"] = df["model_name"].map(MODEL_LABELS).fillna(df["model_name"])
            frames.append(df)
        except FileNotFoundError:
            print(f"  WARNING: {path} not found — skipping")
    return pd.concat(frames, ignore_index=True)


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ── 1. descriptive stats ──────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    tiers_present = [t for t in TIER_ORDER if t in df["disagreement_tier"].unique()]
    rows = []
    for (label, dataset), g in df.groupby(["model_label", "dataset"]):
        for tier in tiers_present:
            sub = g[g["disagreement_tier"] == tier]["llm_flip_rate"]
            if len(sub) == 0:
                continue
            rows.append({
                "model": label, "dataset": dataset, "tier": tier,
                "n": len(sub),
                "mean_flip_rate": round(sub.mean(), 4),
                "std_flip_rate":  round(sub.std(),  4),
                "median_flip_rate": round(sub.median(), 4),
            })
    return pd.DataFrame(rows)


# ── 2. spearman correlation ───────────────────────────────────────────────────

def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, dataset), g in df.groupby(["model_label", "dataset"]):
        rho, p = stats.spearmanr(g["llm_flip_rate"], g["human_entropy"])
        rows.append({
            "model": label, "dataset": dataset,
            "spearman_rho": round(rho, 4),
            "p_value":      round(p,   4),
            "significance": sig_stars(p),
            "n": len(g),
        })
    return pd.DataFrame(rows)


# ── 3. kruskal-wallis ─────────────────────────────────────────────────────────

def kruskal_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, dataset), g in df.groupby(["model_label", "dataset"]):
        groups = [g[g["disagreement_tier"] == t]["llm_flip_rate"].values
                  for t in TIER_ORDER if t in g["disagreement_tier"].unique()]
        if len(groups) < 2:
            continue
        try:
            stat, p = stats.kruskal(*groups)
        except ValueError:
            stat, p = 0.0, 1.0  # all identical — no difference by definition
        rows.append({
            "model": label, "dataset": dataset,
            "H_statistic": round(stat, 4),
            "p_value":     round(p,    4),
            "significance": sig_stars(p),
        })
    return pd.DataFrame(rows)


# ── 4. mann-whitney pairwise ──────────────────────────────────────────────────

def mannwhitney_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, dataset), g in df.groupby(["model_label", "dataset"]):
        tiers = [t for t in TIER_ORDER if t in g["disagreement_tier"].unique()]
        for t1, t2 in combinations(tiers, 2):
            a = g[g["disagreement_tier"] == t1]["llm_flip_rate"].values
            b = g[g["disagreement_tier"] == t2]["llm_flip_rate"].values
            try:
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                stat, p = 0.0, 1.0
            rows.append({
                "model": label, "dataset": dataset,
                "tier_1": t1, "tier_2": t2,
                "U_statistic": round(stat, 2),
                "p_value":     round(p,    4),
                "significance": sig_stars(p),
            })
    return pd.DataFrame(rows)


# ── 5. plots ──────────────────────────────────────────────────────────────────

def _plot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with HateXplain removed for plotting."""
    return df[~df["model_label"].isin(EXCLUDE_MODELS)].copy()


def plot_mean_flip_by_tier(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: mean flip rate ± 95% CI per tier, hue = model."""
    plot_data = _plot_df(df)
    datasets  = sorted(plot_data["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.8 * len(datasets), 4.0),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub   = plot_data[plot_data["dataset"] == dataset]
        tiers = [t for t in TIER_ORDER if t in sub["disagreement_tier"].unique()]

        sns.barplot(
            data=sub, x="disagreement_tier", y="llm_flip_rate",
            hue="model_label", hue_order=MODEL_ORDER,
            order=tiers, palette=MODEL_COLORS,
            errorbar=("ci", 95), capsize=0.08,
            err_kws={"linewidth": 1.2, "color": "#333333"},
            width=0.65, alpha=0.88,
            ax=ax,
        )
        ax.set_title("MHS" if dataset == "mhs" else "CREHate",
                     fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Human Disagreement Tier", fontsize=10, labelpad=5)
        ax.set_ylabel("Mean Flip Rate (95% CI)" if ax is axes[0] else "", fontsize=10)
        ax.set_xticklabels([t.capitalize() for t in tiers])
        ax.set_ylim(bottom=0)
        ax.get_legend().remove()
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODEL_ORDER),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("LLM Flip Rate by Human Disagreement Tier",
                 fontsize=13, fontweight="bold", y=1.10)
    plt.tight_layout()
    fig.savefig(out_dir / "flip_rate_by_tier.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved flip_rate_by_tier.png")


def plot_violin_strip(df: pd.DataFrame, out_dir: Path) -> None:
    """Individual flip rates as a dodged strip plot per tier, coloured by model.

    Replaces the boxplot approach: since most flip rates are 0 (zero-inflated),
    box quartiles collapse to a flat line and convey nothing. The strip shows
    the full empirical distribution — the pile-up at 0 is itself informative.
    """
    plot_data = _plot_df(df)
    datasets  = sorted(plot_data["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 3.8),
                             sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        sub   = plot_data[plot_data["dataset"] == dataset]
        tiers = [t for t in TIER_ORDER if t in sub["disagreement_tier"].unique()]

        sns.stripplot(
            data=sub, x="disagreement_tier", y="llm_flip_rate",
            hue="model_label", hue_order=MODEL_ORDER,
            order=tiers, palette=MODEL_COLORS,
            dodge=True, alpha=0.5, size=3.5, jitter=0.18,
            ax=ax,
        )
        ax.set_title("MHS" if dataset == "mhs" else "CREHate",
                     fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Human Disagreement Tier", fontsize=10, labelpad=5)
        ax.set_ylabel("Flip Rate" if ax is axes[0] else "", fontsize=10)
        ax.set_xticklabels([t.capitalize() for t in tiers])
        ax.set_ylim(bottom=-0.01)
        ax.get_legend().remove()
        sns.despine(ax=ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODEL_ORDER),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Individual Flip Rates by Tier and Model",
                 fontsize=13, fontweight="bold", y=1.10)
    plt.tight_layout()
    fig.savefig(out_dir / "violin_strip.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved violin_strip.png")


def plot_spearman_bars(spearman_df: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal bar chart of Spearman ρ per model, excluding HateXplain."""
    sub = spearman_df[~spearman_df["model"].isin(EXCLUDE_MODELS)].copy()
    sub["spearman_rho"] = pd.to_numeric(sub["spearman_rho"], errors="coerce").fillna(0)
    datasets = sorted(sub["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.2),
                             sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        d = sub[sub["dataset"] == dataset].copy()
        d = d.set_index("model").reindex(MODEL_ORDER[::-1]).reset_index()

        for _, row in d.iterrows():
            rho   = row["spearman_rho"]
            color = MODEL_COLORS.get(row["model"], "#aaa")
            ax.barh(row["model"], rho, color=color, alpha=0.85,
                    edgecolor="none", height=0.45)

            sig  = row["significance"] if not pd.isna(row["significance"]) else "ns"
            txt  = f"ρ = {rho:.3f}  {sig}"
            ha   = "left" if rho >= 0 else "right"
            xoff = 0.007 if rho >= 0 else -0.007
            ax.text(rho + xoff, row.name, txt,
                    va="center", ha=ha, fontsize=9,
                    fontweight="bold" if sig not in ("ns", "") else "normal",
                    color="#1a1a1a")

        ax.axvline(0, color="#555", linewidth=0.8, linestyle="--", alpha=0.55)
        ax.set_xlim(-0.28, 0.38)
        ax.set_xlabel("Spearman ρ", fontsize=10, labelpad=5)
        ax.set_ylabel("")
        ax.set_title("MHS" if dataset == "mhs" else "CREHate",
                     fontsize=12, fontweight="bold", pad=8)
        sns.despine(ax=ax)

    fig.suptitle("Spearman ρ: Flip Rate vs Human Disagreement Entropy",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "spearman_bars.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved spearman_bars.png")


def _regression_band(x_arr: np.ndarray, y_arr: np.ndarray,
                     xs: np.ndarray, alpha: float = 0.95):
    """Return (y_pred, lower_band, upper_band) for xs using OLS + t-CI."""
    from scipy.stats import t as t_dist
    n = len(x_arr)
    m, b = np.polyfit(x_arr, y_arr, 1)
    y_hat = m * x_arr + b
    se    = np.sqrt(np.sum((y_arr - y_hat) ** 2) / (n - 2))
    x_mean = x_arr.mean()
    ss_x   = np.sum((x_arr - x_mean) ** 2)
    se_band = se * np.sqrt(1 / n + (xs - x_mean) ** 2 / ss_x)
    tcrit  = t_dist.ppf((1 + alpha) / 2, df=n - 2)
    y_pred = m * xs + b
    return y_pred, y_pred - tcrit * se_band, y_pred + tcrit * se_band


def plot_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: flip rate vs human entropy per model × dataset, with regression band."""
    plot_data = _plot_df(df)
    datasets  = sorted(plot_data["dataset"].unique())

    fig, axes = plt.subplots(
        len(datasets), len(MODEL_ORDER),
        figsize=(3.8 * len(MODEL_ORDER), 3.3 * len(datasets)),
        sharey=False,
    )
    if len(datasets) == 1:
        axes = [axes]

    all_tiers: list[str] = []
    for row_idx, dataset in enumerate(datasets):
        sub   = plot_data[plot_data["dataset"] == dataset]
        tiers = [t for t in TIER_ORDER if t in sub["disagreement_tier"].unique()]
        if not all_tiers:
            all_tiers = tiers

        for col_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx][col_idx]
            g  = sub[sub["model_label"] == model].dropna(
                    subset=["llm_flip_rate", "human_entropy"])

            sns.scatterplot(
                data=g, x="human_entropy", y="llm_flip_rate",
                hue="disagreement_tier", hue_order=tiers,
                palette={t: TIER_COLORS[t] for t in tiers},
                alpha=0.55, s=24, edgecolor="none",
                legend=False, ax=ax,
            )

            x_arr = g["human_entropy"].values
            y_arr = g["llm_flip_rate"].values
            rho, p = stats.spearmanr(y_arr, x_arr)

            if not np.isnan(rho) and len(g) > 4:
                xs = np.linspace(x_arr.min(), x_arr.max(), 120)
                y_pred, lo, hi = _regression_band(x_arr, y_arr, xs)
                line_color = "#333333"
                ax.plot(xs, y_pred, color=line_color,
                        linewidth=1.2, linestyle="--", alpha=0.8)
                ax.fill_between(xs, lo, hi, color=line_color, alpha=0.08)

            rho_txt = (f"ρ = {rho:.3f}  {sig_stars(p)}"
                       if not np.isnan(rho) else "ρ = N/A")
            bold = not np.isnan(rho) and sig_stars(p) != "ns"
            ax.text(0.97, 0.96, rho_txt, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5,
                    fontweight="bold" if bold else "normal",
                    bbox=dict(boxstyle="round,pad=0.28", fc="white",
                              ec="#cccccc", alpha=0.88, linewidth=0.7))

            ax.set_xlabel("Human Entropy" if row_idx == len(datasets) - 1 else "",
                          fontsize=10, labelpad=4)
            ax.set_ylabel("Flip Rate" if col_idx == 0 else "", fontsize=10)
            ax.set_ylim(bottom=-0.02)
            if row_idx == 0:
                ax.set_title(model, fontsize=10.5, fontweight="bold", pad=7)
            if col_idx == 0:
                ax.annotate(
                    "MHS" if dataset == "mhs" else "CREHate",
                    xy=(-0.36, 0.5), xycoords="axes fraction",
                    fontsize=10.5, fontweight="bold", rotation=90, va="center",
                )
            sns.despine(ax=ax)

    handles = [mpatches.Patch(color=TIER_COLORS[t], label=t.capitalize(), alpha=0.8)
               for t in all_tiers]
    fig.legend(handles=handles, loc="upper center", ncol=len(all_tiers),
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Flip Rate vs Human Entropy",
                 fontsize=13, fontweight="bold", y=1.08)
    plt.tight_layout()
    fig.savefig(out_dir / "scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  saved scatter.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scored CSVs...")
    df = load_all()
    print(f"  {len(df)} rows, {df['model_label'].nunique()} models, "
          f"{df['dataset'].nunique()} datasets\n")

    print("1. Descriptive statistics")
    desc = descriptive_stats(df)
    desc.to_csv(OUT_DIR / "descriptive_stats.csv", index=False)
    print(desc.to_string(index=False))

    print("\n2. Spearman correlation (flip_rate vs human_entropy)")
    spear = spearman_table(df)
    spear.to_csv(OUT_DIR / "spearman.csv", index=False)
    print(spear.to_string(index=False))

    print("\n3. Kruskal-Wallis test (flip_rate across tiers)")
    kw = kruskal_table(df)
    kw.to_csv(OUT_DIR / "kruskal_wallis.csv", index=False)
    print(kw.to_string(index=False))

    print("\n4. Mann-Whitney U pairwise tier comparisons")
    mw = mannwhitney_table(df)
    mw.to_csv(OUT_DIR / "mann_whitney.csv", index=False)
    print(mw.to_string(index=False))

    print("\n5. Plots")
    plot_mean_flip_by_tier(df, OUT_DIR)
    plot_violin_strip(df, OUT_DIR)
    plot_spearman_bars(spear, OUT_DIR)
    plot_scatter(df, OUT_DIR)

    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
