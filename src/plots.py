"""Plotting functions for all experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Global style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (10, 6),
})


def _save_fig(fig, path: str, formats: List[str] = None, dpi: int = 300):
    """Save figure in multiple formats."""
    if formats is None:
        formats = ["png", "pdf"]
    p = Path(path)
    for fmt in formats:
        out = p.with_suffix(f".{fmt}")
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_score_heatmap(
    scores: np.ndarray,
    title: str,
    save_path: str,
    xlabel: str = "Head",
    ylabel: str = "Layer",
    cmap: str = "viridis",
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Heatmap of per-head scores [n_layers, n_heads]."""
    n_layers, n_heads = scores.shape
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.5), max(6, n_layers * 0.3)))

    im = ax.imshow(scores, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Tick labels
    if n_heads <= 32:
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels(range(n_heads), fontsize=8)
    if n_layers <= 40:
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(range(n_layers), fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score")

    _save_fig(fig, save_path, formats, dpi)

    # Also save data as CSV
    df = pd.DataFrame(scores, columns=[f"head_{i}" for i in range(n_heads)])
    df.index.name = "layer"
    df.to_csv(Path(save_path).with_suffix(".csv"))


def plot_overlap_bars(
    jaccard_values: Dict[str, float],
    title: str,
    save_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Bar chart of Jaccard overlaps between head sets."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pairs = list(jaccard_values.keys())
    values = list(jaccard_values.values())
    colors = sns.color_palette("Set2", len(pairs))

    bars = ax.bar(pairs, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Jaccard Index")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3 + 0.05)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    _save_fig(fig, save_path, formats, dpi)

    # Save data
    pd.DataFrame({"pair": pairs, "jaccard": values}).to_csv(
        Path(save_path).with_suffix(".csv"), index=False
    )


def plot_correlation_bars(
    kendall_values: Dict[str, float],
    spearman_values: Dict[str, float],
    title: str,
    save_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Grouped bar chart of rank correlations."""
    fig, ax = plt.subplots(figsize=(10, 5))

    pairs = list(kendall_values.keys())
    x = np.arange(len(pairs))
    width = 0.35

    bars1 = ax.bar(x - width / 2, [kendall_values[p] for p in pairs],
                   width, label="Kendall τ", color="steelblue", edgecolor="black")
    bars2 = ax.bar(x + width / 2, [spearman_values[p] for p in pairs],
                   width, label="Spearman ρ", color="coral", edgecolor="black")

    ax.set_ylabel("Correlation")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    _save_fig(fig, save_path, formats, dpi)

    # Save data
    df = pd.DataFrame({
        "pair": pairs,
        "kendall_tau": [kendall_values[p] for p in pairs],
        "spearman_rho": [spearman_values[p] for p in pairs],
    })
    df.to_csv(Path(save_path).with_suffix(".csv"), index=False)


def plot_conditional_percentile(
    thresholds: List[int],
    tr_means: List[float],
    tl_means: List[float],
    title: str,
    save_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Line chart of conditional mean percentile for top IH heads under TR/TL rankings."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, tr_means, "o-", color="steelblue", label="Avg TR percentile", linewidth=2, markersize=8)
    ax.plot(thresholds, tl_means, "s--", color="coral", label="Avg TL percentile", linewidth=2, markersize=8)

    ax.set_xlabel("Top p% IH heads")
    ax.set_ylabel("Average percentile rank")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, save_path, formats, dpi)

    pd.DataFrame({
        "top_p_percent": thresholds,
        "avg_tr_percentile": tr_means,
        "avg_tl_percentile": tl_means,
    }).to_csv(Path(save_path).with_suffix(".csv"), index=False)


def plot_ablation_results(
    conditions: List[str],
    accuracies: List[float],
    tr_ratios: List[float],
    save_dir: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Plot ablation accuracy and TR ratio bar charts."""
    # --- Accuracy plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(conditions))
    bars = ax.bar(conditions, accuracies, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Study: Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    _save_fig(fig, str(Path(save_dir) / "ablation_accuracy"), formats, dpi)

    # --- TR ratio plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(conditions, tr_ratios, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("TR Ratio")
    ax.set_title("Ablation Study: Task Recognition Ratio", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, tr_ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    _save_fig(fig, str(Path(save_dir) / "ablation_tr_ratio"), formats, dpi)

    # Save combined data
    df = pd.DataFrame({
        "condition": conditions,
        "accuracy": accuracies,
        "tr_ratio": tr_ratios,
    })
    df.to_csv(str(Path(save_dir) / "ablation_results.csv"), index=False)

    # Delta from baseline
    if len(conditions) > 0:
        baseline_acc = accuracies[0]
        baseline_tr = tr_ratios[0]
        delta_df = pd.DataFrame({
            "condition": conditions,
            "accuracy_delta": [a - baseline_acc for a in accuracies],
            "tr_ratio_delta": [t - baseline_tr for t in tr_ratios],
        })
        delta_df.to_csv(str(Path(save_dir) / "ablation_deltas.csv"), index=False)


def plot_top_heads_table(
    tr_heads: List[Tuple[int, int]],
    tl_heads: List[Tuple[int, int]],
    ih_heads: List[Tuple[int, int]],
    save_path: str,
) -> None:
    """Save a table of top head sets."""
    max_len = max(len(tr_heads), len(tl_heads), len(ih_heads))

    def fmt(heads, i):
        if i < len(heads):
            return f"L{heads[i][0]}H{heads[i][1]}"
        return ""

    rows = []
    for i in range(max_len):
        rows.append({
            "Rank": i + 1,
            "TR Head": fmt(tr_heads, i),
            "TL Head": fmt(tl_heads, i),
            "IH Head": fmt(ih_heads, i),
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df


# ── Experiment 4: Layer-wise Distribution Plots ──────────────────────────────


def plot_layer_count_bars(
    layer_counts_df: pd.DataFrame,
    title: str,
    save_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Grouped bar chart of head count per layer for each head type."""
    fig, ax = plt.subplots(figsize=(14, 5))
    head_types = sorted(layer_counts_df["head_type"].unique())
    n_layers = layer_counts_df["layer"].nunique()
    x = np.arange(n_layers)
    width = 0.8 / len(head_types)
    colors = sns.color_palette("Set2", len(head_types))

    for i, ht in enumerate(head_types):
        subset = layer_counts_df[layer_counts_df["head_type"] == ht].sort_values("layer")
        offset = (i - len(head_types) / 2 + 0.5) * width
        ax.bar(x + offset, subset["count"].values, width, label=ht,
               color=colors[i], edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of Selected Heads")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(range(n_layers), fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, save_path, formats, dpi)
    layer_counts_df.to_csv(Path(save_path).with_suffix(".csv"), index=False)


def plot_layer_fraction_lines(
    layer_counts_df: pd.DataFrame,
    title: str,
    save_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Fraction of heads per layer for each head type (line chart)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    head_types = sorted(layer_counts_df["head_type"].unique())
    markers = ["o", "s", "^", "D", "v"]
    colors = sns.color_palette("Set2", len(head_types))

    for i, ht in enumerate(head_types):
        subset = layer_counts_df[layer_counts_df["head_type"] == ht].sort_values("layer")
        ax.plot(subset["layer"].values, subset["fraction"].values,
                marker=markers[i % len(markers)], label=ht, color=colors[i],
                linewidth=2, markersize=6)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of Heads in Layer")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_fig(fig, save_path, formats, dpi)


def plot_layer_distribution_violin(
    head_df: pd.DataFrame,
    title: str,
    save_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Violin plot of layer indices by head type."""
    fig, ax = plt.subplots(figsize=(8, 6))
    order = sorted(head_df["head_type"].unique())
    palette = dict(zip(order, sns.color_palette("Set2", len(order))))

    sns.violinplot(
        data=head_df, x="head_type", y="layer", order=order,
        palette=palette, inner="box", ax=ax, cut=0,
    )
    ax.set_xlabel("Head Type")
    ax.set_ylabel("Layer Index")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    _save_fig(fig, save_path, formats, dpi)


# ── Experiment 5: Attention Distribution Plots ───────────────────────────────


def plot_attention_grouped_bars(
    type_agg_df: pd.DataFrame,
    title: str,
    save_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Grouped bars: demo-label attn vs query attn per head type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    head_types = type_agg_df["head_type"].values
    x = np.arange(len(head_types))
    width = 0.35

    demo_vals = type_agg_df["mean_attn_demo_labels"].values
    query_vals = type_agg_df["mean_attn_query"].values
    demo_err = type_agg_df["se_attn_demo_labels"].values
    query_err = type_agg_df["se_attn_query"].values

    bars1 = ax.bar(x - width / 2, demo_vals, width, yerr=demo_err,
                   label="Demo Labels", color="steelblue", edgecolor="black",
                   linewidth=0.8, capsize=4)
    bars2 = ax.bar(x + width / 2, query_vals, width, yerr=query_err,
                   label="Query Tokens", color="coral", edgecolor="black",
                   linewidth=0.8, capsize=4)

    ax.set_xlabel("Head Type")
    ax.set_ylabel("Mean Attention")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(head_types)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    _save_fig(fig, save_path, formats, dpi)
    type_agg_df.to_csv(Path(save_path).with_suffix(".csv"), index=False)


def plot_attention_scatter(
    head_agg_df: pd.DataFrame,
    title: str,
    save_path: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> None:
    """Scatter: per-head demo-label attn vs query attn, colored by type."""
    fig, ax = plt.subplots(figsize=(8, 7))
    head_types = sorted(head_agg_df["head_type"].unique())
    palette = dict(zip(head_types, sns.color_palette("Set2", len(head_types))))
    markers = {"TR": "o", "TL": "s", "IH": "^", "Random": "D"}

    for ht in head_types:
        subset = head_agg_df[head_agg_df["head_type"] == ht]
        ax.scatter(
            subset["mean_attn_demo_labels"], subset["mean_attn_query"],
            label=ht, color=palette[ht], marker=markers.get(ht, "o"),
            s=60, alpha=0.8, edgecolors="black", linewidth=0.5,
        )

    ax.set_xlabel("Mean Attention to Demo Labels")
    ax.set_ylabel("Mean Attention to Query Tokens")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_fig(fig, save_path, formats, dpi)
