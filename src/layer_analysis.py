"""Experiment 4: Layer-wise distribution of TR, TL, and IH heads."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .utils import Config, write_summary


def build_head_type_df(
    head_sets: Dict[str, List[Tuple[int, int]]],
) -> pd.DataFrame:
    """Flat DataFrame with one row per selected head."""
    rows = []
    for head_type, heads in head_sets.items():
        for layer, head in heads:
            rows.append({"head_type": head_type, "layer": int(layer), "head": int(head)})
    return pd.DataFrame(rows)


def compute_per_layer_counts(
    head_df: pd.DataFrame,
    n_layers: int,
) -> pd.DataFrame:
    """Count and fraction of selected heads per layer for each type."""
    head_types = sorted(head_df["head_type"].unique())
    rows = []
    for ht in head_types:
        subset = head_df[head_df["head_type"] == ht]
        total = len(subset)
        for layer in range(n_layers):
            count = int((subset["layer"] == layer).sum())
            frac = count / total if total > 0 else 0.0
            rows.append({
                "layer": layer,
                "head_type": ht,
                "count": count,
                "fraction": frac,
            })
    return pd.DataFrame(rows)


def compute_layer_summary_stats(
    head_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean, median, std of layer indices per head type."""
    rows = []
    for ht in sorted(head_df["head_type"].unique()):
        layers = head_df[head_df["head_type"] == ht]["layer"].values
        rows.append({
            "head_type": ht,
            "mean_layer": float(np.mean(layers)),
            "median_layer": float(np.median(layers)),
            "std_layer": float(np.std(layers, ddof=1)) if len(layers) > 1 else 0.0,
            "n_heads": len(layers),
        })
    return pd.DataFrame(rows)


def run_layer_comparisons(
    head_df: pd.DataFrame,
    comparisons: List[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """Mann-Whitney U tests comparing layer distributions between head types."""
    if comparisons is None:
        comparisons = [("TL", "TR"), ("IH", "TR"), ("TL", "IH")]

    rows = []
    for type_a, type_b in comparisons:
        layers_a = head_df[head_df["head_type"] == type_a]["layer"].values
        layers_b = head_df[head_df["head_type"] == type_b]["layer"].values

        if len(layers_a) == 0 or len(layers_b) == 0:
            rows.append({
                "comparison": f"{type_a} vs {type_b}",
                "type_a_mean": float(np.mean(layers_a)) if len(layers_a) > 0 else np.nan,
                "type_b_mean": float(np.mean(layers_b)) if len(layers_b) > 0 else np.nan,
                "mean_diff": np.nan,
                "U_statistic": np.nan,
                "p_value": np.nan,
                "significant_005": False,
            })
            continue

        # Two-sided test
        u_stat, p_val = scipy_stats.mannwhitneyu(
            layers_a, layers_b, alternative="two-sided"
        )

        rows.append({
            "comparison": f"{type_a} vs {type_b}",
            "type_a_mean": float(np.mean(layers_a)),
            "type_b_mean": float(np.mean(layers_b)),
            "mean_diff": float(np.mean(layers_b) - np.mean(layers_a)),
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "significant_005": bool(p_val < 0.05),
        })

    return pd.DataFrame(rows)


def generate_layer_analysis_summary(
    summary_stats: pd.DataFrame,
    test_results: pd.DataFrame,
    cfg: Config,
) -> str:
    """Compare observed layer distributions to paper expectations. Returns verdict."""
    # Extract means
    means = {row["head_type"]: row["mean_layer"] for _, row in summary_stats.iterrows()}
    tr_mean = means.get("TR", np.nan)
    tl_mean = means.get("TL", np.nan)
    ih_mean = means.get("IH", np.nan)

    # Check paper hypothesis: TR deeper than TL and IH
    tr_deeper_than_tl = tr_mean > tl_mean
    tr_deeper_than_ih = tr_mean > ih_mean
    tl_ih_closer = abs(tl_mean - ih_mean) < abs(tr_mean - tl_mean)

    checks = [tr_deeper_than_tl, tr_deeper_than_ih, tl_ih_closer]
    n_pass = sum(checks)

    if n_pass == 3:
        verdict = "QUALITATIVE MATCH — TR heads are deeper than TL and IH, and TL/IH are closer to each other."
    elif n_pass >= 1:
        verdict = f"PARTIAL MATCH — {n_pass}/3 expected trends observed."
    else:
        verdict = "MISMATCH — TR heads are not deeper than TL/IH (unexpected)."

    # Significance info
    sig_notes = []
    for _, row in test_results.iterrows():
        sig = "significant" if row["significant_005"] else "not significant"
        sig_notes.append(f"{row['comparison']}: p={row['p_value']:.4f} ({sig})")

    observed_text = (
        f"Mean layers: TR={tr_mean:.2f}, TL={tl_mean:.2f}, IH={ih_mean:.2f}. "
        f"TR deeper than TL: {tr_deeper_than_tl}. "
        f"TR deeper than IH: {tr_deeper_than_ih}. "
        f"TL-IH closer than TR-TL: {tl_ih_closer}. "
        f"Mann-Whitney U: {'; '.join(sig_notes)}."
    )

    write_summary(
        f"{cfg.summaries_dir}/exp4_layer_distribution.txt",
        experiment_name="Experiment 4: Layer-wise Distribution of Special Heads",
        setup=(
            f"Model={cfg.model_name}, top {cfg.top_percent}% TR/TL/IH heads, "
            f"Mann-Whitney U test"
        ),
        expected=(
            "TR heads should occur in deeper layers than TL and IH heads. "
            "TL and IH distributions should be more similar to each other than to TR."
        ),
        observed=observed_text,
        verdict=verdict,
        caveats="With small head sets, statistical power may be limited.",
    )

    return verdict
