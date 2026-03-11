"""Experiment 5: Attention distribution analysis of TR, TL, IH, and random heads."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from .utils import Config, write_summary


def compute_token_spans(
    prompt_text: str,
    query_sentence: str,
    tokenizer,
) -> List[int]:
    """Token indices corresponding to the query sentence in the prompt."""
    # Find the query sentence's last occurrence (it's always at the end)
    q_char_start = prompt_text.rfind(query_sentence)
    if q_char_start < 0:
        return []
    q_char_end = q_char_start + len(query_sentence)

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"][0]
    seq_len = offsets.shape[0]

    query_token_indices = []
    for tok_idx in range(seq_len):
        tok_start, tok_end = offsets[tok_idx].tolist()
        if tok_end == 0 and tok_start == 0 and tok_idx > 0:
            continue
        if tok_start < q_char_end and tok_end > q_char_start:
            query_token_indices.append(tok_idx)

    return query_token_indices


def compute_attention_distribution_single(
    attentions: List[torch.Tensor],
    query_pos: int,
    demo_label_positions: List[List[int]],
    query_token_positions: List[int],
    head_sets: Dict[str, List[Tuple[int, int]]],
    seq_len: int,
) -> List[Dict]:
    """Attention allocation (demo labels, query, other) for each head in each set."""
    # Flatten demo label positions
    demo_label_set = set()
    for positions in demo_label_positions:
        demo_label_set.update(positions)

    query_set = set(query_token_positions)
    # Remove any overlap (query tokens that are also label tokens shouldn't double-count)
    query_only = query_set - demo_label_set
    other_set = set(range(seq_len)) - demo_label_set - query_only

    rows = []
    for head_type, heads in head_sets.items():
        for layer, head in heads:
            if layer >= len(attentions):
                continue
            attn = attentions[layer]
            if head >= attn.shape[1]:
                continue

            attn_row = attn[0, head, query_pos, :].float().cpu().numpy()

            attn_demo = sum(attn_row[p] for p in demo_label_set if p < len(attn_row))
            attn_q = sum(attn_row[p] for p in query_only if p < len(attn_row))
            attn_o = sum(attn_row[p] for p in other_set if p < len(attn_row))

            rows.append({
                "head_type": head_type,
                "layer": int(layer),
                "head": int(head),
                "attn_demo_labels": float(attn_demo),
                "attn_query": float(attn_q),
                "attn_other": float(attn_o),
            })

    return rows


def aggregate_by_head(
    per_prompt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean attention per head across all prompts."""
    grouped = per_prompt_df.groupby(["head_type", "layer", "head"]).agg(
        mean_attn_demo_labels=("attn_demo_labels", "mean"),
        mean_attn_query=("attn_query", "mean"),
        mean_attn_other=("attn_other", "mean"),
        n_prompts=("attn_demo_labels", "count"),
    ).reset_index()
    return grouped


def aggregate_by_head_type(
    per_prompt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean attention per head type across all heads and prompts."""
    grouped = per_prompt_df.groupby("head_type").agg(
        mean_attn_demo_labels=("attn_demo_labels", "mean"),
        mean_attn_query=("attn_query", "mean"),
        mean_attn_other=("attn_other", "mean"),
        std_attn_demo_labels=("attn_demo_labels", "std"),
        std_attn_query=("attn_query", "std"),
        n_observations=("attn_demo_labels", "count"),
    ).reset_index()
    grouped["se_attn_demo_labels"] = grouped["std_attn_demo_labels"] / np.sqrt(grouped["n_observations"])
    grouped["se_attn_query"] = grouped["std_attn_query"] / np.sqrt(grouped["n_observations"])
    return grouped


def generate_attention_distribution_summary(
    type_agg: pd.DataFrame,
    cfg: Config,
) -> str:
    """Compare observed attention patterns to paper expectations. Returns verdict."""
    means = {row["head_type"]: row for _, row in type_agg.iterrows()}

    tr_demo = means.get("TR", {}).get("mean_attn_demo_labels", np.nan)
    tl_demo = means.get("TL", {}).get("mean_attn_demo_labels", np.nan)
    tr_query = means.get("TR", {}).get("mean_attn_query", np.nan)
    tl_query = means.get("TL", {}).get("mean_attn_query", np.nan)
    rand_demo = means.get("Random", {}).get("mean_attn_demo_labels", np.nan)
    rand_query = means.get("Random", {}).get("mean_attn_query", np.nan)

    # Check paper hypotheses
    tr_more_demo_than_tl = tr_demo > tl_demo
    tr_more_demo_than_rand = tr_demo > rand_demo
    tl_more_query_than_tr = tl_query > tr_query
    tl_more_query_than_rand = tl_query > rand_query

    checks = [tr_more_demo_than_tl, tr_more_demo_than_rand, tl_more_query_than_tr, tl_more_query_than_rand]
    n_pass = sum(checks)

    if n_pass == 4:
        verdict = "QUALITATIVE MATCH — TR heads attend more to demo labels, TL heads attend more to query tokens."
    elif n_pass >= 2:
        verdict = f"PARTIAL MATCH — {n_pass}/4 expected attention patterns observed."
    else:
        verdict = "MISMATCH — Attention patterns do not match paper expectations."

    observed_text = (
        f"Mean attn to demo labels: TR={tr_demo:.4f}, TL={tl_demo:.4f}, Random={rand_demo:.4f}. "
        f"Mean attn to query: TR={tr_query:.4f}, TL={tl_query:.4f}, Random={rand_query:.4f}. "
        f"TR > TL on demo labels: {tr_more_demo_than_tl}. "
        f"TL > TR on query: {tl_more_query_than_tr}."
    )

    write_summary(
        f"{cfg.summaries_dir}/exp5_attention_distribution.txt",
        experiment_name="Experiment 5: Attention Distribution of Special Heads",
        setup=(
            f"Model={cfg.model_name}, {cfg.n_shots}-shot ICL on SST-2, "
            f"top {cfg.top_percent}% TR/TL/IH + random heads, "
            f"{cfg.n_scoring_queries} scoring queries"
        ),
        expected=(
            "TR heads should allocate more attention to demonstration label tokens. "
            "TL heads should allocate more attention to query tokens. "
            "Random heads should not show structured patterns."
        ),
        observed=observed_text,
        verdict=verdict,
        caveats="Attention is measured only at the final prediction position.",
    )

    return verdict
