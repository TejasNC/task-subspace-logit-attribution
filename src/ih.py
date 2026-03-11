"""Induction Head (IH) score computation.

IH_{l,k} = sum_i sum_{j in I_i} Attn^{(l,k)}_{s(Q_i), j}
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def compute_ih_score_single(
    attentions: List[torch.Tensor],
    query_pos: int,
    demo_label_positions: List[List[int]],
) -> np.ndarray:
    """IH scores for all heads for a single prompt. Returns [n_layers, n_heads]."""
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    ih_scores = np.zeros((n_layers, n_heads))

    # Flatten all demo label positions
    all_label_pos = []
    for positions in demo_label_positions:
        all_label_pos.extend(positions)

    if len(all_label_pos) == 0:
        return ih_scores

    for l_idx in range(n_layers):
        attn = attentions[l_idx]
        for h_idx in range(n_heads):
            score = 0.0
            for pos in all_label_pos:
                if pos < attn.shape[-1]:
                    score += attn[0, h_idx, query_pos, pos].item()
            ih_scores[l_idx, h_idx] = score

    return ih_scores


def aggregate_ih_scores(all_ih_scores: List[np.ndarray]) -> np.ndarray:
    """Sum IH scores across multiple prompts."""
    return np.sum(np.stack(all_ih_scores, axis=0), axis=0)
