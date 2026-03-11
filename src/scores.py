"""TR and TL score computation (Task-Subspace Logit Attribution)."""

from __future__ import annotations

from typing import Tuple

import torch
import numpy as np


def compute_projection_matrix(W_U_Y: torch.Tensor) -> torch.Tensor:
    """P = W^T (W W^T)^{-1} W  where W = W_U_Y [n_labels, d]."""
    W = W_U_Y.float()
    WWT_inv = torch.linalg.inv(W @ W.T)
    return W.T @ WWT_inv @ W


def compute_tr_score(head_output: torch.Tensor, proj_matrix: torch.Tensor) -> float:
    """TR_{l,k} = || P @ a ||_2"""
    projected = proj_matrix @ head_output.float()
    return projected.norm(p=2).item()


def compute_tl_score(
    head_output: torch.Tensor,
    W_U_Y: torch.Tensor,
    true_label_idx: int,
    proj_matrix: torch.Tensor,
    epsilon: float = 1e-8,
) -> float:
    """TL_{l,k} = a^T (W_U^{y*} - W_U^{y'}) / (|| P @ a ||_2 + eps)"""
    a = head_output.float()
    other_idx = 1 - true_label_idx
    diff = W_U_Y[true_label_idx].float() - W_U_Y[other_idx].float()
    numerator = torch.dot(a, diff).item()
    denominator = (proj_matrix @ a).norm(p=2).item() + epsilon
    return numerator / denominator


def compute_all_scores(
    head_outputs: torch.Tensor,
    W_U_Y: torch.Tensor,
    true_label_idx: int,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute TR and TL scores for all heads. Returns (tr_scores, tl_scores) as [n_layers, n_heads]."""
    n_layers, n_heads, hidden_size = head_outputs.shape
    device = W_U_Y.device
    head_outputs = head_outputs.to(device)
    proj_matrix = compute_projection_matrix(W_U_Y)

    tr_scores = np.zeros((n_layers, n_heads))
    tl_scores = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        for k in range(n_heads):
            a = head_outputs[l, k]  # [hidden_size]
            tr_scores[l, k] = compute_tr_score(a, proj_matrix)
            tl_scores[l, k] = compute_tl_score(a, W_U_Y, true_label_idx, proj_matrix, epsilon)

    return tr_scores, tl_scores
