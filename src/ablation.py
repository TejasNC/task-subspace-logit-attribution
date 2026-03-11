"""Ablation: zero-out selected attention heads during inference."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .model_utils import get_model_info


class HeadAblationHooks:
    """Forward pre-hooks that zero the o_proj input slice of specified heads."""

    def __init__(self, model, heads_to_ablate: Set[Tuple[int, int]]):
        self.model = model
        self.heads_to_ablate = heads_to_ablate
        self._hooks: List[Any] = []

        info = get_model_info(model)
        self.head_dim = info["head_dim"]
        self.n_heads = info["n_heads"]

        # Group by layer for efficiency
        self._layer_heads: Dict[int, List[int]] = {}
        for (l, h) in heads_to_ablate:
            self._layer_heads.setdefault(l, []).append(h)

    def register_hooks(self):
        """Attach hooks that zero selected heads."""
        layers = self.model.model.layers

        for layer_idx, head_indices in self._layer_heads.items():
            o_proj = layers[layer_idx].self_attn.o_proj
            head_dim = self.head_dim

            def make_pre_hook(h_indices, hd):
                def hook_fn(module, args):
                    x = args[0]
                    x_modified = x.clone()
                    for h_idx in h_indices:
                        start = h_idx * hd
                        end = start + hd
                        x_modified[:, :, start:end] = 0.0
                    return (x_modified,) + args[1:]
                return hook_fn

            h = o_proj.register_forward_pre_hook(make_pre_hook(head_indices, head_dim))
            self._hooks.append(h)

    def remove_hooks(self):
        """Remove all ablation hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def run_with_ablation(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    heads_to_ablate: Set[Tuple[int, int]],
) -> torch.Tensor:
    """Forward pass with specified heads zeroed out. Returns logits."""
    if not heads_to_ablate:
        device = next(model.parameters()).device
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
        return outputs.logits.detach().cpu()

    ablation = HeadAblationHooks(model, heads_to_ablate)
    ablation.register_hooks()
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
        return outputs.logits.detach().cpu()
    finally:
        ablation.remove_hooks()
