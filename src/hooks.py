"""Forward hooks for extracting attention weights and per-head outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .model_utils import get_model_info


@dataclass
class HeadExtractionResult:
    """Attentions, per-head outputs, and logits from a single forward pass."""
    attentions: List[torch.Tensor]
    head_outputs: torch.Tensor
    logits: torch.Tensor


class ValueStateCapture:
    """Captures value states via hooks for per-head output reconstruction."""

    def __init__(self, model, n_layers: int, n_heads: int, n_kv_heads: int, head_dim: int):
        self.model = model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.heads_per_kv = n_heads // n_kv_heads

        self._value_states: Dict[int, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _make_v_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            self._value_states[layer_idx] = output.detach()
        return hook_fn

    def register_hooks(self):
        layers = self.model.model.layers
        for l_idx in range(self.n_layers):
            v_proj = layers[l_idx].self_attn.v_proj
            h = v_proj.register_forward_hook(self._make_v_hook(l_idx))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._value_states.clear()

    def compute_head_outputs(
        self,
        attentions: List[torch.Tensor],
        query_pos: int,
    ) -> torch.Tensor:
        """Per-head output vector at query_pos. Returns [n_layers, n_heads, hidden_size]."""
        device = attentions[0].device
        hidden_size = self.n_heads * self.head_dim
        # We'll recompute head_dim from W_o later; for now use model's head_dim
        head_outputs = torch.zeros(
            self.n_layers, self.n_heads, hidden_size,
            dtype=attentions[0].dtype, device=device,
        )

        layers = self.model.model.layers
        for l_idx in range(self.n_layers):
            attn = attentions[l_idx]
            v_states = self._value_states[l_idx]
            batch_size, seq_len, _ = v_states.shape
            v_states = v_states.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            v_states = v_states.permute(0, 2, 1, 3)

            from .model_utils import _dequantize_weight
            W_o = _dequantize_weight(layers[l_idx].self_attn.o_proj)

            for h_idx in range(self.n_heads):
                kv_idx = h_idx // self.heads_per_kv
                attn_weights = attn[0, h_idx, query_pos, :]
                v = v_states[0, kv_idx, :, :]
                weighted_v = torch.matmul(attn_weights, v)
                o_slice = W_o[:, h_idx * self.head_dim : (h_idx + 1) * self.head_dim]
                head_out = torch.matmul(o_slice, weighted_v)
                head_outputs[l_idx, h_idx] = head_out

        return head_outputs


def extract_head_info(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    query_pos: int,
) -> HeadExtractionResult:
    """Forward pass returning attention weights and per-head output vectors."""
    info = get_model_info(model)
    n_layers = info["n_layers"]
    n_heads = info["n_heads"]
    n_kv_heads = info["n_kv_heads"]
    head_dim = info["head_dim"]

    capture = ValueStateCapture(model, n_layers, n_heads, n_kv_heads, head_dim)
    capture.register_hooks()

    try:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
            )

        # outputs.attentions: tuple of [batch, n_heads, seq, seq] per layer
        attentions = [a.detach().cpu() for a in outputs.attentions]
        logits = outputs.logits.detach().cpu()

        # Compute per-head outputs using captured value states + attention weights
        # Use GPU attentions for computation
        gpu_attentions = [a.detach() for a in outputs.attentions]
        head_outputs = capture.compute_head_outputs(gpu_attentions, query_pos)
        head_outputs = head_outputs.cpu()

    finally:
        capture.remove_hooks()

    return HeadExtractionResult(
        attentions=attentions,
        head_outputs=head_outputs,
        logits=logits,
    )
