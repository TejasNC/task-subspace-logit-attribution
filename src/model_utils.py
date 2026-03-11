"""Model loading and architecture helpers for Llama."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import Config, get_device


def load_model_and_tokenizer(cfg: Config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer; forces eager attention for weight extraction."""
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_bnb = "bnb" in cfg.model_name.lower() or "4bit" in cfg.model_name.lower()
    if is_bnb:
        # Quantized model — let the checkpoint's own config handle dtype/quantization
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            device_map="auto",
            attn_implementation="eager",
        )
        dtype = next(model.parameters()).dtype
    else:
        dtype = torch.float16 if (cfg.use_fp16 and device.type == "cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
            attn_implementation="eager",
        )
    model.config.output_attentions = True
    model.eval()

    print(f"Model: {cfg.model_name}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Heads: {model.config.num_attention_heads}")
    print(f"  KV Heads: {getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)}")
    print(f"  Hidden: {model.config.hidden_size}")
    print(f"  Dtype: {dtype}")
    print(f"  Device: {next(model.parameters()).device}")

    return model, tokenizer


def get_model_info(model) -> dict:
    """Extract key architecture info."""
    config = model.config
    return {
        "n_layers": config.num_hidden_layers,
        "n_heads": config.num_attention_heads,
        "n_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "hidden_size": config.hidden_size,
        "head_dim": config.hidden_size // config.num_attention_heads,
        "vocab_size": config.vocab_size,
    }


def _dequantize_weight(module_or_weight) -> torch.Tensor:
    """Return the dequantized weight tensor (handles bnb 4-bit and regular modules)."""
    try:
        import bitsandbytes as bnb
        # Module path (most reliable): check the module type
        if isinstance(module_or_weight, bnb.nn.Linear4bit):
            w = module_or_weight.weight
            return bnb.functional.dequantize_4bit(w.data, w.quant_state).detach()
        # Weight path: check the parameter type
        if isinstance(module_or_weight, bnb.nn.Params4bit):
            return bnb.functional.dequantize_4bit(
                module_or_weight.data, module_or_weight.quant_state
            ).detach()
    except (ImportError, AttributeError):
        pass
    # Fallback: quant_state attribute (other bnb versions)
    if hasattr(module_or_weight, 'quant_state') and module_or_weight.quant_state is not None:
        try:
            import bitsandbytes as bnb
            return bnb.functional.dequantize_4bit(
                module_or_weight.data, module_or_weight.quant_state
            ).detach()
        except Exception:
            pass
    # Regular module or tensor
    if isinstance(module_or_weight, nn.Module):
        return module_or_weight.weight.detach()
    return module_or_weight.detach()


def get_unembedding_matrix(model) -> torch.Tensor:
    """W_U from lm_head, shape [vocab_size, hidden_size]."""
    return _dequantize_weight(model.lm_head)


def get_label_unembed_vectors(
    model, tokenizer, label_words: list[str]
) -> torch.Tensor:
    """Unembedding vectors for each label word, shape [n_labels, hidden_size]."""
    W_U = get_unembedding_matrix(model)

    vectors = []
    for lw in label_words:
        tok_ids = tokenizer(" " + lw, add_special_tokens=False)["input_ids"]
        vectors.append(W_U[tok_ids[0]])

    return torch.stack(vectors, dim=0)


def get_attention_layers(model) -> nn.ModuleList:
    """Return the list of transformer layer modules."""
    return model.model.layers


def get_head_out_proj_slice(model, layer_idx: int, head_idx: int) -> torch.Tensor:
    """W_O slice for a single head, shape [hidden_size, head_dim]."""
    info = get_model_info(model)
    head_dim = info["head_dim"]
    layer = model.model.layers[layer_idx]
    W_o = _dequantize_weight(layer.self_attn.o_proj)  # [hidden_size, hidden_size]
    start = head_idx * head_dim
    end = start + head_dim
    return W_o[:, start:end]  # [hidden_size, head_dim]
