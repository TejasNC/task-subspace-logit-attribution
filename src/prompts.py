"""Prompt construction with token position tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class PromptInfo:
    """Constructed prompt with tracked token positions."""
    prompt_text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    query_position: int
    demo_label_positions: List[List[int]]
    true_label_word: str
    true_label_idx: int


def build_prompt(
    demos: List[Dict],
    query: Dict,
    tokenizer,
    cfg,
) -> PromptInfo:
    """Build an 8-shot ICL prompt and track demo label token positions."""
    separator = cfg.demo_separator

    demo_strings = []
    for d in demos:
        demo_str = cfg.prompt_template.format(sentence=d["sentence"], label=d["label_word"])
        demo_strings.append(demo_str)

    query_str = cfg.query_template.format(sentence=query["sentence"])
    full_prompt = separator.join(demo_strings) + separator + query_str

    # Tokenize with offset mapping to map character spans to token indices
    use_offset_mapping = hasattr(tokenizer, "is_fast") and tokenizer.is_fast
    if use_offset_mapping:
        encoded = tokenizer(
            full_prompt, return_tensors="pt", add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping")[0]
    else:
        encoded = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True)
        offset_mapping = None

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    seq_len = input_ids.shape[1]
    query_position = seq_len - 1

    # Find demo label token positions
    demo_label_positions = []

    char_cursor = 0
    for i, d in enumerate(demos):
        full_demo = cfg.prompt_template.format(sentence=d["sentence"], label=d["label_word"])
        if i > 0:
            char_cursor += len(separator)

        pre_label_part = cfg.prompt_template.split("{label}")[0].format(sentence=d["sentence"])
        label_char_start = char_cursor + len(pre_label_part)
        label_char_end = label_char_start + len(d["label_word"])

        if use_offset_mapping and offset_mapping is not None:
            label_token_indices = []
            for tok_idx in range(seq_len):
                tok_start, tok_end = offset_mapping[tok_idx].tolist()
                if tok_end == 0 and tok_start == 0 and tok_idx > 0:
                    continue
                if tok_start < label_char_end and tok_end > label_char_start:
                    label_token_indices.append(tok_idx)
        else:
            # Fallback: search for label token IDs in the expected region
            label_tok_ids = tokenizer(" " + d["label_word"], add_special_tokens=False)["input_ids"]
            ids_flat = input_ids[0].tolist()
            label_token_indices = []
            approx_tok_pos = int(label_char_start / max(1, len(full_prompt)) * seq_len)
            search_start = max(0, approx_tok_pos - 10)
            search_end = min(seq_len, approx_tok_pos + 20)
            for scan_start in range(search_start, search_end):
                if ids_flat[scan_start:scan_start + len(label_tok_ids)] == label_tok_ids:
                    label_token_indices = list(range(scan_start, scan_start + len(label_tok_ids)))
                    break

        demo_label_positions.append(label_token_indices)
        char_cursor += len(full_demo)

    return PromptInfo(
        prompt_text=full_prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        query_position=query_position,
        demo_label_positions=demo_label_positions,
        true_label_word=query["label_word"],
        true_label_idx=cfg.label_words.index(query["label_word"]),
    )


def get_label_token_ids(tokenizer, label_words: List[str]) -> Dict[str, List[int]]:
    """Token IDs for each label word (space-prefixed tokenization)."""
    result = {}
    for lw in label_words:
        # Tokenize with a space prefix to get the typical in-context tokenization
        toks = tokenizer(" " + lw, add_special_tokens=False)["input_ids"]
        result[lw] = toks
    return result
