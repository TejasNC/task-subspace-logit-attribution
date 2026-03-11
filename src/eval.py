"""Prediction parsing, accuracy, and TR ratio."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def parse_prediction(
    logits: torch.Tensor,
    query_pos: int,
    tokenizer,
    label_words: List[str],
) -> Dict:
    """Map the top predicted token at query_pos to a task label."""
    # Get logits at query position
    query_logits = logits[0, query_pos, :]
    pred_token_id = query_logits.argmax().item()
    pred_token_str = tokenizer.decode([pred_token_id]).strip().lower()

    # Try to map to a label
    parsed_label = None
    for lw in label_words:
        if pred_token_str.startswith(lw.lower()[:3]):
            parsed_label = lw
            break
        # Also check if the token id matches the first token of any label
        label_first_tok = tokenizer(" " + lw, add_special_tokens=False)["input_ids"][0]
        if pred_token_id == label_first_tok:
            parsed_label = lw
            break

    return {
        "predicted_token_id": pred_token_id,
        "predicted_token": pred_token_str,
        "parsed_label": parsed_label,
        "is_valid": parsed_label is not None,
    }


def compute_accuracy(
    predictions: List[Dict],
    queries: List[Dict],
) -> float:
    """Fraction of valid predictions that match the ground truth."""
    correct = 0
    total = 0
    for pred, query in zip(predictions, queries):
        if pred["is_valid"]:
            total += 1
            if pred["parsed_label"] == query["label_word"]:
                correct += 1
        else:
            total += 1  # count invalid as wrong

    return correct / total if total > 0 else 0.0


def compute_tr_ratio(predictions: List[Dict]) -> float:
    """Fraction of predictions mapping to a valid task label."""
    if not predictions:
        return 0.0
    valid = sum(1 for p in predictions if p["is_valid"])
    return valid / len(predictions)


def compute_label_logit_scores(
    logits: torch.Tensor,
    query_pos: int,
    tokenizer,
    label_words: List[str],
) -> Dict[str, float]:
    """Get raw logit scores for each label word at the query position."""
    query_logits = logits[0, query_pos, :]
    result = {}
    for lw in label_words:
        tok_id = tokenizer(" " + lw, add_special_tokens=False)["input_ids"][0]
        result[lw] = query_logits[tok_id].item()
    return result
