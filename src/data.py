"""SST-2 dataset loading and splitting."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset

from .utils import Config


def load_sst2(cfg: Config) -> Dict:
    """Load SST-2 from GLUE."""
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)
    return ds


def sample_demonstrations(
    train_data,
    n_shots: int,
    seed: int,
    label_words: List[str],
) -> List[Dict]:
    """Sample balanced demonstrations (n_shots//2 per class)."""
    rng = np.random.RandomState(seed)
    label_map = {0: label_words[1], 1: label_words[0]}

    demos = []
    per_class = n_shots // 2
    for label_idx in [0, 1]:
        candidates = [ex for ex in train_data if ex["label"] == label_idx]
        chosen_indices = rng.choice(len(candidates), size=per_class, replace=False)
        for idx in chosen_indices:
            ex = candidates[idx]
            demos.append({
                "sentence": ex["sentence"],
                "label_word": label_map[ex["label"]],
                "label_idx": ex["label"],
            })

    # Shuffle
    rng.shuffle(demos)
    return demos


def get_query_examples(
    validation_data,
    n_queries: int,
    seed: int,
    label_words: List[str],
    offset: int = 0,
) -> List[Dict]:
    """Sample query examples from the validation set."""
    label_map = {0: label_words[1], 1: label_words[0]}
    rng = np.random.RandomState(seed + 1)  # Different seed from demos

    all_indices = list(range(len(validation_data)))
    rng.shuffle(all_indices)
    selected = all_indices[offset : offset + n_queries]

    queries = []
    for idx in selected:
        ex = validation_data[idx]
        queries.append({
            "sentence": ex["sentence"],
            "label_word": label_map[ex["label"]],
            "label_idx": ex["label"],
        })
    return queries
