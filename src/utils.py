"""Config, seeds, paths, and summary generation."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class Config:
    """Experiment configuration."""
    model_name: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    n_shots: int = 8
    n_scoring_queries: int = 50
    n_eval_queries: int = 200
    top_percent: float = 3.0
    label_words: List[str] = field(default_factory=lambda: ["positive", "negative"])
    prompt_template: str = "{sentence} Sentiment: {label}"
    query_template: str = "{sentence} Sentiment:"
    demo_separator: str = "\n"
    seed: int = 42
    batch_size: int = 1
    use_fp16: bool = True
    epsilon: float = 1e-8
    ablation_conditions: List[str] = field(
        default_factory=lambda: [
            "baseline", "tr_ablation", "tl_ablation", "ih_ablation", "random_ablation"
        ]
    )
    percentile_thresholds: List[int] = field(default_factory=lambda: [3, 5, 10, 20])
    output_dir: str = "outputs"
    cache_dir: str = "outputs/cache"
    scores_dir: str = "outputs/scores"
    plots_dir: str = "outputs/plots"
    tables_dir: str = "outputs/tables"
    summaries_dir: str = "outputs/summaries"
    plot_dpi: int = 300
    plot_format: List[str] = field(default_factory=lambda: ["png", "pdf"])


def load_config(path: str = "configs/default_config.json") -> Config:
    """Load config from JSON, falling back to defaults for missing keys."""
    cfg = Config()
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: Config) -> None:
    """Create all output directories."""
    for d in [cfg.output_dir, cfg.cache_dir, cfg.scores_dir,
              cfg.plots_dir, cfg.tables_dir, cfg.summaries_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    """Save a JSON-serializable object."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str) -> Any:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def write_summary(
    path: str,
    experiment_name: str,
    setup: str,
    expected: str,
    observed: str,
    verdict: str,
    caveats: str = "",
) -> None:
    """Write an expected-vs-observed summary file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Experiment: {experiment_name}",
        f"{'=' * 60}",
        f"Setup: {setup}",
        "",
        f"Expected: {expected}",
        "",
        f"Observed: {observed}",
        "",
        f"Verdict: {verdict}",
    ]
    if caveats:
        lines += ["", f"Caveats: {caveats}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary saved to {path}")


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_gpu_info() -> None:
    """Print GPU memory info if available."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            name = torch.cuda.get_device_properties(i).name
            print(f"GPU {i}: {name}, {total:.1f} GB")
    else:
        print("No GPU available — running on CPU (will be slow).")
