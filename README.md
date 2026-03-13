# Task-Subspace Logit Attribution: Scaled Reproduction

A Kaggle-friendly reproduction of the core findings from **"Localizing Task Recognition and Task Learning in In-Context Learning via Attention Head Analysis"**, focused on the SST-2 sentiment classification task using Llama-3.2-3B-Instruct-bnb-4bit (4-bit quantized via bitsandbytes).

## Overview

This project identifies three classes of attention heads using Task-Subspace Logit Attribution:

- **Task Recognition (TR) heads**: Project head outputs onto the task-label subspace
- **Task Learning (TL) heads**: Push predictions toward the correct label
- **Induction Heads (IH)**: Attend strongly from the query position to demonstration label tokens

The codebase reproduces three key experiments:
1. **Head identification**: Compute TR, TL, and IH scores and extract top 3% heads
2. **Overlap analysis**: Measure Jaccard overlap, rank correlations, and conditional percentile overlap
3. **Ablation study**: Zero-out selected head sets and measure accuracy/TR-ratio impact

## Project Structure

```
├── main.ipynb                  # Main notebook — run this end-to-end
├── configs/
│   └── default_config.json     # Experiment configuration
├── src/
│   ├── utils.py                # Config loading, seeds, paths
│   ├── data.py                 # Dataset loading and splitting
│   ├── prompts.py              # Prompt construction and token tracking
│   ├── model_utils.py          # Model loading and architecture helpers
│   ├── hooks.py                # Forward hooks for attention/head-output extraction
│   ├── scores.py               # TR and TL score computation
│   ├── ih.py                   # Induction Head score computation
│   ├── ablation.py             # Head ablation during inference
│   ├── eval.py                 # Evaluation metrics and label parsing
│   └── plots.py                # Publication-quality plotting
├── outputs/
│   ├── cache/                  # Cached intermediate results
│   ├── scores/                 # Per-head score CSVs and JSONs
│   ├── plots/                  # Generated figures (PNG + PDF)
│   ├── tables/                 # CSV tables
│   └── summaries/              # Expected-vs-observed text summaries
├── requirements.txt
└── README.md
```

## Quick Start

### On Kaggle
1. Upload this repository as a Kaggle dataset or use `Add Data`
2. Create a new notebook attached to a GPU runtime
3. Open `main.ipynb` and run all cells top-to-bottom

### Locally
```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Configuration

Edit `configs/default_config.json` to change:
- Model checkpoint
- Number of ICL shots
- Number of scoring/eval queries
- Top-head percentage threshold
- Output paths

## Requirements

- Python 3.9+
- PyTorch 2.0+
- bitsandbytes >= 0.46.1
- A GPU with >= 8 GB VRAM (Kaggle T4 works with 4-bit quantization)
- No gated model access required (uses the public `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` checkpoint)

## License

Research reproduction — for educational and academic use.
