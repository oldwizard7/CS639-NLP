# MATH EDA Pipeline

This folder contains a reproducible EDA script for the MATH dataset (Hendrycks et al.) with Hugging Face fallback loading logic.

## Installation

```bash
cd /home/ctong29/cs639/Project/EDA/MATH
python -m pip install -r requirements.txt
```

Minimal install (no tokenizer, no plots):

```bash
python -m pip install -U datasets numpy
```

Tokenizer-aware lengths (recommended):

```bash
python -m pip install -U transformers sentencepiece
```

If you want plots:

```bash
python -m pip install -U matplotlib
```

## One-line Run

```bash
python math_eda.py --outdir outputs --tokenizer Qwen/Qwen2.5-1.5B-Instruct
```

Without tokenizer:

```bash
python math_eda.py --outdir outputs
```

Enable plots:

```bash
python math_eda.py --outdir outputs --plots
```

## Dataset Loading Order

The script tries:

1. `jeggers/competition_math` with config `original` (or your `--dataset --config`)
2. `HuggingFaceTB/MATH`
3. `EleutherAI/hendrycks_math` by concatenating the 7 subject configs into unified train/test

The selected source is logged and written into output files.

## Outputs

All outputs are written under `--outdir` (default: `outputs`):

- `eda_stats.json`: full machine-readable statistics
- `eda_summary.tex`: LaTeX subsection text with filled numbers
- `eda_tables.tex`: LaTeX tables (subject/level/answer type distributions)
- Optional `.png` plots when `--plots` is enabled

## Overleaf Inclusion

Upload `eda_summary.tex` and `eda_tables.tex` to your Overleaf project, then add:

```latex
\input{outputs/eda_summary.tex}
\input{outputs/eda_tables.tex}
```
