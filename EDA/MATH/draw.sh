#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/ctong29/miniconda3/envs/cs639-hw2/bin/python"
PROJECT_DIR="/home/ctong29/cs639/Project/EDA/MATH"
OUTDIR="${PROJECT_DIR}/outputs"
TOKENIZER="Qwen/Qwen2.5-1.5B-Instruct"

"${PYTHON_BIN}" -m pip install -r "${PROJECT_DIR}/requirements.txt"

"${PYTHON_BIN}" "${PROJECT_DIR}/math_eda.py" \
  --outdir "${OUTDIR}" \
  --tokenizer "${TOKENIZER}" \
  --plots
