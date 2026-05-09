#!/bin/bash
set -e

# math_trajectories generate job (HTCondor GPU)
# 用法: condor_submit condor/submit_generate.sub

mkdir -p output

echo "=== Working directory: $(pwd) ==="
ls -la
echo "=========================="

# --- 1. 启动 vLLM ---
# start_vllm.sh 会设置 PYTHONHOME / PATH 指向 vllm_env，并设置 VLLM_PID / VLLM_PORT
CONFIG_FILE="${CONFIG_FILE:-configs/qwen2.5-math-7b-instruct.env}"

if [ -f "local-llm-serve/scripts/start_vllm.sh" ]; then
    source local-llm-serve/scripts/start_vllm.sh "${CONFIG_FILE}"
else
    echo "ERROR: Cannot find local-llm-serve/scripts/start_vllm.sh"
    ls -la
    exit 1
fi

# --- 2. 确保 openai SDK 可用 ---
# vllm_env 不一定自带 openai client，快速安装（通常 <30s）
if ! python -c "import openai" 2>/dev/null; then
    echo "=== Installing openai SDK ==="
    pip install -q openai
fi
python -c "import openai; print(f'openai version: {openai.__version__}')"

# --- 3. 读取任务参数（从 environment 变量或默认值）---
MODEL_TAG="${MODEL_TAG:-sft}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-math-7b-instruct}"
PORT="${VLLM_PORT:-18502}"
INPUT_FILE="${INPUT_FILE:-math_trajectories/data/math_subset.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-output/generations_${MODEL_TAG}.jsonl}"

echo "=== Running generate.py ==="
echo "  MODEL_TAG:  ${MODEL_TAG}"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  PORT:       ${PORT}"
echo "  INPUT_FILE: ${INPUT_FILE}"
echo "  OUTPUT_FILE:${OUTPUT_FILE}"

python math_trajectories/generate.py \
    --model_tag    "${MODEL_TAG}" \
    --model_name   "${MODEL_NAME}" \
    --api_base     "http://localhost:${PORT}/v1" \
    --input_file   "${INPUT_FILE}" \
    --output_file  "${OUTPUT_FILE}" \
    --num_samples  3 \
    --temperature  0.7 \
    --top_p        0.95 \
    --max_tokens   2048 \
    --seed         42

EXIT_CODE=$?

# --- 4. 清理 vLLM ---
echo "=== Cleanup vLLM (PID=${VLLM_PID:-unknown}) ==="
kill "${VLLM_PID}" 2>/dev/null || true
wait "${VLLM_PID}" 2>/dev/null || true

echo "Done (exit code: ${EXIT_CODE})"
exit "${EXIT_CODE}"
