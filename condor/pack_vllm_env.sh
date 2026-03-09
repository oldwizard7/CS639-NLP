#!/bin/bash
# 创建并打包 vLLM conda 环境到 staging
# 用法：bash pack_vllm_env.sh
set -e

ENV_NAME="vllm-serve"
OUTPUT="/staging/ctong29/conda_pack/vllm_env.tgz"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n ${ENV_NAME} python=3.11 -y

echo "=== Installing vLLM ==="
conda activate ${ENV_NAME}
pip install vllm

echo "=== Packing environment ==="
conda-pack -n ${ENV_NAME} -o "${OUTPUT}"

echo "Done! Packed to ${OUTPUT}"
ls -lh "${OUTPUT}"
