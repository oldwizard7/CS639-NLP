#!/bin/bash
# 通用 vLLM 启动脚本 —— 任何项目都可以 source 使用
# 用法：source start_vllm.sh [config_file]
#   启动后设置 VLLM_PID, VLLM_PORT, VLLM_MODEL_NAME 变量
#   调用方负责最后 kill $VLLM_PID
#
# 前置条件：
#   - vllm_env.tgz 在 /staging/ctong29/conda_pack/
#   - 模型权重在 /staging/ 上

CONFIG_FILE="${1:-configs/qwen2.5-7b.env}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载模型配置
source "${SCRIPT_DIR}/../${CONFIG_FILE}"

echo "=== Unpacking vLLM environment ==="
mkdir -p vllm_env
tar xzf /staging/ctong29/conda_pack/vllm_env.tgz -C vllm_env

# --- 设置 Python 环境 ---
# conda-pack 的 Python 二进制内嵌了 build prefix 路径，conda-unpack 只修复文本文件
# 必须始终保持 PYTHONHOME 指向解压目录，Python 才能找到标准库
VLLM_ENV_ABS="$(pwd)/vllm_env"
export PYTHONHOME="${VLLM_ENV_ABS}"
export PATH="${VLLM_ENV_ABS}/bin:${PATH}"

# 运行 conda-unpack 修复文本文件（脚本、.pth、配置等）
"${VLLM_ENV_ABS}/bin/python" "${VLLM_ENV_ABS}/bin/conda-unpack"
echo "conda-unpack done (exit=$?)"

# 验证 Python 完整可用（import encodings 是关键检查）
python -c "import sys; print(f'Python {sys.version}, prefix={sys.prefix}')"

# HTCondor 可能将 CUDA_VISIBLE_DEVICES 设为 UUID 格式（GPU-xxxx），vLLM 需要整数索引
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    echo "Converting CUDA_VISIBLE_DEVICES from UUID ('${CUDA_VISIBLE_DEVICES}') to index 0"
    export CUDA_VISIBLE_DEVICES=0
fi

echo "=== Starting vLLM server (model: ${VLLM_MODEL_NAME}) ==="
python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_PATH}" \
    --served-model-name "${VLLM_MODEL_NAME}" \
    --port ${VLLM_PORT} \
    --enable-auto-tool-choice \
    --tool-call-parser "${VLLM_TOOL_PARSER}" \
    --max-model-len ${VLLM_MAX_MODEL_LEN} \
    --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL} &
VLLM_PID=$!

# 绕过 CHTC Squid 代理（worker 节点默认设置了 http_proxy，会拦截 localhost 请求）
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1"

echo "=== Waiting for vLLM ready ==="
for i in $(seq 1 180); do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready after $((i * 5))s"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "ERROR: vLLM startup timeout (900s)"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done

export VLLM_PID VLLM_PORT VLLM_MODEL_NAME
echo "vLLM running (PID=${VLLM_PID}, port=${VLLM_PORT})"
