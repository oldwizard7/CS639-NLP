# MATH Trajectory Sampling

Post-training 对推理准确率 vs 多样性的影响研究。对比 Qwen2.5-Math-7B 的 base / SFT / RLHF 三个变体。

## Setup

### 1. 下载模型

```bash
huggingface-cli download Qwen/Qwen2.5-Math-7B \
  --local-dir /staging/ctong29/models/Qwen2.5-Math-7B

huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct \
  --local-dir /staging/ctong29/models/Qwen2.5-Math-7B-Instruct

huggingface-cli download sail/Qwen2.5-Math-7B-Oat-Zero \
  --local-dir /staging/ctong29/models/Qwen2.5-Math-7B-Oat-Zero
```

### 2. 创建数据

当前保留两套数据流程：

- `competition_math` 50 题子集：旧实验，保留不动
- `Math500`：新的主评测集

```bash
# 旧的 50 题实验
python make_math_subset.py

# 新的 Math500（500 题）
python make_math500.py
```

会生成：

```bash
data/math_subset.jsonl
data/math500.jsonl
```

### 3. 启动 vLLM 并生成轨迹

每次只 serve 一个模型（单 GPU）：

```bash
# --- SFT (Instruct) ---
cd ~/local-llm-serve
source scripts/start_vllm.sh configs/qwen2.5-math-7b-instruct.env

cd ~/cs639/Project/math_trajectories
python generate.py \
  --model_tag sft \
  --model_name qwen2.5-math-7b-instruct \
  --input_file data/math500.jsonl \
  --output_file generations_math500_sft.jsonl \
  --api_base http://localhost:18502/v1

# --- Base ---
# (kill 上一个 vLLM, 换模型重启)
python generate.py \
  --model_tag base \
  --model_name qwen2.5-math-7b \
  --input_file data/math500.jsonl \
  --output_file generations_math500_base.jsonl \
  --api_base http://localhost:18501/v1

# --- DRPO (Oat-Zero) ---
python generate.py \
  --model_tag drpo \
  --model_name qwen2.5-math-7b-oat-zero \
  --input_file data/math500.jsonl \
  --output_file generations_math500_drpo.jsonl \
  --api_base http://localhost:18503/v1
```

HTCondor 版本同样支持参数化输入/输出路径。Math500 提交示例：

```bash
cd ~/cs639/Project/math_trajectories

condor_submit condor/submit_generate.sub \
  -append 'MODEL_TAG=sft' \
  -append 'MODEL_NAME=qwen2.5-math-7b-instruct' \
  -append 'CONFIG_FILE=configs/qwen2.5-math-7b-instruct.env' \
  -append 'INPUT_FILE=math_trajectories/data/math500.jsonl' \
  -append 'OUTPUT_FILE=output/generations_math500_sft.jsonl'

condor_submit condor/submit_generate.sub \
  -append 'MODEL_TAG=base' \
  -append 'MODEL_NAME=qwen2.5-math-7b' \
  -append 'CONFIG_FILE=configs/qwen2.5-math-7b.env' \
  -append 'INPUT_FILE=math_trajectories/data/math500.jsonl' \
  -append 'OUTPUT_FILE=output/generations_math500_base.jsonl'

condor_submit condor/submit_generate.sub \
  -append 'MODEL_TAG=drpo' \
  -append 'MODEL_NAME=qwen2.5-math-7b-oat-zero' \
  -append 'CONFIG_FILE=configs/qwen2.5-math-7b-oat-zero.env' \
  -append 'INPUT_FILE=math_trajectories/data/math500.jsonl' \
  -append 'OUTPUT_FILE=output/generations_math500_drpo.jsonl'
```

注意：作业运行时文件先写到远端 scratch 的 `output/` 下；作业结束后，Condor 会把这些文件回传到项目根目录，所以你本地最终会看到 `generations_math500_<tag>.jsonl`。

### 4. 评估

```bash
# 旧的 50 题实验
python eval.py --data data/math_subset.jsonl --generations generations_sft.jsonl

# Math500 单个模型
python eval.py --data data/math500.jsonl --generations generations_math500_sft.jsonl --out_dir results/math500

# Math500 所有模型
python eval.py --data data/math500.jsonl --generations generations_math500_*.jsonl --out_dir results/math500
```

## 输出

- `data/math_subset.jsonl` — 旧实验的 50 道 `competition_math` 题
- `data/math500.jsonl` — `Math500` 全部 500 题
- `generations_<tag>.jsonl` — 旧实验输出
- `generations_math500_<tag>.jsonl` — Math500 输出，每模型 1500 条轨迹 (500 题 x 3 samples)
- `results/metrics_<tag>.json` / `results/summary.csv` / `results/per_problem.csv` — 旧实验评估结果
- `results/math500/metrics_<tag>.json` / `results/math500/summary.csv` / `results/math500/per_problem.csv` — Math500 评估结果
