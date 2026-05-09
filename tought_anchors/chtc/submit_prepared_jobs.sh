#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chtc_dir="$repo_root/chtc"

condor_submit "$chtc_dir/generated/math_qwen25_math_7b.sub"
condor_submit "$chtc_dir/generated/math_qwen25_math_7b_instruct.sub"
condor_submit "$chtc_dir/generated/math_qwen25_math_7b_oat_zero.sub"
condor_submit "$chtc_dir/generated/mmlu_qwen25_math_7b.sub"
condor_submit "$chtc_dir/generated/mmlu_qwen25_math_7b_instruct.sub"
condor_submit "$chtc_dir/generated/mmlu_qwen25_math_7b_oat_zero.sub"
