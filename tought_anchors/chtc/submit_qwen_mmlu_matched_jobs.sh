#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chtc_dir="$repo_root/chtc"

condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_correct_default.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_correct_forced_answer.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_incorrect_default.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_incorrect_forced_answer.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_instruct_correct_default.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_instruct_correct_forced_answer.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_instruct_incorrect_default.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_instruct_incorrect_forced_answer.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_oat_zero_correct_default.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_oat_zero_correct_forced_answer.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_oat_zero_incorrect_default.sub"
condor_submit "$chtc_dir/generated_qwen_anchor/mmlu_anchor_qwen25_math_7b_oat_zero_incorrect_forced_answer.sub"
