#!/usr/bin/env python3
"""
Render the Qwen2.5 Thought Anchors follow-up job matrix.

The matrix follows the paper-style black-box setup:
- MATH: selected challenging problems, correct/incorrect base traces, default and
  forced-answer 100-rollout continuations.
- MMLU: saved benchmark traces, correct/incorrect selected examples, default and
  forced-answer 100-rollout continuations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "chtc" / "generated_qwen_anchor"
SUBMIT_ALL_PATH = REPO_ROOT / "chtc" / "submit_qwen_anchor_jobs.sh"
SUBMIT_MATCHED_MMLU_PATH = REPO_ROOT / "chtc" / "submit_qwen_mmlu_matched_jobs.sh"
SUBMIT_MATH_PREPARE_PATH = REPO_ROOT / "chtc" / "submit_qwen_math_prepare_jobs.sh"
SUBMIT_MATH_ROLLOUT_PATH = REPO_ROOT / "chtc" / "submit_qwen_math_rollout_jobs.sh"
SUBMIT_MATH_ROLLOUT_RESUME_PATH = (
    REPO_ROOT / "chtc" / "submit_qwen_math_rollout_resume_jobs.sh"
)
MATCHED_MATH_MANIFEST_PATH = REPO_ROOT / "analysis" / "matched_math" / "math_manifest.json"
SELECTED_PROBLEMS_PATH = REPO_ROOT / "selected_problems.json"

MATH_PREP_SUFFIX = "paper10_baseprep_apr28"
MATH_ROLLOUT_SUFFIX = "paper10_matched_shared_apr28"
MMLU_SUFFIX = "paper10_matched_clean_apr28"

MODELS = [
    ("Qwen/Qwen2.5-Math-7B", "qwen25_math_7b"),
    ("Qwen/Qwen2.5-Math-7B-Instruct", "qwen25_math_7b_instruct"),
    ("sail/Qwen2.5-Math-7B-Oat-Zero", "qwen25_math_7b_oat_zero"),
]

MODEL_ID_BY_SLUG = {slug: model_id for model_id, slug in MODELS}
MODEL_SLUG_BY_SHORT_NAME = {
    model_id.split("/")[-1]: slug for model_id, slug in MODELS
}

GPU_DEVICE_NAMES = [
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100-SXM4-80GB MIG 3g.40gb",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
    "NVIDIA H200 NVL",
    "NVIDIA A40",
    "Quadro RTX 8000",
    "NVIDIA L40",
    "NVIDIA L40S",
]


def gpu_requirements() -> str:
    terms = [
        f'(TARGET.GPUs_DeviceName =?= "{device_name}")'
        for device_name in GPU_DEVICE_NAMES
    ]
    return f"({' || '.join(terms)})"


DEFAULTS = {
    "universe": "vanilla",
    "executable": "chtc/run_repo_job.sh",
    "should_transfer_files": "YES",
    "when_to_transfer_output": "ON_EXIT",
    "transfer_input_files": "chtc/thought-anchors.tar.gz, chtc/python-bundle.tar.gz",
    "requirements": gpu_requirements(),
    "request_gpus": 1,
    "request_cpus": 4,
    "request_memory": "32GB",
    "request_disk": "60GB",
    "queue": 1,
}


def render_submit(job: Dict[str, str]) -> str:
    lines = [
        "run_id = $(Cluster).$(Process)",
        f"log_dir = chtc/logs/{job['name']}/$(run_id)",
        "",
        f"universe = {DEFAULTS['universe']}",
        "",
        f"executable = {DEFAULTS['executable']}",
        f"arguments = {job['arguments']}",
        "",
        f"should_transfer_files = {DEFAULTS['should_transfer_files']}",
        f"when_to_transfer_output = {DEFAULTS['when_to_transfer_output']}",
        f"transfer_input_files = {DEFAULTS['transfer_input_files']}",
        "",
        "log = $(log_dir)/$(run_id).log",
        "output = $(log_dir)/$(run_id).out",
        "error = $(log_dir)/$(run_id).err",
        "",
        f"transfer_output_files = {job['transfer_output_files']}",
    ]
    if job.get("transfer_output_remaps"):
        lines.append(f"transfer_output_remaps = {job['transfer_output_remaps']}")
    lines.extend([
        "",
        f"requirements = {DEFAULTS['requirements']}",
        "",
    ])
    if DEFAULTS.get("concurrency_limits"):
        lines.extend([
            f"concurrency_limits = {DEFAULTS['concurrency_limits']}",
            "",
        ])
    lines.extend([
        "+WantGPULab = true",
        f'+GPUJobLength = "{job.get("gpu_job_length", "long")}"',
        "",
        f"request_gpus = {DEFAULTS['request_gpus']}",
        f"request_cpus = {DEFAULTS['request_cpus']}",
        f"request_memory = {DEFAULTS['request_memory']}",
        f"request_disk = {DEFAULTS['request_disk']}",
        "",
        f"queue {DEFAULTS['queue']}",
        "",
    ])
    return "\n".join(lines)


def output_remap(job_name: str, output_dir_name: str) -> str:
    return (
        f'"repo/{output_dir_name} = '
        f'chtc/outputs/qwen_anchor/{job_name}/{output_dir_name}"'
    )


def load_selected_math_problem_ids(start: int, count: int) -> List[int]:
    payload = json.loads(SELECTED_PROBLEMS_PATH.read_text(encoding="utf-8"))
    problem_ids: List[int] = []
    for item in payload[start : start + count]:
        raw = item.get("problem_idx", item)
        if isinstance(raw, str):
            raw = raw.replace("problem_", "")
        problem_ids.append(int(raw))
    return problem_ids


def math_prepare_jobs(problem_ids: List[int]) -> Iterable[Dict[str, str]]:
    for model_id, slug in MODELS:
        for problem_idx in problem_ids:
            for base_type in ["correct", "incorrect"]:
                name = f"math_prepare_{slug}_{base_type}_p{problem_idx}"
                command: List[str] = [
                    "thought-anchors.tar.gz",
                    "python-bundle.tar.gz",
                    "python3",
                    "generate_rollouts.py",
                    "--provider",
                    "Local",
                    "--model",
                    model_id,
                    "--include_problems",
                    str(problem_idx),
                    "--prepare_only",
                    "--max_chunks",
                    "80",
                    "--batch_size",
                    "1",
                    "--max_tokens",
                    "4096",
                    "--max_base_solution_attempts",
                    "25",
                    "--base_solution_type",
                    base_type,
                    "--output_suffix",
                    MATH_PREP_SUFFIX,
                ]
                yield {
                    "name": name,
                    "arguments": " ".join(command),
                    "transfer_output_files": "repo/math_rollouts",
                    "transfer_output_remaps": output_remap(name, "math_rollouts"),
                    "gpu_job_length": "medium",
                }


def math_rollout_problem_dir(
    short_model_slug: str,
    base_type: str,
    rollout_type: str,
    problem_idx: int,
) -> Path:
    dir_name = f"{base_type}_base_solution"
    if rollout_type == "forced_answer":
        dir_name += "_forced_answer"
    dir_name += f"_{MATH_ROLLOUT_SUFFIX}"
    return (
        REPO_ROOT
        / "math_rollouts"
        / short_model_slug
        / "temperature_0.6_top_p_0.95"
        / dir_name
        / f"problem_{problem_idx}"
    )


def chunk_rollout_complete(
    short_model_slug: str,
    base_type: str,
    rollout_type: str,
    problem_idx: int,
    chunk_idx: int,
) -> bool:
    solutions_path = (
        math_rollout_problem_dir(short_model_slug, base_type, rollout_type, problem_idx)
        / f"chunk_{chunk_idx}"
        / "solutions.json"
    )
    if not solutions_path.exists():
        return False
    try:
        payload = json.loads(solutions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, list) and len(payload) >= 100


def math_rollout_jobs(missing_only: bool = False) -> Iterable[Dict[str, str]]:
    if not MATCHED_MATH_MANIFEST_PATH.exists():
        return

    manifest = json.loads(MATCHED_MATH_MANIFEST_PATH.read_text(encoding="utf-8"))
    for problem in manifest.get("problems", []):
        problem_idx = int(problem["problem_id"])
        for short_model_slug, model_entry in problem["models"].items():
            slug = MODEL_SLUG_BY_SHORT_NAME[short_model_slug]
            model_id = MODEL_ID_BY_SLUG[slug]
            for base_type in ["correct", "incorrect"]:
                base_entry = model_entry["bases"][base_type]
                num_chunks = int(base_entry["num_chunks"])
                base_input_dir = (
                    "analysis/matched_math/base_traces/"
                    f"{short_model_slug}/{base_type}_base_solution_{MATH_PREP_SUFFIX}"
                )
                for rollout_type in ["default", "forced_answer"]:
                    for chunk_idx in range(num_chunks):
                        if missing_only and chunk_rollout_complete(
                            short_model_slug,
                            base_type,
                            rollout_type,
                            problem_idx,
                            chunk_idx,
                        ):
                            continue
                        name = (
                            f"math_rollout_{slug}_{base_type}_{rollout_type}"
                            f"_p{problem_idx}_c{chunk_idx}"
                        )
                        command: List[str] = [
                            "thought-anchors.tar.gz",
                            "python-bundle.tar.gz",
                            "python3",
                            "generate_rollouts.py",
                            "--provider",
                            "Local",
                            "--model",
                            model_id,
                            "--include_problems",
                            str(problem_idx),
                            "--include_chunks",
                            str(chunk_idx),
                            "--base_input_dir",
                            base_input_dir,
                            "--require_base_input",
                            "--num_rollouts",
                            "100",
                            "--batch_size",
                            "1",
                            "--max_tokens",
                            "4096",
                            "--max_chunks",
                            "80",
                            "--base_solution_type",
                            base_type,
                            "--output_suffix",
                            MATH_ROLLOUT_SUFFIX,
                        ]
                        if rollout_type == "forced_answer":
                            command.extend(["--rollout_type", "forced_answer"])
                        yield {
                            "name": name,
                            "arguments": " ".join(command),
                            "transfer_output_files": "repo/math_rollouts",
                            "transfer_output_remaps": output_remap(
                                name, "math_rollouts"
                            ),
                            "gpu_job_length": "medium",
                        }


def mmlu_jobs() -> Iterable[Dict[str, str]]:
    for model_id, slug in MODELS:
        for base_type in ["correct", "incorrect"]:
            for rollout_type in ["default", "forced_answer"]:
                name = f"mmlu_anchor_{slug}_{base_type}_{rollout_type}"
                question_keys_file = (
                    f"analysis/matched_mmlu/{base_type}_question_keys.txt"
                )
                command: List[str] = [
                    "thought-anchors.tar.gz",
                    "python-bundle.tar.gz",
                    "python3",
                    "generate_mmlu_rollouts.py",
                    "--model",
                    model_id,
                    "--base_solution_type",
                    base_type,
                    "--rollout_type",
                    rollout_type,
                    "--question_keys_file",
                    question_keys_file,
                    "--num_questions",
                    "10",
                    "--num_rollouts",
                    "100",
                    "--batch_size",
                    "8",
                    "--max_new_tokens",
                    "256",
                    "--output_suffix",
                    MMLU_SUFFIX,
                ]
                yield {
                    "name": name,
                    "arguments": " ".join(command),
                    "transfer_output_files": "repo/mmlu_rollouts",
                    "gpu_job_length": "medium",
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-prepare-start", type=int, default=0)
    parser.add_argument("--math-prepare-count", type=int, default=20)
    parser.add_argument(
        "--math-rollout-missing-only",
        action="store_true",
        help="Render only MATH rollout chunks that do not have 100 saved rollouts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    math_prepare_job_list = list(
        math_prepare_jobs(
            load_selected_math_problem_ids(
                args.math_prepare_start,
                args.math_prepare_count,
            )
        )
    )
    math_rollout_job_list = list(
        math_rollout_jobs(missing_only=args.math_rollout_missing_only)
    )
    mmlu_job_list = list(mmlu_jobs())
    jobs = math_prepare_job_list + math_rollout_job_list + mmlu_job_list
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_submit in OUTPUT_DIR.glob("*.sub"):
        stale_submit.unlink()

    submit_header = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        'repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
        'chtc_dir="$repo_root/chtc"',
        "",
    ]
    submit_lines = list(submit_header)
    submit_mmlu_lines = list(submit_header)
    submit_math_prepare_lines = list(submit_header)
    submit_math_rollout_lines = list(submit_header)

    for job in jobs:
        submit_path = OUTPUT_DIR / f"{job['name']}.sub"
        submit_path.write_text(render_submit(job), encoding="utf-8")
        submit_lines.append(
            f'condor_submit "$chtc_dir/generated_qwen_anchor/{job["name"]}.sub"'
        )
        if job in mmlu_job_list:
            submit_mmlu_lines.append(
                f'condor_submit "$chtc_dir/generated_qwen_anchor/{job["name"]}.sub"'
            )
        if job in math_prepare_job_list:
            submit_math_prepare_lines.append(
                f'condor_submit "$chtc_dir/generated_qwen_anchor/{job["name"]}.sub"'
            )
        if job in math_rollout_job_list:
            submit_math_rollout_lines.append(
                f'condor_submit "$chtc_dir/generated_qwen_anchor/{job["name"]}.sub"'
            )

    SUBMIT_ALL_PATH.write_text("\n".join(submit_lines) + "\n", encoding="utf-8")
    SUBMIT_ALL_PATH.chmod(0o755)
    SUBMIT_MATCHED_MMLU_PATH.write_text(
        "\n".join(submit_mmlu_lines) + "\n",
        encoding="utf-8",
    )
    SUBMIT_MATCHED_MMLU_PATH.chmod(0o755)
    SUBMIT_MATH_PREPARE_PATH.write_text(
        "\n".join(submit_math_prepare_lines) + "\n",
        encoding="utf-8",
    )
    SUBMIT_MATH_PREPARE_PATH.chmod(0o755)
    SUBMIT_MATH_ROLLOUT_PATH.write_text(
        "\n".join(submit_math_rollout_lines) + "\n",
        encoding="utf-8",
    )
    SUBMIT_MATH_ROLLOUT_PATH.chmod(0o755)
    SUBMIT_MATH_ROLLOUT_RESUME_PATH.write_text(
        "\n".join(submit_math_rollout_lines) + "\n",
        encoding="utf-8",
    )
    SUBMIT_MATH_ROLLOUT_RESUME_PATH.chmod(0o755)

    print(f"Rendered {len(jobs)} submit files into {OUTPUT_DIR}")
    print(f"Wrote helper script: {SUBMIT_ALL_PATH}")
    print(f"Wrote matched MMLU helper script: {SUBMIT_MATCHED_MMLU_PATH}")
    print(f"Wrote MATH prepare helper script: {SUBMIT_MATH_PREPARE_PATH}")
    print(f"Wrote MATH rollout helper script: {SUBMIT_MATH_ROLLOUT_PATH}")
    print(f"Wrote MATH rollout resume helper script: {SUBMIT_MATH_ROLLOUT_RESUME_PATH}")
    if not math_rollout_job_list:
        print(f"No MATH rollout jobs rendered; missing {MATCHED_MATH_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
