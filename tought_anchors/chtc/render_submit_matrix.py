#!/usr/bin/env python3
"""
Render concrete HTCondor submit files from the prepared model x dataset matrix.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "chtc" / "job_matrix.json"
OUTPUT_DIR = REPO_ROOT / "chtc" / "generated"


def render_submit(job: dict, defaults: dict) -> str:
    request_gpus = job.get("request_gpus", defaults["request_gpus"])
    lines = [
        "run_id = $(Cluster).$(Process)",
        f"log_dir = chtc/logs/{job['name']}/$(run_id)",
        "",
        f"universe = {defaults['universe']}",
        "",
        f"executable = {defaults['executable']}",
        f"arguments = {job['arguments']}",
        "",
        f"should_transfer_files = {defaults['should_transfer_files']}",
        f"when_to_transfer_output = {defaults['when_to_transfer_output']}",
        f"transfer_input_files = {defaults['transfer_input_files']}",
        "",
        "log = $(log_dir)/$(run_id).log",
        "output = $(log_dir)/$(run_id).out",
        "error = $(log_dir)/$(run_id).err",
        "",
        f"transfer_output_files = {job['transfer_output_files']}",
        "",
    ]

    requirements = job.get("requirements", defaults.get("requirements"))
    if requirements:
        lines.append(f"requirements = {requirements}")
        lines.append("")

    concurrency_limits = job.get(
        "concurrency_limits", defaults.get("concurrency_limits")
    )
    if concurrency_limits:
        lines.append(f"concurrency_limits = {concurrency_limits}")
        lines.append("")

    if defaults.get("want_gpu_lab", False):
        lines.append("+WantGPULab = true")
    if "gpu_job_length" in defaults:
        lines.append(f'+GPUJobLength = "{defaults["gpu_job_length"]}"')

    lines.extend(
        [
            "",
            f"request_gpus = {request_gpus}",
            f"request_cpus = {job.get('request_cpus', defaults['request_cpus'])}",
            f"request_memory = {job.get('request_memory', defaults['request_memory'])}",
            f"request_disk = {job.get('request_disk', defaults['request_disk'])}",
            "",
            f"queue {job.get('queue', defaults['queue'])}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    defaults = config["defaults"]
    jobs = config["jobs"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submit_all_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        'repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
        'chtc_dir="$repo_root/chtc"',
        "",
    ]

    for job in jobs:
        submit_path = OUTPUT_DIR / f"{job['name']}.sub"
        submit_path.write_text(render_submit(job, defaults), encoding="utf-8")
        submit_all_lines.append(f'condor_submit "$chtc_dir/generated/{job["name"]}.sub"')

    submit_all_path = REPO_ROOT / "chtc" / "submit_prepared_jobs.sh"
    submit_all_path.write_text("\n".join(submit_all_lines) + "\n", encoding="utf-8")
    submit_all_path.chmod(0o755)

    print(f"Rendered {len(jobs)} submit files into {OUTPUT_DIR}")
    print(f"Wrote helper script: {submit_all_path}")


if __name__ == "__main__":
    main()
