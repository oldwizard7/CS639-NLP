#!/usr/bin/env python3
"""Run the Qwen Thought Anchors analysis matrix once rollouts are complete."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List


MODELS = [
    "Qwen2.5-Math-7B",
    "Qwen2.5-Math-7B-Instruct",
    "Qwen2.5-Math-7B-Oat-Zero",
]
BENCHMARKS = [
    ("math", "math_rollouts", "paper10_matched_shared_apr28"),
    ("mmlu", "mmlu_rollouts", "paper10_matched_clean_apr28"),
]


def rollout_dir(root: Path, model_slug: str, base_type: str, forced: bool, suffix: str) -> Path:
    name = f"{base_type}_base_solution"
    if forced:
        name += "_forced_answer"
    name += f"_{suffix}"
    return root / model_slug / "temperature_0.6_top_p_0.95" / name


def problem_dirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_dir() and item.name.startswith("problem_")
    )


def required_dirs(root: Path, model_slug: str, suffix: str) -> Iterable[Path]:
    for base_type in ["correct", "incorrect"]:
        yield rollout_dir(root, model_slug, base_type, False, suffix)
        yield rollout_dir(root, model_slug, base_type, True, suffix)


def analysis_command(
    benchmark: str,
    root: Path,
    model_slug: str,
    suffix: str,
    output_root: Path,
) -> List[str]:
    return [
        "python3",
        "analyze_rollouts.py",
        "--correct_rollouts_dir",
        str(rollout_dir(root, model_slug, "correct", False, suffix)),
        "--correct_forced_answer_rollouts_dir",
        str(rollout_dir(root, model_slug, "correct", True, suffix)),
        "--incorrect_rollouts_dir",
        str(rollout_dir(root, model_slug, "incorrect", False, suffix)),
        "--incorrect_forced_answer_rollouts_dir",
        str(rollout_dir(root, model_slug, "incorrect", True, suffix)),
        "--output_dir",
        str(output_root / benchmark / model_slug),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="analysis/qwen_anchor")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip model/benchmark pairs whose rollout directories are incomplete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    commands: List[List[str]] = []

    for benchmark, root_text, suffix in BENCHMARKS:
        root = Path(root_text)
        for model_slug in MODELS:
            missing = [
                path
                for path in required_dirs(root, model_slug, suffix)
                if not problem_dirs(path)
            ]
            if missing:
                message = (
                    f"{benchmark}/{model_slug} is not ready; missing completed "
                    f"problem dirs under: {', '.join(str(path) for path in missing)}"
                )
                if args.skip_missing:
                    print(f"Skipping: {message}")
                    continue
                raise SystemExit(message)

            commands.append(
                analysis_command(benchmark, root, model_slug, suffix, output_root)
            )

    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)

    summary_command = [
        "python3",
        "summarize_qwen_anchor_results.py",
        "--output-dir",
        str(output_root),
    ]
    print(" ".join(summary_command))
    if not args.dry_run:
        subprocess.run(summary_command, check=True)

    qualitative_command = [
        "python3",
        "qualitative_qwen_anchor_report.py",
        "--output-dir",
        str(output_root),
    ]
    print(" ".join(qualitative_command))
    if not args.dry_run:
        subprocess.run(qualitative_command, check=True)


if __name__ == "__main__":
    main()
