#!/usr/bin/env python3
"""Summarize Qwen Thought Anchors rollout completeness and anchor previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


MODELS = [
    "Qwen2.5-Math-7B",
    "Qwen2.5-Math-7B-Instruct",
    "Qwen2.5-Math-7B-Oat-Zero",
]
BASE_TYPES = ["correct", "incorrect"]
ROLLOUT_TYPES = ["default", "forced_answer"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def rollout_dir(
    root: Path,
    model_slug: str,
    base_type: str,
    rollout_type: str,
    suffix: str,
) -> Path:
    name = f"{base_type}_base_solution"
    if rollout_type == "forced_answer":
        name += "_forced_answer"
    name += f"_{suffix}"
    return root / model_slug / "temperature_0.6_top_p_0.95" / name


def valid_solution_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        solutions = load_json(path)
    except Exception:
        return 0
    return sum(
        1
        for item in solutions
        if "error" not in item and item.get("answer", "") not in {"", "None"}
    )


def solution_accuracy(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        solutions = load_json(path)
    except Exception:
        return None
    valid = [
        item
        for item in solutions
        if "error" not in item and item.get("answer", "") not in {"", "None"}
    ]
    if not valid:
        return None
    return sum(1 for item in valid if item.get("is_correct") is True) / len(valid)


def problem_summary(problem_dir: Path, forced_problem_dir: Optional[Path]) -> Dict[str, Any]:
    chunks_payload = load_json(problem_dir / "chunks.json")
    base_solution = load_json(problem_dir / "base_solution.json")
    chunks = chunks_payload.get("chunks", [])
    base_accuracy = 1.0 if base_solution.get("is_correct") is True else 0.0

    chunk_entries: List[Dict[str, Any]] = []
    for idx, chunk_text in enumerate(chunks):
        solutions_file = problem_dir / f"chunk_{idx}" / "solutions.json"
        forced_solutions_file = (
            forced_problem_dir / f"chunk_{idx}" / "solutions.json"
            if forced_problem_dir
            else None
        )
        accuracy = solution_accuracy(solutions_file)
        forced_accuracy = (
            solution_accuracy(forced_solutions_file)
            if forced_solutions_file is not None
            else None
        )
        valid_count = valid_solution_count(solutions_file)
        forced_valid_count = (
            valid_solution_count(forced_solutions_file)
            if forced_solutions_file is not None
            else 0
        )
        impact = None if accuracy is None else base_accuracy - accuracy
        forced_impact = (
            None if forced_accuracy is None else base_accuracy - forced_accuracy
        )
        chunk_entries.append(
            {
                "chunk_idx": idx,
                "chunk_text": chunk_text,
                "valid_count": valid_count,
                "forced_valid_count": forced_valid_count,
                "accuracy": accuracy,
                "forced_accuracy": forced_accuracy,
                "impact": impact,
                "forced_impact": forced_impact,
                "complete": valid_count >= 100 and forced_valid_count >= 100,
            }
        )

    complete_chunks = sum(1 for item in chunk_entries if item["complete"])
    top_chunks = sorted(
        [
            item
            for item in chunk_entries
            if item["impact"] is not None or item["forced_impact"] is not None
        ],
        key=lambda item: max(
            abs(item["impact"] or 0.0),
            abs(item["forced_impact"] or 0.0),
        ),
        reverse=True,
    )[:5]

    return {
        "problem_id": problem_dir.name.replace("problem_", ""),
        "base_is_correct": base_solution.get("is_correct"),
        "num_chunks": len(chunks),
        "complete_chunks": complete_chunks,
        "is_complete": complete_chunks == len(chunks) and len(chunks) > 0,
        "top_chunks": top_chunks,
    }


def summarize_dataset(
    benchmark: str,
    root: Path,
    suffix: str,
    expected_problem_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    expected = set(str(item) for item in expected_problem_ids or [])
    dataset_summary: Dict[str, Any] = {
        "benchmark": benchmark,
        "suffix": suffix,
        "models": {},
    }

    for model_slug in MODELS:
        model_summary: Dict[str, Any] = {}
        for base_type in BASE_TYPES:
            default_dir = rollout_dir(root, model_slug, base_type, "default", suffix)
            forced_dir = rollout_dir(root, model_slug, base_type, "forced_answer", suffix)
            problem_dirs = sorted(
                path
                for path in default_dir.glob("problem_*")
                if path.is_dir() and (path / "chunks.json").exists()
            )
            if expected:
                problem_dirs = [
                    path
                    for path in problem_dirs
                    if path.name.replace("problem_", "") in expected
                    or path.name in expected
                ]

            problems = []
            for problem_dir in problem_dirs:
                forced_problem_dir = forced_dir / problem_dir.name
                problems.append(
                    problem_summary(
                        problem_dir,
                        forced_problem_dir if forced_problem_dir.exists() else None,
                    )
                )

            model_summary[base_type] = {
                "default_dir": str(default_dir),
                "forced_dir": str(forced_dir),
                "num_problems": len(problems),
                "num_complete_problems": sum(
                    1 for item in problems if item["is_complete"]
                ),
                "problems": problems,
            }
        dataset_summary["models"][model_slug] = model_summary
    return dataset_summary


def markdown_report(payload: Dict[str, Any]) -> str:
    lines = ["# Qwen Thought Anchors Summary", ""]
    for dataset in payload["datasets"]:
        lines.append(f"## {dataset['benchmark']}")
        lines.append("")
        lines.append(f"Suffix: `{dataset['suffix']}`")
        lines.append("")
        for model_slug, model_summary in dataset["models"].items():
            lines.append(f"### {model_slug}")
            for base_type, base_summary in model_summary.items():
                lines.append(
                    f"- {base_type}: {base_summary['num_complete_problems']}/"
                    f"{base_summary['num_problems']} complete problems"
                )
                top_candidates = []
                for problem in base_summary["problems"]:
                    for chunk in problem["top_chunks"][:1]:
                        top_candidates.append((problem, chunk))
                top_candidates = sorted(
                    top_candidates,
                    key=lambda item: max(
                        abs(item[1].get("impact") or 0.0),
                        abs(item[1].get("forced_impact") or 0.0),
                    ),
                    reverse=True,
                )[:3]
                for problem, chunk in top_candidates:
                    lines.append(
                        f"- top {base_type} problem_{problem['problem_id']} "
                        f"chunk_{chunk['chunk_idx']}: "
                        f"impact={chunk.get('impact')} "
                        f"forced={chunk.get('forced_impact')} "
                        f"{chunk['chunk_text'][:160]}"
                    )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-root", default="math_rollouts")
    parser.add_argument("--mmlu-root", default="mmlu_rollouts")
    parser.add_argument("--math-suffix", default="paper10_matched_shared_apr28")
    parser.add_argument("--mmlu-suffix", default="paper10_matched_clean_apr28")
    parser.add_argument("--math-manifest", default="analysis/matched_math/math_manifest.json")
    parser.add_argument("--output-dir", default="analysis/qwen_anchor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    math_expected: Optional[List[str]] = None
    manifest_path = Path(args.math_manifest)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        math_expected = [str(item) for item in manifest.get("selected_problem_ids", [])]

    payload = {
        "datasets": [
            summarize_dataset(
                "MATH",
                Path(args.math_root),
                args.math_suffix,
                expected_problem_ids=math_expected,
            ),
            summarize_dataset("MMLU", Path(args.mmlu_root), args.mmlu_suffix),
        ]
    }
    dump_json(output_dir / "summary_report.json", payload)
    (output_dir / "summary_report.md").write_text(
        markdown_report(payload),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()
