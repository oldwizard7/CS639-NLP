#!/usr/bin/env python3
"""
Prepare lightweight, inspection-friendly thought-anchor artifacts from saved outputs.

This script does two practical things:
1. For saved MATH rollouts, compute chunk-level removal accuracies and simple anchor previews.
2. For saved MMLU generations, split reasoning traces into chunks for manual inspection.

It is intentionally lightweight and does not require the OpenAI-based labeling path in
`analyze_rollouts.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import re
from pathlib import Path
from typing import Any, Dict, List

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def split_solution_into_chunks(solution_text: str) -> List[str]:
    """Local copy of the chunking logic without heavyweight imports."""
    if "<think>" in solution_text:
        after_open = solution_text.split("<think>", 1)[1].strip()
        if "</think>" in after_open:
            inside_think, after_close = after_open.split("</think>", 1)
            solution_text = inside_think.strip() or after_close.strip()
        else:
            solution_text = after_open

    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    chunks: List[str] = []
    current_chunk = ""
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char in {" ", "\n"}:
                is_sentence_end = True

        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    merged: List[str] = []
    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if merged and len(chunk) < 10:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)

    return merged


def compute_chunk_accuracy(solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [item for item in solutions if "error" not in item]
    correct = [item for item in valid if item.get("is_correct") is True]
    accuracy = len(correct) / len(valid) if valid else None
    return {
        "num_solutions": len(solutions),
        "num_valid_solutions": len(valid),
        "num_correct_solutions": len(correct),
        "removal_accuracy": accuracy,
    }


def build_math_preview(model_dir: Path, output_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "source_dir": str(model_dir),
        "problems": [],
    }

    problem_dirs = sorted(
        path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("problem_")
    )

    for problem_dir in problem_dirs:
        chunks_path = problem_dir / "chunks.json"
        base_solution_path = problem_dir / "base_solution.json"
        problem_path = problem_dir / "problem.json"
        if not chunks_path.exists() or not base_solution_path.exists() or not problem_path.exists():
            continue

        chunks_payload = load_json(chunks_path)
        base_solution = load_json(base_solution_path)
        problem = load_json(problem_path)
        chunks = chunks_payload.get("chunks", [])
        if not chunks:
            continue

        base_is_correct = bool(base_solution.get("is_correct"))
        base_accuracy = 1.0 if base_is_correct else 0.0

        chunk_entries: List[Dict[str, Any]] = []
        for chunk_idx, chunk_text in enumerate(chunks):
            solutions_path = problem_dir / f"chunk_{chunk_idx}" / "solutions.json"
            entry: Dict[str, Any] = {
                "chunk_idx": chunk_idx,
                "chunk_text": chunk_text,
            }
            if solutions_path.exists():
                stats = compute_chunk_accuracy(load_json(solutions_path))
                entry.update(stats)
                if stats["removal_accuracy"] is not None:
                    entry["importance_drop_from_base"] = base_accuracy - stats["removal_accuracy"]
            chunk_entries.append(entry)

        for idx, entry in enumerate(chunk_entries[:-1]):
            curr = entry.get("removal_accuracy")
            nxt = chunk_entries[idx + 1].get("removal_accuracy")
            if curr is not None and nxt is not None:
                entry["resampling_importance_accuracy_preview"] = nxt - curr

        ranked_chunks = [
            entry for entry in chunk_entries if entry.get("importance_drop_from_base") is not None
        ]
        ranked_chunks.sort(key=lambda item: item["importance_drop_from_base"], reverse=True)

        problem_report = {
            "problem_idx": problem_dir.name.replace("problem_", ""),
            "problem_text": problem.get("problem"),
            "ground_truth_answer": problem.get("gt_answer"),
            "base_answer": base_solution.get("answer"),
            "base_is_correct": base_is_correct,
            "num_chunks": len(chunks),
            "chunks": chunk_entries,
            "top_anchor_candidates": ranked_chunks[:5],
        }
        summary["problems"].append(problem_report)

        dump_json(output_dir / problem_dir.name / "anchor_preview.json", problem_report)

    all_candidates: List[Dict[str, Any]] = []
    for problem in summary["problems"]:
        for chunk in problem["top_anchor_candidates"]:
            all_candidates.append(
                {
                    "problem_idx": problem["problem_idx"],
                    "chunk_idx": chunk["chunk_idx"],
                    "importance_drop_from_base": chunk["importance_drop_from_base"],
                    "chunk_text": chunk["chunk_text"],
                }
            )
    all_candidates.sort(key=lambda item: item["importance_drop_from_base"], reverse=True)
    summary["top_anchor_candidates_across_problems"] = all_candidates[:20]

    lines = [
        f"# MATH Anchor Preview for {model_dir.parts[-3]}",
        "",
        f"Problems with saved chunks: {len(summary['problems'])}",
        "",
    ]
    for item in summary["top_anchor_candidates_across_problems"][:10]:
        lines.append(
            f"- problem_{item['problem_idx']} chunk_{item['chunk_idx']} "
            f"drop={item['importance_drop_from_base']:.3f}: {item['chunk_text']}"
        )
    write_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    dump_json(output_dir / "summary.json", summary)
    return summary


def extract_mmlu_reasoning(text: str) -> str:
    explicit_marker = "\nAnswer:"
    if explicit_marker in text:
        return text.rsplit(explicit_marker, 1)[1].strip()
    marker = "Answer:"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    return text.strip()


def build_mmlu_preview(results_path: Path, output_dir: Path, examples_per_bucket: int) -> Dict[str, Any]:
    payload = load_json(results_path)
    summary = payload["summary"]
    results = payload["results"]

    enriched = []
    lengths = []
    for row in results:
        reasoning = extract_mmlu_reasoning(row.get("generated_text", ""))
        chunks = split_solution_into_chunks(reasoning)
        lengths.append(len(chunks))
        enriched.append(
            {
                "subject": row["subject"],
                "question_idx": row["question_idx"],
                "question": row["question"],
                "correct_answer": row["correct_answer"],
                "predicted_answer": row.get("predicted_answer"),
                "is_correct": row["is_correct"],
                "num_chunks": len(chunks),
                "chunks": chunks,
                "generated_text": row.get("generated_text", ""),
            }
        )

    correct_examples = sorted(
        [row for row in enriched if row["is_correct"]],
        key=lambda item: item["num_chunks"],
        reverse=True,
    )[:examples_per_bucket]
    incorrect_examples = sorted(
        [row for row in enriched if not row["is_correct"]],
        key=lambda item: item["num_chunks"],
        reverse=True,
    )[:examples_per_bucket]

    report = {
        "summary": summary,
        "num_results": len(results),
        "avg_num_chunks": statistics.mean(lengths) if lengths else 0.0,
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples,
    }

    lines = [
        f"# MMLU Trace Preview for {summary['model'].split('/')[-1]}",
        "",
        f"Accuracy: {summary['accuracy']:.3f}",
        f"Average chunk count: {report['avg_num_chunks']:.2f}",
        "",
        "## Long Correct Traces",
    ]
    for row in correct_examples[:5]:
        lines.append(
            f"- {row['subject']} #{row['question_idx']} chunks={row['num_chunks']} "
            f"pred={row['predicted_answer']} gold={row['correct_answer']}"
        )
    lines.append("")
    lines.append("## Long Incorrect Traces")
    for row in incorrect_examples[:5]:
        lines.append(
            f"- {row['subject']} #{row['question_idx']} chunks={row['num_chunks']} "
            f"pred={row['predicted_answer']} gold={row['correct_answer']}"
        )

    dump_json(output_dir / "trace_preview.json", report)
    write_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare inspection-ready thought-anchor artifacts.")
    parser.add_argument(
        "--math-root",
        default="math_rollouts",
        help="Root directory containing saved MATH rollouts",
    )
    parser.add_argument(
        "--mmlu-root",
        default="mmlu_local",
        help="Root directory containing saved MMLU result JSON files",
    )
    parser.add_argument(
        "--output-root",
        default="analysis/anchor_preview",
        help="Where to save preview artifacts",
    )
    parser.add_argument(
        "--examples-per-bucket",
        type=int,
        default=10,
        help="How many correct/incorrect MMLU examples to save per model",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    math_root = Path(args.math_root)
    mmlu_root = Path(args.mmlu_root)

    if math_root.exists():
        for model_dir in sorted(math_root.iterdir()):
            correct_dir = model_dir / "temperature_0.6_top_p_0.95" / "correct_base_solution"
            if correct_dir.exists():
                build_math_preview(
                    correct_dir,
                    output_root / "math" / model_dir.name,
                )

    if mmlu_root.exists():
        for model_dir in sorted(mmlu_root.iterdir()):
            results_path = model_dir / "test" / "all_results.json"
            if results_path.exists():
                build_mmlu_preview(
                    results_path,
                    output_root / "mmlu" / model_dir.name,
                    args.examples_per_bucket,
                )


if __name__ == "__main__":
    main()
