#!/usr/bin/env python3
"""
Select matched MMLU question sets for fair Thought Anchors comparisons.

The rollout stage should compare the same questions across models. This script
builds question-key manifests for all-correct and all-incorrect buckets, after
cleaning older saved MMLU generations that may contain echoed prompt text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from generate_mmlu_rollouts import create_mmlu_prompt, split_solution_into_chunks


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Math-7B",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "sail/Qwen2.5-Math-7B-Oat-Zero",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matched MMLU question manifests for Qwen anchor runs."
    )
    parser.add_argument(
        "--mmlu-results-root",
        default="mmlu_local",
        help="Root containing run_mmlu_local.py outputs.",
    )
    parser.add_argument("--split", default="test", help="MMLU split to read.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model ids or slugs to match.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/matched_mmlu",
        help="Directory for manifest JSON and question-key text files.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Number of questions to select per correctness bucket.",
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=2,
        help="Require at least this many cleaned chunks for every model.",
    )
    parser.add_argument(
        "--selection",
        choices=["first", "longest_mean", "shortest_mean"],
        default="longest_mean",
        help="How to rank matched candidates before taking num-questions.",
    )
    return parser.parse_args()


def model_slug(model: str) -> str:
    return model.split("/")[-1]


def question_key(row: Dict[str, Any]) -> str:
    return f"{row['subject']}:{int(row['question_idx'])}"


def clean_saved_mmlu_generation(row: Dict[str, Any], model: str) -> str:
    text = row.get("generated_text", "").strip()
    if not text:
        return ""

    question_data = {
        "question": row["question"],
        "choices": row["choices"],
        "answer_letter": row["correct_answer"],
    }
    prompt = create_mmlu_prompt(question_data, model)

    while text.startswith(prompt):
        text = text[len(prompt) :].lstrip()

    for _ in range(3):
        marker_idx = text.find("\n\nAnswer:\n")
        if marker_idx == -1 or marker_idx > 2500:
            break
        text = text[marker_idx + len("\n\nAnswer:\n") :].lstrip()

    return re.sub(r"^Final Answer:\s*<LETTER>\s*", "", text).lstrip()


def load_model_rows(root: Path, model: str, split: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    path = root / model_slug(model) / split / "all_results.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["summary"], payload["results"]


def enrich_rows(model: str, rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    enriched: Dict[str, Dict[str, Any]] = {}
    for order_idx, row in enumerate(rows):
        cleaned = clean_saved_mmlu_generation(row, model)
        chunks = split_solution_into_chunks(cleaned)
        enriched[question_key(row)] = {
            "subject": row["subject"],
            "question_idx": int(row["question_idx"]),
            "correct_answer": row["correct_answer"],
            "predicted_answer": row.get("predicted_answer") or "",
            "is_correct": bool(row.get("is_correct")),
            "num_chunks": len(chunks),
            "order_idx": order_idx,
        }
    return enriched


def candidate_sort_key(candidate: Dict[str, Any], selection: str) -> Tuple[float, int]:
    if selection == "first":
        return (candidate["order_idx"], 0)
    if selection == "shortest_mean":
        return (candidate["mean_chunks"], candidate["order_idx"])
    return (-candidate["mean_chunks"], candidate["order_idx"])


def build_manifest(
    bucket: str,
    models: List[str],
    per_model: Dict[str, Dict[str, Dict[str, Any]]],
    summaries: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    desired_correct = bucket == "correct"
    common_keys = set.intersection(*(set(rows.keys()) for rows in per_model.values()))
    candidates: List[Dict[str, Any]] = []

    for key in common_keys:
        model_entries = {
            model_slug(model): per_model[model][key]
            for model in models
        }
        if any(not entry["predicted_answer"] for entry in model_entries.values()):
            continue
        if any(entry["num_chunks"] < args.min_chunks for entry in model_entries.values()):
            continue
        if any(entry["is_correct"] != desired_correct for entry in model_entries.values()):
            continue

        first_entry = next(iter(model_entries.values()))
        chunk_counts = [entry["num_chunks"] for entry in model_entries.values()]
        order_idx = min(entry["order_idx"] for entry in model_entries.values())
        candidates.append(
            {
                "question_key": key,
                "subject": first_entry["subject"],
                "question_idx": first_entry["question_idx"],
                "correct_answer": first_entry["correct_answer"],
                "mean_chunks": sum(chunk_counts) / len(chunk_counts),
                "min_chunks": min(chunk_counts),
                "max_chunks": max(chunk_counts),
                "order_idx": order_idx,
                "per_model": model_entries,
            }
        )

    candidates.sort(key=lambda item: candidate_sort_key(item, args.selection))
    selected = candidates[: args.num_questions]

    return {
        "bucket": bucket,
        "split": args.split,
        "selection": args.selection,
        "num_requested": args.num_questions,
        "min_chunks": args.min_chunks,
        "num_candidates": len(candidates),
        "question_keys": [item["question_key"] for item in selected],
        "selected": selected,
        "model_summaries": {
            model_slug(model): summaries[model]
            for model in models
        },
    }


def write_outputs(manifest: Dict[str, Any], output_dir: Path) -> None:
    bucket = manifest["bucket"]
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"{bucket}_manifest.json"
    keys_path = output_dir / f"{bucket}_question_keys.txt"

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    keys_path.write_text(
        "\n".join(manifest["question_keys"]) + "\n",
        encoding="utf-8",
    )

    print(
        f"{bucket}: selected {len(manifest['question_keys'])} questions "
        f"from {manifest['num_candidates']} candidates"
    )
    print(f"  manifest: {manifest_path}")
    print(f"  keys: {keys_path}")


def main() -> None:
    args = parse_args()
    root = Path(args.mmlu_results_root)
    output_dir = Path(args.output_dir)
    models = args.models

    summaries: Dict[str, Dict[str, Any]] = {}
    per_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model in models:
        summary, rows = load_model_rows(root, model, args.split)
        summaries[model] = summary
        per_model[model] = enrich_rows(model, rows)

    for bucket in ["correct", "incorrect"]:
        manifest = build_manifest(bucket, models, per_model, summaries, args)
        write_outputs(manifest, output_dir)


if __name__ == "__main__":
    main()
