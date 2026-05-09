#!/usr/bin/env python3
"""Select and export matched MATH base traces for fair Thought Anchors runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


MODELS = [
    ("Qwen/Qwen2.5-Math-7B", "Qwen2.5-Math-7B"),
    ("Qwen/Qwen2.5-Math-7B-Instruct", "Qwen2.5-Math-7B-Instruct"),
    ("sail/Qwen2.5-Math-7B-Oat-Zero", "Qwen2.5-Math-7B-Oat-Zero"),
]
BASE_TYPES = ["correct", "incorrect"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_selected_problem_ids(path: Path) -> List[int]:
    payload = load_json(path)
    problem_ids: List[int] = []
    for item in payload:
        raw = item.get("problem_idx", item)
        if isinstance(raw, str):
            raw = raw.replace("problem_", "")
        problem_ids.append(int(raw))
    return problem_ids


def source_base_dir(
    source_root: Path,
    model_slug: str,
    base_type: str,
    source_suffix: str,
) -> Path:
    return (
        source_root
        / model_slug
        / "temperature_0.6_top_p_0.95"
        / f"{base_type}_base_solution_{source_suffix}"
    )


def validate_problem_dir(
    problem_dir: Path,
    base_type: str,
    min_chunks: int,
    max_chunks: int,
) -> Optional[Dict[str, Any]]:
    problem_file = problem_dir / "problem.json"
    base_solution_file = problem_dir / "base_solution.json"
    chunks_file = problem_dir / "chunks.json"
    if not (problem_file.exists() and base_solution_file.exists() and chunks_file.exists()):
        return None

    base_solution = load_json(base_solution_file)
    is_correct = base_solution.get("is_correct")
    if base_type == "correct" and is_correct is not True:
        return None
    if base_type == "incorrect" and is_correct is not False:
        return None

    chunks_payload = load_json(chunks_file)
    chunks = chunks_payload.get("chunks", [])
    if not (min_chunks <= len(chunks) <= max_chunks):
        return None

    return {
        "problem_dir": problem_dir,
        "num_chunks": len(chunks),
        "answer": base_solution.get("answer"),
        "is_correct": is_correct,
    }


def collect_candidate(
    problem_id: int,
    source_root: Path,
    source_suffix: str,
    min_chunks: int,
    max_chunks: int,
) -> Optional[Dict[str, Any]]:
    problem_name = f"problem_{problem_id}"
    model_entries: Dict[str, Any] = {}

    for model_id, model_slug in MODELS:
        base_entries: Dict[str, Any] = {}
        for base_type in BASE_TYPES:
            problem_dir = (
                source_base_dir(source_root, model_slug, base_type, source_suffix)
                / problem_name
            )
            validation = validate_problem_dir(
                problem_dir, base_type, min_chunks, max_chunks
            )
            if validation is None:
                return None
            base_entries[base_type] = validation
        model_entries[model_slug] = {
            "model_id": model_id,
            "bases": base_entries,
        }

    return {
        "problem_id": problem_id,
        "problem_name": problem_name,
        "models": model_entries,
    }


def export_base_trace(
    source_problem_dir: Path,
    output_root: Path,
    model_slug: str,
    base_type: str,
    source_suffix: str,
    problem_name: str,
) -> Path:
    target_dir = (
        output_root
        / "base_traces"
        / model_slug
        / f"{base_type}_base_solution_{source_suffix}"
        / problem_name
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["problem.json", "base_solution.json", "chunks.json"]:
        shutil.copy2(source_problem_dir / filename, target_dir / filename)
    return target_dir


def select_and_export(
    selected_problem_ids: Iterable[int],
    source_root: Path,
    output_root: Path,
    source_suffix: str,
    target_count: int,
    min_chunks: int,
    max_chunks: int,
) -> Dict[str, Any]:
    selected: List[Dict[str, Any]] = []
    considered = 0

    for problem_id in selected_problem_ids:
        considered += 1
        candidate = collect_candidate(
            problem_id,
            source_root,
            source_suffix,
            min_chunks,
            max_chunks,
        )
        if candidate is None:
            continue

        for model_slug, model_entry in candidate["models"].items():
            for base_type, base_entry in model_entry["bases"].items():
                export_dir = export_base_trace(
                    base_entry["problem_dir"],
                    output_root,
                    model_slug,
                    base_type,
                    source_suffix,
                    candidate["problem_name"],
                )
                base_entry["source_dir"] = str(base_entry["problem_dir"])
                base_entry["export_dir"] = str(export_dir)
                del base_entry["problem_dir"]

        selected.append(candidate)
        if len(selected) >= target_count:
            break

    manifest = {
        "source_root": str(source_root),
        "source_suffix": source_suffix,
        "base_traces_root": str(output_root / "base_traces"),
        "target_count": target_count,
        "selected_count": len(selected),
        "considered_count": considered,
        "min_chunks": min_chunks,
        "max_chunks": max_chunks,
        "models": [{"model_id": model_id, "slug": slug} for model_id, slug in MODELS],
        "base_types": BASE_TYPES,
        "selected_problem_ids": [item["problem_id"] for item in selected],
        "problems": selected,
    }

    dump_json(output_root / "math_manifest.json", manifest)
    problem_ids_text = "\n".join(str(item["problem_id"]) for item in selected)
    (output_root / "problem_ids.txt").write_text(
        f"{problem_ids_text}\n" if problem_ids_text else "",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select matched MATH problems and export base traces."
    )
    parser.add_argument("--source-root", default="math_rollouts")
    parser.add_argument("--output-root", default="analysis/matched_math")
    parser.add_argument("--selected-problems", default="selected_problems.json")
    parser.add_argument("--source-suffix", default="paper10_baseprep_apr28")
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--min-chunks", type=int, default=2)
    parser.add_argument("--max-chunks", type=int, default=80)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write a partial manifest instead of failing when target-count is unmet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = select_and_export(
        selected_problem_ids=load_selected_problem_ids(Path(args.selected_problems)),
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        source_suffix=args.source_suffix,
        target_count=args.target_count,
        min_chunks=args.min_chunks,
        max_chunks=args.max_chunks,
    )

    print(
        f"Selected {manifest['selected_count']}/{manifest['target_count']} "
        f"matched MATH problems after considering {manifest['considered_count']}."
    )
    print(f"Wrote {Path(args.output_root) / 'math_manifest.json'}")
    if manifest["selected_count"] < manifest["target_count"] and not args.allow_partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
