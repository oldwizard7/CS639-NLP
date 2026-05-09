#!/usr/bin/env python3
"""Render a qualitative Thought Anchors report from completed rollout chunks."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def safe_load_json(path: Path) -> Any:
    try:
        return load_json(path)
    except Exception:
        return None


def clean_text(text: str, max_len: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


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


def valid_solutions(path: Path) -> List[Dict[str, Any]]:
    payload = safe_load_json(path)
    if not isinstance(payload, list):
        return []
    return [
        item
        for item in payload
        if "error" not in item
        and str(item.get("answer", "")) not in {"", "None"}
        and item.get("is_correct") is not None
    ]


def accuracy(items: List[Dict[str, Any]]) -> Optional[float]:
    if not items:
        return None
    return sum(1 for item in items if item.get("is_correct") is True) / len(items)


def fmt_float(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.2f}"


def classify_chunk(chunk_text: str) -> str:
    text = chunk_text.lower()
    stripped = chunk_text.strip()
    if "final answer" in text or "\\boxed" in text:
        return "final-answer commitment"
    if re.fullmatch(r"[A-D][.)]?", stripped) or re.match(r"^[A-D][.)]\s", stripped):
        return "answer-option token"
    if "python" in text or "```" in text or "print(" in text:
        return "code/tool artifact"
    if any(marker in text for marker in ["choice", "option", "choices", "a.", "b.", "c.", "d."]):
        return "option comparison"
    if any(marker in text for marker in ["therefore", "thus", "based on", "so the"]):
        return "conclusion bridge"
    if any(marker in text for marker in ["not ", "except", "false", "incorrect", "does not"]):
        return "negation or exception"
    if any(marker in text for marker in ["define", "defined", "means", "refers to", "is a", "are "]):
        return "definition/factual bridge"
    if any(ch.isdigit() for ch in text) or any(marker in text for marker in ["=", "\\(", "\\)", "calculate"]):
        return "calculation/formal step"
    if any(marker in text for marker in ["to solve", "we need", "let's", "analyze", "determine"]):
        return "problem framing"
    return "local reasoning step"


def option_letter_text(problem: Dict[str, Any], answer: Optional[str]) -> str:
    if not answer:
        return "NA"
    choices = problem.get("choices") or []
    if len(answer) == 1 and answer in "ABCD":
        idx = ord(answer) - ord("A")
        if 0 <= idx < len(choices):
            return f"{answer}. {choices[idx]}"
    return str(answer)


def problem_dirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_dir() and item.name.startswith("problem_")
    )


def scan_model_base(
    root: Path,
    model_slug: str,
    base_type: str,
    suffix: str,
    min_valid: int,
) -> Dict[str, Any]:
    default_dir = rollout_dir(root, model_slug, base_type, "default", suffix)
    forced_dir = rollout_dir(root, model_slug, base_type, "forced_answer", suffix)
    default_problems = {path.name: path for path in problem_dirs(default_dir)}
    forced_problems = {path.name: path for path in problem_dirs(forced_dir)}
    problem_names = sorted(set(default_problems) | set(forced_problems))

    chunks: List[Dict[str, Any]] = []
    problem_summaries: List[Dict[str, Any]] = []

    for problem_name in problem_names:
        default_problem_dir = default_problems.get(problem_name)
        forced_problem_dir = forced_problems.get(problem_name)
        meta_dir = default_problem_dir or forced_problem_dir
        if meta_dir is None:
            continue

        problem = safe_load_json(meta_dir / "problem.json") or {}
        base_solution = safe_load_json(meta_dir / "base_solution.json") or {}
        chunks_payload = safe_load_json(meta_dir / "chunks.json") or {}
        chunk_texts = chunks_payload.get("chunks", [])
        base_accuracy = 1.0 if base_solution.get("is_correct") is True else 0.0

        paired_count = 0
        full_count = 0
        default_count = 0
        forced_count = 0

        for chunk_idx, chunk_text in enumerate(chunk_texts):
            default_values = (
                valid_solutions(default_problem_dir / f"chunk_{chunk_idx}" / "solutions.json")
                if default_problem_dir
                else []
            )
            forced_values = (
                valid_solutions(forced_problem_dir / f"chunk_{chunk_idx}" / "solutions.json")
                if forced_problem_dir
                else []
            )
            default_accuracy = accuracy(default_values)
            forced_accuracy = accuracy(forced_values)
            n_default = len(default_values)
            n_forced = len(forced_values)

            if n_default:
                default_count += 1
            if n_forced:
                forced_count += 1
            if n_default and n_forced:
                paired_count += 1
            if n_default >= min_valid and n_forced >= min_valid:
                full_count += 1

            default_delta = (
                None if default_accuracy is None else default_accuracy - base_accuracy
            )
            forced_delta = (
                None if forced_accuracy is None else forced_accuracy - base_accuracy
            )
            if base_type == "correct":
                anchor_strength = (
                    None
                    if default_accuracy is None
                    else max(0.0, base_accuracy - default_accuracy)
                )
                forced_strength = (
                    None
                    if forced_accuracy is None
                    else max(0.0, base_accuracy - forced_accuracy)
                )
                qualitative_label = "positive anchor"
            else:
                anchor_strength = (
                    None if default_accuracy is None else max(0.0, default_accuracy)
                )
                forced_strength = (
                    None if forced_accuracy is None else max(0.0, forced_accuracy)
                )
                qualitative_label = "misleading anchor"

            chunks.append(
                {
                    "model": model_slug,
                    "base_type": base_type,
                    "problem_name": problem_name,
                    "subject": problem.get("subject"),
                    "question_idx": problem.get("question_idx"),
                    "question": problem.get("problem", ""),
                    "gt_answer": problem.get("gt_answer"),
                    "gt_answer_text": option_letter_text(problem, problem.get("gt_answer")),
                    "base_answer": base_solution.get("answer"),
                    "base_answer_text": option_letter_text(
                        problem, base_solution.get("answer")
                    ),
                    "base_is_correct": base_solution.get("is_correct"),
                    "chunk_idx": chunk_idx,
                    "chunk": chunk_text,
                    "chunk_type": classify_chunk(chunk_text),
                    "n_default": n_default,
                    "n_forced": n_forced,
                    "default_accuracy": default_accuracy,
                    "forced_accuracy": forced_accuracy,
                    "default_delta": default_delta,
                    "forced_delta": forced_delta,
                    "anchor_strength": anchor_strength,
                    "forced_strength": forced_strength,
                    "qualitative_label": qualitative_label,
                    "full_paired": n_default >= min_valid and n_forced >= min_valid,
                }
            )

        problem_summaries.append(
            {
                "problem_name": problem_name,
                "subject": problem.get("subject"),
                "num_chunks": len(chunk_texts),
                "chunks_with_default": default_count,
                "chunks_with_forced": forced_count,
                "paired_chunks": paired_count,
                "full_paired_chunks": full_count,
            }
        )

    full_chunks = [chunk for chunk in chunks if chunk["full_paired"]]
    default_counts = [chunk["n_default"] for chunk in chunks if chunk["n_default"]]
    forced_counts = [chunk["n_forced"] for chunk in chunks if chunk["n_forced"]]
    default_accuracies = [
        chunk["default_accuracy"]
        for chunk in full_chunks
        if chunk["default_accuracy"] is not None
    ]
    forced_accuracies = [
        chunk["forced_accuracy"]
        for chunk in full_chunks
        if chunk["forced_accuracy"] is not None
    ]

    return {
        "model": model_slug,
        "base_type": base_type,
        "default_dir": str(default_dir),
        "forced_dir": str(forced_dir),
        "num_default_problems": len(default_problems),
        "num_forced_problems": len(forced_problems),
        "num_any_problems": len(problem_names),
        "num_chunks": len(chunks),
        "num_full_paired_chunks": len(full_chunks),
        "median_default_valid": statistics.median(default_counts) if default_counts else 0,
        "median_forced_valid": statistics.median(forced_counts) if forced_counts else 0,
        "mean_default_accuracy": (
            statistics.mean(default_accuracies) if default_accuracies else None
        ),
        "mean_forced_accuracy": (
            statistics.mean(forced_accuracies) if forced_accuracies else None
        ),
        "problem_summaries": problem_summaries,
        "chunks": chunks,
    }


def select_top_chunks(
    chunks: Iterable[Dict[str, Any]],
    base_type: str,
    min_valid: int,
    limit: int = 6,
    exclude_final: bool = False,
) -> List[Dict[str, Any]]:
    candidates = [
        chunk
        for chunk in chunks
        if chunk["n_default"] >= min_valid
        and chunk["n_forced"] >= min_valid
        and chunk["anchor_strength"] is not None
    ]
    if exclude_final:
        candidates = [
            chunk
            for chunk in candidates
            if chunk["chunk_type"]
            not in {"final-answer commitment", "answer-option token"}
        ]

    def key(chunk: Dict[str, Any]) -> Tuple[float, float, int]:
        forced_strength = chunk["forced_strength"] or 0.0
        return (
            chunk["anchor_strength"] or 0.0,
            forced_strength,
            chunk["n_default"] + chunk["n_forced"],
        )

    return sorted(candidates, key=key, reverse=True)[:limit]


def chunk_markdown(chunk: Dict[str, Any], rank: int) -> List[str]:
    direction = (
        "removal damages correctness"
        if chunk["base_type"] == "correct"
        else "removal repairs the wrong trace"
    )
    lines = [
        (
            f"{rank}. `{chunk['problem_name']}` chunk `{chunk['chunk_idx']}` "
            f"({chunk['chunk_type']}): {direction}."
        ),
        (
            f"   Evidence: default acc `{fmt_float(chunk['default_accuracy'])}` "
            f"from `{chunk['n_default']}` rollouts; forced acc "
            f"`{fmt_float(chunk['forced_accuracy'])}` from `{chunk['n_forced']}`."
        ),
        f"   Chunk: \"{clean_text(chunk['chunk'])}\"",
        (
            f"   Question: {clean_text(chunk['question'], 220)} "
            f"Correct answer: `{chunk['gt_answer_text']}`; base answer: "
            f"`{chunk['base_answer_text']}`."
        ),
    ]
    return lines


def coverage_sentence(summary: Dict[str, Any]) -> str:
    return (
        f"default problems `{summary['num_default_problems']}`, forced problems "
        f"`{summary['num_forced_problems']}`, full paired chunks "
        f"`{summary['num_full_paired_chunks']}/{summary['num_chunks']}`, median "
        f"valid rollouts default/forced "
        f"`{summary['median_default_valid']}/{summary['median_forced_valid']}`"
    )


def interpret_model_base(summary: Dict[str, Any], min_valid: int) -> List[str]:
    chunks = [
        chunk
        for chunk in summary["chunks"]
        if chunk["n_default"] >= min_valid and chunk["n_forced"] >= min_valid
    ]
    if not chunks:
        return [
            "No high-confidence paired chunks yet, so qualitative claims should wait."
        ]

    default_accs = [
        chunk["default_accuracy"]
        for chunk in chunks
        if chunk["default_accuracy"] is not None
    ]
    forced_accs = [
        chunk["forced_accuracy"]
        for chunk in chunks
        if chunk["forced_accuracy"] is not None
    ]
    if summary["base_type"] == "correct":
        fragile = sum(1 for value in default_accs if value <= 0.5)
        robust = sum(1 for value in default_accs if value >= 0.9)
        forced_fragile = sum(1 for value in forced_accs if value <= 0.5)
        return [
            (
                f"Most chunks are redundant if default median accuracy stays high; "
                f"here `{robust}/{len(default_accs)}` full paired chunks retain "
                "at least 90% accuracy after removal."
            ),
            (
                f"The meaningful anchors are the brittle minority: "
                f"`{fragile}/{len(default_accs)}` chunks fall to <=50% accuracy "
                "under default resampling."
            ),
            (
                f"Forced-answer continuation is harsher in this slice: "
                f"`{forced_fragile}/{len(forced_accs)}` chunks fall to <=50% "
                "forced accuracy."
            ),
        ]

    repaired = sum(1 for value in default_accs if value >= 0.5)
    stubborn = sum(1 for value in default_accs if value <= 0.1)
    forced_repaired = sum(1 for value in forced_accs if value >= 0.5)
    return [
        (
            "For incorrect bases, high accuracy after chunk removal is the key "
            "qualitative signal: the removed chunk was helping lock in a wrong "
            "trajectory."
        ),
        (
            f"`{repaired}/{len(default_accs)}` full paired chunks repair to at "
            "least 50% default accuracy, while "
            f"`{stubborn}/{len(default_accs)}` remain near-zero."
        ),
        (
            f"Forced-answer repair is weaker here: `{forced_repaired}/{len(forced_accs)}` "
            "chunks reach at least 50% forced accuracy."
        ),
    ]


def render_markdown(payload: Dict[str, Any], min_valid: int) -> str:
    lines: List[str] = [
        "# Qwen Thought Anchors Qualitative Report",
        "",
        (
            "This report is intentionally qualitative: it separates robust chunks, "
            "positive anchors, misleading anchors, and final-answer artifacts. "
            f"Examples below require at least `{min_valid}` valid default and "
            f"`{min_valid}` valid forced rollouts."
        ),
        "",
        "## Current Readiness",
        "",
    ]

    for summary in payload["summaries"]:
        lines.append(
            f"- `{summary['model']}` `{summary['base_type']}`: "
            f"{coverage_sentence(summary)}."
        )

    lines.extend(["", "## Qualitative Findings", ""])

    for summary in payload["summaries"]:
        lines.append(f"### {summary['model']} / {summary['base_type']} base")
        lines.append("")
        lines.append(f"Coverage: {coverage_sentence(summary)}.")
        lines.append("")
        for sentence in interpret_model_base(summary, min_valid):
            lines.append(f"- {sentence}")
        lines.append("")

        full_chunks = [
            chunk
            for chunk in summary["chunks"]
            if chunk["n_default"] >= min_valid and chunk["n_forced"] >= min_valid
        ]
        if not full_chunks:
            lines.append("No case studies yet.")
            lines.append("")
            continue

        lines.append("Most interpretable anchors, excluding trivial final-answer tokens:")
        non_final = select_top_chunks(
            summary["chunks"],
            summary["base_type"],
            min_valid,
            limit=3,
            exclude_final=True,
        )
        if non_final:
            for idx, chunk in enumerate(non_final, 1):
                lines.extend(chunk_markdown(chunk, idx))
        else:
            lines.append("No non-final high-confidence anchors yet.")
        lines.append("")

        lines.append("Strongest raw anchors, including final-answer artifacts:")
        for idx, chunk in enumerate(
            select_top_chunks(
                summary["chunks"],
                summary["base_type"],
                min_valid,
                limit=3,
                exclude_final=False,
            ),
            1,
        ):
            lines.extend(chunk_markdown(chunk, idx))
        lines.append("")

    lines.extend(
        [
            "## What Not To Overclaim Yet",
            "",
            "- MATH is still in shared-base preparation/rollout, so fair MATH anchor conclusions are not available yet.",
            "- The MMLU slice is matched and complete for this study design, but it is still 10 questions per base type, so treat the examples as qualitative evidence rather than benchmark-wide accuracy estimates.",
            "- Chunk-type labels in this report are heuristic. The numeric MMLU Thought Anchors metrics and plots live under `analysis/qwen_anchor/mmlu/`.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlu-root", default="mmlu_rollouts")
    parser.add_argument("--mmlu-suffix", default="paper10_matched_clean_apr28")
    parser.add_argument("--output-dir", default="analysis/qwen_anchor")
    parser.add_argument("--min-valid", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.mmlu_root)
    summaries = [
        scan_model_base(root, model, base_type, args.mmlu_suffix, args.min_valid)
        for model in MODELS
        for base_type in BASE_TYPES
    ]
    payload = {"summaries": summaries, "min_valid": args.min_valid}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qualitative_report.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "qualitative_report.md").write_text(
        render_markdown(payload, args.min_valid),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'qualitative_report.md'}")


if __name__ == "__main__":
    main()
