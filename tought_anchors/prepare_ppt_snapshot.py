#!/usr/bin/env python3
"""Create PPT-ready current-result summaries and circular anchor visuals.

This intentionally uses only already-computed local MMLU qualitative data.
MATH remains partial, so the generated narrative treats it as appendix evidence.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path("analysis/qwen_anchor")
VIS_DIR = ROOT / "visualizations"
QUAL_PATH = ROOT / "qualitative_report.json"
SUMMARY_PATH = ROOT / "summary_report.json"
OUT_MD = ROOT / "ppt_current_snapshot.md"
MATH_PARTIAL_QUAL_PATH = ROOT / "math_partial" / "qualitative_report.json"
MMLU_ROOT = Path("mmlu_rollouts")
MATH_ROOT = Path("math_rollouts")
MMLU_SUFFIX = "paper10_matched_clean_apr28"
MATH_SUFFIX = "paper10_matched_shared_apr28"

MODEL_LABELS = {
    "Qwen2.5-Math-7B": "Base",
    "Qwen2.5-Math-7B-Instruct": "Instruct",
    "Qwen2.5-Math-7B-Oat-Zero": "Oat-Zero",
}

MODEL_COLORS = {
    "Qwen2.5-Math-7B": "#2f6f73",
    "Qwen2.5-Math-7B-Instruct": "#d07a2d",
    "Qwen2.5-Math-7B-Oat-Zero": "#6b5ca5",
}

TYPE_COLORS = {
    "correct": "#d45b4f",
    "incorrect": "#3778bf",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(text: str, limit: int = 130) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def full_chunks(summary: Dict[str, Any], *, exclude_answer_tokens: bool = True) -> List[Dict[str, Any]]:
    return [
        chunk
        for chunk in summary.get("chunks", [])
        if chunk.get("full_paired")
        and chunk.get("default_accuracy") is not None
        and chunk.get("forced_accuracy") is not None
        and (not exclude_answer_tokens or chunk.get("chunk_type") != "answer-option token")
    ]


def pct(num: int, denom: int) -> str:
    if denom == 0:
        return "NA"
    return f"{num / denom:.0%}"


def f2(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def f1(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.1f}"


def by_model_base(qual: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        (summary["model"], summary["base_type"]): summary
        for summary in qual.get("summaries", [])
    }


def rollout_dir(
    root: Path,
    model: str,
    base_type: str,
    suffix: str,
    *,
    forced: bool = False,
) -> Path:
    dirname = f"{base_type}_base_solution"
    if forced:
        dirname += "_forced_answer"
    dirname += f"_{suffix}"
    return root / model / "temperature_0.6_top_p_0.95" / dirname


def solutions_path(
    root: Path,
    suffix: str,
    chunk: Dict[str, Any],
    *,
    forced: bool = False,
) -> Path:
    return (
        rollout_dir(root, chunk["model"], chunk["base_type"], suffix, forced=forced)
        / chunk["problem_name"]
        / f"chunk_{chunk['chunk_idx']}"
        / "solutions.json"
    )


def answer_counts(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}
    counts: Dict[str, int] = {}
    for item in payload:
        if not isinstance(item, dict) or "error" in item:
            continue
        answer = str(item.get("answer", "")).strip()
        if answer in {"", "None"}:
            continue
        counts[answer] = counts.get(answer, 0) + 1
    return counts


def diversity_metrics(counts: Dict[str, int]) -> Dict[str, Any]:
    total = sum(counts.values())
    if total <= 0:
        return {"n": 0, "unique": 0, "effective": None, "top_share": None, "top_answer": "NA"}
    probs = [count / total for count in counts.values()]
    entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0)
    top_answer, top_count = max(counts.items(), key=lambda item: item[1])
    return {
        "n": total,
        "unique": len(counts),
        "effective": math.exp(entropy),
        "top_share": top_count / total,
        "top_answer": top_answer,
    }


def format_counts(counts: Dict[str, int], *, limit: int = 5) -> str:
    if not counts:
        return "NA"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{answer}:{count}" for answer, count in ordered)


def rollout_items_for_chunk(
    chunk: Dict[str, Any],
    *,
    root: Path = MMLU_ROOT,
    suffix: str = MMLU_SUFFIX,
) -> List[Dict[str, Any]]:
    path = solutions_path(root, suffix, chunk)
    if not path.exists():
        return []
    try:
        payload = load_json(path)
    except Exception:
        return []
    return [item for item in payload if isinstance(item, dict) and "error" not in item]


def answer_representative_snippet(
    items: Iterable[Dict[str, Any]],
    answer: str,
    *,
    limit: int = 220,
) -> str:
    candidates = [item for item in items if str(item.get("answer")) == answer]
    if not candidates:
        return "NA"

    def candidate_score(item: Dict[str, Any]) -> Tuple[int, int]:
        text = str(item.get("rollout") or item.get("full_cot") or "")
        spam_penalty = text.count("Final Answer") + text.count("\\]") + text.count("```")
        return (spam_penalty, -len(text))

    for item in sorted(candidates, key=candidate_score):
        if str(item.get("answer")) != answer:
            continue
        text = str(item.get("rollout") or item.get("full_cot") or "")
        text = re.sub(r"Final Answer:.*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"`+", "", text)
        text_value = clean_text(text, limit)
        if text_value:
            return text_value
    return "NA"


def answer_distribution_phrase(counts: Dict[str, int]) -> str:
    total = sum(counts.values())
    if total <= 0:
        return "NA"
    parts = []
    for answer, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        parts.append(f"{answer}: {count}/{total}")
    return ", ".join(parts)


def philosophy_cases(index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """A same-question, model-specific-anchor comparison for the main qualitative slide."""
    return [
        (
            "Base",
            find_chunk(index, "Qwen2.5-Math-7B", "correct", "problem_philosophy_64", 9),
        ),
        (
            "Instruct",
            find_chunk(index, "Qwen2.5-Math-7B-Instruct", "correct", "problem_philosophy_64", 12),
        ),
        (
            "Oat-Zero",
            find_chunk(index, "Qwen2.5-Math-7B-Oat-Zero", "correct", "problem_philosophy_64", 9),
        ),
    ]


def philosophy_problem_context_lines(index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[str]:
    cases = philosophy_cases(index)
    first = cases[0][1]
    base_answers = ", ".join(
        f"{model_label}: `{answer_text(chunk, 'base')}`"
        for model_label, chunk in cases
    )
    return [
        "**Problem used for all three models.**",
        "",
        f"> {clean_text(first.get('question', ''), 650)}",
        "",
        f"**Correct answer.** `{answer_text(first, 'gt')}`",
        "",
        f"**Before ablation.** All three base traces answer correctly: {base_answers}.",
        "",
        "**Why this example is in the main talk.** It is a matched same-question comparison. The removed chunk is model-specific because each model wrote a different base trace, but each removed chunk sits in the same conceptual area: whether Baier's morality rule requires universal teachability / non-taboo / all-of-the-above. After that local perturbation, we compare 100 resampled continuations per model.",
        "",
    ]


def philosophy_motif_table_rows(
    index: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[List[str]]:
    cases = [
        (
            "Base",
            find_chunk(index, "Qwen2.5-Math-7B", "correct", "problem_philosophy_64", 9),
        ),
        (
            "Instruct",
            find_chunk(index, "Qwen2.5-Math-7B-Instruct", "correct", "problem_philosophy_64", 12),
        ),
        (
            "Oat-Zero",
            find_chunk(index, "Qwen2.5-Math-7B-Oat-Zero", "correct", "problem_philosophy_64", 9),
        ),
    ]
    motif_labels = {
        "A": "A-only path: rejects all-of-above as too strict",
        "B": "B-centered path: universal teachability emphasized",
        "C": "C-centered path: taboo condition emphasized",
        "D": "D/all-of-above path: accepts A/B/C as requirements",
    }
    rows = [["Model", "Removed thought", "Path split", "Representative internal reasoning"]]
    for model_label, chunk in cases:
        items = rollout_items_for_chunk(chunk)
        counts = answer_counts(solutions_path(MMLU_ROOT, MMLU_SUFFIX, chunk))
        motif_parts = []
        for answer, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:2]:
            snippet = answer_representative_snippet(items, answer, limit=150)
            motif_parts.append(f"{motif_labels.get(answer, f'answer {answer} path')}: \"{snippet}\"")
        rows.append(
            [
                model_label,
                clean_text(chunk.get("chunk", ""), 120),
                answer_distribution_phrase(counts),
                " / ".join(motif_parts),
            ]
        )
    return rows


def problem_full_nonanswer_chunks(
    index: Dict[Tuple[str, str], Dict[str, Any]],
    model: str,
    base_type: str,
    problem_name: str,
) -> List[Dict[str, Any]]:
    return [
        chunk
        for chunk in index[(model, base_type)].get("chunks", [])
        if chunk.get("problem_name") == problem_name
        and chunk.get("full_paired")
        and chunk.get("chunk_type") != "answer-option token"
    ]


def philosophy_problem_audit_table(index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[List[str]]:
    rows = [["Model", "Full chunks on this problem", "Mean effective paths", "Collapsed chunks", "Reasoning-path pattern"]]
    patterns = {
        "Qwen2.5-Math-7B": "No local perturbation collapses the trace; several A/B/C/D interpretations stay alive.",
        "Qwen2.5-Math-7B-Instruct": "Intermediate: mostly still multi-path, but one requirement chunk collapses to A in 91/100 rollouts.",
        "Qwen2.5-Math-7B-Oat-Zero": "Clear drop: later chunks repeatedly collapse to the same A-only story.",
    }
    for model in MODEL_LABELS:
        chunks = problem_full_nonanswer_chunks(index, model, "correct", "problem_philosophy_64")
        metrics = [diversity_for_chunk(chunk) for chunk in chunks]
        effective_values = [metric["effective"] for metric in metrics if metric["effective"] is not None]
        collapsed = sum(
            1
            for metric in metrics
            if metric["top_share"] is not None and metric["top_share"] >= 0.9
        )
        rows.append(
            [
                MODEL_LABELS[model],
                str(len(chunks)),
                f2(sum(effective_values) / len(effective_values) if effective_values else None),
                f"{collapsed}/{len(chunks)}",
                patterns[model],
            ]
        )
    return rows


def diagnostic_philosophy_cases(
    index: Dict[Tuple[str, str], Dict[str, Any]]
) -> List[Tuple[str, str, Dict[str, Any], int, str]]:
    return [
        (
            "Base: perturbation opens several reasoning paths",
            "Qwen2.5-Math-7B",
            find_chunk(index, "Qwen2.5-Math-7B", "correct", "problem_philosophy_64", 9),
            4,
            "The base model does not lock into one explanation. After the universal-teachability thought is removed, it alternates between minimal-requirement, taboo-centered, all-of-the-above, and universal-teachability interpretations.",
        ),
        (
            "Instruct: one dominant path appears",
            "Qwen2.5-Math-7B-Instruct",
            find_chunk(index, "Qwen2.5-Math-7B-Instruct", "correct", "problem_philosophy_64", 12),
            3,
            "The instruction-tuned model mostly turns the same conceptual area into a single A-only story: A is necessary, while B/C/D are too strong.",
        ),
        (
            "Oat-Zero: reasoning path collapse",
            "Qwen2.5-Math-7B-Oat-Zero",
            find_chunk(index, "Qwen2.5-Math-7B-Oat-Zero", "correct", "problem_philosophy_64", 14),
            2,
            "The RL-style model gives the cleanest collapse example: all 100 continuations repeat the same A-only rationale after this local thought is removed.",
        ),
    ]


def branch_lines_for_chunk(chunk: Dict[str, Any], *, limit: int) -> List[str]:
    counts = answer_counts(solutions_path(MMLU_ROOT, MMLU_SUFFIX, chunk))
    items = rollout_items_for_chunk(chunk)
    branch_names = {
        "A": "A-only/minimal-requirement path",
        "B": "universal-teachability path",
        "C": "non-taboo path",
        "D": "all-of-the-above path",
    }
    lines = []
    for answer, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        snippet = answer_representative_snippet(items, answer, limit=260)
        lines.append(
            f"- {answer} path `{count}/100`: {branch_names.get(answer, 'other path')}. Representative reasoning: \"{snippet}\""
        )
    return lines


def philosophy_motif_lines(index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    lines.extend(philosophy_problem_context_lines(index))
    lines.append(
        "**Selection rule.** This was not chosen as a random sample. Among the common MMLU correct problems in the current slice, this question has the largest Base-vs-Oat-Zero gap in problem-level path diversity."
    )
    lines.append("")
    lines.append(markdown_table(philosophy_problem_audit_table(index)))
    lines.append("")
    lines.append(
        "**Path codebook.** The answer letters below are used as shorthand for internal reasoning branches: A = only `part of the mores` is treated as necessary; B = universal teachability becomes decisive; C = non-taboo becomes decisive; D = all three requirements are accepted."
    )
    lines.append("")
    for title, _model, chunk, branch_limit, reading in diagnostic_philosophy_cases(index):
        diversity = diversity_for_chunk(chunk)
        lines.append(f"### {title}")
        lines.append(f"- Removed thought: `{clean_text(chunk.get('chunk', ''), 260)}`")
        lines.append(
            f"- Path split after removal: `{format_counts(diversity['counts'], limit=6)}`; effective paths `{f2(diversity.get('effective'))}`; dominant share `{f2(diversity.get('top_share'))}`."
        )
        lines.append("- Internal branches observed in rollout text:")
        lines.extend(branch_lines_for_chunk(chunk, limit=branch_limit))
        lines.append(f"- Reading: {reading}")
        lines.append("")
    lines.append(
        "**Slide reading.** The drop is visible at the reasoning-path level: Base keeps multiple interpretations alive on the same question, Instruct narrows toward one A-only explanation, and Oat-Zero can collapse completely into that same explanation across all 100 continuations. This is the qualitative example to use for the thesis slide."
    )
    lines.append("")
    return lines


def chunk_strength(chunk: Dict[str, Any]) -> float:
    value = chunk.get("anchor_strength")
    return 0.0 if value is None else float(value)


def chunk_forced_strength(chunk: Dict[str, Any]) -> float:
    value = chunk.get("forced_strength")
    return 0.0 if value is None else float(value)


def sorted_top_chunks(
    chunks: Iterable[Dict[str, Any]],
    *,
    limit: int = 5,
    exclude_answer_tokens: bool = True,
) -> List[Dict[str, Any]]:
    items = []
    for chunk in chunks:
        if exclude_answer_tokens and chunk.get("chunk_type") == "answer-option token":
            continue
        items.append(chunk)
    return sorted(
        items,
        key=lambda item: max(chunk_strength(item), chunk_forced_strength(item)),
        reverse=True,
    )[:limit]


def compact_problem_name(name: str) -> str:
    name = name.replace("problem_", "")
    name = re.sub(r"_\d+$", "", name)
    return name.replace("_", " ")


def collect_stats(qual: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in qual.get("summaries", []):
        chunks = full_chunks(summary)
        total_non_answer = sum(
            1
            for chunk in summary.get("chunks", [])
            if chunk.get("chunk_type") != "answer-option token"
        )
        mean_default = (
            sum(float(c["default_accuracy"]) for c in chunks) / len(chunks)
            if chunks
            else None
        )
        mean_forced = (
            sum(float(c["forced_accuracy"]) for c in chunks) / len(chunks)
            if chunks
            else None
        )
        base_type = summary["base_type"]
        if base_type == "correct":
            redundant = sum(1 for c in chunks if c["default_accuracy"] >= 0.9)
            brittle = sum(1 for c in chunks if c["default_accuracy"] <= 0.5)
            forced_brittle = sum(1 for c in chunks if c["forced_accuracy"] <= 0.5)
            rows.append(
                {
                    "model": summary["model"],
                    "base_type": base_type,
                    "full": len(chunks),
                    "total": total_non_answer,
                    "main_a": redundant,
                    "main_a_label": "redundant >=90%",
                    "main_b": brittle,
                    "main_b_label": "brittle <=50%",
                    "forced": forced_brittle,
                    "forced_label": "forced <=50%",
                    "mean_default": mean_default,
                    "mean_forced": mean_forced,
                }
            )
        else:
            repair = sum(1 for c in chunks if c["default_accuracy"] >= 0.5)
            stuck = sum(1 for c in chunks if c["default_accuracy"] <= 0.1)
            forced_repair = sum(1 for c in chunks if c["forced_accuracy"] >= 0.5)
            rows.append(
                {
                    "model": summary["model"],
                    "base_type": base_type,
                    "full": len(chunks),
                    "total": total_non_answer,
                    "main_a": repair,
                    "main_a_label": "repairs >=50%",
                    "main_b": stuck,
                    "main_b_label": "stuck <=10%",
                    "forced": forced_repair,
                    "forced_label": "forced repairs >=50%",
                    "mean_default": mean_default,
                    "mean_forced": mean_forced,
                }
            )
    return rows


def collect_math_partial_stats() -> List[Dict[str, Any]]:
    if not MATH_PARTIAL_QUAL_PATH.exists():
        return []
    payload = load_json(MATH_PARTIAL_QUAL_PATH)
    rows: List[Dict[str, Any]] = []
    for summary in payload.get("summaries", []):
        chunks = full_chunks(summary, exclude_answer_tokens=False)
        base_type = summary["base_type"]
        if base_type == "correct":
            main = sum(1 for chunk in chunks if chunk["default_accuracy"] <= 0.5)
            secondary = sum(1 for chunk in chunks if chunk["default_accuracy"] >= 0.9)
            forced = sum(1 for chunk in chunks if chunk["forced_accuracy"] <= 0.5)
            main_label = "brittle <=50%"
            secondary_label = "robust >=90%"
            forced_label = "forced <=50%"
        else:
            main = sum(1 for chunk in chunks if chunk["default_accuracy"] >= 0.5)
            secondary = sum(1 for chunk in chunks if chunk["default_accuracy"] <= 0.1)
            forced = sum(1 for chunk in chunks if chunk["forced_accuracy"] >= 0.5)
            main_label = "repair >=50%"
            secondary_label = "stuck <=10%"
            forced_label = "forced repair >=50%"
        rows.append(
            {
                "model": summary["model"],
                "base_type": base_type,
                "full": len(chunks),
                "total": summary.get("num_chunks", 0),
                "main": main,
                "secondary": secondary,
                "forced": forced,
                "main_label": main_label,
                "secondary_label": secondary_label,
                "forced_label": forced_label,
            }
        )
    return rows


def collect_answer_diversity_stats(
    qual: Dict[str, Any],
    *,
    root: Path = MMLU_ROOT,
    suffix: str = MMLU_SUFFIX,
    exclude_answer_tokens: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary in qual.get("summaries", []):
        chunk_rows: List[Dict[str, Any]] = []
        for chunk in full_chunks(summary, exclude_answer_tokens=exclude_answer_tokens):
            counts = answer_counts(solutions_path(root, suffix, chunk))
            metrics = diversity_metrics(counts)
            if metrics["n"] <= 0 or metrics["effective"] is None or metrics["top_share"] is None:
                continue
            chunk_rows.append({"chunk": chunk, "counts": counts, **metrics})
        if not chunk_rows:
            continue
        rows.append(
            {
                "model": summary["model"],
                "base_type": summary["base_type"],
                "chunks": len(chunk_rows),
                "mean_effective": sum(row["effective"] for row in chunk_rows) / len(chunk_rows),
                "median_effective": sorted(row["effective"] for row in chunk_rows)[len(chunk_rows) // 2],
                "mean_top_share": sum(row["top_share"] for row in chunk_rows) / len(chunk_rows),
                "collapse": sum(1 for row in chunk_rows if row["top_share"] >= 0.9),
                "single_answer": sum(1 for row in chunk_rows if row["unique"] == 1),
            }
        )
    return rows


def diversity_for_chunk(
    chunk: Dict[str, Any],
    *,
    root: Path = MMLU_ROOT,
    suffix: str = MMLU_SUFFIX,
) -> Dict[str, Any]:
    counts = answer_counts(solutions_path(root, suffix, chunk))
    metrics = diversity_metrics(counts)
    return {"counts": counts, **metrics}


def markdown_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(header)) + " |")
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def find_chunk(
    index: Dict[Tuple[str, str], Dict[str, Any]],
    model: str,
    base_type: str,
    problem_name: str,
    chunk_idx: int,
) -> Dict[str, Any]:
    summary = index[(model, base_type)]
    for chunk in summary.get("chunks", []):
        if chunk.get("problem_name") == problem_name and chunk.get("chunk_idx") == chunk_idx:
            return chunk
    raise KeyError((model, base_type, problem_name, chunk_idx))


def answer_text(chunk: Dict[str, Any], key: str) -> str:
    return str(chunk.get(f"{key}_answer_text") or chunk.get(f"{key}_answer") or "NA")


def case_study_lines(
    title: str,
    chunk: Dict[str, Any],
    *,
    lesson: str,
    max_question_len: int = 420,
    root: Path = MMLU_ROOT,
    suffix: str = MMLU_SUFFIX,
) -> List[str]:
    default_accuracy = chunk.get("default_accuracy")
    if chunk["base_type"] == "correct":
        if default_accuracy is not None and default_accuracy <= 0.5:
            direction = "correctness drops after removal"
        elif default_accuracy is not None and default_accuracy >= 0.9:
            direction = "answer remains correct, but diversity can still be low"
        else:
            direction = "correct trace becomes unstable after removal"
    elif default_accuracy is not None and default_accuracy >= 0.5:
        direction = "wrong trace repairs after removal"
    elif default_accuracy is not None and default_accuracy <= 0.1:
        direction = "wrong trace stays stuck after removal"
    else:
        direction = "wrong trace changes only partially after removal"
    diversity = diversity_for_chunk(chunk, root=root, suffix=suffix)
    top_share = diversity.get("top_share")
    effective = diversity.get("effective")
    top_answer = diversity.get("top_answer")
    items = rollout_items_for_chunk(chunk, root=root, suffix=suffix)
    representative = (
        answer_representative_snippet(items, str(top_answer), limit=260)
        if top_answer and top_answer != "NA"
        else "NA"
    )
    diversity_line = (
        f"Answer distribution after removal: `{format_counts(diversity['counts'])}`; "
        f"dominant answer share `{f2(top_share)}`; effective answer count `{f2(effective)}`."
    )
    return [
        f"### {title}",
        "",
        f"**Question.** {clean_text(chunk.get('question', ''), max_question_len)}",
        "",
        f"**Correct answer.** `{answer_text(chunk, 'gt')}`",
        "",
        f"**Model base answer.** `{answer_text(chunk, 'base')}` (`base_is_correct={chunk.get('base_is_correct')}`)",
        "",
        f"**Removed thought chunk.** `{clean_text(chunk.get('chunk', ''), 260)}`",
        "",
        (
            f"**After removing this chunk.** default accuracy `{f2(chunk.get('default_accuracy'))}` "
            f"over `{chunk.get('n_default', 'NA')}` rollouts; forced-answer accuracy "
            f"`{f2(chunk.get('forced_accuracy'))}` over `{chunk.get('n_forced', 'NA')}` rollouts."
        ),
        "",
        f"**Diversity signal.** {diversity_line}",
        "",
        f"**Representative internal continuation.** Dominant answer `{top_answer}` follows this reasoning: \"{representative}\"",
        "",
        f"**Interpretation.** {direction}. {lesson}",
        "",
    ]


def diversity_glossary_lines() -> List[str]:
    return [
        "### How to read this table",
        "",
        "- This is the main table for the presentation thesis: lower diversity means the 100 resampled rollouts keep following one dominant answer trajectory.",
        "- This numeric table is an answer-path diversity proxy. Pair it with the next qualitative section, which reads representative internal continuations from the rollout text.",
        "- `Effective answers` is the entropy-based number of answers effectively used after a chunk is removed. `1.0` means all rollouts chose the same answer; `4.0` would mean A/B/C/D were used evenly.",
        "- `Dominant answer share` is the average fraction of rollouts choosing the most common answer. Higher means more collapse onto one path.",
        "- `Answer-collapsed chunks` counts chunks where at least 90 of 100 rollouts choose the same answer. Higher means lower reasoning-path diversity.",
        "- The cleanest thesis-relevant comparison is the `correct` trace row: Oat-Zero has the highest collapse rate and lowest effective answer count, so its correct reasoning is less exploratory after perturbation.",
        "",
    ]


def anchor_metric_glossary_lines(*, math_partial: bool = False) -> List[str]:
    chunk_name = "Full chunks" if math_partial else "Non-answer chunks"
    chunk_desc = (
        "completed chunks with both 100 default rollouts and 100 forced-answer rollouts"
        if math_partial
        else "completed non-answer reasoning chunks with both default and forced-answer rollout results"
    )
    return [
        "### How to read the anchor table",
        "",
        f"- `{chunk_name}` means `{chunk_desc}`. The denominator is all chunks found in those traces, so low coverage means the row is still partial.",
        "- `correct` rows start from a base trace that originally answered correctly. If accuracy stays high after removing a chunk, that chunk was redundant; if it drops, that chunk was a positive anchor.",
        "- `incorrect` rows start from a base trace that originally answered incorrectly. If accuracy rises after removing a chunk, that chunk was a misleading anchor; if it stays near zero, the wrong trace is stuck.",
        "- `default accuracy` is the fraction of normal resampled rollouts that answer correctly after one chunk is removed.",
        "- `forced-answer accuracy` is a secondary diagnostic/comparison, not the paper's main Thought Anchors method. The main black-box method is resampling-based counterfactual importance.",
        "",
    ]


def mmlu_anchor_outcome_lines(stats: List[Dict[str, Any]]) -> List[str]:
    """Make the anchor-effect table readable as two explicit causal questions."""
    correct_rows = [row for row in stats if row["base_type"] == "correct"]
    incorrect_rows = [row for row in stats if row["base_type"] == "incorrect"]
    lines: List[str] = []
    lines.append("## Thought-Anchor Outcome Tables")
    lines.append("")
    lines.append(
        "These tables answer a different question from the diversity table. Diversity asks `how many paths remain?`; anchor outcomes ask `what does removing this thought do to correctness?`"
    )
    lines.append("")
    lines.append(
        "**Correct base traces.** These start from a correct model solution. A `positive anchor` is a thought whose removal makes the model lose the answer in at least half of the 100 resampled rollouts. A `robust / redundant` chunk can be removed while the answer stays correct in at least 90 rollouts."
    )
    lines.append("")
    table = [["Model", "Tested chunks", "Mean accuracy after removal", "Robust / redundant", "Positive anchors", "Presentation reading"]]
    for row in correct_rows:
        label = MODEL_LABELS.get(row["model"], row["model"])
        reading = (
            "mostly robust, but paths are highly collapsed"
            if label == "Oat-Zero"
            else "more alternative paths remain after perturbation"
            if label == "Base"
            else "between Base and Oat-Zero"
        )
        table.append(
            [
                label,
                f"{row['full']}/{row['total']}",
                f2(row["mean_default"]),
                f"{row['main_a']} ({pct(row['main_a'], row['full'])})",
                f"{row['main_b']} ({pct(row['main_b'], row['full'])})",
                reading,
            ]
        )
    lines.append(markdown_table(table))
    lines.append("")
    lines.append(
        "**Incorrect base traces.** These start from a wrong model solution. A `misleading anchor / repair` is a thought whose removal lets at least half of the rollouts recover the correct answer. `Stuck` means the trace remains wrong in at least 90 of 100 rollouts."
    )
    lines.append("")
    table = [["Model", "Tested chunks", "Mean accuracy after removal", "Misleading anchors / repairs", "Stuck chunks", "Presentation reading"]]
    for row in incorrect_rows:
        label = MODEL_LABELS.get(row["model"], row["model"])
        reading = (
            "wrong traces often become correct when a bad local thought is removed"
            if row["main_a"] / max(1, row["full"]) >= 0.5
            else "wrong traces usually stay wrong after local perturbation"
        )
        table.append(
            [
                label,
                f"{row['full']}/{row['total']}",
                f2(row["mean_default"]),
                f"{row['main_a']} ({pct(row['main_a'], row['full'])})",
                f"{row['main_b']} ({pct(row['main_b'], row['full'])})",
                reading,
            ]
        )
    lines.append(markdown_table(table))
    lines.append("")
    lines.append(
        "**Forced-answer note.** Forced-answer rollouts are not the central Thought Anchors result in this deck. In our code they are an answer-format/prompted-continuation stress test, useful for appendix checks, but the main anchor construction here is the default resampling/counterfactual accuracy change after removing a thought chunk."
    )
    lines.append("")
    return lines


def math_partial_outcome_lines(rows: List[Dict[str, Any]]) -> List[str]:
    correct_rows = [row for row in rows if row["base_type"] == "correct"]
    incorrect_rows = [row for row in rows if row["base_type"] == "incorrect"]
    lines: List[str] = []
    lines.append("### MATH partial Thought-Anchor outcomes")
    lines.append("")
    lines.append(
        "Appendix only. These numbers use completed chunks so far, not a fair complete MATH comparison."
    )
    lines.append("")
    table = [["Model", "Correct-trace chunks", "Robust >=90%", "Positive anchors <=50%", "Reading"]]
    for row in correct_rows:
        label = MODEL_LABELS.get(row["model"], row["model"])
        table.append(
            [
                label,
                f"{row['full']}/{row['total']}",
                f"{row['secondary']} ({pct(row['secondary'], row['full'])})",
                f"{row['main']} ({pct(row['main'], row['full'])})",
                "partial formal-reasoning bottleneck evidence",
            ]
        )
    lines.append(markdown_table(table))
    lines.append("")
    table = [["Model", "Incorrect-trace chunks", "Repair >=50%", "Stuck <=10%", "Reading"]]
    for row in incorrect_rows:
        label = MODEL_LABELS.get(row["model"], row["model"])
        table.append(
            [
                label,
                f"{row['full']}/{row['total']}",
                f"{row['main']} ({pct(row['main'], row['full'])})",
                f"{row['secondary']} ({pct(row['secondary'], row['full'])})",
                "partial only; do not rank models from this",
            ]
        )
    lines.append(markdown_table(table))
    lines.append("")
    lines.append(
        "Forced-answer MATH diagnostics are intentionally left out of the main snapshot table for the same reason as MMLU: they are a stress test, not the central anchor-construction signal."
    )
    lines.append("")
    return lines


def render_slide_ready_examples(index: Dict[Tuple[str, str], Dict[str, Any]]) -> List[str]:
    base_philosophy = find_chunk(index, "Qwen2.5-Math-7B", "correct", "problem_philosophy_64", 9)
    oat_philosophy = find_chunk(index, "Qwen2.5-Math-7B-Oat-Zero", "correct", "problem_philosophy_64", 14)
    examples = [
        (
            "MMLU Case 1: Base Model Explores Several Alternative Answers",
            base_philosophy,
            "On the same philosophy question, the non-RL base model spreads across A/C/D/B after perturbation, so the ablation exposes multiple competing reasoning trajectories.",
        ),
        (
            "MMLU Case 2: Instruct Mostly Narrows To One Path",
            find_chunk(index, "Qwen2.5-Math-7B-Instruct", "correct", "problem_philosophy_64", 12),
            "The instruction-tuned model is already much less diverse than Base on the same question: most rollouts argue that all conditions are too strict and choose A.",
        ),
        (
            "MMLU Case 3: Oat-Zero Fully Collapses To One Wrong Path",
            oat_philosophy,
            "This is the thesis slide: after a local perturbation on the same question, the RL-style Oat-Zero trace collapses to answer A in all 100 continuations rather than exploring alternatives.",
        ),
        (
            "MMLU Case 4: Low Diversity Can Also Produce A Clean Repair",
            find_chunk(index, "Qwen2.5-Math-7B-Instruct", "incorrect", "problem_high_school_statistics_8", 10),
            "This case is useful as a bridge to Thought Anchors: removal repairs the answer, but the repaired rollouts also concentrate almost entirely on one answer path.",
        ),
        (
            "MMLU Case 5: A Fully Collapsed Robust Trace",
            find_chunk(index, "Qwen2.5-Math-7B-Oat-Zero", "correct", "problem_college_chemistry_91", 4),
            "This is not an important anchor, but it is a clear diversity example: perturbing this step still gives the exact same answer in all 100 rollouts.",
        ),
    ]

    lines = ["## Backup qualitative cards", ""]
    lines.append(
        "The same-question audit above is the main qualitative result. These cards are backup speaker notes with full question/answer context for the same rollouts and a couple of secondary examples."
    )
    lines.append("")
    for title, chunk, lesson in examples:
        lines.extend(case_study_lines(title, chunk, lesson=lesson))
    return lines


def render_math_case_studies() -> List[str]:
    if not MATH_PARTIAL_QUAL_PATH.exists():
        return []
    math_payload = load_json(MATH_PARTIAL_QUAL_PATH)
    math_index = by_model_base(math_payload)
    examples = [
        (
            "MATH Appendix Case 1: Locally True Rule, Globally Wrong Answer",
            find_chunk(math_index, "Qwen2.5-Math-7B", "incorrect", "problem_4682", 1),
            "The base trace over-applies the `one hex digit = four bits` rule and misses that the leading hex digit `6` only needs three bits.",
        ),
        (
            "MATH Appendix Case 2: Formal Bottleneck In A Correct Trace",
            find_chunk(math_index, "Qwen2.5-Math-7B", "correct", "problem_4019", 16),
            "Removing this algebraic implication breaks the route from polynomial root equality to the possible values of `c`.",
        ),
        (
            "MATH Appendix Case 3: Partial Evidence Of Repairable Algebraic Errors",
            find_chunk(math_index, "Qwen2.5-Math-7B-Oat-Zero", "incorrect", "problem_3448", 8),
            "This is a partial-result example where removing a sign-handling step lets the rollout recover the correct binomial expression.",
        ),
    ]
    lines = ["## MATH appendix case studies", ""]
    lines.append(
        "These are not final benchmark conclusions. They are slide-ready examples from the chunks completed so far."
    )
    lines.append("")
    for title, chunk, lesson in examples:
        lines.extend(
            case_study_lines(
                title,
                chunk,
                lesson=lesson,
                max_question_len=520,
                root=MATH_ROOT,
                suffix=MATH_SUFFIX,
            )
        )
    return lines


def render_copy_paste_slide_deck() -> List[str]:
    """Concrete slide draft for the current deck narrative."""
    lines: List[str] = []
    lines.append("## Copy-paste slide deck draft")
    lines.append("")
    lines.append(
        "Use this section directly when building the PPT. The `On-slide copy` text is intentionally short enough to paste into slides; `Speaker note` is what to say while presenting."
    )
    lines.append("")

    slides = [
        {
            "title": "Slide 1: Main Claim",
            "layout": "Title slide with one strong takeaway.",
            "on_slide": [
                "RL-style tuning can narrow reasoning paths",
                "Thought Anchors on Qwen2.5-Math variants",
                "Key finding: Oat-Zero correct traces show the strongest post-perturbation path collapse in the current matched MMLU slice.",
            ],
            "visual": "Optional: use `analysis/qwen_anchor/visualizations/mmlu_model_tendency_circles.svg` as a small background/side visual.",
            "speaker": "이 발표의 핵심은 accuracy가 아니라 reasoning path diversity입니다. 같은 문제를 맞히더라도, thought 하나를 제거했을 때 모델이 여러 reasoning route를 탐색하는지, 아니면 하나의 route로 빨려 들어가는지를 봅니다.",
        },
        {
            "title": "Slide 2: Method",
            "layout": "Simple pipeline diagram.",
            "on_slide": [
                "Take a base reasoning trace",
                "Split it into thought chunks",
                "Remove one chunk",
                "Resample 100 continuations",
                "Measure correctness change and path diversity",
            ],
            "visual": "Draw this as: `Base trace -> chunk removal -> 100 continuations -> accuracy + diversity metrics`.",
            "speaker": "Thought Anchors의 메인 black-box signal은 forced answer가 아니라 chunk removal 이후의 resampled continuation입니다. Forced-answer는 여기서 보조 stress test로만 다룹니다.",
        },
        {
            "title": "Slide 3: Experimental Scope",
            "layout": "Compact setup table.",
            "on_slide": [
                "Models: Qwen2.5-Math-7B Base, Instruct, Oat-Zero",
                "Main benchmark: matched MMLU slice",
                "Rollouts: 100 continuations per removed chunk",
                "Main evidence: full-paired non-answer chunks",
                "MATH: partial appendix only",
            ],
            "visual": "Paste the `Current data status` coverage table from this file.",
            "speaker": "여기서는 final benchmark-wide theorem이라고 말하면 안 됩니다. 현재 완성된 full-paired chunk slice를 기반으로 한 snapshot이라고 프레이밍하는 것이 안전합니다.",
        },
        {
            "title": "Slide 4: Quantitative Takeaway",
            "layout": "Three-row table plus one takeaway sentence.",
            "on_slide": [
                "Correct traces after thought perturbation",
                "Base: effective paths 1.34, collapsed chunks 71%",
                "Instruct: effective paths 1.40, collapsed chunks 67%",
                "Oat-Zero: effective paths 1.24, collapsed chunks 85%",
                "Takeaway: Oat-Zero is the most path-collapsed when it starts from a correct trace.",
            ],
            "visual": "Paste only the `correct` rows from `Primary Table: Reasoning Path Diversity`.",
            "speaker": "여기서 effective paths는 entropy 기반으로 실제로 몇 개의 answer trajectory를 쓰는지 보는 값입니다. 낮을수록 하나의 path에 몰립니다. Oat-Zero는 correct trace에서 가장 낮고, collapsed chunk 비율은 가장 높습니다.",
        },
        {
            "title": "Slide 5: Circle Visualization",
            "layout": "One large figure with a two-line caption.",
            "on_slide": [
                "Each circle summarizes chunk-level perturbation behavior",
                "Darker / stronger marks indicate larger Thought Anchor effects",
                "Use this as the visual bridge from aggregate metrics to local thought chunks",
            ],
            "visual": "Insert `analysis/qwen_anchor/visualizations/mmlu_model_tendency_circles.svg`.",
            "speaker": "이 그림은 논문 웹사이트 스타일의 circle view를 우리 결과에 맞춰 만든 것입니다. 표가 aggregate claim이라면, 이 그림은 thought chunk 단위로 어디서 효과가 생기는지 보여주는 bridge 역할을 합니다.",
        },
        {
            "title": "Slide 6: Qualitative Example Setup",
            "layout": "Problem card on left, mini audit table on right.",
            "on_slide": [
                "Same MMLU philosophy question for all three models",
                "Correct answer: D. all of the above",
                "All three base traces originally answer D correctly",
                "Selected because it has the largest Base-vs-Oat-Zero path-diversity gap in the current correct-trace slice",
            ],
            "visual": "Paste the problem text and the small audit table from `Same-question Internal Motif Comparison`.",
            "speaker": "이 예시는 임의 샘플이 아닙니다. 같은 문제에서 Base와 Oat-Zero의 path diversity 차이가 가장 크게 보이는 대표 사례라서 선택했습니다.",
        },
        {
            "title": "Slide 7: Qualitative Result",
            "layout": "Three model columns: Base, Instruct, Oat-Zero.",
            "on_slide": [
                "Base: A:46, C:25, D:21, B:8; effective paths 3.43",
                "Instruct: A:91, D:8, B:1; effective paths 1.40",
                "Oat-Zero: A:100; effective paths 1.00",
                "Interpretation: the same conceptual perturbation moves from multi-branch reasoning to a single repeated A-only rationale.",
            ],
            "visual": "Paste the three `Path split after removal` blocks from the qualitative section.",
            "speaker": "여기가 발표의 핵심 qualitative slide입니다. Base는 여러 내부 해석이 살아 있습니다. Instruct는 하나의 dominant story로 좁아지고, Oat-Zero는 100개 continuation 전부가 같은 A-only reasoning으로 collapse됩니다.",
        },
        {
            "title": "Slide 8: What Thought Anchors Add",
            "layout": "Two small tables or two side-by-side callouts.",
            "on_slide": [
                "Diversity asks: how many reasoning paths remain?",
                "Anchor outcomes ask: what does removing this thought do to correctness?",
                "Correct traces: Oat-Zero is often robust, but highly path-collapsed",
                "Incorrect traces: some bad local thoughts can be removed to repair the answer",
            ],
            "visual": "Paste the two `Thought-Anchor Outcome Tables`, or simplify to the `Positive anchors` and `Misleading anchors / repairs` columns.",
            "speaker": "이 슬라이드는 diversity와 anchor effect를 구분하기 위한 것입니다. 낮은 diversity가 항상 낮은 accuracy를 뜻하지는 않습니다. Thought Anchors는 어떤 thought가 답을 깨는지, 또는 오답을 고치는지를 추가로 보여줍니다.",
        },
        {
            "title": "Slide 9: MATH Appendix",
            "layout": "Appendix status slide, not a main result slide.",
            "on_slide": [
                "MATH is not yet fair-complete",
                "Use as partial appendix evidence only",
                "Completed chunks show formal bottlenecks and repairable algebraic errors",
                "Do not use current MATH numbers for final model ranking",
            ],
            "visual": "Optional: insert `analysis/qwen_anchor/visualizations/math_partial_tendency_cards.svg` or one MATH appendix case card.",
            "speaker": "MATH는 아직 incomplete라서 본문 결론에 넣으면 위험합니다. 다만 algebraic bottleneck이 thought anchor로 잡히는 예시는 appendix로 보여줄 수 있습니다.",
        },
        {
            "title": "Slide 10: Conclusion",
            "layout": "Three takeaways plus next step.",
            "on_slide": [
                "Thought Anchors let us test reasoning-path stability, not just final-answer accuracy",
                "On the current matched MMLU slice, Oat-Zero correct traces show the strongest path collapse",
                "The representative example shows collapse inside the generated reasoning, not only in final answers",
                "Next: finish MATH, rerun matched analysis, and optionally add white-box mechanism analysis",
            ],
            "visual": "Use no dense table; repeat the one-line quantitative result `Oat-Zero: effective paths 1.24, collapsed chunks 85%`.",
            "speaker": "마무리는 신중하게 해야 합니다. 우리가 보인 것은 현재 matched MMLU slice에서의 강한 black-box evidence입니다. claim은 reasoning path diversity가 RL-style 모델에서 낮아지는 경향이지, 모든 상황에서 항상 그렇다는 결론은 아닙니다.",
        },
    ]

    for slide in slides:
        lines.append(f"### {slide['title']}")
        lines.append(f"**Layout.** {slide['layout']}")
        lines.append("")
        lines.append("**On-slide copy.**")
        for item in slide["on_slide"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append(f"**Visual / table.** {slide['visual']}")
        lines.append("")
        lines.append(f"**Speaker note.** {slide['speaker']}")
        lines.append("")

    return lines


def render_ppt_markdown(qual: Dict[str, Any], summary_report: Dict[str, Any]) -> str:
    stats = collect_stats(qual)
    diversity_stats = collect_answer_diversity_stats(qual)
    index = by_model_base(qual)
    lines: List[str] = []
    lines.append("# PPT Snapshot: Qwen Thought Anchors")
    lines.append("")
    lines.append("## One-slide thesis")
    lines.append("")
    lines.append(
        "- 발표의 핵심 메시지: RL-style 모델인 Oat-Zero는 perturbation 이후 reasoning path diversity가 더 낮게 보입니다. "
        "즉, 같은 base trace에서 한 thought chunk를 제거하고 100번 다시 굴려도 여러 답/경로로 탐색하기보다 한 dominant trajectory로 몰리는 경향이 큽니다."
    )
    lines.append(
        "- Thought Anchors 방법론은 이 주장을 보여주기에 적합합니다: 동일 base trace에서 thought chunk 하나를 제거하고 100개 default continuation을 다시 샘플링해 "
        "`그 thought가 답을 바꾸는가`와 `남은 reasoning이 얼마나 다양한 경로로 갈라지는가`를 봅니다."
    )
    lines.append(
        "- 단, 논문 기준의 메인 black-box method는 forced-answer가 아니라 sentence resampling/counterfactual importance입니다. "
        "forced-answer는 기존 방법과의 비교 및 보조 진단으로만 사용합니다."
    )
    lines.append(
        "- MATH는 아직 fair-complete 상태가 아니므로 본문 결론에는 넣지 말고, "
        "PPT appendix에서 partial evidence로만 보여주는 편이 안전합니다."
    )
    lines.append("")
    lines.append("## Evidence tier for PPT")
    lines.append("")
    lines.append("- Main claim: MMLU matched slice에서 Oat-Zero correct traces의 post-ablation answer/path collapse가 가장 강합니다.")
    lines.append("- Mechanism evidence: Thought Anchor ablations show whether one removed thought makes a correct trace brittle, repairs an incorrect trace, or leaves several alternative paths alive.")
    lines.append("- Qualitative evidence: use the same-question philosophy example below because it includes the problem, correct answer, all model base answers, removed thought, 100-rollout split, and representative internal reasoning.")
    lines.append("- Appendix: MATH partial results are useful for showing algebraic/formal examples, but not final cross-model evidence.")
    lines.append("- Do not claim: white-box attention/activation conclusions, final MATH model ranking, or benchmark-wide generalization beyond this matched slice.")
    lines.append("")
    lines.append("## Current data status")
    lines.append("")
    lines.append(
        "- MMLU is usable for a current PPT snapshot at the full-paired chunk level: every number below uses chunks with both default and forced diagnostic rollout files available, and the diversity claim uses the default 100 resamples."
    )
    lines.append(
        "- Important caveat: this is not an exhaustive every-chunk/every-problem completion report. Several problems still have some incomplete chunks, so phrase the result as `current matched MMLU chunk slice`, not as a final benchmark-wide theorem."
    )
    coverage_table = [["Model", "Trace type", "Full-paired non-answer chunks"]]
    for row in stats:
        coverage_table.append(
            [
                MODEL_LABELS.get(row["model"], row["model"]),
                row["base_type"],
                f"{row['full']}/{row['total']}",
            ]
        )
    lines.append(markdown_table(coverage_table))
    lines.append("")
    lines.append("## Primary Table: Reasoning Path Diversity")
    lines.append("")
    lines.extend(diversity_glossary_lines())
    table = [["Model", "Trace type", "Chunks", "Effective answers", "Dominant answer share", "Answer-collapsed chunks", "Single-answer chunks"]]
    for row in diversity_stats:
        table.append(
            [
                MODEL_LABELS.get(row["model"], row["model"]),
                row["base_type"],
                str(row["chunks"]),
                f2(row["mean_effective"]),
                f"{row['mean_top_share']:.0%}",
                f"{row['collapse']} ({pct(row['collapse'], row['chunks'])})",
                f"{row['single_answer']} ({pct(row['single_answer'], row['chunks'])})",
            ]
        )
    lines.append(markdown_table(table))
    lines.append("")
    lines.append("Practical reading: the `correct` rows best support the presentation thesis. Oat-Zero has the lowest effective answer count (`1.24`) and the highest collapse rate (`76/89 = 85%`), compared with Base (`1.34`, `50/70 = 71%`) and Instruct (`1.40`, `52/78 = 67%`).")
    lines.append("")
    lines.append("## Same-question Internal Motif Comparison")
    lines.append("")
    lines.append("This is the qualitative example to use in the main talk. It uses the same MMLU philosophy question for all three models, was selected because it maximizes the Base-vs-Oat-Zero path-diversity contrast in the current correct-trace slice, and inspects the rollout text rather than only final answers.")
    lines.append("")
    lines.extend(philosophy_motif_lines(index))
    lines.append("")
    lines.extend(mmlu_anchor_outcome_lines(stats))
    lines.extend(render_copy_paste_slide_deck())
    lines.extend(render_slide_ready_examples(index))
    lines.append("## Current MATH framing")
    lines.append("")
    math = next((d for d in summary_report.get("datasets", []) if d.get("benchmark") == "MATH"), None)
    if math:
        lines.append("- MATH selected shared-base pipeline has partial completed chunks, but not enough for fair final comparison.")
        for model, model_summary in math.get("models", {}).items():
            correct = model_summary.get("correct", {})
            incorrect = model_summary.get("incorrect", {})
            lines.append(
                f"- {MODEL_LABELS.get(model, model)}: correct complete problems "
                f"`{correct.get('num_complete_problems', 0)}/{correct.get('num_problems', 0)}`, "
                f"incorrect complete problems `{incorrect.get('num_complete_problems', 0)}/{incorrect.get('num_problems', 0)}`."
            )
    math_partial_rows = collect_math_partial_stats()
    if math_partial_rows:
        lines.append("- Chunk-level partial MATH summary is now available; use it only as appendix/early evidence.")
        lines.append("")
        lines.extend(math_partial_outcome_lines(math_partial_rows))
        lines.append("- Suggested appendix visual: `analysis/qwen_anchor/visualizations/math_partial_tendency_cards.svg`.")
        lines.append("- Suggested appendix report: `analysis/qwen_anchor/math_partial_summary.md`.")
    lines.append("")
    lines.extend(render_math_case_studies())
    lines.append("")
    lines.append("## Files generated")
    lines.append("")
    lines.append("- `analysis/qwen_anchor/ppt_current_snapshot.md`")
    lines.append("- `analysis/qwen_anchor/visualizations/mmlu_model_tendency_cards.svg`")
    lines.append("- `analysis/qwen_anchor/visualizations/mmlu_model_tendency_slope.svg`")
    lines.append("- `analysis/qwen_anchor/visualizations/mmlu_model_tendency_compass.svg`")
    lines.append("- `analysis/qwen_anchor/visualizations/mmlu_model_tendency_circles.svg`")
    lines.append("- `analysis/qwen_anchor/visualizations/mmlu_thought_anchor_circles.svg`")
    lines.append("- `analysis/qwen_anchor/visualizations/mmlu_case_study_circles.svg`")
    lines.append("- `analysis/qwen_anchor/visualizations/math_partial_tendency_cards.svg`")
    lines.append("- `analysis/qwen_anchor/visualizations/mmlu_positive_anchor_philosophy64_circles.svg`")
    lines.append("- `analysis/qwen_anchor/visualizations/mmlu_misleading_anchor_statistics8_circles.svg`")
    lines.append("- `analysis/qwen_anchor/math_partial_summary.md`")
    lines.append("- These SVGs are paper-style static replicas of the website circle view.")
    lines.append("- PNG versions are also written automatically for the older summary charts if `matplotlib` is available.")
    return "\n".join(lines) + "\n"


def try_import_pyplot():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D

        return plt, Circle, Line2D
    except Exception:
        return None, None, None


def node_size(strength: float) -> float:
    return 22 + 360 * max(0.0, min(1.0, strength)) ** 1.35


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    anchor: str = "middle",
    fill: str = "#2f2a25",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(text)}</text>'
    )


def svg_page(width: int, height: int, elements: List[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fbfaf7"/>',
            *elements,
            "</svg>",
        ]
    )


def write_svg(path: Path, width: int, height: int, elements: List[str]) -> None:
    path.write_text(svg_page(width, height, elements), encoding="utf-8")


def svg_circle(
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str = "none",
    stroke: str = "#756f67",
    stroke_width: float = 1.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}" '
        f'opacity="{opacity:.3f}"/>'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = "#756f67",
    stroke_width: float = 1.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" opacity="{opacity:.3f}"/>'
    )


def draw_overview_circle_map_svg(qual: Dict[str, Any]) -> None:
    width, height = 1600, 1920
    panel_w, panel_h = 700, 540
    lefts = [420, 1180]
    tops = [330, 870, 1410]
    elements: List[str] = [
        svg_text(width / 2, 70, "MMLU Thought Anchor Circle Map", size=34, weight="700"),
        svg_text(
            width / 2,
            106,
            "Each dot is one full-paired reasoning chunk. Size/opacity = anchor strength.",
            size=17,
            fill="#4f4a45",
        ),
        svg_text(
            width / 2,
            132,
            "Red: removing a correct-trace chunk hurts. Blue: removing a wrong-trace chunk repairs.",
            size=17,
            fill="#4f4a45",
        ),
    ]
    index = by_model_base(qual)
    for row_idx, model in enumerate(MODEL_LABELS):
        for col_idx, base_type in enumerate(["correct", "incorrect"]):
            cx, cy = lefts[col_idx], tops[row_idx]
            summary = index[(model, base_type)]
            chunks = full_chunks(summary)
            color = TYPE_COLORS[base_type]
            for radius, opacity in [(205, 0.35), (150, 0.18), (95, 0.12)]:
                elements.append(svg_circle(cx, cy, radius, stroke_width=2.0 if radius == 205 else 1.2, opacity=opacity))

            chunks_by_problem: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for chunk in chunks:
                chunks_by_problem[chunk["problem_name"]].append(chunk)
            problems = sorted(chunks_by_problem)
            total_slots = sum(len(chunks_by_problem[p]) + 1 for p in problems)
            cursor = 0
            labeled = set()
            top_labels = {
                (c["problem_name"], c["chunk_idx"])
                for c in sorted_top_chunks(chunks, limit=3, exclude_answer_tokens=True)
            }

            for problem in problems:
                problem_chunks = sorted(chunks_by_problem[problem], key=lambda c: c["chunk_idx"])
                start = cursor
                for local_idx, chunk in enumerate(problem_chunks):
                    cursor += 1
                    theta = (cursor / max(1, total_slots)) * 2 * math.pi - math.pi / 2
                    local_fraction = local_idx / max(1, len(problem_chunks) - 1)
                    radius = 95 + 90 * local_fraction
                    x = cx + radius * math.cos(theta)
                    y = cy + radius * math.sin(theta)
                    strength = chunk_strength(chunk)
                    r = 3.4 + 12.0 * max(0.0, min(1.0, strength)) ** 1.2
                    opacity = alpha_for_strength(strength)
                    elements.append(
                        svg_circle(
                            x,
                            y,
                            r,
                            fill=color,
                            stroke="#fffdf7",
                            stroke_width=1.0,
                            opacity=opacity,
                        )
                    )
                    if (problem, chunk["chunk_idx"]) in top_labels and problem not in labeled:
                        lx = cx + 232 * math.cos(theta)
                        ly = cy + 232 * math.sin(theta)
                        anchor = "start" if lx >= cx else "end"
                        label = f"{compact_problem_name(problem)} c{chunk['chunk_idx']}"
                        elements.append(svg_line(x, y, cx + 212 * math.cos(theta), cy + 212 * math.sin(theta), stroke=color, opacity=0.45))
                        elements.append(svg_text(lx, ly, label, size=13, anchor=anchor, fill="#2f2a25"))
                        labeled.add(problem)
                cursor += 1
                if problem_chunks:
                    theta_mid = ((start + len(problem_chunks) / 2) / max(1, total_slots)) * 2 * math.pi - math.pi / 2
                    elements.append(
                        svg_line(
                            cx + 205 * math.cos(theta_mid),
                            cy + 205 * math.sin(theta_mid),
                            cx + 215 * math.cos(theta_mid),
                            cy + 215 * math.sin(theta_mid),
                            opacity=0.18,
                        )
                    )

            if base_type == "correct":
                signal = sum(1 for c in chunks if c["default_accuracy"] <= 0.5)
                descriptor = "brittle anchors"
            else:
                signal = sum(1 for c in chunks if c["default_accuracy"] >= 0.5)
                descriptor = "repair anchors"
            elements.append(svg_text(cx, cy - 4, MODEL_LABELS[model], size=27, weight="700", fill=MODEL_COLORS[model]))
            elements.append(svg_text(cx, cy + 28, f"{base_type} base", size=15, fill="#4f4a45"))
            elements.append(svg_text(cx, cy + 52, f"{signal}/{len(chunks)} {descriptor}", size=15, fill="#4f4a45"))

    elements.extend(
        [
            svg_circle(620, 1840, 10, fill=TYPE_COLORS["correct"], stroke="#fffdf7"),
            svg_text(645, 1845, "Correct base: removal hurts", size=16, anchor="start"),
            svg_circle(940, 1840, 10, fill=TYPE_COLORS["incorrect"], stroke="#fffdf7"),
            svg_text(965, 1845, "Incorrect base: removal repairs", size=16, anchor="start"),
        ]
    )
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    write_svg(VIS_DIR / "mmlu_thought_anchor_circles.svg", width, height, elements)


def draw_trace_circle_svg(
    qual: Dict[str, Any],
    problem_name: str,
    base_type: str,
    output_stem: str,
    title: str,
) -> None:
    by_model = trace_chunks_for_problem(qual, problem_name, base_type)
    if not by_model:
        return
    width, height = 1600, 620
    centers = [(290, 330), (800, 330), (1310, 330)]
    color = TYPE_COLORS[base_type]
    elements: List[str] = [
        svg_text(width / 2, 64, title, size=30, weight="700"),
        svg_text(
            width / 2,
            100,
            "Nodes are chunks in reasoning order. Larger nodes mark stronger Thought Anchor effects.",
            size=16,
            fill="#4f4a45",
        ),
    ]
    for (cx, cy), model in zip(centers, MODEL_LABELS):
        chunks = by_model.get(model, [])
        elements.append(svg_circle(cx, cy, 168, stroke_width=2.0, opacity=0.35))
        if not chunks:
            elements.append(svg_text(cx, cy, f"{MODEL_LABELS[model]}\nno full-paired chunks", size=15))
            continue
        n = len(chunks)
        top_ids = {
            (chunk["problem_name"], chunk["chunk_idx"])
            for chunk in sorted(chunks, key=chunk_strength, reverse=True)[:3]
        }
        points: List[Tuple[float, float]] = []
        for i, chunk in enumerate(chunks):
            theta = (i / max(1, n)) * 2 * math.pi - math.pi / 2
            x = cx + 168 * math.cos(theta)
            y = cy + 168 * math.sin(theta)
            points.append((x, y))
            strength = chunk_strength(chunk)
            r = 5 + 16 * max(0.0, min(1.0, strength)) ** 1.2
            elements.append(svg_circle(x, y, r, fill=color, stroke="#fffdf7", stroke_width=1.2, opacity=alpha_for_strength(strength)))
            elements.append(svg_text(cx + 120 * math.cos(theta), cy + 120 * math.sin(theta) + 4, str(chunk["chunk_idx"]), size=11, fill="#4f4a45"))
            if (chunk["problem_name"], chunk["chunk_idx"]) in top_ids:
                lx = cx + 218 * math.cos(theta)
                ly = cy + 218 * math.sin(theta)
                anchor = "start" if lx >= cx else "end"
                elements.append(svg_text(lx, ly, f"c{chunk['chunk_idx']} acc {f2(chunk['default_accuracy'])}", size=13, anchor=anchor))
        for first, second in zip(points, points[1:] + points[:1]):
            elements.append(svg_line(first[0], first[1], second[0], second[1], opacity=0.16))
        elements.append(svg_text(cx, cy - 4, MODEL_LABELS[model], size=26, weight="700", fill=MODEL_COLORS[model]))
        elements.append(svg_text(cx, cy + 28, f"{len(chunks)} chunks", size=15, fill="#4f4a45"))
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    write_svg(VIS_DIR / f"{output_stem}.svg", width, height, elements)


def alpha_for_strength(strength: float) -> float:
    return 0.18 + 0.78 * max(0.0, min(1.0, strength))


def draw_overview_circle_map(qual: Dict[str, Any]) -> None:
    plt, Circle, Line2D = try_import_pyplot()
    if plt is None:
        draw_overview_circle_map_svg(qual)
        return

    summaries = qual.get("summaries", [])
    fig, axes = plt.subplots(3, 2, figsize=(15.8, 19.0), facecolor="#fbfaf7")
    fig.suptitle(
        "MMLU Thought Anchor Circle Map (full-paired chunks only)",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )
    subtitle = (
        "Each dot is one reasoning chunk. Size/opacity = anchor strength. "
        "Red: correct trace breaks when removed. Blue: wrong trace repairs when removed."
    )
    fig.text(0.5, 0.958, subtitle, ha="center", va="center", fontsize=12, color="#4f4a45")

    index = by_model_base(qual)
    for row_idx, model in enumerate(MODEL_LABELS):
        for col_idx, base_type in enumerate(["correct", "incorrect"]):
            ax = axes[row_idx][col_idx]
            summary = index[(model, base_type)]
            chunks = full_chunks(summary)
            color = TYPE_COLORS[base_type]
            ax.set_aspect("equal")
            ax.set_facecolor("#fbfaf7")
            ax.axis("off")

            for radius, lw, alpha in [(1.0, 1.3, 0.35), (0.72, 0.8, 0.18), (0.44, 0.8, 0.12)]:
                ax.add_patch(Circle((0, 0), radius, fill=False, lw=lw, ec="#756f67", alpha=alpha))

            chunks_by_problem: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for chunk in chunks:
                chunks_by_problem[chunk["problem_name"]].append(chunk)
            problems = sorted(chunks_by_problem)
            total_slots = sum(len(chunks_by_problem[p]) + 1 for p in problems)
            cursor = 0
            labeled = set()
            top_labels = {
                (c["problem_name"], c["chunk_idx"])
                for c in sorted_top_chunks(chunks, limit=3, exclude_answer_tokens=True)
            }

            for problem in problems:
                problem_chunks = sorted(chunks_by_problem[problem], key=lambda c: c["chunk_idx"])
                problem_start = cursor
                for local_idx, chunk in enumerate(problem_chunks):
                    cursor += 1
                    theta = (cursor / total_slots) * 2 * math.pi - math.pi / 2
                    # Slightly move later chunks outward; this preserves reasoning order
                    # while avoiding a perfectly uniform clock-face look.
                    local_fraction = local_idx / max(1, len(problem_chunks) - 1)
                    radius = 0.45 + 0.45 * local_fraction
                    x = radius * math.cos(theta)
                    y = radius * math.sin(theta)
                    strength = chunk_strength(chunk)
                    ax.scatter(
                        [x],
                        [y],
                        s=node_size(strength),
                        color=color,
                        alpha=alpha_for_strength(strength),
                        edgecolors="#fffdf7",
                        linewidths=0.7,
                        zorder=3,
                    )
                    if (problem, chunk["chunk_idx"]) in top_labels and problem not in labeled:
                        lx = 1.12 * math.cos(theta)
                        ly = 1.12 * math.sin(theta)
                        label = f"{compact_problem_name(problem)} c{chunk['chunk_idx']}"
                        ax.plot([x, lx * 0.93], [y, ly * 0.93], color=color, alpha=0.45, lw=0.8)
                        ax.text(
                            lx,
                            ly,
                            label,
                            ha="left" if lx >= 0 else "right",
                            va="center",
                            fontsize=8.5,
                            color="#2f2a25",
                        )
                        labeled.add(problem)
                cursor += 1
                if problem_chunks:
                    theta_mid = ((problem_start + len(problem_chunks) / 2) / total_slots) * 2 * math.pi - math.pi / 2
                    ax.plot(
                        [0.98 * math.cos(theta_mid), 1.03 * math.cos(theta_mid)],
                        [0.98 * math.sin(theta_mid), 1.03 * math.sin(theta_mid)],
                        color="#756f67",
                        alpha=0.18,
                        lw=1,
                    )

            if base_type == "correct":
                signal = sum(1 for c in chunks if c["default_accuracy"] <= 0.5)
                descriptor = "brittle anchors"
            else:
                signal = sum(1 for c in chunks if c["default_accuracy"] >= 0.5)
                descriptor = "repair anchors"
            ax.text(
                0,
                0.03,
                MODEL_LABELS[model],
                ha="center",
                va="center",
                fontsize=17,
                color=MODEL_COLORS[model],
                fontweight="bold",
            )
            ax.text(
                0,
                -0.13,
                f"{base_type} base\n{signal}/{len(chunks)} {descriptor}",
                ha="center",
                va="center",
                fontsize=10.5,
                color="#4f4a45",
            )
            ax.set_xlim(-1.32, 1.32)
            ax.set_ylim(-1.32, 1.32)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLORS["correct"], markersize=9, label="Correct base: removal hurts"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLORS["incorrect"], markersize=9, label="Incorrect base: removal repairs"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.012))
    fig.tight_layout(rect=(0.03, 0.035, 0.97, 0.94))
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        fig.savefig(VIS_DIR / f"mmlu_thought_anchor_circles.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def trace_chunks_for_problem(
    qual: Dict[str, Any],
    problem_name: str,
    base_type: str,
) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for summary in qual.get("summaries", []):
        if summary["base_type"] != base_type:
            continue
        chunks = [
            chunk
            for chunk in full_chunks(summary)
            if chunk["problem_name"] == problem_name
        ]
        if chunks:
            result[summary["model"]] = sorted(chunks, key=lambda c: c["chunk_idx"])
    return result


def draw_trace_circle(
    qual: Dict[str, Any],
    problem_name: str,
    base_type: str,
    output_stem: str,
    title: str,
) -> None:
    plt, Circle, Line2D = try_import_pyplot()
    if plt is None:
        draw_trace_circle_svg(qual, problem_name, base_type, output_stem, title)
        return

    by_model = trace_chunks_for_problem(qual, problem_name, base_type)
    if not by_model:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), facecolor="#fbfaf7")
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.93,
        "Nodes are chunks in reasoning order. Larger nodes mark stronger Thought Anchor effects.",
        ha="center",
        va="center",
        fontsize=11,
        color="#4f4a45",
    )
    color = TYPE_COLORS[base_type]
    for ax, model in zip(axes, MODEL_LABELS):
        chunks = by_model.get(model, [])
        ax.set_aspect("equal")
        ax.set_facecolor("#fbfaf7")
        ax.axis("off")
        ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.2, ec="#756f67", alpha=0.35))
        if not chunks:
            ax.text(0, 0, f"{MODEL_LABELS[model]}\n(no full-paired chunks)", ha="center", va="center")
            continue

        n = len(chunks)
        top = sorted(chunks, key=chunk_strength, reverse=True)[:3]
        top_ids = {(c["problem_name"], c["chunk_idx"]) for c in top}
        points = []
        for i, chunk in enumerate(chunks):
            theta = (i / n) * 2 * math.pi - math.pi / 2
            x, y = math.cos(theta), math.sin(theta)
            strength = chunk_strength(chunk)
            points.append((x, y))
            ax.scatter(
                [x],
                [y],
                s=node_size(strength) * 1.2,
                color=color,
                alpha=alpha_for_strength(strength),
                edgecolors="#fffdf7",
                linewidths=0.9,
                zorder=3,
            )
            ax.text(x * 0.76, y * 0.76, str(chunk["chunk_idx"]), ha="center", va="center", fontsize=8, color="#4f4a45")
            if (chunk["problem_name"], chunk["chunk_idx"]) in top_ids:
                ax.text(
                    x * 1.2,
                    y * 1.2,
                    f"c{chunk['chunk_idx']}\nacc {f2(chunk['default_accuracy'])}",
                    ha="left" if x >= 0 else "right",
                    va="center",
                    fontsize=9,
                    color="#2f2a25",
                )
        for a, b in zip(points, points[1:] + points[:1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#756f67", alpha=0.16, lw=0.8, zorder=1)
        ax.text(0, 0.07, MODEL_LABELS[model], ha="center", va="center", fontsize=16, color=MODEL_COLORS[model], fontweight="bold")
        ax.text(0, -0.1, f"{len(chunks)} chunks", ha="center", va="center", fontsize=10.5, color="#4f4a45")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.9))
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        fig.savefig(VIS_DIR / f"{output_stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    qual = load_json(QUAL_PATH)
    summary = load_json(SUMMARY_PATH)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_ppt_markdown(qual, summary), encoding="utf-8")
    draw_overview_circle_map(qual)
    draw_trace_circle(
        qual,
        problem_name="problem_philosophy_64",
        base_type="correct",
        output_stem="mmlu_philosophy64_trace_circles",
        title="Correct-base anchor case: Philosophy question 64",
    )
    draw_trace_circle(
        qual,
        problem_name="problem_clinical_knowledge_36",
        base_type="incorrect",
        output_stem="mmlu_clinical36_repair_circles",
        title="Incorrect-base repair case: Clinical knowledge question 36",
    )
    # Keep the public-facing circle SVGs faithful to the Thought Anchors website
    # circle view. This call intentionally overwrites the fallback summary
    # circles above with paper-style static replicas.
    try:
        import render_ppt_trend_figures

        render_ppt_trend_figures.main()
    except Exception as exc:
        print(f"Warning: PPT trend figure rendering failed: {exc}")
    try:
        import render_paper_style_anchor_viz

        render_paper_style_anchor_viz.main()
    except Exception as exc:
        print(f"Warning: paper-style circle rendering failed: {exc}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote circle visualizations under {VIS_DIR}")


if __name__ == "__main__":
    main()
