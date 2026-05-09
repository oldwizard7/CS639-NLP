#!/usr/bin/env python3
"""Create a PPT-friendly partial MATH Thought Anchors summary.

The MATH rollout set is not complete yet, so this report deliberately uses
chunk-level partial evidence and labels every claim as provisional.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


INPUT = Path("analysis/qwen_anchor/math_partial/qualitative_report.json")
OUT_MD = Path("analysis/qwen_anchor/math_partial_summary.md")
VIS_DIR = Path("analysis/qwen_anchor/visualizations")
OUT_SVG = VIS_DIR / "math_partial_tendency_cards.svg"

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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clean_text(value: str, limit: int = 150) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def f2(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}"


def pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "NA"
    return f"{num / denom:.0%}"


def full_chunks(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        chunk
        for chunk in summary.get("chunks", [])
        if chunk.get("full_paired")
        and chunk.get("default_accuracy") is not None
        and chunk.get("forced_accuracy") is not None
    ]


def summary_stats(summary: Dict[str, Any]) -> Dict[str, Any]:
    chunks = full_chunks(summary)
    base_type = summary["base_type"]
    if base_type == "correct":
        robust = sum(1 for chunk in chunks if chunk["default_accuracy"] >= 0.9)
        brittle = sum(1 for chunk in chunks if chunk["default_accuracy"] <= 0.5)
        forced_brittle = sum(1 for chunk in chunks if chunk["forced_accuracy"] <= 0.5)
        main = brittle
        secondary = robust
        forced = forced_brittle
    else:
        repair = sum(1 for chunk in chunks if chunk["default_accuracy"] >= 0.5)
        stuck = sum(1 for chunk in chunks if chunk["default_accuracy"] <= 0.1)
        forced_repair = sum(1 for chunk in chunks if chunk["forced_accuracy"] >= 0.5)
        main = repair
        secondary = stuck
        forced = forced_repair
    return {
        "full": len(chunks),
        "total": summary.get("num_chunks", 0),
        "coverage": len(chunks) / summary.get("num_chunks", 1) if summary.get("num_chunks") else 0.0,
        "mean_default": summary.get("mean_default_accuracy"),
        "mean_forced": summary.get("mean_forced_accuracy"),
        "main": main,
        "secondary": secondary,
        "forced": forced,
    }


def top_chunks(summary: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    chunks = full_chunks(summary)
    candidates = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_type") not in {"final-answer commitment", "answer-option token"}
        and not low_ppt_value_chunk(chunk.get("chunk", ""))
    ]
    if not candidates:
        candidates = [
            chunk
            for chunk in chunks
            if chunk.get("chunk_type") not in {"final-answer commitment", "answer-option token"}
        ]
    return sorted(
        candidates,
        key=lambda item: (
            item.get("anchor_strength") or 0.0,
            item.get("forced_strength") or 0.0,
        ),
        reverse=True,
    )[:limit]


def low_ppt_value_chunk(text: str) -> bool:
    low = text.lower().strip()
    return any(
        marker in low
        for marker in [
            "```",
            "print(",
            "math.",
            "valid_arrangements",
            "adjacent_arrangements",
            "</th>",
            "<table",
            "don't need to go into that detail",
        ]
    ) or low.startswith("# ")


def problem_level_rows(summary: Dict[str, Any]) -> List[Tuple[str, int, float, int, int]]:
    by_problem: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in full_chunks(summary):
        by_problem.setdefault(chunk["problem_name"], []).append(chunk)

    rows: List[Tuple[str, int, float, int, int]] = []
    for problem_name, chunks in sorted(by_problem.items()):
        mean_default = sum(chunk["default_accuracy"] for chunk in chunks) / len(chunks)
        if summary["base_type"] == "correct":
            signal = sum(chunk["default_accuracy"] <= 0.5 for chunk in chunks)
            secondary = sum(chunk["forced_accuracy"] <= 0.5 for chunk in chunks)
        else:
            signal = sum(chunk["default_accuracy"] >= 0.5 for chunk in chunks)
            secondary = sum(chunk["default_accuracy"] <= 0.1 for chunk in chunks)
        rows.append((problem_name, len(chunks), mean_default, signal, secondary))
    return rows


def markdown_table(rows: List[List[str]]) -> str:
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    lines = []
    lines.append("| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(rows[0])) + " |")
    lines.append("| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |")
    return "\n".join(lines)


def metric_glossary_lines() -> List[str]:
    return [
        "### How to read this table",
        "",
        "- `Full chunks` means chunks with 100 valid default rollouts and 100 valid forced-answer rollouts after one thought chunk is removed.",
        "- `Mean def` is the mean default accuracy after chunk removal across those completed chunks.",
        "- `Mean forced` is the mean forced-answer accuracy after chunk removal; this is a stricter continuation check.",
        "- In `correct` rows, `brittle <=50%` means removing the chunk often breaks a previously correct trace; `robust >=90%` means the trace usually survives without it.",
        "- In `incorrect` rows, `repair >=50%` means removing the chunk often repairs a previously wrong trace; `stuck <=10%` means removal still leaves the model near-zero.",
        "",
    ]


def render_markdown(payload: Dict[str, Any]) -> str:
    summaries = payload["summaries"]
    lines: List[str] = [
        "# Partial MATH Thought Anchors Summary",
        "",
        "Status: provisional. This uses only full-paired chunks already present locally, with 100 valid default rollouts and 100 valid forced-answer rollouts.",
        "",
        "## One-slide Takeaway",
        "",
        "- Coverage is still uneven, so this should be an appendix or early-evidence slide, not the main final comparison.",
        "- Correct-base traces show the same basic pattern as MMLU: many chunks remain robust, while a smaller set of formal steps act as positive anchors.",
        "- Incorrect-base traces are currently most repairable for Oat-Zero, much less repairable for Base and Instruct in the completed slice.",
        "- Many MATH anchors are calculation/formal-step anchors, unlike MMLU where option comparison and factual/negation bridges were more prominent.",
        "",
        "## Model-level Partial Metrics",
        "",
    ]

    lines.extend(metric_glossary_lines())
    table = [["Model", "Base", "Full chunks", "Mean def", "Mean forced", "Main signal", "Secondary", "Forced signal"]]
    for summary in summaries:
        stats = summary_stats(summary)
        model_label = MODEL_LABELS.get(summary["model"], summary["model"])
        if summary["base_type"] == "correct":
            main = f"{stats['main']} ({pct(stats['main'], stats['full'])}) brittle <=50%"
            secondary = f"{stats['secondary']} ({pct(stats['secondary'], stats['full'])}) robust >=90%"
            forced = f"{stats['forced']} ({pct(stats['forced'], stats['full'])}) forced <=50%"
        else:
            main = f"{stats['main']} ({pct(stats['main'], stats['full'])}) repair >=50%"
            secondary = f"{stats['secondary']} ({pct(stats['secondary'], stats['full'])}) stuck <=10%"
            forced = f"{stats['forced']} ({pct(stats['forced'], stats['full'])}) forced repair >=50%"
        table.append(
            [
                model_label,
                summary["base_type"],
                f"{stats['full']}/{stats['total']}",
                f2(stats["mean_default"]),
                f2(stats["mean_forced"]),
                main,
                secondary,
                forced,
            ]
        )
    lines.append(markdown_table(table))
    lines.extend(["", "## What This Suggests", ""])
    lines.append("- Positive anchors in MATH are usually formal bottlenecks: root-set equivalence, recurrence periodicity, binary-length framing, or combinatorial subtraction.")
    lines.append("- Misleading anchors often come from locally plausible but globally wrong shortcuts, such as treating every hex digit as four bits without handling the leading digit, or locking onto a false recurrence period.")
    lines.append("- Oat-Zero currently has the strongest incorrect-trace repair signal: `35/61` completed incorrect chunks repair to at least 50% default accuracy, versus Base `2/28` and Instruct `8/62`.")
    lines.append("- Correct-trace robustness is highest in the completed Oat-Zero slice: `62/93` chunks stay at >=90% default accuracy after removal, but this is not final because coverage is uneven.")
    lines.extend(["", "## Best Current Case Studies", ""])

    for summary in summaries:
        label = MODEL_LABELS.get(summary["model"], summary["model"])
        lines.append(f"### {label} / {summary['base_type']}")
        for idx, chunk in enumerate(top_chunks(summary, limit=2), 1):
            direction = "breaks a correct trace" if summary["base_type"] == "correct" else "repairs a wrong trace"
            lines.append(
                f"{idx}. `{chunk['problem_name']}` chunk `{chunk['chunk_idx']}` ({chunk['chunk_type']}): removal {direction}; "
                f"default acc `{f2(chunk['default_accuracy'])}`, forced acc `{f2(chunk['forced_accuracy'])}`."
            )
            lines.append(f"   Question: {clean_text(chunk.get('question', ''), 260)}")
            lines.append(f"   Chunk: \"{clean_text(chunk['chunk'], 190)}\"")
            lines.append(f"   Correct answer `{chunk.get('gt_answer')}`; base answer `{chunk.get('base_answer')}`.")
        lines.append("")

    lines.extend(["## Problem-level Notes", ""])
    for summary in summaries:
        label = MODEL_LABELS.get(summary["model"], summary["model"])
        signal_name = "brittle" if summary["base_type"] == "correct" else "repair"
        secondary_name = "forced-brittle" if summary["base_type"] == "correct" else "stuck"
        rows = [["Problem", "Chunks", "Mean def", signal_name, secondary_name]]
        for problem_name, count, mean_default, signal, secondary in problem_level_rows(summary):
            rows.append([problem_name, str(count), f"{mean_default:.2f}", str(signal), str(secondary)])
        lines.append(f"### {label} / {summary['base_type']}")
        lines.append(markdown_table(rows))
        lines.append("")

    lines.extend(
        [
            "## Safe PPT Framing",
            "",
            "- Use this as `MATH partial evidence`, not as a final benchmark result.",
            "- The safe headline is: `MATH anchors are more algebraic/formal, and partial incorrect traces already show repairable misleading steps.`",
            "- Do not claim a final model ranking until the remaining MATH chunks finish and coverage is matched across models/base types.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def svg_text(x: float, y: float, value: str, *, size: int = 14, fill: str = "#332f2b", anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
        f"{escape(value)}</text>"
    )


def svg_rect(x: float, y: float, w: float, h: float, *, fill: str, stroke: str = "#ddd", rx: float = 12.0, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" ry="{rx:.1f}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.3f}"/>'
    )


def svg_circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = "#fff", sw: float = 2.0, opacity: float = 1.0) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity:.3f}"/>'
    )


def render_bar(x: float, y: float, w: float, h: float, value: float, color: str, label: str) -> List[str]:
    value = max(0.0, min(1.0, value))
    return [
        svg_rect(x, y, w, h, fill="#eee8df", stroke="#eee8df", rx=h / 2),
        svg_rect(x, y, w * value, h, fill=color, stroke=color, rx=h / 2),
        svg_text(x + w + 10, y + h - 3, label, size=12, fill="#5b544d"),
    ]


def render_svg(payload: Dict[str, Any]) -> str:
    summaries = payload["summaries"]
    index = {(item["model"], item["base_type"]): item for item in summaries}
    width, height = 1700, 980
    elements: List[str] = [
        svg_rect(0, 0, width, height, fill="#fbfaf7", stroke="#fbfaf7", rx=0),
        svg_text(width / 2, 54, "Partial MATH Thought Anchors", size=34, anchor="middle", weight="700"),
        svg_text(width / 2, 86, "Full-paired chunks completed so far; use as appendix/early evidence, not final model ranking.", size=16, anchor="middle", fill="#655e57"),
    ]

    elements.append(svg_rect(60, 116, 1580, 58, fill="#fff4df", stroke="#e7c275", rx=16))
    elements.append(svg_text(88, 152, "Caution: coverage is uneven. Base has far fewer completed MATH chunks than Oat-Zero.", size=18, fill="#7a4f00", weight="700"))

    card_w, card_h = 500, 330
    xs = [70, 600, 1130]
    ys = [210, 585]
    for col, model in enumerate(MODEL_LABELS):
        for row, base_type in enumerate(["correct", "incorrect"]):
            summary = index[(model, base_type)]
            stats = summary_stats(summary)
            x, y = xs[col], ys[row]
            color = MODEL_COLORS[model]
            accent = "#d45b4f" if base_type == "correct" else "#3778bf"
            elements.append(svg_rect(x, y, card_w, card_h, fill="#ffffff", stroke="#e4ddd5", rx=18))
            elements.append(svg_rect(x, y, card_w, 74, fill="#fffdf8", stroke="#ebe3d9", rx=18))
            elements.append(svg_text(x + 24, y + 32, MODEL_LABELS[model], size=25, fill=color, weight="700"))
            elements.append(svg_text(x + card_w - 24, y + 32, base_type, size=15, fill=accent, anchor="end", weight="700"))
            elements.append(svg_text(x + 24, y + 58, f"coverage {stats['full']}/{stats['total']} full-paired chunks", size=13, fill="#655e57"))

            if base_type == "correct":
                main_label = f"brittle {stats['main']}/{stats['full']} ({pct(stats['main'], stats['full'])})"
                secondary_label = f"robust {stats['secondary']}/{stats['full']} ({pct(stats['secondary'], stats['full'])})"
                forced_label = f"forced-brittle {stats['forced']}/{stats['full']} ({pct(stats['forced'], stats['full'])})"
                main_value = stats["main"] / stats["full"] if stats["full"] else 0.0
                secondary_value = stats["secondary"] / stats["full"] if stats["full"] else 0.0
                forced_value = stats["forced"] / stats["full"] if stats["full"] else 0.0
            else:
                main_label = f"repair {stats['main']}/{stats['full']} ({pct(stats['main'], stats['full'])})"
                secondary_label = f"stuck {stats['secondary']}/{stats['full']} ({pct(stats['secondary'], stats['full'])})"
                forced_label = f"forced-repair {stats['forced']}/{stats['full']} ({pct(stats['forced'], stats['full'])})"
                main_value = stats["main"] / stats["full"] if stats["full"] else 0.0
                secondary_value = stats["secondary"] / stats["full"] if stats["full"] else 0.0
                forced_value = stats["forced"] / stats["full"] if stats["full"] else 0.0

            elements.extend(render_bar(x + 28, y + 105, 270, 20, main_value, accent, main_label))
            elements.extend(render_bar(x + 28, y + 145, 270, 20, secondary_value, "#8c8177", secondary_label))
            elements.extend(render_bar(x + 28, y + 185, 270, 20, forced_value, "#f0ad3d", forced_label))
            elements.append(svg_text(x + 28, y + 240, f"mean default acc after removal: {f2(stats['mean_default'])}", size=14, fill="#3f3a35"))
            elements.append(svg_text(x + 28, y + 266, f"mean forced acc after removal: {f2(stats['mean_forced'])}", size=14, fill="#3f3a35"))

            dots = min(30, stats["full"])
            for idx in range(dots):
                cx = x + 330 + (idx % 10) * 14
                cy = y + 112 + (idx // 10) * 14
                fill = accent if idx < min(dots, stats["main"]) else "#d8d1c8"
                elements.append(svg_circle(cx, cy, 4.6, fill=fill, stroke="#fff", sw=1.0, opacity=0.9))
            elements.append(svg_text(x + 330, y + 174, "dots: completed chunks", size=11, fill="#7b736b"))

    elements.append(svg_text(850, 946, "Reading: red = correct trace breaks after removal; blue = wrong trace repairs after removal; gold = forced-answer sensitivity.", size=14, anchor="middle", fill="#5e574f"))
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            *elements,
            "</svg>",
        ]
    )


def main() -> None:
    payload = load_json(INPUT)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_SVG.write_text(render_svg(payload), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
