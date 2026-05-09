#!/usr/bin/env python3
"""Render paper-style Thought Anchors circle visualizations as static SVG.

This mirrors the public Thought Anchors website circle view:
- chunks are nodes arranged around a circle in step order
- sequential links are solid dark arrows
- causal/dependency links are dashed gray arrows
- node radius and color intensity scale with normalized importance
- selected nodes receive a dark outline and a detail panel

The source site uses React + D3. This script writes static SVG directly so it
works in the minimal login-node environment where matplotlib is unavailable.
"""

from __future__ import annotations

import json
import math
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


OUT_DIR = Path("analysis/qwen_anchor/visualizations")
QUAL_PATH = Path("analysis/qwen_anchor/qualitative_report.json")
MMLU_ROOT = Path("mmlu_rollouts")
SUFFIX = "paper10_matched_clean_apr28"

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

FUNCTION_TAG_COLORS = {
    "problem_setup": "#4285F4",
    "plan_generation": "#EA4335",
    "fact_retrieval": "#FBBC05",
    "active_computation": "#34A853",
    "uncertainty_management": "#9C27B0",
    "self_checking": "#FF9800",
    "result_consolidation": "#00BCD4",
    "final_answer_emission": "#795548",
    "other": "#795548",
    "unknown": "#9E9E9E",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_svg(path: Path, width: int, height: int, body: List[str]) -> None:
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            *body,
            "</svg>",
        ]
    )
    path.write_text(svg, encoding="utf-8")


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    fill: str = "#333",
    anchor: str = "start",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{escape(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "#fff",
    stroke: str = "#ddd",
    sw: float = 1.0,
    rx: float = 8,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" ry="{rx:.1f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw:.1f}" opacity="{opacity:.3f}"/>'
    )


def circle(
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str,
    stroke: str = "#fff",
    sw: float = 2.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity:.3f}"/>'
    )


def path_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    sw: float = 2.0,
    dash: Optional[str] = None,
    opacity: float = 1.0,
    marker: Optional[str] = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<path d="M{x1:.1f},{y1:.1f}L{x2:.1f},{y2:.1f}" fill="none" '
        f'stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity:.3f}"'
        f'{dash_attr}{marker_attr}/>'
    )


def wrap_words(value: str, width: int, max_lines: int) -> List[str]:
    value = re.sub(r"\s+", " ", str(value)).strip()
    words = value.split(" ")
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            lines.append(word[: max(1, width - 1)] + "...")
            current = ""
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1][: max(1, width - 3)].rstrip() + "..."
    return lines or [""]


def color_blend_with_white(hex_color: str, intensity: float) -> str:
    intensity = max(0.0, min(1.0, intensity))
    raw = hex_color.lstrip("#")
    r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    out = (
        round(r * intensity + 255 * (1 - intensity)),
        round(g * intensity + 255 * (1 - intensity)),
        round(b * intensity + 255 * (1 - intensity)),
    )
    return f"#{out[0]:02x}{out[1]:02x}{out[2]:02x}"


def format_tag(tag: str, abbrev: bool = False) -> str:
    if abbrev:
        return "".join(part[:1].upper() for part in tag.split("_"))
    return " ".join(part.capitalize() for part in tag.split("_"))


def heuristic_tag(chunk_text: str) -> str:
    low = chunk_text.lower()
    stripped = chunk_text.strip()
    if "final answer" in low or "\\boxed" in low:
        return "final_answer_emission"
    if re.fullmatch(r"[A-D][.)]?", stripped) or re.match(r"^[A-D][.)]\s", stripped):
        return "final_answer_emission"
    if any(marker in low for marker in ["therefore", "thus", "based on", "so the"]):
        return "result_consolidation"
    if any(marker in low for marker in ["not ", "except", "false", "incorrect", "does not"]):
        return "self_checking"
    if any(marker in low for marker in ["choice", "option", "a.", "b.", "c.", "d."]):
        return "fact_retrieval"
    if any(marker in low for marker in ["define", "defined", "means", "refers to", "is a", "are "]):
        return "fact_retrieval"
    if any(ch.isdigit() for ch in low) or any(marker in low for marker in ["=", "\\(", "\\)", "calculate"]):
        return "active_computation"
    if any(marker in low for marker in ["to solve", "we need", "let's", "analyze", "determine"]):
        return "problem_setup"
    return "active_computation"


def rollout_dir(model: str, base_type: str) -> Path:
    return (
        MMLU_ROOT
        / model
        / "temperature_0.6_top_p_0.95"
        / f"{base_type}_base_solution_{SUFFIX}"
    )


def qual_lookup() -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
    data = load_json(QUAL_PATH)
    lookup: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for summary in data.get("summaries", []):
        model = summary["model"]
        base_type = summary["base_type"]
        for chunk in summary.get("chunks", []):
            lookup[(model, base_type, chunk["problem_name"], chunk["chunk_idx"])] = chunk
    return lookup


def load_case(model: str, base_type: str, problem_name: str) -> Dict[str, Any]:
    problem_dir = rollout_dir(model, base_type) / problem_name
    chunks = load_json(problem_dir / "chunks_labeled.json")
    problem = load_json(problem_dir / "problem.json")
    base = load_json(problem_dir / "base_solution.json")
    qlookup = qual_lookup()

    raw_importances: List[float] = []
    enriched: List[Dict[str, Any]] = []
    for chunk in chunks:
        idx = int(chunk.get("chunk_idx", len(enriched)))
        qchunk = qlookup.get((model, base_type, problem_name, idx), {})
        raw = abs(
            chunk.get("counterfactual_importance_kl")
            or chunk.get("counterfactual_importance_accuracy")
            or qchunk.get("anchor_strength")
            or 0.0
        )
        raw_importances.append(raw)
        tag = (chunk.get("function_tags") or ["unknown"])[0]
        if tag == "unknown":
            tag = heuristic_tag(chunk.get("chunk", ""))
        enriched.append(
            {
                **chunk,
                "chunk_idx": idx,
                "raw_importance": raw,
                "chunk_type": qchunk.get("chunk_type"),
                "full_paired": qchunk.get("full_paired"),
                "default_accuracy": qchunk.get("default_accuracy"),
                "forced_accuracy": qchunk.get("forced_accuracy"),
                "anchor_strength": qchunk.get("anchor_strength"),
                "forced_strength": qchunk.get("forced_strength"),
                "tag": tag,
            }
        )

    min_imp = min(raw_importances) if raw_importances else 0.0
    max_imp = max(raw_importances) if raw_importances else 1.0
    denom = max(max_imp - min_imp, 1e-9)
    for chunk in enriched:
        norm = (chunk["raw_importance"] - min_imp) / denom
        chunk["importance"] = norm
        chunk["radius"] = max(10.0, 10.0 + math.log(1 + norm * 20.0) * 3.5)
        chunk["color_intensity"] = min(1.0, max(0.6, norm * 3.0))
    return {
        "model": model,
        "base_type": base_type,
        "problem_name": problem_name,
        "problem": problem,
        "base": base,
        "chunks": enriched,
    }


def link_endpoint(
    src: Dict[str, Any],
    dst: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    dx = dst["x"] - src["x"]
    dy = dst["y"] - src["y"]
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    return (
        src["x"] + ux * (src["radius"] + 2),
        src["y"] + uy * (src["radius"] + 2),
        dst["x"] - ux * (dst["radius"] + 8),
        dst["y"] - uy * (dst["radius"] + 8),
    )


def build_links(chunks: List[Dict[str, Any]], causal_links_count: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_id = {chunk["chunk_idx"]: chunk for chunk in chunks}
    sequential = []
    for left, right in zip(chunks, chunks[1:]):
        sequential.append({"source": left["chunk_idx"], "target": right["chunk_idx"], "type": "sequential", "weight": 1.0})

    causal = []
    seen = set()
    for chunk in chunks:
        deps = [dep for dep in chunk.get("depends_on", []) if dep in by_id and dep != chunk["chunk_idx"]]
        deps = sorted(deps, key=lambda dep: by_id[dep]["importance"], reverse=True)[:causal_links_count]
        for dep in deps:
            key = (dep, chunk["chunk_idx"])
            if key in seen or dep + 1 == chunk["chunk_idx"]:
                continue
            seen.add(key)
            causal.append(
                {
                    "source": dep,
                    "target": chunk["chunk_idx"],
                    "type": "causal",
                    "weight": max(by_id[dep]["importance"], chunk["importance"]),
                }
            )
    return sequential, causal


def layout_circle(chunks: List[Dict[str, Any]], cx: float, cy: float, radius: float) -> None:
    count = max(1, len(chunks))
    for i, chunk in enumerate(sorted(chunks, key=lambda item: item["chunk_idx"])):
        angle = (i / count) * 2 * math.pi - math.pi / 2
        chunk["angle"] = angle
        chunk["x"] = cx + radius * math.cos(angle)
        chunk["y"] = cy + radius * math.sin(angle)


def marker_defs() -> str:
    return """
<defs>
  <marker id="arrow-sequential" viewBox="0 -5 10 10" refX="8" refY="0" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M0,-5L10,0L0,5" fill="#333"/>
  </marker>
  <marker id="arrow-causal" viewBox="0 -5 10 10" refX="8" refY="0" markerWidth="5" markerHeight="5" orient="auto">
    <path d="M0,-5L10,0L0,5" fill="#999"/>
  </marker>
</defs>
""".strip()


def render_circle_graph(
    case: Dict[str, Any],
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    selected_idx: Optional[int] = None,
    title: Optional[str] = None,
) -> List[str]:
    chunks = sorted(case["chunks"], key=lambda item: item["chunk_idx"])
    cx, cy = x + w / 2, y + h * 0.47
    radius = min(w, h) * 0.39
    layout_circle(chunks, cx, cy, radius)
    by_id = {chunk["chunk_idx"]: chunk for chunk in chunks}
    sequential, causal = build_links(chunks)
    selected_idx = selected_idx if selected_idx is not None else max(chunks, key=lambda c: c["importance"])["chunk_idx"]

    body: List[str] = []
    body.append(rect(x, y, w, h, fill="#f9f9f9", stroke="#ddd", rx=8))
    if title:
        body.append(text(cx, y + 28, title, size=18, anchor="middle", weight="700"))

    body.append(marker_defs())
    # Causal links first.
    for link in causal:
        src, dst = by_id[link["source"]], by_id[link["target"]]
        x1, y1, x2, y2 = link_endpoint(src, dst)
        related = selected_idx in {src["chunk_idx"], dst["chunk_idx"]}
        body.append(
            path_line(
                x1,
                y1,
                x2,
                y2,
                stroke="#999",
                sw=2,
                dash="3,3",
                opacity=(0.85 if related else max(0.18, min(0.55, link["weight"] * 1.2))),
                marker="arrow-causal",
            )
        )

    for link in sequential:
        src, dst = by_id[link["source"]], by_id[link["target"]]
        x1, y1, x2, y2 = link_endpoint(src, dst)
        related = selected_idx in {src["chunk_idx"], dst["chunk_idx"]}
        body.append(
            path_line(
                x1,
                y1,
                x2,
                y2,
                stroke="#333",
                sw=2,
                opacity=(0.9 if related else 0.8),
                marker="arrow-sequential",
            )
        )

    for chunk in chunks:
        base_color = FUNCTION_TAG_COLORS.get(chunk["tag"], "#999")
        fill = color_blend_with_white(base_color, chunk["color_intensity"])
        stroke = "#333" if chunk["chunk_idx"] == selected_idx else "#fff"
        sw = 3.0 if chunk["chunk_idx"] == selected_idx else 2.0
        body.append(circle(chunk["x"], chunk["y"], chunk["radius"], fill=fill, stroke=stroke, sw=sw))
        body.append(
            text(
                chunk["x"],
                chunk["y"] + 4,
                str(chunk["chunk_idx"]),
                size=max(10, int(chunk["radius"] * 0.65)),
                fill="#fff",
                anchor="middle",
                weight="700",
            )
        )
    return body


def is_non_answer_chunk(chunk: Dict[str, Any]) -> bool:
    return chunk.get("chunk_type") != "answer-option token"


def is_full_paired_signal(chunk: Dict[str, Any]) -> bool:
    return bool(chunk.get("full_paired")) and chunk.get("default_accuracy") is not None and is_non_answer_chunk(chunk)


def aggregate_trend_stats() -> Dict[Tuple[str, str], Dict[str, Any]]:
    data = load_json(QUAL_PATH)
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for summary in data.get("summaries", []):
        chunks = [
            chunk
            for chunk in summary.get("chunks", [])
            if chunk.get("full_paired")
            and chunk.get("default_accuracy") is not None
            and chunk.get("forced_accuracy") is not None
            and chunk.get("chunk_type") != "answer-option token"
        ]
        model = summary["model"]
        base_type = summary["base_type"]
        if base_type == "correct":
            signal = sum(1 for chunk in chunks if chunk["default_accuracy"] <= 0.5)
            forced = sum(1 for chunk in chunks if chunk["forced_accuracy"] <= 0.5)
            stats[(model, base_type)] = {
                "full": len(chunks),
                "signal": signal,
                "forced": forced,
                "signal_label": "break",
                "forced_label": "forced-sensitive",
            }
        else:
            signal = sum(1 for chunk in chunks if chunk["default_accuracy"] >= 0.5)
            forced = sum(1 for chunk in chunks if chunk["forced_accuracy"] >= 0.5)
            stuck = sum(1 for chunk in chunks if chunk["default_accuracy"] <= 0.1)
            stats[(model, base_type)] = {
                "full": len(chunks),
                "signal": signal,
                "forced": forced,
                "stuck": stuck,
                "signal_label": "repair",
                "forced_label": "forced-repair",
            }
    return stats


def trend_signal(chunk: Dict[str, Any], base_type: str) -> str:
    if not is_full_paired_signal(chunk):
        return "unpaired"
    default_accuracy = chunk.get("default_accuracy")
    forced_accuracy = chunk.get("forced_accuracy")
    if base_type == "correct":
        if default_accuracy <= 0.5:
            return "break"
        if forced_accuracy is not None and forced_accuracy <= 0.5:
            return "forced_sensitive"
        return "robust"
    if default_accuracy >= 0.5:
        return "repair"
    if default_accuracy <= 0.1:
        return "stuck"
    if forced_accuracy is not None and forced_accuracy >= 0.5:
        return "forced_repair"
    return "partial"


def trend_case_counts(case: Dict[str, Any]) -> Dict[str, int]:
    counts = {
        "full": 0,
        "break": 0,
        "forced_sensitive": 0,
        "repair": 0,
        "forced_repair": 0,
        "stuck": 0,
    }
    for chunk in case["chunks"]:
        signal = trend_signal(chunk, case["base_type"])
        if signal == "unpaired":
            continue
        counts["full"] += 1
        if signal in {"break", "repair", "stuck"}:
            counts[signal] += 1
        if case["base_type"] == "correct" and chunk.get("forced_accuracy") is not None and chunk["forced_accuracy"] <= 0.5:
            counts["forced_sensitive"] += 1
        if case["base_type"] == "incorrect" and chunk.get("forced_accuracy") is not None and chunk["forced_accuracy"] >= 0.5:
            counts["forced_repair"] += 1
    return counts


def pct_text(num: int, denom: int) -> str:
    if denom <= 0:
        return "NA"
    return f"{num / denom:.0%}"


def render_trend_circle_panel(
    case: Dict[str, Any],
    x: float,
    y: float,
    w: float,
    h: float,
    aggregate: Dict[str, Any],
) -> List[str]:
    panel_x, panel_y = x, y
    chunks = sorted(case["chunks"], key=lambda item: item["chunk_idx"])
    cx, cy = panel_x + w / 2, panel_y + h * 0.53
    radius = min(w, h) * 0.29
    layout_circle(chunks, cx, cy, radius)
    by_id = {chunk["chunk_idx"]: chunk for chunk in chunks}
    sequential, causal = build_links(chunks, causal_links_count=2)
    counts = trend_case_counts(case)
    base_type = case["base_type"]
    model = case["model"]

    if base_type == "correct":
        accent = "#d45b4f"
        aggregate_line = (
            f"Overall correct traces: {aggregate['signal']}/{aggregate['full']} "
            f"break ({pct_text(aggregate['signal'], aggregate['full'])})"
        )
        case_line = (
            f"Shown trace: {counts['break']}/{counts['full']} red break anchors, "
            f"{counts['forced_sensitive']} forced-sensitive"
        )
    else:
        accent = "#3778bf"
        aggregate_line = (
            f"Overall incorrect traces: {aggregate['signal']}/{aggregate['full']} "
            f"repair ({pct_text(aggregate['signal'], aggregate['full'])})"
        )
        case_line = (
            f"Shown trace: {counts['repair']}/{counts['full']} blue repair anchors, "
            f"{counts['stuck']} stuck"
        )

    body: List[str] = []
    body.append(rect(panel_x, panel_y, w, h, fill="#fbfaf7", stroke="#ddd", rx=14))
    body.append(rect(panel_x, panel_y, w, 82, fill="#fff", stroke="#e3dfd8", rx=14))
    body.append(text(panel_x + 22, panel_y + 31, MODEL_LABELS[model], size=22, weight="700", fill=MODEL_COLORS[model]))
    body.append(text(panel_x + w - 22, panel_y + 31, base_type, size=14, anchor="end", fill=accent, weight="700"))
    body.append(text(panel_x + 22, panel_y + 56, aggregate_line, size=13, fill="#3f3a35"))
    body.append(text(panel_x + 22, panel_y + 76, case_line, size=12, fill="#6a625a"))
    body.append(marker_defs())

    for link in causal:
        src, dst = by_id[link["source"]], by_id[link["target"]]
        x1, link_y1, x2, link_y2 = link_endpoint(src, dst)
        body.append(path_line(x1, link_y1, x2, link_y2, stroke="#999", sw=1.8, dash="3,3", opacity=0.22, marker="arrow-causal"))
    for link in sequential:
        src, dst = by_id[link["source"]], by_id[link["target"]]
        x1, link_y1, x2, link_y2 = link_endpoint(src, dst)
        body.append(path_line(x1, link_y1, x2, link_y2, stroke="#333", sw=1.6, opacity=0.42, marker="arrow-sequential"))

    for chunk in chunks:
        signal = trend_signal(chunk, base_type)
        base_color = FUNCTION_TAG_COLORS.get(chunk["tag"], "#999")
        fill = color_blend_with_white(base_color, max(0.45, chunk["color_intensity"] * 0.85))
        r = max(7.5, min(18.0, chunk["radius"] * 0.82))

        if signal == "break":
            body.append(circle(chunk["x"], chunk["y"], r + 7.0, fill="#d45b4f", stroke="#d45b4f", sw=1.0, opacity=0.23))
            stroke, sw = "#d45b4f", 4.0
        elif signal == "repair":
            body.append(circle(chunk["x"], chunk["y"], r + 7.0, fill="#3778bf", stroke="#3778bf", sw=1.0, opacity=0.23))
            stroke, sw = "#3778bf", 4.0
        elif signal in {"forced_sensitive", "forced_repair"}:
            body.append(circle(chunk["x"], chunk["y"], r + 5.0, fill="#f0ad3d", stroke="#f0ad3d", sw=1.0, opacity=0.18))
            stroke, sw = "#f0ad3d", 2.4
        elif signal == "stuck":
            stroke, sw = "#9c948b", 2.2
        else:
            stroke, sw = "#fff", 1.8

        body.append(circle(chunk["x"], chunk["y"], r, fill=fill, stroke=stroke, sw=sw, opacity=0.96))
        label_fill = "#fff" if signal in {"break", "repair"} else "#3d3934"
        body.append(text(chunk["x"], chunk["y"] + 3.8, str(chunk["chunk_idx"]), size=9, fill=label_fill, anchor="middle", weight="700"))

    # A compact footer keeps the plot readable in slides without forcing people
    # to inspect the question text.
    problem_short = case["problem_name"].replace("problem_", "").replace("_", " ")
    body.append(text(cx, panel_y + h - 42, problem_short, size=13, anchor="middle", fill="#5c554e", weight="700"))
    body.append(text(cx, panel_y + h - 20, "same-question example; aggregate count is across all matched MMLU chunks", size=10, anchor="middle", fill="#7b736b"))
    return body


def render_model_tendency_storyboard() -> List[str]:
    stats = aggregate_trend_stats()
    correct_cases = [
        load_case("Qwen2.5-Math-7B", "correct", "problem_philosophy_64"),
        load_case("Qwen2.5-Math-7B-Instruct", "correct", "problem_philosophy_64"),
        load_case("Qwen2.5-Math-7B-Oat-Zero", "correct", "problem_philosophy_64"),
    ]
    incorrect_cases = [
        load_case("Qwen2.5-Math-7B", "incorrect", "problem_human_aging_168"),
        load_case("Qwen2.5-Math-7B-Instruct", "incorrect", "problem_human_aging_168"),
        load_case("Qwen2.5-Math-7B-Oat-Zero", "incorrect", "problem_human_aging_168"),
    ]

    body: List[str] = []
    body.append(text(950, 48, "Model tendencies as Thought Anchor circles", size=28, anchor="middle", weight="700"))
    body.append(
        text(
            950,
            78,
            "Same circle grammar as the paper website; colored halos expose the model-level pattern without reading every chunk.",
            size=14,
            anchor="middle",
            fill="#5c554e",
        )
    )
    body.append(circle(92, 106, 10, fill="#d45b4f", stroke="#d45b4f", sw=1, opacity=0.25))
    body.append(circle(92, 106, 6, fill="#fff", stroke="#d45b4f", sw=3))
    body.append(text(112, 111, "red halo: removing this thought breaks a correct trace", size=13, fill="#4a4540"))
    body.append(circle(520, 106, 10, fill="#3778bf", stroke="#3778bf", sw=1, opacity=0.25))
    body.append(circle(520, 106, 6, fill="#fff", stroke="#3778bf", sw=3))
    body.append(text(540, 111, "blue halo: removing this thought repairs a wrong trace", size=13, fill="#4a4540"))
    body.append(circle(920, 106, 8, fill="#fff", stroke="#f0ad3d", sw=2))
    body.append(text(938, 111, "gold: forced-answer sensitive", size=13, fill="#4a4540"))

    panel_w, panel_h = 560, 445
    xs = [62, 670, 1278]
    row_ys = [150, 660]
    row_titles = [
        ("Correct-base traces", "Across models, correct traces are mostly robust, but a small set of semantic/negation anchors can collapse the trajectory."),
        ("Incorrect-base traces", "Repairability separates the models: Base repairs often, Instruct stays stuck, Oat-Zero is mixed."),
    ]
    for row_idx, (title_value, subtitle) in enumerate(row_titles):
        y = row_ys[row_idx]
        body.append(text(34, y + 20, title_value, size=17, fill="#2f2a25", weight="700"))
        body.append(text(34, y + 43, subtitle, size=12, fill="#6a625a"))
        cases = correct_cases if row_idx == 0 else incorrect_cases
        for x, case in zip(xs, cases):
            body.extend(render_trend_circle_panel(case, x, y + 58, panel_w, panel_h, stats[(case["model"], case["base_type"])]))
    return body


def render_chain_panel(case: Dict[str, Any], x: float, y: float, w: float, h: float, selected_idx: int) -> List[str]:
    body = [rect(x, y, w, h, fill="#fff", stroke="#ddd", rx=8)]
    body.append(rect(x, y, w, 58, fill="#f8f9fa", stroke="#ddd", rx=8))
    body.append(text(x + 18, y + 36, "Chain-of-thought", size=20, weight="700"))
    chunks = sorted(case["chunks"], key=lambda item: item["chunk_idx"])
    card_h = min(65, (h - 78) / max(1, len(chunks)) - 5)
    card_h = max(43, card_h)
    gap = 6
    cursor_y = y + 74
    for chunk in chunks:
        color = FUNCTION_TAG_COLORS.get(chunk["tag"], "#999")
        opacity_hex = f"{round(min(0.8, max(0.12, chunk['importance'] * 2)) * 255):02x}"
        fill = f"{color}{opacity_hex}"
        is_selected = chunk["chunk_idx"] == selected_idx
        body.append(rect(x + 14, cursor_y, w - 28, card_h, fill=fill, stroke=(color if is_selected else "#e6e6e6"), sw=(2 if is_selected else 1), rx=6))
        body.append(circle(x + 20, cursor_y + 2, 12, fill=color, stroke="#fff", sw=2))
        body.append(text(x + 20, cursor_y + 6, str(chunk["chunk_idx"]), size=10, fill="#fff", anchor="middle", weight="700"))
        body.append(text(x + 40, cursor_y + 18, format_tag(chunk["tag"]), size=11, weight="700", fill="#444"))
        body.append(text(x + w - 24, cursor_y + 18, f"Importance: {chunk['importance']:.3f}", size=10, fill="#666", anchor="end", weight="700"))
        for i, line in enumerate(wrap_words(chunk.get("chunk", ""), 46, 2)):
            body.append(text(x + 40, cursor_y + 38 + i * 14, line, size=11, fill="#333"))
        cursor_y += card_h + gap
        if cursor_y > y + h - card_h:
            break
    return body


def incoming_outgoing(case: Dict[str, Any], selected_idx: int, limit: int = 3) -> Tuple[List[int], List[int]]:
    chunks = sorted(case["chunks"], key=lambda item: item["chunk_idx"])
    by_id = {chunk["chunk_idx"]: chunk for chunk in chunks}
    _, causal = build_links(chunks)
    incoming = sorted(
        [link for link in causal if link["target"] == selected_idx],
        key=lambda item: item["weight"],
        reverse=True,
    )[:limit]
    outgoing = sorted(
        [link for link in causal if link["source"] == selected_idx],
        key=lambda item: item["weight"],
        reverse=True,
    )[:limit]
    return [item["source"] for item in incoming], [item["target"] for item in outgoing]


def render_detail_panel(case: Dict[str, Any], x: float, y: float, w: float, h: float, selected_idx: int) -> List[str]:
    chunks = {chunk["chunk_idx"]: chunk for chunk in case["chunks"]}
    selected = chunks[selected_idx]
    color = FUNCTION_TAG_COLORS.get(selected["tag"], "#999")
    body = [rect(x, y, w, h, fill="#fff", stroke="#ddd", rx=8)]
    body.append(text(x + 18, y + 32, "Selected step", size=20, weight="700"))
    body.append(rect(x + 18, y + 54, w - 36, 148, fill=f"{color}22", stroke=color, rx=6))
    body.append(text(x + 34, y + 82, f"Step {selected_idx} ({format_tag(selected['tag'], True)})", size=15, weight="700", fill="#0066cc"))
    body.append(text(x + w - 34, y + 82, f"Importance: {selected['importance']:.3f}", size=12, fill="#666", anchor="end", weight="700"))
    for i, line in enumerate(wrap_words(selected.get("chunk", ""), 52, 4)):
        body.append(text(x + 34, y + 112 + i * 18, line, size=13, fill="#333"))

    default_acc = selected.get("default_accuracy")
    forced_acc = selected.get("forced_accuracy")
    body.append(text(x + 18, y + 232, "Rollout evidence", size=15, weight="700"))
    body.append(text(x + 28, y + 258, f"default accuracy after removal: {default_acc:.2f}" if default_acc is not None else "default accuracy after removal: NA", size=13, fill="#444"))
    body.append(text(x + 28, y + 280, f"forced-answer accuracy after removal: {forced_acc:.2f}" if forced_acc is not None else "forced-answer accuracy after removal: NA", size=13, fill="#444"))

    incoming, outgoing = incoming_outgoing(case, selected_idx)
    body.append(text(x + 18, y + 326, "Incoming connections", size=15, weight="700"))
    if incoming:
        for i, idx in enumerate(incoming):
            source = chunks[idx]
            scolor = FUNCTION_TAG_COLORS.get(source["tag"], "#999")
            body.append(rect(x + 22, y + 344 + i * 34, w - 44, 26, fill=f"{scolor}22", stroke=scolor, rx=4))
            body.append(text(x + 32, y + 362 + i * 34, f"Step {idx} ({format_tag(source['tag'], True)})", size=12, fill="#0066cc", weight="700"))
    else:
        body.append(text(x + 28, y + 358, "No dependency links found.", size=12, fill="#777"))

    base = case["base"]
    answer = base.get("answer", "NA")
    correct = base.get("is_correct")
    body.append(text(x + 18, y + h - 88, "Base answer", size=15, weight="700"))
    body.append(text(x + 28, y + h - 62, f"answer: {answer}", size=13, fill="#444"))
    body.append(text(x + 28, y + h - 40, f"is_correct: {correct}", size=13, fill="#444"))
    return body


def render_problem_interface(case: Dict[str, Any], selected_idx: Optional[int] = None) -> List[str]:
    chunks = sorted(case["chunks"], key=lambda item: item["chunk_idx"])
    selected_idx = selected_idx if selected_idx is not None else max(chunks, key=lambda c: c["importance"])["chunk_idx"]
    body: List[str] = []
    body.append(text(900, 44, "Thought Anchors circle view", size=28, anchor="middle", weight="700"))
    body.append(text(900, 74, f"{MODEL_LABELS[case['model']]} / {case['base_type']} / {case['problem_name']}", size=15, anchor="middle", fill="#666"))
    body.append(rect(40, 96, 1720, 76, fill="#fff", stroke="#ddd", rx=8))
    question = case["problem"].get("problem") or case["problem"].get("question") or ""
    for i, line in enumerate(wrap_words(question, 180, 2)):
        body.append(text(62, 126 + i * 20, line, size=13, fill="#333"))
    body.extend(render_chain_panel(case, 40, 190, 420, 820, selected_idx))
    body.extend(render_circle_graph(case, 482, 190, 830, 820, selected_idx=selected_idx))
    body.extend(render_detail_panel(case, 1334, 190, 426, 820, selected_idx))

    # Legend, matching the website's basic visual language.
    lx, ly = 514, 972
    body.append(path_line(lx, ly, lx + 56, ly, stroke="#333", sw=2, marker="arrow-sequential"))
    body.append(text(lx + 70, ly + 5, "Sequential reasoning", size=12, fill="#555"))
    body.append(path_line(lx + 250, ly, lx + 306, ly, stroke="#999", sw=2, dash="3,3", marker="arrow-causal"))
    body.append(text(lx + 320, ly + 5, "Causal/dependency link", size=12, fill="#555"))
    for i, (tag, color) in enumerate(
        [
            ("problem_setup", FUNCTION_TAG_COLORS["problem_setup"]),
            ("fact_retrieval", FUNCTION_TAG_COLORS["fact_retrieval"]),
            ("active_computation", FUNCTION_TAG_COLORS["active_computation"]),
            ("self_checking", FUNCTION_TAG_COLORS["self_checking"]),
            ("result_consolidation", FUNCTION_TAG_COLORS["result_consolidation"]),
            ("final_answer_emission", FUNCTION_TAG_COLORS["final_answer_emission"]),
        ]
    ):
        x = 514 + (i % 3) * 190
        y = 1010 + (i // 3) * 26
        body.append(circle(x, y - 5, 6, fill=color, stroke="#fff", sw=1))
        body.append(text(x + 14, y, format_tag(tag), size=11, fill="#555"))
    return body


def render_montage(cases: List[Dict[str, Any]]) -> List[str]:
    body: List[str] = []
    body.append(text(900, 48, "Paper-style circle views on current MMLU results", size=27, anchor="middle", weight="700"))
    body.append(text(900, 76, "Same layout as the Thought Anchors website: circular steps, solid sequential arrows, dashed dependency arrows.", size=14, anchor="middle", fill="#666"))
    panel_w, panel_h = 540, 420
    positions = [(55, 115), (630, 115), (1205, 115), (55, 590), (630, 590), (1205, 590)]
    for case, (x, y) in zip(cases, positions):
        chunks = sorted(case["chunks"], key=lambda item: item["chunk_idx"])
        selected = max(chunks, key=lambda c: c["importance"])["chunk_idx"]
        title = f"{MODEL_LABELS[case['model']]} / {case['base_type']} / {case['problem_name'].replace('problem_', '')}"
        body.extend(render_circle_graph(case, x, y, panel_w, panel_h, selected_idx=selected, title=title))
    return body


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    positive_anchor = load_case("Qwen2.5-Math-7B-Oat-Zero", "correct", "problem_philosophy_64")
    misleading_anchor = load_case("Qwen2.5-Math-7B-Instruct", "incorrect", "problem_high_school_statistics_8")

    write_svg(
        OUT_DIR / "mmlu_positive_anchor_philosophy64_circles.svg",
        1800,
        1040,
        render_problem_interface(positive_anchor, selected_idx=9),
    )
    write_svg(
        OUT_DIR / "mmlu_misleading_anchor_statistics8_circles.svg",
        1800,
        1040,
        render_problem_interface(misleading_anchor, selected_idx=10),
    )
    # Backward-compatible filenames used by earlier PPT notes. They now point to
    # the stronger examples above rather than the older weaker case selection.
    write_svg(
        OUT_DIR / "mmlu_philosophy64_trace_circles.svg",
        1800,
        1040,
        render_problem_interface(positive_anchor, selected_idx=9),
    )
    write_svg(
        OUT_DIR / "mmlu_statistics8_repair_circles.svg",
        1800,
        1040,
        render_problem_interface(misleading_anchor, selected_idx=10),
    )

    montage_cases = [
        load_case("Qwen2.5-Math-7B", "correct", "problem_high_school_government_and_politics_22"),
        load_case("Qwen2.5-Math-7B-Instruct", "correct", "problem_philosophy_64"),
        load_case("Qwen2.5-Math-7B-Oat-Zero", "correct", "problem_philosophy_64"),
        load_case("Qwen2.5-Math-7B", "incorrect", "problem_clinical_knowledge_36"),
        load_case("Qwen2.5-Math-7B-Instruct", "incorrect", "problem_high_school_statistics_8"),
        load_case("Qwen2.5-Math-7B-Oat-Zero", "incorrect", "problem_high_school_statistics_8"),
    ]
    write_svg(
        OUT_DIR / "mmlu_case_study_circles.svg",
        1800,
        1060,
        render_montage(montage_cases),
    )
    tendency_storyboard = render_model_tendency_storyboard()
    write_svg(
        OUT_DIR / "mmlu_model_tendency_circles.svg",
        1900,
        1180,
        tendency_storyboard,
    )
    # Main PPT-facing circle figure: same paper-style visual grammar, arranged
    # to show model tendencies at a glance rather than isolated case studies.
    write_svg(
        OUT_DIR / "mmlu_thought_anchor_circles.svg",
        1900,
        1180,
        tendency_storyboard,
    )
    print("Wrote paper-style SVG visualizations to", OUT_DIR)


if __name__ == "__main__":
    main()
