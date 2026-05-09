#!/usr/bin/env python3
"""Render PPT-ready model tendency figures for Qwen Thought Anchors results.

The figures intentionally avoid dense per-chunk detail. They compress the MMLU
Thought Anchors interventions into a few model-level tendencies that can be
read from a slide at a glance.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


QUAL_PATH = Path("analysis/qwen_anchor/qualitative_report.json")
OUT_DIR = Path("analysis/qwen_anchor/visualizations")

MODEL_ORDER = [
    "Qwen2.5-Math-7B",
    "Qwen2.5-Math-7B-Instruct",
    "Qwen2.5-Math-7B-Oat-Zero",
]

MODEL_LABELS = {
    "Qwen2.5-Math-7B": "Base",
    "Qwen2.5-Math-7B-Instruct": "Instruct",
    "Qwen2.5-Math-7B-Oat-Zero": "Oat-Zero",
}

MODEL_COLORS = {
    "Qwen2.5-Math-7B": "#237277",
    "Qwen2.5-Math-7B-Instruct": "#d5792d",
    "Qwen2.5-Math-7B-Oat-Zero": "#6f5ca8",
}

BG = "#fbfaf6"
INK = "#27231f"
MUTED = "#665f57"
GRID = "#ded7cb"
CORRECT = "#cf534c"
INCORRECT = "#2878a8"
SENS = "#8b6f32"
GREEN = "#2e8068"

FONT = "Noto Sans CJK KR, Apple SD Gothic Neo, Malgun Gothic, Arial, sans-serif"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg_text(
    x: float,
    y: float,
    text: object,
    *,
    size: int = 24,
    anchor: str = "start",
    weight: str = "400",
    fill: str = INK,
    opacity: float = 1.0,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{fill}" opacity="{opacity:.3f}">{esc(text)}</text>'
    )


def svg_tspan_lines(
    x: float,
    y: float,
    lines: Iterable[str],
    *,
    size: int = 22,
    anchor: str = "start",
    weight: str = "400",
    fill: str = INK,
    line_gap: int = 28,
) -> str:
    lines = list(lines)
    tspans = []
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_gap
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{fill}">' + "".join(tspans) + "</text>"
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = "none",
    stroke: str = "none",
    sw: float = 1.0,
    rx: float = 0.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{sw:.1f}" opacity="{opacity:.3f}"/>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = GRID,
    sw: float = 1.0,
    opacity: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{sw:.1f}" opacity="{opacity:.3f}"{dash_attr}/>'
    )


def circle(
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str,
    stroke: str = "none",
    sw: float = 1.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}" '
        f'opacity="{opacity:.3f}"/>'
    )


def path(points: List[Tuple[float, float]], *, stroke: str, sw: float = 4.0) -> str:
    if not points:
        return ""
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    return f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw:.1f}" stroke-linecap="round" stroke-linejoin="round"/>'


def page(width: int, height: int, elements: List[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="100%" height="100%" fill="{BG}"/>',
            *elements,
            "</svg>",
            "",
        ]
    )


def full_chunks(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        chunk
        for chunk in summary.get("chunks", [])
        if chunk.get("full_paired")
        and chunk.get("default_accuracy") is not None
        and chunk.get("forced_accuracy") is not None
    ]


def collect_metrics(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summaries = {
        (summary["model"], summary["base_type"]): summary
        for summary in payload.get("summaries", [])
    }
    metrics: Dict[str, Dict[str, Any]] = {}
    for model in MODEL_ORDER:
        correct_chunks = full_chunks(summaries[(model, "correct")])
        incorrect_chunks = full_chunks(summaries[(model, "incorrect")])
        all_chunks = correct_chunks + incorrect_chunks
        correct_break = [
            chunk for chunk in correct_chunks if chunk["default_accuracy"] <= 0.5
        ]
        correct_catastrophic = [
            chunk for chunk in correct_chunks if chunk["default_accuracy"] <= 0.2
        ]
        incorrect_repair = [
            chunk for chunk in incorrect_chunks if chunk["default_accuracy"] >= 0.5
        ]
        incorrect_strong_repair = [
            chunk for chunk in incorrect_chunks if chunk["default_accuracy"] >= 0.8
        ]
        forced_gaps = [
            abs(chunk["forced_accuracy"] - chunk["default_accuracy"])
            for chunk in all_chunks
        ]
        correct_mean_fragility = (
            sum(1.0 - chunk["default_accuracy"] for chunk in correct_chunks)
            / len(correct_chunks)
            if correct_chunks
            else 0.0
        )
        incorrect_mean_repair = (
            sum(chunk["default_accuracy"] for chunk in incorrect_chunks)
            / len(incorrect_chunks)
            if incorrect_chunks
            else 0.0
        )
        metrics[model] = {
            "label": MODEL_LABELS[model],
            "color": MODEL_COLORS[model],
            "correct_n": len(correct_chunks),
            "incorrect_n": len(incorrect_chunks),
            "correct_break_n": len(correct_break),
            "correct_break_rate": len(correct_break) / len(correct_chunks)
            if correct_chunks
            else 0.0,
            "correct_catastrophic_n": len(correct_catastrophic),
            "incorrect_repair_n": len(incorrect_repair),
            "incorrect_repair_rate": len(incorrect_repair) / len(incorrect_chunks)
            if incorrect_chunks
            else 0.0,
            "incorrect_strong_repair_n": len(incorrect_strong_repair),
            "forced_gap": sum(forced_gaps) / len(forced_gaps) if forced_gaps else 0.0,
            "correct_mean_fragility": correct_mean_fragility,
            "incorrect_mean_repair": incorrect_mean_repair,
        }
    return metrics


def pct(value: float) -> str:
    return f"{round(100 * value):d}%"


def small_pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def pill(x: float, y: float, label: str, *, fill: str, text: str = "#ffffff") -> List[str]:
    width = 18 + 9.8 * len(label)
    return [
        rect(x, y - 22, width, 32, fill=fill, rx=16, opacity=0.96),
        svg_text(x + width / 2, y, label, size=15, anchor="middle", weight="700", fill=text),
    ]


def bar(
    x: float,
    y: float,
    w: float,
    h: float,
    value: float,
    *,
    fill: str,
    bg: str = "#ebe4d8",
) -> List[str]:
    value = max(0.0, min(1.0, value))
    return [
        rect(x, y, w, h, fill=bg, rx=h / 2),
        rect(x, y, w * value, h, fill=fill, rx=h / 2),
    ]


def render_model_cards(metrics: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    width, height = 1600, 900
    elements: List[str] = [
        svg_text(80, 84, "MMLU에서 보이는 모델별 Thought Anchor 성향", size=42, weight="800"),
        svg_text(
            80,
            126,
            "동일 base trace에서 chunk 하나를 제거하고 100 default / 100 forced-answer rollout을 비교",
            size=22,
            fill=MUTED,
        ),
        *pill(80, 176, "full-paired chunks only", fill="#51483f"),
        *pill(300, 176, "black-box intervention", fill="#51483f"),
    ]
    card_y, card_w, card_h = 235, 450, 520
    card_xs = [80, 575, 1070]
    punchlines = {
        "Qwen2.5-Math-7B": ("오답 trace가", "가장 잘 풀린다"),
        "Qwen2.5-Math-7B-Instruct": ("오답 trace가", "가장 고집스럽다"),
        "Qwen2.5-Math-7B-Oat-Zero": ("중간 성향 +", "정답 anchor pockets"),
    }
    detail_lines = {
        "Qwen2.5-Math-7B": "removing a misleading chunk often repairs the wrong chain",
        "Qwen2.5-Math-7B-Instruct": "wrong chains rarely repair from one local deletion",
        "Qwen2.5-Math-7B-Oat-Zero": "between Base and Instruct, with visible brittle pockets",
    }
    for x, model in zip(card_xs, MODEL_ORDER):
        m = metrics[model]
        color = m["color"]
        elements.append(rect(x, card_y, card_w, card_h, fill="#ffffff", stroke="#e1d7ca", sw=2, rx=32))
        elements.append(rect(x, card_y, card_w, 16, fill=color, rx=8))
        elements.append(svg_text(x + 34, card_y + 72, m["label"], size=34, weight="800", fill=color))
        elements.append(
            svg_tspan_lines(
                x + 34,
                card_y + 126,
                punchlines[model],
                size=30,
                weight="800",
                fill=INK,
                line_gap=38,
            )
        )
        elements.append(svg_text(x + 34, card_y + 210, detail_lines[model], size=15, fill=MUTED))

        big_y = card_y + 292
        elements.append(svg_text(x + 34, big_y, pct(m["incorrect_repair_rate"]), size=74, weight="900", fill=INCORRECT))
        elements.append(svg_text(x + 190, big_y - 12, "오답 trace", size=23, weight="800", fill=INCORRECT))
        elements.append(svg_text(x + 190, big_y + 22, "repair anchors", size=23, weight="800", fill=INCORRECT))
        elements.append(svg_text(x + 34, big_y + 48, f"{m['incorrect_repair_n']}/{m['incorrect_n']} chunks", size=17, fill=MUTED))

        y2 = card_y + 385
        elements.append(svg_text(x + 34, y2, "정답 trace breaks", size=18, weight="700", fill=CORRECT))
        elements.extend(bar(x + 210, y2 - 18, 180, 18, m["correct_break_rate"], fill=CORRECT))
        elements.append(svg_text(x + 404, y2 - 2, pct(m["correct_break_rate"]), size=18, weight="800", fill=CORRECT))
        elements.append(svg_text(x + 34, y2 + 44, "forced 민감도", size=18, weight="700", fill=SENS))
        elements.extend(bar(x + 210, y2 + 26, 180, 18, m["forced_gap"], fill=SENS))
        elements.append(svg_text(x + 404, y2 + 42, pct(m["forced_gap"]), size=18, weight="800", fill=SENS))

    elements.append(
        svg_text(
            width / 2,
            832,
            "읽는 법: 파란 숫자 = 오답 base trace에서 chunk 제거만으로 정답으로 회복되는 비율. 빨간/갈색 막대는 정답 trace 취약성 및 forced-answer 민감도.",
            size=19,
            anchor="middle",
            fill=MUTED,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mmlu_model_tendency_cards.svg").write_text(page(width, height, elements), encoding="utf-8")


def render_slope_summary(metrics: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    width, height = 1600, 900
    left, right = 210, 1260
    top, bottom = 190, 710
    model_x = {
        model: left + idx * ((right - left) / 2.0)
        for idx, model in enumerate(MODEL_ORDER)
    }

    def y_for(value: float) -> float:
        return bottom - max(0.0, min(0.7, value)) / 0.7 * (bottom - top)

    series = [
        ("오답 trace repair anchors", "incorrect_repair_rate", INCORRECT, 7),
        ("forced-answer 민감도", "forced_gap", SENS, 7),
        ("정답 trace breaks", "correct_break_rate", CORRECT, 7),
    ]
    elements: List[str] = [
        svg_text(80, 84, "한 장 요약: 모델 차이는 주로 오답 trace에서 벌어진다", size=42, weight="800"),
        svg_text(
            80,
            126,
            "정답 trace 취약성은 세 모델이 비슷하지만, 오답 trace repairability는 크게 갈립니다.",
            size=23,
            fill=MUTED,
        ),
        rect(140, 160, 1220, 600, fill="#ffffff", stroke="#e1d7ca", sw=2, rx=34),
    ]
    for tick in [0.0, 0.2, 0.4, 0.6]:
        y = y_for(tick)
        elements.append(line(left, y, right, y, stroke=GRID, sw=1.4))
        elements.append(svg_text(left - 24, y + 7, small_pct(tick), size=18, anchor="end", fill=MUTED))
    for model in MODEL_ORDER:
        x = model_x[model]
        elements.append(line(x, top - 25, x, bottom + 8, stroke="#eee7dc", sw=1.1))
        elements.append(svg_text(x, bottom + 56, metrics[model]["label"], size=28, anchor="middle", weight="800", fill=metrics[model]["color"]))

    for label, key, color, radius in series:
        points = [(model_x[model], y_for(metrics[model][key])) for model in MODEL_ORDER]
        elements.append(path(points, stroke=color, sw=5.5 if key == "incorrect_repair_rate" else 4.0))
        for model, (x, y) in zip(MODEL_ORDER, points):
            elements.append(circle(x, y, radius + (4 if key == "incorrect_repair_rate" else 0), fill=color, stroke="#ffffff", sw=2.2))
            elements.append(svg_text(x, y - 20, pct(metrics[model][key]), size=20, anchor="middle", weight="800", fill=color))

    legend_y = 816
    legend_xs = [260, 650, 1015]
    for (label, _key, color, _radius), x in zip(series, legend_xs):
        elements.append(line(x, legend_y - 7, x + 58, legend_y - 7, stroke=color, sw=5))
        elements.append(circle(x + 29, legend_y - 7, 7, fill=color, stroke="#ffffff", sw=1.5))
        elements.append(svg_text(x + 72, legend_y, label, size=19, fill=INK, weight="700"))

    callout_x, callout_y = 1330, 278
    elements.append(rect(callout_x, callout_y, 210, 330, fill="#f4efe7", stroke="#e1d7ca", rx=24))
    elements.append(svg_text(callout_x + 24, callout_y + 50, "발표용 takeaway", size=22, weight="800"))
    elements.append(
        svg_tspan_lines(
            callout_x + 24,
            callout_y + 92,
            [
                "Base: 잘못된 CoT도",
                "한 chunk 삭제로",
                "자주 회복됨",
                "",
                "Instruct: 오답 경로가",
                "더 고정되어 있음",
            ],
            size=17,
            fill=MUTED,
            line_gap=26,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mmlu_model_tendency_slope.svg").write_text(page(width, height, elements), encoding="utf-8")


def render_compass(metrics: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    width, height = 1600, 900
    plot_x, plot_y, plot_w, plot_h = 210, 175, 860, 600

    def sx(value: float) -> float:
        return plot_x + max(0.0, min(0.45, value)) / 0.45 * plot_w

    def sy(value: float) -> float:
        return plot_y + plot_h - max(0.0, min(0.7, value)) / 0.7 * plot_h

    elements: List[str] = [
        svg_text(80, 84, "Thought Anchor tendency map", size=42, weight="800"),
        svg_text(
            80,
            126,
            "x = forced/default rollout gap, y = 오답 trace repairability, bubble = catastrophic correct anchors",
            size=22,
            fill=MUTED,
        ),
        rect(plot_x, plot_y, plot_w, plot_h, fill="#ffffff", stroke="#e1d7ca", sw=2, rx=28),
    ]
    for tick in [0.0, 0.15, 0.30, 0.45]:
        x = sx(tick)
        elements.append(line(x, plot_y, x, plot_y + plot_h, stroke=GRID, sw=1.2))
        elements.append(svg_text(x, plot_y + plot_h + 34, small_pct(tick), size=17, anchor="middle", fill=MUTED))
    for tick in [0.0, 0.2, 0.4, 0.6]:
        y = sy(tick)
        elements.append(line(plot_x, y, plot_x + plot_w, y, stroke=GRID, sw=1.2))
        elements.append(svg_text(plot_x - 20, y + 6, small_pct(tick), size=17, anchor="end", fill=MUTED))

    elements.append(svg_text(plot_x + plot_w / 2, plot_y + plot_h + 78, "forced-answer 민감도", size=22, anchor="middle", weight="800"))
    elements.append(svg_text(plot_x - 82, plot_y + plot_h / 2, "오답 trace repairability", size=22, anchor="middle", weight="800"))
    elements.append(svg_text(plot_x + 34, plot_y + 42, "오답 경로가 잘 풀림", size=21, fill=INCORRECT, weight="800"))
    elements.append(svg_text(plot_x + plot_w - 34, plot_y + plot_h - 26, "forced 영향 작음", size=18, fill=MUTED, anchor="end"))

    for model in MODEL_ORDER:
        m = metrics[model]
        x = sx(m["forced_gap"])
        y = sy(m["incorrect_repair_rate"])
        radius = 26 + 8 * m["correct_catastrophic_n"]
        elements.append(circle(x, y, radius, fill=m["color"], stroke="#ffffff", sw=4, opacity=0.92))
        elements.append(svg_text(x, y + 8, m["label"], size=21, anchor="middle", weight="900", fill="#ffffff"))
        elements.append(svg_text(x, y + radius + 30, f"repair {pct(m['incorrect_repair_rate'])}, gap {pct(m['forced_gap'])}", size=17, anchor="middle", fill=INK, weight="700"))

    note_x, note_y = 1135, 220
    elements.append(rect(note_x, note_y, 355, 420, fill="#ffffff", stroke="#e1d7ca", sw=2, rx=28))
    elements.append(svg_text(note_x + 28, note_y + 55, "그림만 보고 읽는 법", size=25, weight="900"))
    elements.append(
        svg_tspan_lines(
            note_x + 28,
            note_y + 105,
            [
                "위로 갈수록:",
                "오답 CoT에서 misleading",
                "anchor를 지우면 정답으로",
                "회복되는 경우가 많음",
                "",
                "오른쪽으로 갈수록:",
                "forced-answer 조건이",
                "결과를 크게 바꿈",
            ],
            size=20,
            fill=MUTED,
            line_gap=30,
        )
    )
    elements.append(svg_text(note_x + 28, note_y + 365, "Bubble size = 정답 trace catastrophic anchors", size=16, fill=MUTED, weight="700"))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mmlu_model_tendency_compass.svg").write_text(page(width, height, elements), encoding="utf-8")


def write_readme(metrics: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# PPT Trend Figures",
        "",
        "Generated from `analysis/qwen_anchor/qualitative_report.json` using full-paired MMLU chunks only.",
        "",
        "Metrics:",
        "- `correct trace breaks`: correct-base chunks with default rollout accuracy <= 0.5 after chunk removal.",
        "- `incorrect trace repair anchors`: incorrect-base chunks with default rollout accuracy >= 0.5 after chunk removal.",
        "- `forced-answer sensitivity`: mean absolute gap between default and forced-answer rollout accuracy across full-paired chunks.",
        "",
        "| Model | Correct breaks | Incorrect repairs | Forced sensitivity |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model in MODEL_ORDER:
        m = metrics[model]
        lines.append(
            f"| {m['label']} | {m['correct_break_n']}/{m['correct_n']} ({pct(m['correct_break_rate'])}) "
            f"| {m['incorrect_repair_n']}/{m['incorrect_n']} ({pct(m['incorrect_repair_rate'])}) "
            f"| {pct(m['forced_gap'])} |"
        )
    lines.extend(
        [
            "",
            "Recommended PPT files:",
            "- `mmlu_model_tendency_cards.svg`: best one-slide, audience-friendly summary.",
            "- `mmlu_model_tendency_slope.svg`: best for showing model trend lines.",
            "- `mmlu_model_tendency_compass.svg`: best conceptual map of model behavior.",
        ]
    )
    (output_dir / "PPT_TREND_FIGURES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", default=str(QUAL_PATH))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(Path(args.report_json))
    output_dir = Path(args.output_dir)
    metrics = collect_metrics(payload)
    render_model_cards(metrics, output_dir)
    render_slope_summary(metrics, output_dir)
    render_compass(metrics, output_dir)
    write_readme(metrics, output_dir)
    print(f"Wrote PPT trend figures to {output_dir}")


if __name__ == "__main__":
    main()
