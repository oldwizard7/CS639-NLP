#!/usr/bin/env python3
"""Create compact Thought Anchors visualizations from Qwen analysis reports."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # Keep visual generation usable on minimal CHTC/login envs.
    plt = None


MODEL_LABELS = {
    "Qwen2.5-Math-7B": "Base",
    "Qwen2.5-Math-7B-Instruct": "Instruct",
    "Qwen2.5-Math-7B-Oat-Zero": "Oat-Zero",
}

COLORS = {
    "Qwen2.5-Math-7B": "#355C7D",
    "Qwen2.5-Math-7B-Instruct": "#C06C84",
    "Qwen2.5-Math-7B-Oat-Zero": "#6C5B7B",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def full_chunks(summary: Dict[str, Any], min_valid: int) -> List[Dict[str, Any]]:
    return [
        chunk
        for chunk in summary["chunks"]
        if chunk["n_default"] >= min_valid and chunk["n_forced"] >= min_valid
    ]


def fmt_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 14,
    anchor: str = "start",
    weight: str = "400",
    fill: str = "#222222",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'
    )


def svg_page(width: int, height: int, elements: List[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#fbfaf7"/>',
            *elements,
            "</svg>",
        ]
    )


def write_svg(path: Path, width: int, height: int, elements: List[str]) -> None:
    path.write_text(svg_page(width, height, elements), encoding="utf-8")


def plot_anchor_scatter_svg(
    summaries: List[Dict[str, Any]], min_valid: int, output_dir: Path
) -> None:
    width, height = 1180, 560
    plot_w, plot_h = 420, 340
    panels = [("correct", 90, 110), ("incorrect", 650, 110)]
    titles = {
        "correct": "Correct base: removal damages true trace",
        "incorrect": "Incorrect base: removal repairs wrong trace",
    }
    elements = [
        svg_text(width / 2, 42, "Thought Anchor Chunk Map (MMLU)", 24, "middle", "700")
    ]

    for base_type, left, top in panels:
        elements.append(svg_text(left + plot_w / 2, top - 28, titles[base_type], 16, "middle", "700"))
        elements.append(
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
            'fill="#ffffff" stroke="#ddd6cc"/>'
        )
        for tick in [0.0, 0.5, 1.0]:
            x = left + tick * plot_w
            y = top + plot_h - tick * plot_h
            elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#ece6dc"/>')
            elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ece6dc"/>')
            elements.append(svg_text(x, top + plot_h + 24, f"{tick:.1f}", 12, "middle"))
            elements.append(svg_text(left - 12, y + 4, f"{tick:.1f}", 12, "end"))
        mid_x = left + 0.5 * plot_w
        mid_y = top + plot_h - 0.5 * plot_h
        elements.append(f'<line x1="{mid_x:.1f}" y1="{top}" x2="{mid_x:.1f}" y2="{top + plot_h}" stroke="#898989" stroke-dasharray="6 6"/>')
        elements.append(f'<line x1="{left}" y1="{mid_y:.1f}" x2="{left + plot_w}" y2="{mid_y:.1f}" stroke="#898989" stroke-dasharray="6 6"/>')

        for summary in summaries:
            if summary["base_type"] != base_type:
                continue
            color = COLORS.get(summary["model"], "#333333")
            for chunk in full_chunks(summary, min_valid):
                x_val = chunk["default_accuracy"]
                y_val = chunk["forced_accuracy"]
                if x_val is None or y_val is None:
                    continue
                if base_type == "correct":
                    strength = max(0.0, 1.0 - x_val)
                else:
                    strength = max(0.0, x_val)
                radius = 4.0 + 9.0 * strength
                x = left + x_val * plot_w
                y = top + plot_h - y_val * plot_h
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                    f'fill="{color}" fill-opacity="0.58" stroke="#ffffff" stroke-width="1"/>'
                )

        elements.append(svg_text(left + plot_w / 2, top + plot_h + 54, "Default rollout accuracy", 13, "middle"))
        elements.append(svg_text(left - 58, top + plot_h / 2, "Forced-answer accuracy", 13, "middle"))

    legend_x, legend_y = 870, 490
    for idx, model in enumerate(MODEL_LABELS):
        y = legend_y + idx * 22
        elements.append(f'<circle cx="{legend_x}" cy="{y - 4}" r="6" fill="{COLORS[model]}"/>')
        elements.append(svg_text(legend_x + 16, y, fmt_model(model), 13))

    write_svg(output_dir / "mmlu_anchor_chunk_map.svg", width, height, elements)


def plot_outcome_bars_svg(
    summaries: List[Dict[str, Any]], min_valid: int, output_dir: Path
) -> None:
    width, height = 1120, 500
    elements = [
        svg_text(width / 2, 42, "Thought Anchor Outcomes by Model (MMLU)", 24, "middle", "700")
    ]
    correct_summaries = [summary for summary in summaries if summary["base_type"] == "correct"]
    incorrect_summaries = [summary for summary in summaries if summary["base_type"] == "incorrect"]
    labels = [fmt_model(summary["model"]) for summary in correct_summaries]

    correct_robust = []
    correct_brittle = []
    incorrect_repaired = []
    incorrect_stubborn = []
    for summary in correct_summaries:
        chunks = full_chunks(summary, min_valid)
        correct_robust.append(sum(1 for chunk in chunks if (chunk["default_accuracy"] or 0.0) >= 0.9))
        correct_brittle.append(sum(1 for chunk in chunks if chunk["default_accuracy"] is not None and chunk["default_accuracy"] <= 0.5))
    for summary in incorrect_summaries:
        chunks = full_chunks(summary, min_valid)
        incorrect_repaired.append(sum(1 for chunk in chunks if (chunk["default_accuracy"] or 0.0) >= 0.5))
        incorrect_stubborn.append(sum(1 for chunk in chunks if chunk["default_accuracy"] is not None and chunk["default_accuracy"] <= 0.1))

    panels = [
        ("Correct-base chunks", correct_robust, correct_brittle, "Robust >=90%", "Brittle <=50%", "#4E937A", "#D95D39", 80),
        ("Incorrect-base chunks", incorrect_repaired, incorrect_stubborn, "Repairs >=50%", "Stubborn <=10%", "#3B8EA5", "#5D576B", 620),
    ]
    plot_w, plot_h, top = 420, 300, 105
    for title, series_a, series_b, label_a, label_b, color_a, color_b, left in panels:
        max_y = max(series_a + series_b + [1])
        elements.append(svg_text(left + plot_w / 2, top - 24, title, 16, "middle", "700"))
        elements.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#ddd6cc"/>')
        for tick in [0, max_y // 2, max_y]:
            y = top + plot_h - (tick / max_y) * plot_h
            elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ece6dc"/>')
            elements.append(svg_text(left - 10, y + 4, str(tick), 12, "end"))
        group_w = plot_w / max(1, len(labels))
        bar_w = group_w * 0.28
        for idx, label in enumerate(labels):
            center = left + group_w * (idx + 0.5)
            for offset, value, color in [(-bar_w * 0.6, series_a[idx], color_a), (bar_w * 0.6, series_b[idx], color_b)]:
                h = (value / max_y) * plot_h
                x = center + offset - bar_w / 2
                y = top + plot_h - h
                elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
                elements.append(svg_text(x + bar_w / 2, y - 6, str(value), 11, "middle"))
            elements.append(svg_text(center, top + plot_h + 26, label, 12, "middle"))
        legend_y = top + plot_h + 58
        elements.append(f'<rect x="{left + 60}" y="{legend_y - 12}" width="12" height="12" fill="{color_a}"/>')
        elements.append(svg_text(left + 78, legend_y, label_a, 12))
        elements.append(f'<rect x="{left + 230}" y="{legend_y - 12}" width="12" height="12" fill="{color_b}"/>')
        elements.append(svg_text(left + 248, legend_y, label_b, 12))

    write_svg(output_dir / "mmlu_anchor_outcomes_by_model.svg", width, height, elements)


def heat_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    lo = (242, 248, 228)
    hi = (43, 130, 169)
    rgb = tuple(round(lo[i] + (hi[i] - lo[i]) * value) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def plot_coverage_heatmap_svg(
    summaries: List[Dict[str, Any]], min_valid: int, output_dir: Path
) -> None:
    width, height = 720, 430
    models = list(MODEL_LABELS)
    columns = ["correct", "incorrect"]
    lookup = {(summary["model"], summary["base_type"]): summary for summary in summaries}
    left, top, cell_w, cell_h = 210, 110, 190, 72
    elements = [
        svg_text(width / 2, 42, "Full Paired Chunk Coverage (MMLU)", 22, "middle", "700")
    ]
    for x, label in enumerate(["Correct base", "Incorrect base"]):
        elements.append(svg_text(left + x * cell_w + cell_w / 2, top - 20, label, 14, "middle", "700"))
    for y, model in enumerate(models):
        elements.append(svg_text(left - 18, top + y * cell_h + cell_h / 2 + 5, fmt_model(model), 14, "end", "700"))
        for x, base_type in enumerate(columns):
            summary = lookup[(model, base_type)]
            full = len(full_chunks(summary, min_valid))
            total = summary["num_chunks"]
            ratio = full / total if total else 0.0
            color = heat_color(ratio)
            cell_x = left + x * cell_w
            cell_y = top + y * cell_h
            elements.append(
                f'<rect x="{cell_x}" y="{cell_y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{color}" stroke="#fbfaf7" stroke-width="3"/>'
            )
            elements.append(svg_text(cell_x + cell_w / 2, cell_y + cell_h / 2 + 5, f"{full}/{total}", 18, "middle", "700", "#111111"))
    elements.append(svg_text(left + cell_w, height - 34, "Cell text = full paired chunks / total chunks", 13, "middle"))
    write_svg(output_dir / "mmlu_full_paired_coverage.svg", width, height, elements)


def write_svg_visualizations(
    summaries: List[Dict[str, Any]], min_valid: int, output_dir: Path
) -> None:
    plot_anchor_scatter_svg(summaries, min_valid, output_dir)
    plot_outcome_bars_svg(summaries, min_valid, output_dir)
    plot_coverage_heatmap_svg(summaries, min_valid, output_dir)


def plot_anchor_scatter(
    summaries: List[Dict[str, Any]], min_valid: int, output_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    base_titles = {
        "correct": "Correct base: removal damages true trace",
        "incorrect": "Incorrect base: removal repairs wrong trace",
    }

    for ax, base_type in zip(axes, ["correct", "incorrect"]):
        for summary in summaries:
            if summary["base_type"] != base_type:
                continue
            chunks = full_chunks(summary, min_valid)
            xs = [chunk["default_accuracy"] for chunk in chunks]
            ys = [chunk["forced_accuracy"] for chunk in chunks]
            strengths = []
            for chunk in chunks:
                if base_type == "correct":
                    strength = max(0.0, 1.0 - (chunk["default_accuracy"] or 0.0))
                else:
                    strength = max(0.0, chunk["default_accuracy"] or 0.0)
                strengths.append(30 + 170 * strength)
            ax.scatter(
                xs,
                ys,
                s=strengths,
                alpha=0.58,
                label=fmt_model(summary["model"]),
                color=COLORS.get(summary["model"], "#333333"),
                edgecolors="white",
                linewidths=0.5,
            )

        ax.axvline(0.5, color="#8a8a8a", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="#8a8a8a", linestyle="--", linewidth=1)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(base_titles[base_type], fontsize=12, weight="bold")
        ax.set_xlabel("Default rollout accuracy after chunk removal")
        ax.grid(alpha=0.18)

    axes[0].set_ylabel("Forced-answer rollout accuracy after chunk removal")
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle("Thought Anchor Chunk Map (MMLU)", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "mmlu_anchor_chunk_map.png", dpi=180)
    plt.close(fig)


def plot_outcome_bars(
    summaries: List[Dict[str, Any]], min_valid: int, output_dir: Path
) -> None:
    labels = [fmt_model(summary["model"]) for summary in summaries if summary["base_type"] == "correct"]
    correct_summaries = [summary for summary in summaries if summary["base_type"] == "correct"]
    incorrect_summaries = [summary for summary in summaries if summary["base_type"] == "incorrect"]

    correct_robust = []
    correct_brittle = []
    incorrect_repaired = []
    incorrect_stubborn = []
    for summary in correct_summaries:
        chunks = full_chunks(summary, min_valid)
        correct_robust.append(sum(1 for chunk in chunks if (chunk["default_accuracy"] or 0.0) >= 0.9))
        correct_brittle.append(
            sum(
                1
                for chunk in chunks
                if chunk["default_accuracy"] is not None
                and chunk["default_accuracy"] <= 0.5
            )
        )
    for summary in incorrect_summaries:
        chunks = full_chunks(summary, min_valid)
        incorrect_repaired.append(sum(1 for chunk in chunks if (chunk["default_accuracy"] or 0.0) >= 0.5))
        incorrect_stubborn.append(
            sum(
                1
                for chunk in chunks
                if chunk["default_accuracy"] is not None
                and chunk["default_accuracy"] <= 0.1
            )
        )

    x = list(range(len(labels)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    axes[0].bar([i - width / 2 for i in x], correct_robust, width, label="Robust >=90%", color="#4E937A")
    axes[0].bar([i + width / 2 for i in x], correct_brittle, width, label="Brittle <=50%", color="#D95D39")
    axes[0].set_title("Correct-base chunks", weight="bold")
    axes[0].set_ylabel("Number of full paired chunks")
    axes[0].set_xticks(x, labels)
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x], incorrect_repaired, width, label="Repairs >=50%", color="#3B8EA5")
    axes[1].bar([i + width / 2 for i in x], incorrect_stubborn, width, label="Stubborn <=10%", color="#5D576B")
    axes[1].set_title("Incorrect-base chunks", weight="bold")
    axes[1].set_xticks(x, labels)
    axes[1].legend()

    for ax in axes:
        ax.grid(axis="y", alpha=0.18)

    fig.suptitle("Thought Anchor Outcomes by Model (MMLU)", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "mmlu_anchor_outcomes_by_model.png", dpi=180)
    plt.close(fig)


def plot_coverage_heatmap(
    summaries: List[Dict[str, Any]], min_valid: int, output_dir: Path
) -> None:
    models = list(MODEL_LABELS)
    columns = ["correct", "incorrect"]
    matrix = []
    annotations = []
    lookup = {(summary["model"], summary["base_type"]): summary for summary in summaries}
    for model in models:
        row = []
        annotation_row = []
        for base_type in columns:
            summary = lookup[(model, base_type)]
            full = len(full_chunks(summary, min_valid))
            total = summary["num_chunks"]
            row.append(full / total if total else 0.0)
            annotation_row.append(f"{full}/{total}")
        matrix.append(row)
        annotations.append(annotation_row)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(columns)), ["Correct base", "Incorrect base"])
    ax.set_yticks(range(len(models)), [fmt_model(model) for model in models])
    ax.set_title("Full Paired Chunk Coverage (MMLU)", fontsize=14, weight="bold")
    for y, row in enumerate(annotations):
        for x, text in enumerate(row):
            ax.text(x, y, text, ha="center", va="center", color="#111111", weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Full paired chunk ratio")
    fig.tight_layout()
    fig.savefig(output_dir / "mmlu_full_paired_coverage.png", dpi=180)
    plt.close(fig)


def write_readme(output_dir: Path, image_ext: str) -> None:
    readme = f"""# Thought Anchor Visualizations

Generated from `analysis/qwen_anchor/qualitative_report.json`.

- `mmlu_anchor_chunk_map.{image_ext}`: each point is one full paired chunk. For correct bases, points near the lower-left are stronger positive anchors because removing the chunk damages accuracy. For incorrect bases, points near the upper-right are misleading anchors because removing the chunk repairs the wrong trace.
- `mmlu_anchor_outcomes_by_model.{image_ext}`: compact model comparison of robust/brittle correct-base chunks and repaired/stubborn incorrect-base chunks.
- `mmlu_full_paired_coverage.{image_ext}`: how much of each model/base slice has 100 valid default and 100 valid forced rollouts.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-json", default="analysis/qwen_anchor/qualitative_report.json")
    parser.add_argument("--output-dir", default="analysis/qwen_anchor/visualizations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_json(Path(args.report_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = payload["summaries"]
    min_valid = int(payload.get("min_valid", 100))

    if plt is None:
        write_svg_visualizations(summaries, min_valid, output_dir)
        write_readme(output_dir, "svg")
    else:
        plot_anchor_scatter(summaries, min_valid, output_dir)
        plot_outcome_bars(summaries, min_valid, output_dir)
        plot_coverage_heatmap(summaries, min_valid, output_dir)
        write_readme(output_dir, "png")
    print(f"Wrote visualizations to {output_dir}")


if __name__ == "__main__":
    main()
