#!/usr/bin/env python3
"""Reproducible EDA pipeline for the MATH dataset (Hendrycks et al.)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np


SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

ANSWER_TYPES = ["int", "fraction", "expression", "set_or_list", "other", "missing"]

SUBJECT_DISPLAY = {
    "algebra": "Algebra",
    "counting_and_probability": "Counting \\& Probability",
    "geometry": "Geometry",
    "intermediate_algebra": "Intermediate Algebra",
    "number_theory": "Number Theory",
    "prealgebra": "Prealgebra",
    "precalculus": "Precalculus",
    "unknown": "Unknown",
}

ANSWER_DISPLAY = {
    "int": "Integer",
    "fraction": "Fraction",
    "expression": "Expression",
    "set_or_list": "Set/List",
    "other": "Other",
    "missing": "Missing",
}


def log(msg: str) -> None:
    print(f"[math_eda] {msg}")


def normalize_subject_name(raw: str) -> str:
    x = raw.strip().lower()
    x = x.replace("&", "and")
    x = re.sub(r"[^\w]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def canonical_subject(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    norm = normalize_subject_name(value)
    aliases = {
        "counting_probability": "counting_and_probability",
        "counting_and_prob": "counting_and_probability",
        "counting_and_probability": "counting_and_probability",
        "number_theory": "number_theory",
        "intermediate_algebra": "intermediate_algebra",
    }
    if norm in aliases:
        norm = aliases[norm]
    return norm if norm in SUBJECTS else "unknown"


def parse_level(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        iv = int(value)
        return iv if 1 <= iv <= 5 else None
    if isinstance(value, str):
        m = re.search(r"([1-5])", value)
        if m:
            return int(m.group(1))
    return None


def has_asymptote(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    lower = text.lower()
    return (
        "\\begin{asy}" in lower
        or "\\begin{asymptote}" in lower
        or "asymptote" in lower
    )


def extract_last_boxed(solution: str) -> str | None:
    boxed_spans: list[str] = []
    for m in re.finditer(r"\\boxed\s*{", solution):
        start = m.end() - 1
        depth = 0
        for idx in range(start, len(solution)):
            ch = solution[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    boxed_spans.append(solution[start + 1 : idx].strip())
                    break
    if boxed_spans:
        return boxed_spans[-1] or None
    return None


def extract_answer(example: dict[str, Any], solution_text: str) -> str | None:
    for key in ("answer", "extracted_solution", "final_answer"):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    boxed = extract_last_boxed(solution_text)
    return boxed.strip() if isinstance(boxed, str) and boxed.strip() else None


def answer_type(answer: str | None) -> str:
    if answer is None or not answer.strip():
        return "missing"
    a = answer.strip().strip("$")
    if re.fullmatch(r"-?\d+", a):
        return "int"
    if "\\frac" in a or re.fullmatch(r"-?\d+\s*/\s*-?\d+", a):
        return "fraction"
    if ("{" in a and "}" in a) or ("," in a and len(a) <= 120):
        return "set_or_list"
    if re.search(r"[A-Za-z]", a) or any(op in a for op in ("\\sqrt", "\\pi", "^", "_")):
        return "expression"
    return "other"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100.0 * count / total


def summarize_lengths(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"median": 0.0, "q25": 0.0, "q75": 0.0, "iqr": 0.0, "mean": 0.0}
    arr = np.asarray(lengths, dtype=float)
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "median": float(np.median(arr)),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "mean": float(np.mean(arr)),
    }


def proxy_token_len(text: str) -> int:
    if not text:
        return 0
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return len(tokens)


def build_token_counter(
    tokenizer_name: str | None,
) -> tuple[Callable[[str], int], str]:
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer

            log(f"Loading tokenizer: {tokenizer_name}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

            def count_tokens(text: str) -> int:
                if not text:
                    return 0
                return len(tokenizer.encode(text, add_special_tokens=False))

            return count_tokens, f"hf_tokenizer:{tokenizer_name}"
        except Exception as exc:
            log(f"Tokenizer load failed ({exc}); fallback to proxy tokenization.")
    return proxy_token_len, "proxy_regex"


def safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def import_datasets_api() -> tuple[Any, Any, Any, Any]:
    try:
        from datasets import DatasetDict, concatenate_datasets, load_dataset

        return DatasetDict, concatenate_datasets, load_dataset, None
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Install with `pip install -U datasets numpy`."
        ) from exc


def add_subject_column(ds: Dataset, subject: str) -> Dataset:
    if "type" in ds.column_names:
        return ds
    return ds.add_column("type", [subject] * len(ds))


def ensure_train_test(ds: Any, dataset_dict_cls: Any) -> Any:
    if "train" in ds and "test" in ds:
        return dataset_dict_cls({"train": ds["train"], "test": ds["test"]})
    if "train" in ds and "validation" in ds:
        return dataset_dict_cls({"train": ds["train"], "test": ds["validation"]})
    raise ValueError(f"Dataset does not provide train/test-like splits: {list(ds.keys())}")


def load_with_fallback(primary_repo: str, primary_config: str | None) -> tuple[DatasetDict, str]:
    DatasetDict, concatenate_datasets, load_dataset, _ = import_datasets_api()
    errors: list[str] = []

    try:
        log(f"Trying dataset: {primary_repo} config={primary_config}")
        ds = load_dataset(primary_repo, primary_config)
        ds = ensure_train_test(ds, DatasetDict)
        return ds, f"{primary_repo}:{primary_config or 'default'}"
    except Exception as exc:
        errors.append(f"{primary_repo}:{primary_config or 'default'} -> {exc}")

    try:
        repo = "HuggingFaceTB/MATH"
        log(f"Trying fallback dataset: {repo}")
        ds = load_dataset(repo)
        ds = ensure_train_test(ds, DatasetDict)
        return ds, f"{repo}:default"
    except Exception as exc:
        errors.append(f"HuggingFaceTB/MATH:default -> {exc}")

    try:
        repo = "EleutherAI/hendrycks_math"
        log(f"Trying fallback dataset (subject-concat): {repo}")
        train_parts: list[Dataset] = []
        test_parts: list[Dataset] = []
        for subject in SUBJECTS:
            part = load_dataset(repo, subject)
            part = ensure_train_test(part, DatasetDict)
            train_parts.append(add_subject_column(part["train"], subject))
            test_parts.append(add_subject_column(part["test"], subject))
        ds = DatasetDict(
            {"train": concatenate_datasets(train_parts), "test": concatenate_datasets(test_parts)}
        )
        return ds, f"{repo}:subject_concat"
    except Exception as exc:
        errors.append(f"EleutherAI/hendrycks_math:subject_concat -> {exc}")

    raise RuntimeError("All dataset loading attempts failed:\n- " + "\n- ".join(errors))


def distribution_with_pct(
    counter: Counter[str], keys: list[str], total: int
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for k in keys:
        c = int(counter.get(k, 0))
        out[k] = {"count": c, "pct": pct(c, total)}
    return out


def compute_split_stats(
    split_ds: Dataset, token_len: Callable[[str], int]
) -> dict[str, Any]:
    n = len(split_ds)
    subject_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    answer_counter: Counter[str] = Counter()
    asym_count = 0

    problem_lens: list[int] = []
    solution_lens: list[int] = []

    for ex in split_ds:
        problem = safe_str(ex.get("problem"))
        solution = safe_str(ex.get("solution"))

        subject = canonical_subject(ex.get("type") or ex.get("subject"))
        subject_counter[subject] += 1

        level = parse_level(ex.get("level"))
        level_key = str(level) if level is not None else "missing"
        level_counter[level_key] += 1

        if has_asymptote(problem) or has_asymptote(solution):
            asym_count += 1

        problem_lens.append(token_len(problem))
        solution_lens.append(token_len(solution))

        ans = extract_answer(ex, solution)
        answer_counter[answer_type(ans)] += 1

    subject_dist = distribution_with_pct(subject_counter, SUBJECTS + ["unknown"], n)
    level_dist = distribution_with_pct(level_counter, ["1", "2", "3", "4", "5", "missing"], n)
    answer_dist = distribution_with_pct(answer_counter, ANSWER_TYPES, n)

    return {
        "n_examples": n,
        "subject_distribution": subject_dist,
        "level_distribution": level_dist,
        "answer_type_distribution": answer_dist,
        "asymptote": {"count": asym_count, "pct": pct(asym_count, n)},
        "problem_length": summarize_lengths(problem_lens),
        "solution_length": summarize_lengths(solution_lens),
    }


def fmt_float(x: float) -> str:
    return f"{x:.2f}"


def biggest_smallest_subject(subject_dist: dict[str, dict[str, float | int]]) -> tuple[str, str]:
    filtered = [(k, int(v["count"])) for k, v in subject_dist.items() if k != "unknown"]
    if not filtered:
        return "unknown", "unknown"
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[0][0], filtered[-1][0]


def render_distribution_table(
    title: str,
    row_order: list[str],
    train_dist: dict[str, dict[str, float | int]],
    test_dist: dict[str, dict[str, float | int]],
    display_map: dict[str, str] | None = None,
) -> str:
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{latex_escape(title)}}}",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Category & Train Count & Train (\\%) & Test Count & Test (\\%)\\\\",
        "\\hline",
    ]
    for key in row_order:
        label = display_map[key] if display_map and key in display_map else latex_escape(key)
        tr = train_dist.get(key, {"count": 0, "pct": 0.0})
        te = test_dist.get(key, {"count": 0, "pct": 0.0})
        lines.append(
            f"{label} & {int(tr['count'])} & {fmt_float(float(tr['pct']))} & "
            f"{int(te['count'])} & {fmt_float(float(te['pct']))}\\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def generate_latex_summary(stats: dict[str, Any]) -> str:
    train = stats["splits"]["train"]
    test = stats["splits"]["test"]
    source = stats["source"]
    token_mode = stats["tokenization"]

    train_sub = train["subject_distribution"]
    big_sub, small_sub = biggest_smallest_subject(train_sub)

    train_levels = train["level_distribution"]
    train_ans = train["answer_type_distribution"]
    train_asym = train["asymptote"]

    p_train = train["problem_length"]
    p_test = test["problem_length"]
    s_train = train["solution_length"]
    s_test = test["solution_length"]

    paragraph = (
        "\\subsection{Exploratory Data Analysis on MATH}\n"
        f"We analyzed the MATH benchmark loaded from \\texttt{{{latex_escape(source)}}}, "
        f"with {stats['total_examples']} examples in total: {train['n_examples']} train and "
        f"{test['n_examples']} test. On the training split, the largest subject is "
        f"{SUBJECT_DISPLAY.get(big_sub, latex_escape(big_sub))} "
        f"({fmt_float(float(train_sub[big_sub]['pct']))}\\%), while the smallest is "
        f"{SUBJECT_DISPLAY.get(small_sub, latex_escape(small_sub))} "
        f"({fmt_float(float(train_sub[small_sub]['pct']))}\\%). "
        "Difficulty is spread across Levels 1--5 with training proportions of "
        f"L1={fmt_float(float(train_levels['1']['pct']))}\\%, "
        f"L2={fmt_float(float(train_levels['2']['pct']))}\\%, "
        f"L3={fmt_float(float(train_levels['3']['pct']))}\\%, "
        f"L4={fmt_float(float(train_levels['4']['pct']))}\\%, and "
        f"L5={fmt_float(float(train_levels['5']['pct']))}\\%.\n\n"
        "For text length, training problems have median token length "
        f"{fmt_float(float(p_train['median']))} with IQR "
        f"[{fmt_float(float(p_train['q25']))}, {fmt_float(float(p_train['q75']))}], "
        "and training solutions have median "
        f"{fmt_float(float(s_train['median']))} with IQR "
        f"[{fmt_float(float(s_train['q25']))}, {fmt_float(float(s_train['q75']))}]. "
        "On the test split, problem median/IQR is "
        f"{fmt_float(float(p_test['median']))} "
        f"[{fmt_float(float(p_test['q25']))}, {fmt_float(float(p_test['q75']))}], and "
        "solution median/IQR is "
        f"{fmt_float(float(s_test['median']))} "
        f"[{fmt_float(float(s_test['q25']))}, {fmt_float(float(s_test['q75']))}]. "
        f"Asymptote appears in {fmt_float(float(train_asym['pct']))}\\% of training examples. "
        "Using coarse answer-type heuristics on training data, we observe "
        f"integer={fmt_float(float(train_ans['int']['pct']))}\\%, "
        f"fraction={fmt_float(float(train_ans['fraction']['pct']))}\\%, "
        f"expression={fmt_float(float(train_ans['expression']['pct']))}\\%, "
        f"set/list={fmt_float(float(train_ans['set_or_list']['pct']))}\\%, "
        f"other={fmt_float(float(train_ans['other']['pct']))}\\%, "
        f"missing={fmt_float(float(train_ans['missing']['pct']))}\\%.\n\n"
        f"\\noindent\\textit{{Data source used: {latex_escape(source)}; token counting mode: "
        f"{latex_escape(token_mode)}.}}"
    )
    return paragraph + "\n"


def write_outputs(outdir: Path, stats: dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    stats_path = outdir / "eda_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    summary_path = outdir / "eda_summary.tex"
    summary_path.write_text(generate_latex_summary(stats), encoding="utf-8")

    train = stats["splits"]["train"]
    test = stats["splits"]["test"]

    tables = [
        render_distribution_table(
            "Subject Distribution",
            SUBJECTS + ["unknown"],
            train["subject_distribution"],
            test["subject_distribution"],
            display_map=SUBJECT_DISPLAY,
        ),
        render_distribution_table(
            "Difficulty Distribution",
            ["1", "2", "3", "4", "5", "missing"],
            train["level_distribution"],
            test["level_distribution"],
            display_map={
                "1": "Level 1",
                "2": "Level 2",
                "3": "Level 3",
                "4": "Level 4",
                "5": "Level 5",
                "missing": "Missing",
            },
        ),
        render_distribution_table(
            "Answer Type Distribution",
            ANSWER_TYPES,
            train["answer_type_distribution"],
            test["answer_type_distribution"],
            display_map=ANSWER_DISPLAY,
        ),
    ]
    (outdir / "eda_tables.tex").write_text("\n".join(tables), encoding="utf-8")

    log(f"Wrote {stats_path}")
    log(f"Wrote {summary_path}")
    log(f"Wrote {outdir / 'eda_tables.tex'}")


def plot_distributions(outdir: Path, stats: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"Plot requested but matplotlib is unavailable ({exc}). Skipping plots.")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    def bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, values)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def hist_plot(path: Path, values: list[float], title: str, xlabel: str) -> None:
        plt.figure(figsize=(8, 4.5))
        plt.hist(values, bins=40)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    for split in ("train", "test"):
        split_stats = stats["splits"][split]
        subject_vals = split_stats["subject_distribution"]
        level_vals = split_stats["level_distribution"]
        problem_summary = split_stats["problem_length"]
        solution_summary = split_stats["solution_length"]

        bar_plot(
            outdir / f"subject_dist_{split}.png",
            [SUBJECT_DISPLAY[s] for s in SUBJECTS],
            [float(subject_vals[s]["pct"]) for s in SUBJECTS],
            f"Subject Distribution ({split})",
            "Percent",
        )
        bar_plot(
            outdir / f"level_dist_{split}.png",
            [f"L{i}" for i in range(1, 6)],
            [float(level_vals[str(i)]["pct"]) for i in range(1, 6)],
            f"Difficulty Distribution ({split})",
            "Percent",
        )
        # Reconstruct approximate histogram-compatible arrays from summary stats is not valid;
        # histogram files are generated from raw lengths only in the main function.
        _ = problem_summary, solution_summary

    raw = stats.get("raw_lengths", {})
    for split in ("train", "test"):
        if split in raw:
            hist_plot(
                outdir / f"problem_len_hist_{split}.png",
                raw[split]["problem"],
                f"Problem Token Length Histogram ({split})",
                "Token Length",
            )
            hist_plot(
                outdir / f"solution_len_hist_{split}.png",
                raw[split]["solution"],
                f"Solution Token Length Histogram ({split})",
                "Token Length",
            )

    log("Plots generated.")


def print_console_summary(stats: dict[str, Any]) -> None:
    train = stats["splits"]["train"]
    test = stats["splits"]["test"]
    print("\n=== MATH EDA SUMMARY ===")
    print(f"Source: {stats['source']}")
    print(f"Token mode: {stats['tokenization']}")
    print(
        f"Size: total={stats['total_examples']} "
        f"(train={train['n_examples']}, test={test['n_examples']})"
    )
    print("\n[Train subject distribution]")
    for s in SUBJECTS + ["unknown"]:
        v = train["subject_distribution"][s]
        print(f"- {s}: {v['count']} ({v['pct']:.2f}%)")
    print("\n[Train level distribution]")
    for lv in ["1", "2", "3", "4", "5", "missing"]:
        v = train["level_distribution"][lv]
        print(f"- L{lv if lv != 'missing' else 'missing'}: {v['count']} ({v['pct']:.2f}%)")
    print("\n[Train answer type distribution]")
    for at in ANSWER_TYPES:
        v = train["answer_type_distribution"][at]
        print(f"- {at}: {v['count']} ({v['pct']:.2f}%)")
    print(
        "\n[Lengths]\n"
        f"- Train problem median/IQR: {train['problem_length']['median']:.2f} "
        f"[{train['problem_length']['q25']:.2f}, {train['problem_length']['q75']:.2f}]\n"
        f"- Train solution median/IQR: {train['solution_length']['median']:.2f} "
        f"[{train['solution_length']['q25']:.2f}, {train['solution_length']['q75']:.2f}]\n"
        f"- Test problem median/IQR: {test['problem_length']['median']:.2f} "
        f"[{test['problem_length']['q25']:.2f}, {test['problem_length']['q75']:.2f}]\n"
        f"- Test solution median/IQR: {test['solution_length']['median']:.2f} "
        f"[{test['solution_length']['q25']:.2f}, {test['solution_length']['q75']:.2f}]"
    )
    print(f"\nAsymptote rate (train): {train['asymptote']['pct']:.2f}%")
    print("========================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA pipeline for MATH dataset.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="jeggers/competition_math",
        help="Primary HF dataset repo id.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="original",
        help="Primary dataset config.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional HF tokenizer for exact token counting.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Enable plot generation (requires matplotlib).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="plots",
        help="Disable plot generation (default).",
    )
    parser.set_defaults(plots=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    token_len_fn, token_mode = build_token_counter(args.tokenizer)
    ds, source = load_with_fallback(args.dataset, args.config)
    train_ds = ds["train"]
    test_ds = ds["test"]
    log(f"Using dataset source: {source}")
    log(f"Split sizes: train={len(train_ds)}, test={len(test_ds)}")

    # Keep raw lengths for optional histogram output.
    raw_lengths: dict[str, dict[str, list[int]]] = {
        "train": {"problem": [], "solution": []},
        "test": {"problem": [], "solution": []},
    }

    def compute_with_raw(split_name: str, split_ds: Dataset) -> dict[str, Any]:
        n = len(split_ds)
        subject_counter: Counter[str] = Counter()
        level_counter: Counter[str] = Counter()
        answer_counter: Counter[str] = Counter()
        asym_count = 0
        problem_lens: list[int] = []
        solution_lens: list[int] = []

        for ex in split_ds:
            problem = safe_str(ex.get("problem"))
            solution = safe_str(ex.get("solution"))

            p_len = token_len_fn(problem)
            s_len = token_len_fn(solution)
            problem_lens.append(p_len)
            solution_lens.append(s_len)
            raw_lengths[split_name]["problem"].append(p_len)
            raw_lengths[split_name]["solution"].append(s_len)

            subject = canonical_subject(ex.get("type") or ex.get("subject"))
            subject_counter[subject] += 1

            level = parse_level(ex.get("level"))
            level_counter[str(level) if level is not None else "missing"] += 1

            if has_asymptote(problem) or has_asymptote(solution):
                asym_count += 1

            ans = extract_answer(ex, solution)
            answer_counter[answer_type(ans)] += 1

        return {
            "n_examples": n,
            "subject_distribution": distribution_with_pct(subject_counter, SUBJECTS + ["unknown"], n),
            "level_distribution": distribution_with_pct(level_counter, ["1", "2", "3", "4", "5", "missing"], n),
            "answer_type_distribution": distribution_with_pct(answer_counter, ANSWER_TYPES, n),
            "asymptote": {"count": asym_count, "pct": pct(asym_count, n)},
            "problem_length": summarize_lengths(problem_lens),
            "solution_length": summarize_lengths(solution_lens),
        }

    log("Computing train split statistics...")
    train_stats = compute_with_raw("train", train_ds)
    log("Computing test split statistics...")
    test_stats = compute_with_raw("test", test_ds)

    stats = {
        "source": source,
        "tokenization": token_mode,
        "total_examples": len(train_ds) + len(test_ds),
        "splits": {"train": train_stats, "test": test_stats},
        "raw_lengths": raw_lengths if args.plots else {},
    }

    write_outputs(outdir, stats)
    if args.plots:
        plot_distributions(outdir, stats)
    print_console_summary(stats)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log(f"Failed: {exc}")
        sys.exit(1)
