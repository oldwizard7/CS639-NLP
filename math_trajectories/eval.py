"""Evaluate generation logs: pass@1, pass@k, distinct answers, output length."""

import argparse
import csv
import glob
import json
import os
import re
from math import comb


# ---------------------------------------------------------------------------
# Answer extraction (reused from hw3_math_pilot/scripts/eval.py)
# ---------------------------------------------------------------------------

def extract_last_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} span from text (nested-brace aware)."""
    if text is None:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip() or None


def extract_gold_from_solution(sol: str) -> str | None:
    """Extract last \\boxed{...} from solution (nested-brace aware)."""
    return extract_last_boxed(sol)


def cleanup_extracted_answer(ans: str | None) -> str | None:
    """Trim common math-mode wrappers around an extracted answer."""
    if ans is None:
        return None
    ans = ans.strip()
    if ans.startswith("\\[") and ans.endswith("\\]"):
        ans = ans[2:-2].strip()
    if ans.startswith("\\(") and ans.endswith("\\)"):
        ans = ans[2:-2].strip()
    if ans.startswith("$") and ans.endswith("$"):
        ans = ans[1:-1].strip()
    ans = ans.rstrip(".")
    return ans or None


def extract_pred_from_text(text: str) -> str | None:
    """Extract predicted answer from model output."""
    if text is None:
        return None

    # 1) \boxed{...}
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed

    # 2) answer markers on the final lines
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue

        m = re.search(r"Final Answer:\s*(.+)$", stripped, flags=re.IGNORECASE)
        if m:
            return cleanup_extracted_answer(m.group(1))

        m = re.search(r"final answer is\s*(.+)$", stripped, flags=re.IGNORECASE)
        if m:
            return cleanup_extracted_answer(m.group(1))

        m = re.search(r"answer is\s*(.+)$", stripped, flags=re.IGNORECASE)
        if m:
            return cleanup_extracted_answer(m.group(1))

    # 3) last number fallback
    m = re.findall(r"(-?\d+\.?\d*)", text)
    if m:
        return m[-1].strip()

    return None


def normalize(ans: str) -> str | None:
    """Lightweight normalization for answer comparison."""
    if ans is None:
        return None
    ans = ans.strip()
    boxed = extract_last_boxed(ans)
    if boxed is not None and ans.startswith("\\boxed{"):
        ans = boxed
    ans = ans.replace("$", "")
    ans = ans.replace("\\,", "")
    ans = ans.replace("\\!", "")
    ans = ans.replace(" ", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    ans = ans.replace(",", "")
    ans = ans.strip(".")
    return ans


# ---------------------------------------------------------------------------
# pass@k estimator (Chen et al. 2021)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    if k > n:
        return 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(problems: list[dict], generations: list[dict], k: int):
    """Compute per-problem and per-model metrics."""
    # Build gold map
    gold_map = {}
    prob_meta = {}
    for p in problems:
        pid = p["problem_id"]
        gold_raw = p.get("answer") or extract_gold_from_solution(p.get("solution", ""))
        if gold_raw is None:
            nums = re.findall(r"(-?\d+\.?\d*)", p.get("solution", ""))
            gold_raw = nums[-1] if nums else None
        gold_map[pid] = normalize(gold_raw)
        prob_meta[pid] = {"subject": p.get("subject"), "level": p.get("level")}

    # Group generations by (model_tag, problem_id)
    groups: dict[tuple[str, str], list[dict]] = {}
    for g in generations:
        key = (g["model_tag"], g["problem_id"])
        groups.setdefault(key, []).append(g)

    per_problem_rows = []
    for (model_tag, pid), samples in sorted(groups.items()):
        gold = gold_map.get(pid)
        preds = []
        correct_count = 0
        total_len = 0

        for s in samples:
            raw = s.get("raw_output", "")
            pred = normalize(extract_pred_from_text(raw))
            preds.append(pred)
            if gold is not None and pred is not None and pred == gold:
                correct_count += 1
            total_len += len(raw) if raw else 0

        n = len(samples)
        distinct = len(set(p for p in preds if p is not None))

        per_problem_rows.append({
            "model_tag": model_tag,
            "problem_id": pid,
            "subject": prob_meta.get(pid, {}).get("subject"),
            "level": prob_meta.get(pid, {}).get("level"),
            "n": n,
            "c": correct_count,
            "pass_at_1": pass_at_k(n, correct_count, 1),
            f"pass_at_{k}": pass_at_k(n, correct_count, k),
            "distinct_answers": distinct,
            "avg_output_len": round(total_len / n, 1) if n > 0 else 0,
            "gold": gold,
            "preds": json.dumps(preds),
        })

    # Aggregate per model
    model_summaries = {}
    for row in per_problem_rows:
        tag = row["model_tag"]
        if tag not in model_summaries:
            model_summaries[tag] = {
                "pass_at_1": [],
                f"pass_at_{k}": [],
                "distinct_answers": [],
                "avg_output_len": [],
            }
        m = model_summaries[tag]
        m["pass_at_1"].append(row["pass_at_1"])
        m[f"pass_at_{k}"].append(row[f"pass_at_{k}"])
        m["distinct_answers"].append(row["distinct_answers"])
        m["avg_output_len"].append(row["avg_output_len"])

    summary_rows = []
    for tag, m in sorted(model_summaries.items()):
        n_problems = len(m["pass_at_1"])
        summary_rows.append({
            "model_tag": tag,
            "pass_at_1": round(sum(m["pass_at_1"]) / n_problems, 4),
            f"pass_at_{k}": round(sum(m[f"pass_at_{k}"]) / n_problems, 4),
            "mean_distinct_answers": round(sum(m["distinct_answers"]) / n_problems, 4),
            "mean_output_len": round(sum(m["avg_output_len"]) / n_problems, 1),
            "n_problems": n_problems,
            "k": k,
        })

    return per_problem_rows, summary_rows


def write_csv(rows: list[dict], path: str):
    """Write list of dicts as CSV."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate math generation logs")
    parser.add_argument("--data", default="data/math_subset.jsonl", help="Gold problems JSONL")
    parser.add_argument("--generations", nargs="+", default=["logs/generations_*.jsonl"],
                        help="Generation JSONL files (supports glob)")
    parser.add_argument("--out_dir", default="results/", help="Output directory")
    parser.add_argument("--k", type=int, default=3, help="k for pass@k")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    problems = load_jsonl(args.data)
    print(f"Loaded {len(problems)} problems from {args.data}")

    # Expand globs
    gen_files = []
    for pattern in args.generations:
        expanded = glob.glob(pattern)
        if expanded:
            gen_files.extend(expanded)
        elif os.path.exists(pattern):
            gen_files.append(pattern)
    gen_files = sorted(set(gen_files))

    if not gen_files:
        print("No generation files found!")
        return

    all_generations = []
    for gf in gen_files:
        rows = load_jsonl(gf)
        all_generations.extend(rows)
        print(f"  Loaded {len(rows)} rows from {gf}")

    per_problem, summary = evaluate(problems, all_generations, args.k)

    # Write per-model JSON files
    model_tags = set(r["model_tag"] for r in summary)
    for tag in model_tags:
        tag_summary = [r for r in summary if r["model_tag"] == tag][0]
        json_path = os.path.join(args.out_dir, f"metrics_{tag}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tag_summary, f, indent=2)
        print(f"Wrote {json_path}")

    # Write summary CSV
    csv_path = os.path.join(args.out_dir, "summary.csv")
    write_csv(summary, csv_path)
    print(f"Wrote {csv_path}")

    # Write per-problem CSV
    per_prob_path = os.path.join(args.out_dir, "per_problem.csv")
    write_csv(per_problem, per_prob_path)
    print(f"Wrote {per_prob_path}")

    # Print summary
    print("\n=== Summary ===")
    for row in summary:
        print(f"  {row['model_tag']:>8s}  pass@1={row['pass_at_1']:.4f}  "
              f"pass@{args.k}={row[f'pass_at_{args.k}']:.4f}  "
              f"distinct={row['mean_distinct_answers']:.2f}  "
              f"len={row['mean_output_len']:.0f}")


if __name__ == "__main__":
    main()
