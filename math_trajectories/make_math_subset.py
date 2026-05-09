"""Stratified sampling of 50 MATH problems from qwedsacf/competition_math."""

import json
import os
import random
import re
from collections import defaultdict
from datasets import load_dataset

DATASET_NAME = "qwedsacf/competition_math"
SPLIT = "train"
SEED = 42
N = 50
OUT_PATH = "data/math_subset.jsonl"


def extract_boxed(solution: str) -> str | None:
    """Extract the last \\boxed{...} from a solution string (nested-brace aware)."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(solution) and depth > 0:
        ch = solution[i]
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
    return "".join(out).strip() or None


def main():
    random.seed(SEED)
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    # Group indices by subject
    by_subject = defaultdict(list)
    for i, ex in enumerate(ds):
        subj = ex.get("type", "Unknown")
        by_subject[subj].append(i)

    total = sum(len(v) for v in by_subject.values())

    # Proportional allocation, at least 1 per subject
    allocations = {}
    remaining = N
    for subj, idxs in sorted(by_subject.items()):
        alloc = max(1, round(len(idxs) / total * N))
        allocations[subj] = alloc
        remaining -= alloc

    # Adjust if rounding caused over/under allocation
    subjects_sorted = sorted(allocations.keys(), key=lambda s: len(by_subject[s]), reverse=True)
    i = 0
    while remaining != 0:
        subj = subjects_sorted[i % len(subjects_sorted)]
        if remaining > 0:
            allocations[subj] += 1
            remaining -= 1
        elif allocations[subj] > 1:
            allocations[subj] -= 1
            remaining += 1
        i += 1

    # Sample within each subject
    sampled_indices = []
    for subj, idxs in sorted(by_subject.items()):
        random.shuffle(idxs)
        sampled_indices.extend(idxs[: allocations[subj]])

    # Write output
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for idx in sampled_indices:
            ex = ds[idx]
            sol = ex.get("solution", "")
            answer = extract_boxed(sol)
            # Fallback: last number in solution
            if answer is None:
                nums = re.findall(r"(-?\d+\.?\d*)", sol)
                answer = nums[-1] if nums else None

            row = {
                "problem_id": f"math_{SPLIT}_{idx}",
                "problem": ex["problem"],
                "solution": sol,
                "answer": answer,
                "subject": ex.get("type", None),
                "level": ex.get("level", None),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Print summary
    print(f"Wrote {len(sampled_indices)} problems to {OUT_PATH}")
    for subj in sorted(allocations.keys()):
        print(f"  {subj}: {allocations[subj]}")


if __name__ == "__main__":
    main()
