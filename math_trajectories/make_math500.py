"""Build a JSONL copy of HuggingFaceH4/MATH-500 for trajectory evaluation."""

import json
import os
import re
from collections import Counter

from datasets import load_dataset

DATASET_NAME = "HuggingFaceH4/MATH-500"
SPLIT = "test"
OUT_PATH = "data/math500.jsonl"


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


def normalize_level(level) -> str | int | None:
    """Preserve level information while keeping legacy output readable."""
    if level is None:
        return None
    if isinstance(level, int):
        return level
    m = re.search(r"(\d+)", str(level))
    if m:
        return int(m.group(1))
    return level


def main():
    ds = load_dataset(DATASET_NAME, split=SPLIT)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    subject_counts = Counter()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in ds:
            solution = ex.get("solution", "")
            answer = ex.get("answer") or extract_boxed(solution)
            if answer is None:
                nums = re.findall(r"(-?\d+\.?\d*)", solution)
                answer = nums[-1] if nums else None

            subject = ex.get("subject")
            row = {
                "problem_id": ex["unique_id"],
                "problem": ex["problem"],
                "solution": solution,
                "answer": answer,
                "subject": subject,
                "level": normalize_level(ex.get("level")),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            subject_counts[subject or "Unknown"] += 1

    print(f"Wrote {len(ds)} problems to {OUT_PATH}")
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count}")


if __name__ == "__main__":
    main()
