import json
import os
import random
from datasets import load_dataset

DATASET_NAME = "qwedsacf/competition_math"
SPLIT = "train"          # this dataset mirror only provides 'train'
SEED = 0
N = 20
OUT_PATH = "data/math_problems.jsonl"

def main():
    random.seed(SEED)
    ds = load_dataset(DATASET_NAME, split=SPLIT)  # 12,500 rows

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[:N]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i in idxs:
            ex = ds[i]
            row = {
                "problem_id": f"math_{SPLIT}_{i}",
                "source_dataset": DATASET_NAME,
                "source_split": SPLIT,
                "source_index": int(i),
                "subject": ex.get("type", None),
                "level": ex.get("level", None),
                "problem": ex["problem"],
                "solution": ex["solution"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {N} problems to {OUT_PATH} from {DATASET_NAME}:{SPLIT}")

if __name__ == "__main__":
    main()
