import json
import os
import re
import pandas as pd

PROB_PATH = "data/math_problems.jsonl"
GEN_PATH  = "logs/generations_seed2.jsonl"
OUT_CSV   = "results/metrics.csv"


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


# --- Gold answer extraction from MATH solution ---
# Prefer the last \boxed{...} (supports nested braces).
def extract_gold_from_solution(sol: str):
    if sol is None:
        return None
    idx = sol.rfind("\\boxed{")
    if idx == -1:
        return None

    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(sol) and depth > 0:
        ch = sol[i]
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
    return "".join(out).strip()


# --- Prediction extraction from model output ---
def extract_pred_from_text(text: str):
    if text is None:
        return None

    # 1) \boxed{...} (common for SFT/DRPO)
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()

    # 2) "final answer is ..." (common for base)
    m = re.search(r"final answer is\s*\$?(.+?)\$?[\.\n]", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 3) "answer is ..." (fallback)
    m = re.search(r"answer is\s*\$?(.+?)\$?[\.\n]", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 4) last number-like token (proposal-level fallback)
    m = re.findall(r"(-?\d+\.?\d*)", text)
    if m:
        return m[-1].strip()

    return None


def normalize(ans: str):
    """Lightweight normalization for proposal-stage scoring."""
    if ans is None:
        return None
    ans = ans.strip()

    # remove common latex wrappers/spaces
    ans = ans.replace("$", "")
    ans = ans.replace("\\,", "")
    ans = ans.replace(" ", "")
    ans = ans.replace("\\left", "").replace("\\right", "")

    # unwrap \boxed{...} if still present (non-nested)
    ans = re.sub(r"\\boxed\{([^}]*)\}", r"\1", ans)

    # remove commas in numbers and trailing punctuation
    ans = ans.replace(",", "")
    ans = ans.strip(".")
    return ans


def main():
    probs = load_jsonl(PROB_PATH)
    # Map problem_id -> gold answer
    gold_map = {}
    for p in probs:
        gold_raw = extract_gold_from_solution(p.get("solution", ""))
        # If no boxed found, fallback to last number in solution
        if gold_raw is None:
            nums = re.findall(r"(-?\d+\.?\d*)", p.get("solution", ""))
            gold_raw = nums[-1] if nums else None
        gold_map[p["problem_id"]] = normalize(gold_raw)

    gens = load_jsonl(GEN_PATH)

    # Attach pred + correctness
    for g in gens:
        pid = g["problem_id"]
        gold = gold_map.get(pid)
        pred = normalize(extract_pred_from_text(g.get("output_text", "")))
        g["gold"] = gold
        g["pred_norm"] = pred
        g["is_correct"] = (gold is not None and pred is not None and pred == gold)

    df = pd.DataFrame(gens)

    # infer k from max sample_id + 1
    k = int(df["sample_id"].max() + 1) if len(df) else 0

    # pass@1 = sample_id==0 accuracy
    pass1 = (
        df[df["sample_id"] == 0]
        .groupby("model_tag")["is_correct"]
        .mean()
        .reset_index(name="pass_at_1")
    )

    # pass@k = any correct among k samples per problem
    passk = (
        df.groupby(["model_tag", "problem_id"])["is_correct"]
        .any()
        .groupby("model_tag")
        .mean()
        .reset_index(name="pass_at_k")
    )

    summary = pass1.merge(passk, on="model_tag", how="outer")
    summary["k"] = k

    os.makedirs("results", exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)

    print(summary)
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
