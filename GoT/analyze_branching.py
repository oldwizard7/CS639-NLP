"""Offline Graph-of-Thoughts analysis on existing Math500 trajectories.

Treats each model's K=3 samples as branches and evaluates Score+Select+Aggregate
strategies without any new GPU inference:

  1. Per-model pass@1 / pass@3 / maj@3 (self-consistency on own branches).
  2. Diversity-killed set: problems base solves at least once but DRPO never does.
  3. Cross-model branching: pool branches across models and vote.

Outputs CSVs into results/ alongside this file.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from itertools import combinations

# Reuse extraction/normalization from the trajectory project so we stay aligned
# with the original eval.py numbers.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "math_trajectories"))
from eval import (  # noqa: E402
    extract_gold_from_solution,
    extract_pred_from_text,
    normalize,
    pass_at_k,
)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_gold_map(problems):
    gold = {}
    meta = {}
    for p in problems:
        pid = p["problem_id"]
        raw = p.get("answer") or extract_gold_from_solution(p.get("solution", ""))
        gold[pid] = normalize(raw)
        meta[pid] = {"subject": p.get("subject"), "level": p.get("level")}
    return gold, meta


def index_branches(generations):
    """Group samples as preds[(model_tag, pid)] -> list[normalized_pred]."""
    preds = {}
    for g in generations:
        key = (g["model_tag"], g["problem_id"])
        pred = normalize(extract_pred_from_text(g.get("raw_output", "")))
        preds.setdefault(key, []).append(pred)
    return preds


def majority_vote(branches):
    """Return the most common non-None pred; None if all None."""
    counts = Counter(p for p in branches if p is not None)
    if not counts:
        return None
    # Counter.most_common is stable on insertion order for ties -> deterministic.
    return counts.most_common(1)[0][0]


def per_model_summary(preds, gold_map):
    """Per-model pass@1, pass@3, maj@3, distinct."""
    by_model = {}
    for (tag, pid), branches in preds.items():
        by_model.setdefault(tag, []).append((pid, branches))

    rows = []
    for tag, items in sorted(by_model.items()):
        n_correct_p1 = 0.0
        n_correct_pk = 0.0
        n_correct_maj = 0
        distincts = []
        n_problems = len(items)
        k = 3
        for pid, branches in items:
            gold = gold_map.get(pid)
            c = sum(1 for p in branches if p is not None and gold is not None and p == gold)
            n_correct_p1 += pass_at_k(len(branches), c, 1)
            n_correct_pk += pass_at_k(len(branches), c, k)
            voted = majority_vote(branches)
            if voted is not None and gold is not None and voted == gold:
                n_correct_maj += 1
            distincts.append(len(set(p for p in branches if p is not None)))

        rows.append({
            "model_tag": tag,
            "pass_at_1": round(n_correct_p1 / n_problems, 4),
            "pass_at_3": round(n_correct_pk / n_problems, 4),
            "maj_at_3": round(n_correct_maj / n_problems, 4),
            "maj_minus_pass1": round(n_correct_maj / n_problems - n_correct_p1 / n_problems, 4),
            "mean_distinct": round(sum(distincts) / n_problems, 4),
            "n_problems": n_problems,
        })
    return rows


def killed_set(preds, gold_map, source="base", target="drpo"):
    """Problems where source got >=1/3 correct but target got 0/3."""
    pids = sorted({pid for (_, pid) in preds.keys()})
    killed = []
    for pid in pids:
        s_branches = preds.get((source, pid), [])
        t_branches = preds.get((target, pid), [])
        gold = gold_map.get(pid)
        if gold is None or not s_branches or not t_branches:
            continue
        s_c = sum(1 for p in s_branches if p == gold)
        t_c = sum(1 for p in t_branches if p == gold)
        if s_c >= 1 and t_c == 0:
            killed.append(pid)
    return killed


def cross_model_intervention(preds, gold_map, problem_ids, target, donors):
    """Evaluate intervention strategies on a fixed problem set.

    Strategies pool the target's branches with branches from one or more donor
    models, then majority-vote. We also run a pure target self-consistency
    baseline as the no-intervention control.
    """
    strategies = {}
    strategies[f"{target}_self_maj3"] = [target]
    for donor in donors:
        if donor == target:
            continue
        strategies[f"{target}+{donor}"] = [target, donor]
    strategies[f"{target}+all_donors"] = [target] + [d for d in donors if d != target]

    rows = []
    for name, model_set in strategies.items():
        recovered = 0
        voted_correct = 0
        for pid in problem_ids:
            pool = []
            for m in model_set:
                pool.extend(preds.get((m, pid), []))
            voted = majority_vote(pool)
            gold = gold_map.get(pid)
            if voted is not None and gold is not None and voted == gold:
                voted_correct += 1
                recovered += 1
        rows.append({
            "strategy": name,
            "models": "+".join(model_set),
            "n_branches_per_problem": sum(3 for _ in model_set),
            "n_problems": len(problem_ids),
            "voted_correct": voted_correct,
            "recovered_pct": round(voted_correct / len(problem_ids), 4) if problem_ids else 0.0,
        })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def print_table(rows, title):
    if not rows:
        return
    print(f"\n=== {title} ===")
    keys = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../math_trajectories/data/math500.jsonl")
    parser.add_argument("--gen_dir", default="../math_trajectories",
                        help="Directory containing generations_math500_<tag>.jsonl files")
    parser.add_argument("--tags", nargs="+",
                        default=["base", "sft", "drpo", "openai_nano"])
    parser.add_argument("--target", default="drpo",
                        help="Collapsed model that intervention tries to rescue")
    parser.add_argument("--source", default="base",
                        help="Reference model used to define the killed set")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    data_path = os.path.join(HERE, args.data)
    gen_dir = os.path.join(HERE, args.gen_dir)
    out_dir = os.path.join(HERE, args.out_dir)

    problems = load_jsonl(data_path)
    gold_map, _ = build_gold_map(problems)
    print(f"Loaded {len(problems)} problems")

    all_gens = []
    for tag in args.tags:
        path = os.path.join(gen_dir, f"generations_math500_{tag}.jsonl")
        rows = load_jsonl(path)
        all_gens.extend(rows)
        print(f"  {tag}: {len(rows)} rows from {path}")

    preds = index_branches(all_gens)

    # 1) Per-model summary including maj@3.
    summary = per_model_summary(preds, gold_map)
    write_csv(summary, os.path.join(out_dir, "branching_summary.csv"))
    print_table(summary, "Per-model self-consistency (maj@3)")

    # 2) Diversity-killed set (default: base solves, drpo never).
    killed = killed_set(preds, gold_map, source=args.source, target=args.target)
    print(f"\nKilled set |{args.source} >=1 ∧ {args.target} = 0|: {len(killed)} problems")
    write_csv(
        [{"problem_id": pid} for pid in killed],
        os.path.join(out_dir, f"killed_{args.source}_vs_{args.target}.csv"),
    )

    # 3) Cross-model intervention on the killed set.
    donors = [t for t in args.tags if t != args.target]
    inter = cross_model_intervention(preds, gold_map, killed, args.target, donors)
    write_csv(inter, os.path.join(out_dir, f"intervention_{args.target}_killed.csv"))
    print_table(inter, f"Cross-model intervention on killed set (rescue {args.target})")

    # 4) Same intervention on the full Math500 (sanity: does intervention also
    # help globally, or only on the killed set?).
    all_pids = sorted({pid for (_, pid) in preds.keys()})
    full = cross_model_intervention(preds, gold_map, all_pids, args.target, donors)
    write_csv(full, os.path.join(out_dir, f"intervention_{args.target}_full.csv"))
    print_table(full, f"Cross-model intervention on full Math500 (target {args.target})")


if __name__ == "__main__":
    main()
