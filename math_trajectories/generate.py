"""Generate K reasoning trajectories per MATH problem via a vLLM-served model."""

import argparse
import json
import os
import time

from openai import OpenAI

PROMPT_TMPL = """\
Solve the following math problem step by step. Show numbered reasoning steps.
End with: Final Answer: $YOUR_ANSWER$

Problem:
{problem}"""


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_successful_sample(row: dict) -> bool:
    """Treat rows with real text and non-error finish reasons as resumable output."""
    return row.get("raw_output") is not None and row.get("finish_reason") != "error"


def prepare_resume(output_path: str, model_tag: str, expected_samples: int) -> set[str]:
    """Drop partial/error rows for model_tag, then return completed problem_ids."""
    if not os.path.exists(output_path):
        return set()

    rows = load_jsonl(output_path)
    keep_rows = []
    grouped: dict[str, list[dict]] = {}

    for row in rows:
        if row.get("model_tag") == model_tag:
            grouped.setdefault(row["problem_id"], []).append(row)
        else:
            keep_rows.append(row)

    done_pids = set()
    dropped_rows = 0
    for pid, samples in grouped.items():
        is_complete = len(samples) == expected_samples and all(is_successful_sample(s) for s in samples)
        if is_complete:
            done_pids.add(pid)
            keep_rows.extend(samples)
        else:
            dropped_rows += len(samples)

    if dropped_rows:
        with open(output_path, "w", encoding="utf-8") as f:
            for row in keep_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Removed {dropped_rows} partial/error rows from {output_path} before resume")

    return done_pids


def generate_for_problem(
    client: OpenAI,
    model_name: str,
    problem_text: str,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
) -> list[dict]:
    """Call chat completions with n samples. Returns list of {text, finish_reason}."""
    messages = [{"role": "user", "content": PROMPT_TMPL.format(problem=problem_text)}]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )

    results = []
    for choice in resp.choices:
        results.append(
            {
                "text": choice.message.content,
                "finish_reason": choice.finish_reason,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate math reasoning trajectories")
    parser.add_argument("--model_tag", required=True, help="Tag: base, sft, or drpo")
    parser.add_argument("--model_name", required=True, help="vLLM served-model-name")
    parser.add_argument("--api_base", default="http://localhost:18502/v1", help="vLLM API base URL")
    parser.add_argument("--api_key", default="EMPTY", help="API key (default: EMPTY for local vLLM)")
    parser.add_argument("--input_file", default="data/math_subset.jsonl")
    parser.add_argument("--output_file", default=None, help="Default: logs/generations_<model_tag>.jsonl")
    parser.add_argument("--num_samples", type=int, default=3, help="K samples per problem")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"logs/generations_{args.model_tag}.jsonl"

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    problems = load_jsonl(args.input_file)
    print(f"Loaded {len(problems)} problems from {args.input_file}")

    done_pids = prepare_resume(args.output_file, args.model_tag, args.num_samples)
    if done_pids:
        print(f"Resuming: {len(done_pids)} problems already done, skipping them")

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    gen_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "num_samples": args.num_samples,
        "seed": args.seed,
    }

    with open(args.output_file, "a", encoding="utf-8") as out_f:
        for i, prob in enumerate(problems):
            pid = prob["problem_id"]
            if pid in done_pids:
                continue

            t0 = time.time()
            prompt_text = PROMPT_TMPL.format(problem=prob["problem"])

            # Retry logic
            results = None
            for attempt in range(3):
                try:
                    results = generate_for_problem(
                        client=client,
                        model_name=args.model_name,
                        problem_text=prob["problem"],
                        n=args.num_samples,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        seed=args.seed,
                    )
                    break
                except Exception as e:
                    print(f"  [attempt {attempt+1}/3] Error for {pid}: {e}")
                    if attempt < 2:
                        time.sleep(5)

            elapsed = round(time.time() - t0, 3)

            if results is None:
                # Write null outputs on total failure
                results = [{"text": None, "finish_reason": "error"}] * args.num_samples

            for sample_id, res in enumerate(results):
                row = {
                    "problem_id": pid,
                    "model_tag": args.model_tag,
                    "model_name": args.model_name,
                    "sample_id": sample_id,
                    "prompt": prompt_text,
                    "raw_output": res["text"],
                    "finish_reason": res["finish_reason"],
                    "generation_config": gen_config,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            n_done = i + 1 - len(done_pids)
            n_total = len(problems) - len(done_pids)
            print(f"[{n_done}/{n_total}] {pid} done ({elapsed}s)")

    print(f"\nDone. Output: {args.output_file}")


if __name__ == "__main__":
    main()
