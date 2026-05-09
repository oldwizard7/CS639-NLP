"""Generate Math500 trajectories via the OpenAI API."""

import argparse
import json
import os
import time

from openai import OpenAI

from generate import PROMPT_TMPL, load_jsonl, prepare_resume


def resolve_api_key(explicit_key: str | None, env_var: str) -> str:
    """Resolve the API key from CLI or environment."""
    if explicit_key:
        return explicit_key

    api_key = os.getenv(env_var)
    if api_key:
        return api_key

    raise SystemExit(
        f"Missing OpenAI API key. Pass --api_key or set {env_var} in the environment."
    )


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
    """Call OpenAI chat completions and return standardized rows."""
    messages = [{"role": "user", "content": PROMPT_TMPL.format(problem=problem_text)}]

    # GPT-5.4 chat completions expect max_completion_tokens instead of max_tokens.
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
        seed=seed,
    )

    return [
        {
            "text": choice.message.content,
            "finish_reason": choice.finish_reason,
        }
        for choice in resp.choices
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate Math500 trajectories via OpenAI API")
    parser.add_argument("--model_tag", default="openai_nano", help="Output tag, e.g. openai_nano")
    parser.add_argument("--model_name", default="gpt-5.4-nano", help="OpenAI model name")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (defaults to env var)")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Environment variable for API key")
    parser.add_argument("--input_file", default="data/math500.jsonl")
    parser.add_argument(
        "--output_file",
        default="generations_math500_openai_nano.jsonl",
        help="JSONL output path",
    )
    parser.add_argument("--num_samples", type=int, default=3, help="K samples per problem")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=10.0)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    api_key = resolve_api_key(args.api_key, args.api_key_env)
    client = OpenAI(api_key=api_key)

    problems = load_jsonl(args.input_file)
    print(f"Loaded {len(problems)} problems from {args.input_file}")

    done_pids = prepare_resume(args.output_file, args.model_tag, args.num_samples)
    if done_pids:
        print(f"Resuming: {len(done_pids)} problems already done, skipping them")

    gen_config = {
        "provider": "openai",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_completion_tokens": args.max_tokens,
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

            results = None
            for attempt in range(args.max_retries):
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
                    print(f"  [attempt {attempt+1}/{args.max_retries}] Error for {pid}: {e}")
                    if attempt < args.max_retries - 1:
                        time.sleep(args.retry_sleep)

            elapsed = round(time.time() - t0, 3)

            if results is None:
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
