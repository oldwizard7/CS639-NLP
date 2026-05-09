#!/usr/bin/env python3
"""
Generate sentence-resampling rollouts for saved MMLU reasoning traces.

This mirrors the black-box Thought Anchors rollout format used by
generate_rollouts.py, but uses completed MMLU generations as the base traces.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm


ANSWER_LETTERS = ["A", "B", "C", "D"]
ANSWER_RE = re.compile(
    r"Final Answer:\s*\[?([A-D])\]?",
    re.IGNORECASE,
)


def create_mmlu_prompt(
    question_data: Dict[str, Any],
    model: Optional[str] = None,
    enable_thinking: bool = True,
) -> str:
    del model
    lines = [
        "You are solving a multiple-choice question.",
        "Think carefully and then give exactly one final answer letter.",
        f"Question: {question_data['question']}",
        "",
        "Choices:",
    ]
    for letter, choice in zip(ANSWER_LETTERS, question_data["choices"]):
        lines.append(f"{letter}. {choice}")

    lines.append("")
    if enable_thinking:
        lines.append(
            "After your reasoning, end with the line: Final Answer: <LETTER>"
        )
    else:
        lines.append("Respond with exactly one line: Final Answer: <LETTER>")
    return "\n".join(lines) + "\n\nAnswer:\n"


def extract_answer_letter(text: str) -> Optional[str]:
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper()

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped[:1].upper() in ANSWER_LETTERS:
            return stripped[:1].upper()

    match = re.search(r"\b([A-D])\b", text[-80:], re.IGNORECASE)
    return match.group(1).upper() if match else None


def split_solution_into_chunks(solution_text: str) -> List[str]:
    """Split a solution into sentence-like chunks for rollout generation."""
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()

    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]
    chunks: List[str] = []
    current_chunk = ""

    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char in {" ", "\n"}:
                is_sentence_end = True

        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Thought Anchors style MMLU rollouts."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument(
        "--mmlu-results-root",
        default="mmlu_local",
        help="Root containing saved run_mmlu_local.py outputs",
    )
    parser.add_argument(
        "--output_dir",
        default="mmlu_rollouts",
        help="Directory to save MMLU rollout data",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="MMLU split name under the saved results directory",
    )
    parser.add_argument(
        "-b",
        "--base_solution_type",
        default="correct",
        choices=["correct", "incorrect"],
        help="Use saved MMLU traces with this correctness",
    )
    parser.add_argument(
        "-r",
        "--rollout_type",
        default="default",
        choices=["default", "forced_answer"],
        help="Generate normal resampling or forced-answer continuations",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Number of saved traces to process",
    )
    parser.add_argument(
        "--question_keys",
        default=None,
        help="Comma-separated subject:index keys. Overrides automatic selection.",
    )
    parser.add_argument(
        "--question_keys_file",
        default=None,
        help=(
            "Text or JSON file containing subject:index keys. Use this to run "
            "the exact same matched question set across models."
        ),
    )
    parser.add_argument(
        "--selection",
        default="longest",
        choices=["longest", "first"],
        help="How to select traces when question_keys is not provided",
    )
    parser.add_argument(
        "--min_chunks",
        type=int,
        default=2,
        help="Ignore saved traces with fewer chunks",
    )
    parser.add_argument(
        "-nr",
        "--num_rollouts",
        type=int,
        default=100,
        help="Number of rollouts per chunk",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for rollout generation",
    )
    parser.add_argument(
        "-tp",
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for rollout generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per rollout",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for local generation",
    )
    parser.add_argument(
        "-ic",
        "--include_chunks",
        default=None,
        help="Comma-separated chunk ids/ranges, e.g. 0,2,5-8",
    )
    parser.add_argument(
        "-os",
        "--output_suffix",
        default=None,
        help="Suffix to add to the output directory",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Regenerate existing artifacts",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only save selected base traces and chunks",
    )
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def output_subdir(args: argparse.Namespace) -> Path:
    base_dir = (
        Path(args.output_dir)
        / model_slug(args.model)
        / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
    )
    name = args.base_solution_type + "_base_solution"
    if args.rollout_type == "forced_answer":
        name += "_forced_answer"
    if args.output_suffix:
        name += f"_{args.output_suffix}"
    return base_dir / name


def parse_include_chunks(include_chunks: Optional[str]) -> Optional[Set[int]]:
    if not include_chunks:
        return None

    selected: Set[int] = set()
    for part in include_chunks.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def make_problem_text(row: Dict[str, Any]) -> str:
    lines = [row["question"], "", "Choices:"]
    for letter, choice in zip(ANSWER_LETTERS, row["choices"]):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def problem_dir_name(row: Dict[str, Any]) -> str:
    subject = re.sub(r"[^A-Za-z0-9_]+", "_", str(row["subject"]))
    return f"problem_{subject}_{row['question_idx']}"


def question_key(row: Dict[str, Any]) -> str:
    return f"{row['subject']}:{int(row['question_idx'])}"


def load_question_keys(args: argparse.Namespace) -> Optional[List[Tuple[str, int]]]:
    raw_keys: List[str] = []

    if args.question_keys_file:
        path = Path(args.question_keys_file)
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix == ".json":
                payload = json.load(handle)
                if isinstance(payload, dict):
                    raw_keys = payload.get("question_keys", [])
                elif isinstance(payload, list):
                    raw_keys = payload
                else:
                    raise ValueError(f"Unsupported key file format in {path}")
            else:
                raw_keys = [
                    line.strip()
                    for line in handle
                    if line.strip() and not line.lstrip().startswith("#")
                ]

    if args.question_keys:
        raw_keys.extend(item.strip() for item in args.question_keys.split(","))

    if not raw_keys:
        return None

    keys: List[Tuple[str, int]] = []
    seen: Set[Tuple[str, int]] = set()
    for item in raw_keys:
        subject, idx = item.strip().rsplit(":", 1)
        key = (subject, int(idx))
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return keys


def clean_saved_mmlu_generation(row: Dict[str, Any], args: argparse.Namespace) -> str:
    """Return only the model's reasoning/answer text, not an echoed prompt."""
    text = row.get("generated_text", "").strip()
    if not text:
        return ""

    question_data = {
        "question": row["question"],
        "choices": row["choices"],
        "answer_letter": row["correct_answer"],
    }
    prompt = create_mmlu_prompt(question_data, args.model)

    # Future runs should be clean after the decoding fix in run_mmlu_local.py.
    # This loop also protects us from older saved runs where left padding caused
    # the tail of the prompt to be decoded as part of generated_text.
    while text.startswith(prompt):
        text = text[len(prompt) :].lstrip()

    for _ in range(3):
        marker_idx = text.find("\n\nAnswer:\n")
        if marker_idx == -1 or marker_idx > 2500:
            break
        text = text[marker_idx + len("\n\nAnswer:\n") :].lstrip()

    text = re.sub(r"^Final Answer:\s*<LETTER>\s*", "", text).lstrip()
    return text


def first_chunk_or_text(text: str) -> str:
    chunks = split_solution_into_chunks(text)
    return chunks[0] if chunks else text.strip()


def remove_one_chunk(full_prefix: str, chunk: str) -> str:
    return full_prefix.replace(chunk, "", 1).strip()


def load_saved_results(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results_path = (
        Path(args.mmlu_results_root)
        / model_slug(args.model)
        / args.split
        / "all_results.json"
    )
    with results_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["summary"], payload["results"]


def select_rows(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    desired_correctness = args.base_solution_type == "correct"
    candidates = [row for row in rows if bool(row.get("is_correct")) == desired_correctness]
    selected_keys = load_question_keys(args)

    if selected_keys:
        key_order = {key: idx for idx, key in enumerate(selected_keys)}
        candidates = [
            row
            for row in candidates
            if (row["subject"], int(row["question_idx"])) in key_order
        ]
        candidates.sort(
            key=lambda row: key_order[(row["subject"], int(row["question_idx"]))]
        )
    else:
        enriched = []
        for row in candidates:
            chunks = split_solution_into_chunks(clean_saved_mmlu_generation(row, args))
            if len(chunks) >= args.min_chunks and row.get("predicted_answer"):
                enriched.append((len(chunks), row))

        if args.selection == "longest":
            enriched.sort(key=lambda item: item[0], reverse=True)
        candidates = [row for _, row in enriched]

    return candidates[: args.num_questions]


def make_base_solution(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    question_data = {
        "question": row["question"],
        "choices": row["choices"],
        "answer_letter": row["correct_answer"],
    }
    prompt = create_mmlu_prompt(question_data, args.model)
    generated_text = clean_saved_mmlu_generation(row, args)
    return {
        "solution": generated_text,
        "full_cot": f"{prompt}{generated_text}",
        "answer": row.get("predicted_answer"),
        "is_correct": bool(row.get("is_correct")),
    }


def generate_with_local_model_batch(
    tokenizer: Any,
    model: Any,
    prompts: List[str],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    import torch

    results: List[Dict[str, Any]] = []
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        print(
            f"Processing batch {start // args.batch_size + 1}/"
            f"{(len(prompts) + args.batch_size - 1) // args.batch_size}"
        )
        encoded = tokenizer(batch_prompts, padding=True, return_tensors="pt")
        if torch.cuda.is_available():
            encoded = {key: value.to("cuda") for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_config)

        input_length = encoded["input_ids"].shape[1]
        for output_ids in outputs:
            generated_text = tokenizer.decode(
                output_ids[input_length:], skip_special_tokens=True
            )
            results.append({"text": generated_text})
    return results


def save_problem_metadata(problem_dir: Path, row: Dict[str, Any], args: argparse.Namespace) -> None:
    problem_payload = {
        "problem": make_problem_text(row),
        "gt_answer": row["correct_answer"],
        "subject": row["subject"],
        "question_idx": row["question_idx"],
        "choices": row["choices"],
    }
    problem_path = problem_dir / "problem.json"
    if args.force or not problem_path.exists():
        with problem_path.open("w", encoding="utf-8") as handle:
            json.dump(problem_payload, handle, indent=2)

    base_path = problem_dir / "base_solution.json"
    if args.force or not base_path.exists():
        with base_path.open("w", encoding="utf-8") as handle:
            json.dump(make_base_solution(row, args), handle, indent=2)

    chunks_path = problem_dir / "chunks.json"
    if args.force or not chunks_path.exists():
        solution_text = clean_saved_mmlu_generation(row, args)
        chunks = split_solution_into_chunks(solution_text)
        with chunks_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "source_text": solution_text,
                    "solution_text": solution_text,
                    "chunks": chunks,
                },
                handle,
                indent=2,
            )
        print(f"{problem_dir.name}: saved {len(chunks)} chunks")


def rollout_prompt(row: Dict[str, Any], args: argparse.Namespace, prefix: str) -> str:
    question_data = {
        "question": row["question"],
        "choices": row["choices"],
        "answer_letter": row["correct_answer"],
    }
    prompt = create_mmlu_prompt(question_data, args.model)
    if prefix:
        prompt = f"{prompt}{prefix}"
    if args.rollout_type == "forced_answer":
        prompt += "\n\nTherefore, the final answer is: Final Answer: "
    return prompt


def process_problem(
    row: Dict[str, Any],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    args: argparse.Namespace,
    selected_chunks: Optional[Set[int]],
) -> None:
    problem_dir = output_subdir(args) / problem_dir_name(row)
    problem_dir.mkdir(parents=True, exist_ok=True)
    save_problem_metadata(problem_dir, row, args)

    with (problem_dir / "chunks.json").open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)["chunks"]

    if not chunks:
        print(f"{problem_dir.name}: no chunks; skipping rollouts")
        return
    if args.prepare_only:
        print(f"{problem_dir.name}: prepare_only set; skipping rollouts")
        return

    cumulative_chunks = []
    current = ""
    for chunk in chunks:
        current += chunk + " "
        cumulative_chunks.append(current.strip())

    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_chunks)):
        if selected_chunks is not None and chunk_idx not in selected_chunks:
            print(f"{problem_dir.name} chunk_{chunk_idx}: skipped by include_chunks")
            continue

        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        solutions_path = chunk_dir / "solutions.json"

        existing: List[Dict[str, Any]] = []
        valid_existing: List[Dict[str, Any]] = []
        if solutions_path.exists() and not args.force:
            with solutions_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            valid_existing = [
                item for item in existing if "error" not in item and item.get("answer")
            ]
            print(
                f"{problem_dir.name} chunk_{chunk_idx}: "
                f"found {len(valid_existing)} valid solutions"
            )

        needed = args.num_rollouts - len(valid_existing)
        if needed <= 0:
            continue

        print(f"{problem_dir.name} chunk_{chunk_idx}: generating {needed} rollouts")
        prefix_without_chunk = remove_one_chunk(full_prefix, chunk)
        prompts = [
            rollout_prompt(row, args, prefix_without_chunk)
            for _ in tqdm(range(needed), desc="Preparing prompts")
        ]
        batch_results = generate_with_local_model_batch(tokenizer, model, prompts, args)

        new_solutions: List[Dict[str, Any]] = []
        for prompt, result in zip(prompts, batch_results):
            if "error" in result:
                new_solutions.append({"error": result["error"]})
                continue

            rollout_text = result.get("text", "")
            answer_source = f"{prompt}{rollout_text}" if args.rollout_type == "forced_answer" else rollout_text
            answer = extract_answer_letter(answer_source)
            is_correct = answer == row["correct_answer"]
            new_solutions.append(
                {
                    "chunk_removed": chunk,
                    "prefix_without_chunk": prefix_without_chunk,
                    "chunk_resampled": first_chunk_or_text(rollout_text),
                    "rollout": rollout_text,
                    "full_cot": f"{prompt}{rollout_text}",
                    "answer": answer or "",
                    "is_correct": bool(is_correct),
                }
            )

        with solutions_path.open("w", encoding="utf-8") as handle:
            json.dump(existing + new_solutions, handle, indent=2)
        print(
            f"{problem_dir.name} chunk_{chunk_idx}: "
            f"saved {len(existing) + len(new_solutions)} solutions"
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    summary, rows = load_saved_results(args)
    selected_rows = select_rows(args, rows)
    if not selected_rows:
        raise SystemExit("No matching MMLU traces selected.")

    output_subdir(args).mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_summary": summary,
        "model": args.model,
        "base_solution_type": args.base_solution_type,
        "rollout_type": args.rollout_type,
        "selected": [
            {
                "subject": row["subject"],
                "question_idx": row["question_idx"],
                "is_correct": row["is_correct"],
                "predicted_answer": row.get("predicted_answer"),
                "correct_answer": row["correct_answer"],
                "num_chunks": len(
                    split_solution_into_chunks(clean_saved_mmlu_generation(row, args))
                ),
                "question_key": question_key(row),
            }
            for row in selected_rows
        ],
    }
    with (output_subdir(args) / "selection_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Selected {len(selected_rows)} MMLU traces.")
    for row in selected_rows:
        print(
            f"- {row['subject']} #{row['question_idx']} "
            f"pred={row.get('predicted_answer')} gold={row['correct_answer']}"
        )

    tokenizer = None
    model = None
    if not args.prepare_only:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading local model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        model.eval()
        print("Local model loaded successfully")

    selected_chunks = parse_include_chunks(args.include_chunks)
    for row in selected_rows:
        process_problem(row, tokenizer, model, args, selected_chunks)


if __name__ == "__main__":
    main()
