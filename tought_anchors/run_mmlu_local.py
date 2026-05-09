#!/usr/bin/env python3
"""
Run MMLU with a local Hugging Face causal LM.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from masking_graphs.load_mmlu import ANSWER_LETTERS, MMLU_SUBJECTS, create_mmlu_prompt


ANSWER_RE = re.compile(
    r"Final Answer:\s*\[?([A-D])\]?",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local model on MMLU.")
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument(
        "--output_dir",
        default="results/mmlu_local",
        help="Base directory for MMLU outputs",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "validation", "dev", "auxiliary_train"],
        help="MMLU split to evaluate",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject subset",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Optional cap across all subjects",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 uses greedy decoding",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p used when sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Random seed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results directory",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_answer_letter(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if match:
        return match.group(1).upper()

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped[:1].upper() in ANSWER_LETTERS:
            return stripped[:1].upper()

    match = re.search(r"\b([A-D])\b", text[-80:], re.IGNORECASE)
    return match.group(1).upper() if match else None


def load_questions(split: str, subjects: List[str], max_questions: int | None) -> List[Dict]:
    questions: List[Dict] = []
    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split=split)
        for idx, item in enumerate(dataset):
            answer_index = int(item["answer"])
            questions.append(
                {
                    "subject": subject,
                    "question_idx": idx,
                    "question": item["question"],
                    "choices": list(item["choices"]),
                    "answer_index": answer_index,
                    "answer_letter": ANSWER_LETTERS[answer_index],
                }
            )
            if max_questions is not None and len(questions) >= max_questions:
                return questions
    return questions


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model_name = args.model
    model_slug = model_name.split("/")[-1]
    output_dir = Path(args.output_dir) / model_slug / args.split
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"Output directory {output_dir} already exists. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading local model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    model.eval()
    print("Local model loaded successfully")

    subjects = args.subjects if args.subjects else MMLU_SUBJECTS
    questions = load_questions(args.split, subjects, args.max_questions)
    print(f"Loaded {len(questions)} MMLU questions from {len(subjects)} subjects.")

    results: List[Dict] = []
    num_correct = 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p if args.temperature > 0 else None,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    for start in range(0, len(questions), args.batch_size):
        batch_questions = questions[start : start + args.batch_size]
        prompts = [create_mmlu_prompt(question, model_name) for question in batch_questions]

        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            encoded = {key: value.to("cuda") for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model.generate(**encoded, **generation_kwargs)

        # With left padding, the prompt occupies the full padded input width in
        # `outputs`. Slicing by each row's non-pad length leaks prompt suffixes
        # into shorter examples, which contaminates downstream chunking.
        prompt_width = encoded["input_ids"].shape[1]
        for idx, question in enumerate(batch_questions):
            generated_ids = outputs[idx][prompt_width:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            answer_letter = extract_answer_letter(generated_text)
            is_correct = answer_letter == question["answer_letter"]
            if is_correct:
                num_correct += 1

            result = {
                "subject": question["subject"],
                "question_idx": question["question_idx"],
                "question": question["question"],
                "choices": question["choices"],
                "correct_answer": question["answer_letter"],
                "predicted_answer": answer_letter,
                "is_correct": is_correct,
                "generated_text": generated_text,
            }
            results.append(result)

        processed = len(results)
        accuracy = num_correct / processed if processed else 0.0
        print(f"Processed {processed}/{len(questions)} questions; accuracy={accuracy:.3f}")

    summary = {
        "model": model_name,
        "split": args.split,
        "num_questions": len(results),
        "num_correct": num_correct,
        "accuracy": (num_correct / len(results)) if results else 0.0,
        "subjects": subjects,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
    }

    with (output_dir / "all_results.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": results}, handle, indent=2)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved results to {output_dir / 'all_results.json'}")
    print(f"Final accuracy: {summary['accuracy']:.3f}")


if __name__ == "__main__":
    main()
