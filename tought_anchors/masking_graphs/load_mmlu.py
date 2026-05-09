#!/usr/bin/env python3
"""
Utilities for loading and formatting the MMLU dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

ANSWER_LETTERS = ["A", "B", "C", "D"]


def _normalize_question(item: Dict[str, Any], subject: str, index: int) -> Dict[str, Any]:
    answer_idx = int(item["answer"])
    return {
        "subject": subject,
        "question_idx": index,
        "question": item["question"],
        "choices": list(item["choices"]),
        "answer_index": answer_idx,
        "answer_letter": ANSWER_LETTERS[answer_idx],
    }


def create_mmlu_prompt(
    question_data: Dict[str, Any],
    model: Optional[str] = None,
    enable_thinking: bool = True,
    just_prompt: bool = False,
) -> str:
    """
    Build a prompt for a single MMLU multiple-choice question.
    """
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

    prompt = "\n".join(lines)
    if just_prompt:
        return prompt
    return prompt + "\n\nAnswer:\n"


def load_all_mmlu(
    split: str = "test",
    subjects: Optional[Iterable[str]] = None,
    max_questions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load all requested MMLU questions from Hugging Face.
    """
    loaded_questions: List[Dict[str, Any]] = []
    selected_subjects = list(subjects) if subjects is not None else MMLU_SUBJECTS

    for subject in selected_subjects:
        dataset = load_dataset("cais/mmlu", subject, split=split)
        for index, item in enumerate(dataset):
            loaded_questions.append(_normalize_question(item, subject, index))
            if max_questions is not None and len(loaded_questions) >= max_questions:
                return loaded_questions

    return loaded_questions


def load_mmlu_below_accuracy_threshold(
    max_questions: Optional[int],
    max_prob_correct: float,
    baseline_csv: str = "results/mmlu_results/baseline_probabilities.csv",
) -> List[Dict[str, Any]]:
    """
    Load a filtered subset of MMLU questions based on a saved baseline CSV.
    """
    csv_path = Path(baseline_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found at {csv_path}. "
            "Run the non-thinking MMLU baseline first or pass max_nonthinking_accuracy=None."
        )

    selected_keys = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                prob_correct = float(row["prob_correct"])
            except (KeyError, ValueError):
                continue
            if prob_correct <= max_prob_correct:
                selected_keys.append((row["subject"], int(row["question_idx"])))
                if max_questions is not None and len(selected_keys) >= max_questions:
                    break

    selected_lookup = set(selected_keys)
    questions = load_all_mmlu(split="test")
    filtered = [
        q for q in questions if (q["subject"], q["question_idx"]) in selected_lookup
    ]
    if max_questions is not None:
        filtered = filtered[:max_questions]
    return filtered


def save_questions_json(path: str, questions: List[Dict[str, Any]]) -> None:
    """
    Convenience helper for debugging prepared question sets.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(questions, handle, indent=2)
