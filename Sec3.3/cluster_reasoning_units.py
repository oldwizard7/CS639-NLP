#!/usr/bin/env python3
"""Cluster reasoning units into higher-level logical steps with OpenAI.

Default workflow:
  python cluster_reasoning_units.py --pilot 50
  python cluster_reasoning_units.py --all
  python cluster_reasoning_units.py --validate-only

The script stores append-only checkpoints in output_dataset/*.partial.jsonl.
Reruns skip completed rows, so pilot work is reused by the full run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input_dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output_dataset"
DEFAULT_INPUT_FILES = (
    "reasoning_units_math500_base.jsonl",
    "reasoning_units_math500_drpo.jsonl",
    "reasoning_units_math500_sft.jsonl",
)

GPT4O_MINI_INPUT_PER_MILLION = 0.15
GPT4O_MINI_OUTPUT_PER_MILLION = 0.60

SYSTEM_PROMPT = """
You are a careful research data annotation assistant.

Your task is to cluster consecutive reasoning units into larger,
semantically coherent logical reasoning steps.

You must output valid JSON only.
No markdown.
No explanation.
""".strip()


class StepValidationError(ValueError):
    """Raised when the model output cannot be used as valid step clusters."""


@dataclass(frozen=True)
class FileSpec:
    input_path: Path
    output_path: Path
    partial_path: Path


@dataclass(frozen=True)
class WorkItem:
    spec: FileSpec
    source_index: int
    row: dict[str, Any]


@dataclass
class ProcessResult:
    item: WorkItem
    ok: bool
    enriched_row: dict[str, Any] | None
    attempts: int
    error: str | None
    usage: dict[str, int]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSONL row: {exc}") from exc
    return rows


def jsonl_dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def make_row_key(row: dict[str, Any]) -> str:
    key = {
        "problem_id": row.get("problem_id"),
        "model_tag": row.get("model_tag"),
        "sample_id": row.get("sample_id"),
    }
    return json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_file_specs(input_dir: Path, output_dir: Path, files: list[str]) -> list[FileSpec]:
    specs: list[FileSpec] = []
    for name in files:
        input_path = input_dir / name
        output_stem = f"clustered_{input_path.stem}"
        specs.append(
            FileSpec(
                input_path=input_path,
                output_path=output_dir / f"{output_stem}.jsonl",
                partial_path=output_dir / f"{output_stem}.partial.jsonl",
            )
        )
    return specs


def load_completed(partial_path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not partial_path.exists():
        return completed

    with partial_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: skipping corrupt checkpoint line {partial_path}:{line_num}", file=sys.stderr)
                continue

            checkpoint = obj.get("_checkpoint")
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get("key"), str):
                key = checkpoint["key"]
            else:
                key = make_row_key(obj)
            completed[key] = obj
    return completed


def strip_checkpoint(row: dict[str, Any]) -> dict[str, Any]:
    clean = dict(row)
    clean.pop("_checkpoint", None)
    return clean


def choose_pilot_indices(total_rows: int, count: int) -> list[int]:
    if count <= 0 or total_rows <= 0:
        return []
    if count >= total_rows:
        return list(range(total_rows))
    if count == 1:
        return [0]

    indices: list[int] = []
    seen: set[int] = set()
    for i in range(count):
        idx = round(i * (total_rows - 1) / (count - 1))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)

    candidate = 0
    while len(indices) < count and candidate < total_rows:
        if candidate not in seen:
            indices.append(candidate)
            seen.add(candidate)
        candidate += 1

    return sorted(indices)


def allocate_pilot_counts(total: int, num_files: int) -> list[int]:
    if total <= 0:
        raise ValueError("--pilot must be positive")
    base = total // num_files
    remainder = total % num_files
    return [base + (1 if i < remainder else 0) for i in range(num_files)]


def make_user_prompt(row: dict[str, Any], last_error: str | None = None) -> str:
    units = row["reasoning_units"]
    m = len(units)
    lower = max(1, round(m / 3))
    upper = min(m, max(lower, round(2 * m / 3)))
    numbered_units = "\n".join(f"[{i}] {unit}" for i, unit in enumerate(units))

    retry_note = ""
    if last_error:
        retry_note = f"""

Previous attempt failed validation:
{last_error}

Return a corrected JSON object only.
""".rstrip()

    return f"""
You are given a sequence of reasoning units from a math model solution.

Cluster consecutive reasoning units into higher-level logical reasoning steps.

Rules:
1. Clusters must preserve the original order.
2. Each cluster must contain consecutive unit_ids only.
3. Every input unit id from 0 to {m - 1} must appear exactly once.
4. Do not skip, duplicate, or reorder unit ids.
5. Use the zero-based unit ids exactly as shown in square brackets.
   Do not use one-based numbering. The largest valid unit id is {m - 1}.
6. Prefer grouping semantically connected units, such as:
   - problem setup
   - formula selection
   - computation
   - simplification
   - verification
   - final answer
7. Avoid over-fragmentation.
8. Avoid over-merging.
9. The number of clusters K should usually be between {lower} and {upper},
   unless the reasoning structure clearly requires otherwise.
10. Do not invent new reasoning.
11. Do not include a content field.
12. Output JSON only.

Expected JSON format:
{{
  "steps": [
    {{"title": "short title", "unit_ids": [0, 1]}},
    {{"title": "short title", "unit_ids": [2, 3]}}
  ]
}}

Metadata:
problem_id: {row.get("problem_id")}
model_tag: {row.get("model_tag")}
sample_id: {row.get("sample_id")}
num_reasoning_units: {m}

Reasoning units:
{numbered_units}{retry_note}
""".strip()


def parse_model_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])

    if not isinstance(obj, dict):
        raise StepValidationError("model output must be a JSON object")
    return obj


def coerce_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = payload.get("steps")
    if isinstance(raw_steps, list):
        return raw_steps
    if isinstance(raw_steps, dict):
        return ordered_step_dict_values(raw_steps)

    if payload and all(k.startswith("s") for k in payload):
        return ordered_step_dict_values(payload)

    raise StepValidationError("model output must contain a steps array")


def ordered_step_dict_values(raw_steps: dict[str, Any]) -> list[dict[str, Any]]:
    def sort_key(key: str) -> tuple[int, str]:
        if len(key) > 1 and key[0] == "s" and key[1:].isdigit():
            return int(key[1:]), key
        return 10**9, key

    values: list[dict[str, Any]] = []
    for key in sorted(raw_steps, key=sort_key):
        value = raw_steps[key]
        if not isinstance(value, dict):
            raise StepValidationError(f"{key} must be an object")
        values.append(value)
    return values


def validate_and_normalize_steps(payload: dict[str, Any], num_units: int) -> list[dict[str, Any]]:
    raw_steps = coerce_steps(payload)
    raw_steps = maybe_convert_one_based_ids(raw_steps, num_units)
    if num_units == 0:
        if raw_steps:
            raise StepValidationError("rows with zero reasoning units must have no steps")
        return []
    if not raw_steps:
        raise StepValidationError("steps must be non-empty")

    normalized: list[dict[str, Any]] = []
    spans: list[tuple[str, int, int]] = []
    all_ids: list[int] = []

    for step_idx, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            raise StepValidationError(f"steps[{step_idx}] must be an object")

        title = step.get("title")
        if not isinstance(title, str) or not title.strip():
            raise StepValidationError(f"steps[{step_idx}].title must be a non-empty string")
        title = " ".join(title.strip().split())

        unit_ids = step.get("unit_ids")
        if not isinstance(unit_ids, list) or not unit_ids:
            raise StepValidationError(f"steps[{step_idx}].unit_ids must be a non-empty list")
        if any(isinstance(unit_id, bool) or not isinstance(unit_id, int) for unit_id in unit_ids):
            raise StepValidationError(f"steps[{step_idx}].unit_ids must contain integers only")

        first = unit_ids[0]
        last = unit_ids[-1]
        if first < 0 or last >= num_units:
            raise StepValidationError(f"steps[{step_idx}].unit_ids out of range")
        if any(left >= right for left, right in zip(unit_ids, unit_ids[1:])):
            raise StepValidationError(f"steps[{step_idx}].unit_ids must be strictly increasing")

        # Models sometimes return an ordered semantic span while omitting a short
        # bridge unit, e.g. [1, 2, 3, 5, 6]. The task requires clusters to be
        # consecutive, so conservatively expand such spans and let the global
        # coverage check reject any overlap or impossible ordering.
        unit_ids = list(range(first, last + 1))

        spans.append((title, first, last))
        normalized.append({"title": title, "unit_ids": unit_ids})
        all_ids.extend(unit_ids)

    expected_all = list(range(num_units))
    if all_ids == expected_all:
        return normalized

    repaired = repair_steps_from_starts(spans, num_units)
    repaired_ids = [unit_id for step in repaired for unit_id in step["unit_ids"]]
    if repaired_ids != expected_all:
        raise StepValidationError(f"unit id coverage must be exactly {expected_all}, got {all_ids}")

    return repaired


def maybe_convert_one_based_ids(raw_steps: list[dict[str, Any]], num_units: int) -> list[dict[str, Any]]:
    all_ids: list[int] = []
    for step in raw_steps:
        if not isinstance(step, dict):
            return raw_steps
        unit_ids = step.get("unit_ids")
        if not isinstance(unit_ids, list):
            return raw_steps
        for unit_id in unit_ids:
            if isinstance(unit_id, bool) or not isinstance(unit_id, int):
                return raw_steps
            all_ids.append(unit_id)

    if not all_ids:
        return raw_steps
    if min(all_ids) >= 1 and max(all_ids) == num_units:
        converted: list[dict[str, Any]] = []
        for step in raw_steps:
            converted_step = dict(step)
            converted_step["unit_ids"] = [unit_id - 1 for unit_id in step["unit_ids"]]
            converted.append(converted_step)
        return converted
    return raw_steps


def repair_steps_from_starts(spans: list[tuple[str, int, int]], num_units: int) -> list[dict[str, Any]]:
    starts = [start for _, start, _ in spans]
    if not starts:
        return []
    if starts[0] != 0:
        raise StepValidationError("first step must start at unit id 0")
    if any(left >= right for left, right in zip(starts, starts[1:])):
        raise StepValidationError("step start ids must be strictly increasing")

    repaired: list[dict[str, Any]] = []
    for idx, (title, start, _) in enumerate(spans):
        end = starts[idx + 1] - 1 if idx + 1 < len(starts) else num_units - 1
        if end < start:
            raise StepValidationError("step boundaries are invalid")
        repaired.append({"title": title, "unit_ids": list(range(start, end + 1))})
    return repaired


def fallback_steps_from_payload(payload: dict[str, Any], num_units: int) -> list[dict[str, Any]]:
    if num_units <= 0:
        return []

    try:
        raw_steps = maybe_convert_one_based_ids(coerce_steps(payload), num_units)
    except Exception:
        raw_steps = []

    starts: dict[int, str] = {0: "Problem Setup"}
    for step_idx, step in enumerate(raw_steps):
        if not isinstance(step, dict):
            continue
        title = step.get("title")
        if not isinstance(title, str) or not title.strip():
            title = f"Reasoning Step {step_idx + 1}"
        title = " ".join(title.strip().split())

        unit_ids = step.get("unit_ids")
        if not isinstance(unit_ids, list):
            continue
        int_ids = [
            unit_id
            for unit_id in unit_ids
            if isinstance(unit_id, int) and not isinstance(unit_id, bool)
        ]
        in_range = [unit_id for unit_id in int_ids if 0 <= unit_id < num_units]
        if not in_range:
            continue
        starts.setdefault(min(in_range), title)

    if len(starts) == 1:
        for start, title in heuristic_starts(num_units).items():
            starts.setdefault(start, title)

    sorted_starts = sorted(starts)
    steps: list[dict[str, Any]] = []
    for idx, start in enumerate(sorted_starts):
        end = sorted_starts[idx + 1] - 1 if idx + 1 < len(sorted_starts) else num_units - 1
        if end < start:
            continue
        title = starts[start]
        if idx == len(sorted_starts) - 1 and end == num_units - 1 and title.startswith("Reasoning Step"):
            title = "Final Answer"
        steps.append({"title": title, "unit_ids": list(range(start, end + 1))})

    repaired_ids = [unit_id for step in steps for unit_id in step["unit_ids"]]
    if repaired_ids != list(range(num_units)):
        raise StepValidationError("fallback could not build full unit coverage")
    return steps


def heuristic_starts(num_units: int) -> dict[int, str]:
    if num_units <= 1:
        return {0: "Complete Reasoning Step"}
    if num_units <= 4:
        return {0: "Problem Setup", num_units - 1: "Final Answer"}

    target_clusters = max(2, min(num_units, round(num_units / 3)))
    starts: dict[int, str] = {0: "Problem Setup"}
    for cluster_idx in range(1, target_clusters):
        start = round(cluster_idx * num_units / target_clusters)
        start = min(max(start, 1), num_units - 1)
        title = "Final Answer" if cluster_idx == target_clusters - 1 else f"Reasoning Step {cluster_idx + 1}"
        starts.setdefault(start, title)
    return starts


def build_logical_steps(units: list[str], steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    logical_steps: dict[str, dict[str, Any]] = {}
    for idx, step in enumerate(steps):
        unit_ids = step["unit_ids"]
        logical_steps[f"s{idx}"] = {
            "title": step["title"],
            "unit_ids": unit_ids,
            "content": "\n".join(units[unit_id] for unit_id in unit_ids),
        }
    return logical_steps


def enrich_row(row: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    units = row.get("reasoning_units")
    if not isinstance(units, list) or any(not isinstance(unit, str) for unit in units):
        raise ValueError("row.reasoning_units must be a list of strings")
    enriched = dict(row)
    enriched["logical_steps"] = build_logical_steps(units, steps)
    return enriched


def add_usage(total: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total[key] = total.get(key, 0) + int(value)


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    result: dict[str, int] = {}
    if prompt_tokens is not None:
        result["prompt_tokens"] = int(prompt_tokens)
    if completion_tokens is not None:
        result["completion_tokens"] = int(completion_tokens)
    if total_tokens is not None:
        result["total_tokens"] = int(total_tokens)
    return result


def is_non_retryable_api_error(error: str) -> bool:
    markers = (
        "insufficient_quota",
        "invalid_api_key",
        "Incorrect API key",
        "AuthenticationError",
        "PermissionDeniedError",
        "model_not_found",
    )
    return any(marker in error for marker in markers)


async def call_model_once(
    client: Any,
    row: dict[str, Any],
    model: str,
    temperature: float,
    last_error: str | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(row, last_error=last_error)},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    return parse_model_json(text), usage_from_response(response)


async def process_item(
    client: Any,
    item: WorkItem,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float,
    max_attempts: int,
) -> ProcessResult:
    units = item.row.get("reasoning_units")
    if not isinstance(units, list) or any(not isinstance(unit, str) for unit in units):
        return ProcessResult(
            item=item,
            ok=False,
            enriched_row=None,
            attempts=0,
            error="row.reasoning_units must be a list of strings",
            usage={},
        )

    if len(units) == 0:
        return ProcessResult(
            item=item,
            ok=True,
            enriched_row=enrich_row(item.row, []),
            attempts=0,
            error=None,
            usage={},
        )
    if len(units) == 1:
        return ProcessResult(
            item=item,
            ok=True,
            enriched_row=enrich_row(
                item.row,
                [{"title": "Complete Reasoning Step", "unit_ids": [0]}],
            ),
            attempts=0,
            error=None,
            usage={},
        )

    total_usage: dict[str, int] = {}
    last_error: str | None = None
    last_payload: dict[str, Any] | None = None
    attempts = 0

    async with semaphore:
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                payload, usage = await call_model_once(
                    client=client,
                    row=item.row,
                    model=model,
                    temperature=temperature,
                    last_error=last_error,
                )
                add_usage(total_usage, usage)
                last_payload = payload
                steps = validate_and_normalize_steps(payload, len(units))
                return ProcessResult(
                    item=item,
                    ok=True,
                    enriched_row=enrich_row(item.row, steps),
                    attempts=attempts,
                    error=None,
                    usage=total_usage,
                )
            except Exception as exc:  # noqa: BLE001 - all failures are retryable here.
                last_error = f"{type(exc).__name__}: {exc}"
                if is_non_retryable_api_error(last_error):
                    break
                if attempt < max_attempts:
                    await asyncio.sleep(min(60.0, (2 ** (attempt - 1)) + random.random()))

    if last_payload is not None and last_error and "StepValidationError" in last_error:
        try:
            fallback_steps = fallback_steps_from_payload(last_payload, len(units))
            return ProcessResult(
                item=item,
                ok=True,
                enriched_row=enrich_row(item.row, fallback_steps),
                attempts=attempts,
                error=None,
                usage=total_usage,
            )
        except Exception as exc:  # noqa: BLE001 - report original model failure below.
            last_error = f"{last_error}; fallback failed: {type(exc).__name__}: {exc}"

    return ProcessResult(
        item=item,
        ok=False,
        enriched_row=None,
        attempts=attempts,
        error=last_error,
        usage=total_usage,
    )


def completed_keys_by_file(specs: list[FileSpec]) -> dict[Path, dict[str, dict[str, Any]]]:
    return {spec.input_path: load_completed(spec.partial_path) for spec in specs}


def build_work_items(
    specs: list[FileSpec],
    mode: str,
    pilot_total: int | None,
) -> list[WorkItem]:
    completed_by_file = completed_keys_by_file(specs)
    work_items: list[WorkItem] = []

    if mode == "pilot":
        assert pilot_total is not None
        pilot_counts = allocate_pilot_counts(pilot_total, len(specs))
    else:
        pilot_counts = [0 for _ in specs]

    for spec_idx, spec in enumerate(specs):
        if not spec.input_path.exists():
            raise FileNotFoundError(f"missing input file: {spec.input_path}")

        rows = load_jsonl(spec.input_path)
        completed = completed_by_file[spec.input_path]

        if mode == "pilot":
            selected_indices = set(choose_pilot_indices(len(rows), pilot_counts[spec_idx]))
        elif mode == "all":
            selected_indices = set(range(len(rows)))
        else:
            raise ValueError(f"unsupported mode: {mode}")

        for source_index, row in enumerate(rows):
            if source_index not in selected_indices:
                continue
            if make_row_key(row) in completed:
                continue
            work_items.append(WorkItem(spec=spec, source_index=source_index, row=row))

    return work_items


def checkpoint_record(result: ProcessResult) -> dict[str, Any]:
    if result.enriched_row is None:
        raise ValueError("cannot checkpoint failed result")

    record = dict(result.enriched_row)
    record["_checkpoint"] = {
        "input_file": result.item.spec.input_path.name,
        "source_index": result.item.source_index,
        "key": make_row_key(result.item.row),
        "attempts": result.attempts,
        "usage": result.usage,
        "completed_at": utc_now_iso(),
    }
    return record


def failed_record(result: ProcessResult, run_id: str) -> dict[str, Any]:
    row = result.item.row
    return {
        "run_id": run_id,
        "failed_at": utc_now_iso(),
        "input_file": result.item.spec.input_path.name,
        "source_index": result.item.source_index,
        "key": make_row_key(row),
        "problem_id": row.get("problem_id"),
        "model_tag": row.get("model_tag"),
        "sample_id": row.get("sample_id"),
        "attempts": result.attempts,
        "usage": result.usage,
        "error": (result.error or "unknown error")[:2000],
    }


async def run_processing(args: argparse.Namespace, specs: list[FileSpec], mode: str) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is not set", file=sys.stderr)
        return 2

    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError:
        print(
            "error: the openai package is not installed in this Python environment.\n"
            "Install it with: python -m pip install openai\n"
            "Or run this script with a Python environment that already has openai installed.",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_items = build_work_items(specs, mode=mode, pilot_total=args.pilot if mode == "pilot" else None)
    if not work_items:
        print(f"No rows to process for mode={mode}; existing checkpoints cover the selection.")
        if mode == "pilot":
            write_pilot_preview(specs, args.pilot, output_dir)
        elif mode == "all":
            finalize_all_outputs(specs)
        return 0

    print(
        f"Processing {len(work_items)} rows with model={args.model}, "
        f"concurrency={args.concurrency}, max_attempts={args.max_retries}"
    )

    client = AsyncOpenAI(timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            process_item(
                client=client,
                item=item,
                semaphore=semaphore,
                model=args.model,
                temperature=args.temperature,
                max_attempts=args.max_retries,
            )
        )
        for item in work_items
    ]

    partial_handles = {
        spec.partial_path: spec.partial_path.open("a", encoding="utf-8") for spec in specs
    }
    failed_path = output_dir / "failed_rows.jsonl"
    failed_handle = failed_path.open("a", encoding="utf-8")
    run_id = f"{mode}-{int(time.time())}"

    success_count = 0
    failure_count = 0
    retried_count = 0
    total_usage: dict[str, int] = {}

    try:
        for completed_count, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            add_usage(total_usage, result.usage)
            if result.attempts > 1:
                retried_count += 1

            if result.ok:
                handle = partial_handles[result.item.spec.partial_path]
                handle.write(jsonl_dumps(checkpoint_record(result)) + "\n")
                handle.flush()
                success_count += 1
            else:
                failed_handle.write(jsonl_dumps(failed_record(result, run_id)) + "\n")
                failed_handle.flush()
                failure_count += 1

            if completed_count == len(tasks) or completed_count % args.progress_every == 0:
                print(
                    f"progress {completed_count}/{len(tasks)} "
                    f"ok={success_count} failed={failure_count} retried={retried_count}"
                )
    finally:
        for handle in partial_handles.values():
            handle.close()
        failed_handle.close()
        await client.close()

    print_summary(
        mode=mode,
        processed=len(work_items),
        success_count=success_count,
        failure_count=failure_count,
        retried_count=retried_count,
        total_usage=total_usage,
        model=args.model,
    )

    if mode == "pilot":
        write_pilot_preview(specs, args.pilot, output_dir)
    elif mode == "all":
        finalize_all_outputs(specs)

    return 1 if failure_count else 0


def print_summary(
    mode: str,
    processed: int,
    success_count: int,
    failure_count: int,
    retried_count: int,
    total_usage: dict[str, int],
    model: str,
) -> None:
    prompt_tokens = total_usage.get("prompt_tokens", 0)
    completion_tokens = total_usage.get("completion_tokens", 0)
    total_tokens = total_usage.get("total_tokens", prompt_tokens + completion_tokens)
    print("summary")
    print(f"  mode: {mode}")
    print(f"  processed: {processed}")
    print(f"  succeeded: {success_count}")
    print(f"  failed: {failure_count}")
    print(f"  rows_with_retries: {retried_count}")
    print(f"  prompt_tokens: {prompt_tokens}")
    print(f"  completion_tokens: {completion_tokens}")
    print(f"  total_tokens: {total_tokens}")
    if model == "gpt-4o-mini" and (prompt_tokens or completion_tokens):
        cost = (
            prompt_tokens / 1_000_000 * GPT4O_MINI_INPUT_PER_MILLION
            + completion_tokens / 1_000_000 * GPT4O_MINI_OUTPUT_PER_MILLION
        )
        print(f"  estimated_cost_usd_for_gpt-4o-mini: {cost:.4f}")


def write_pilot_preview(specs: list[FileSpec], pilot_total: int, output_dir: Path) -> None:
    pilot_counts = allocate_pilot_counts(pilot_total, len(specs))
    preview_rows: list[dict[str, Any]] = []

    for spec_idx, spec in enumerate(specs):
        input_rows = load_jsonl(spec.input_path)
        selected_indices = set(choose_pilot_indices(len(input_rows), pilot_counts[spec_idx]))
        completed = load_completed(spec.partial_path)

        for source_index, row in enumerate(input_rows):
            if source_index not in selected_indices:
                continue
            enriched = completed.get(make_row_key(row))
            if enriched is None:
                continue
            preview = strip_checkpoint(enriched)
            preview["_source_file"] = spec.input_path.name
            preview["_source_index"] = source_index
            preview_rows.append(preview)

    preview_path = output_dir / "pilot_preview.jsonl"
    tmp_path = preview_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in preview_rows:
            f.write(jsonl_dumps(row) + "\n")
    tmp_path.replace(preview_path)
    print(f"wrote pilot preview: {preview_path} ({len(preview_rows)} rows)")


def finalize_all_outputs(specs: list[FileSpec]) -> bool:
    all_complete = True
    for spec in specs:
        input_rows = load_jsonl(spec.input_path)
        completed = load_completed(spec.partial_path)
        output_rows: list[dict[str, Any]] = []
        missing: list[int] = []

        for source_index, row in enumerate(input_rows):
            enriched = completed.get(make_row_key(row))
            if enriched is None:
                missing.append(source_index)
                continue
            output_rows.append(strip_checkpoint(enriched))

        if missing:
            all_complete = False
            print(
                f"not finalizing {spec.output_path.name}: "
                f"{len(missing)} missing rows; first missing indices={missing[:10]}",
                file=sys.stderr,
            )
            continue

        tmp_path = spec.output_path.with_suffix(".jsonl.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(jsonl_dumps(row) + "\n")
        tmp_path.replace(spec.output_path)
        print(f"wrote final output: {spec.output_path} ({len(output_rows)} rows)")

    return all_complete


def logical_steps_to_steps(logical_steps: Any) -> list[dict[str, Any]]:
    if not isinstance(logical_steps, dict):
        raise StepValidationError("logical_steps must be an object")
    if not logical_steps:
        return []

    expected_keys = [f"s{i}" for i in range(len(logical_steps))]
    if list(logical_steps.keys()) != expected_keys:
        raise StepValidationError(f"logical_steps keys must be exactly {expected_keys}")

    steps: list[dict[str, Any]] = []
    for key in expected_keys:
        step = logical_steps[key]
        if not isinstance(step, dict):
            raise StepValidationError(f"{key} must be an object")
        steps.append({"title": step.get("title"), "unit_ids": step.get("unit_ids")})
    return steps


def validate_enriched_row(input_row: dict[str, Any], output_row: dict[str, Any]) -> None:
    if make_row_key(input_row) != make_row_key(output_row):
        raise StepValidationError("output row key does not match input row")

    units = input_row.get("reasoning_units")
    output_units = output_row.get("reasoning_units")
    if units != output_units:
        raise StepValidationError("output reasoning_units must match input exactly")
    if not isinstance(units, list) or any(not isinstance(unit, str) for unit in units):
        raise StepValidationError("reasoning_units must be a list of strings")

    logical_steps = output_row.get("logical_steps")
    steps = logical_steps_to_steps(logical_steps)
    normalized = validate_and_normalize_steps({"steps": steps}, len(units))

    assert isinstance(logical_steps, dict)
    for idx, step in enumerate(normalized):
        key = f"s{idx}"
        expected_content = "\n".join(units[unit_id] for unit_id in step["unit_ids"])
        actual_content = logical_steps[key].get("content")
        if actual_content != expected_content:
            raise StepValidationError(f"{key}.content does not match source reasoning units")


def validate_outputs(specs: list[FileSpec]) -> int:
    errors: list[str] = []
    for spec in specs:
        if not spec.output_path.exists():
            errors.append(f"missing output file: {spec.output_path}")
            continue

        input_rows = load_jsonl(spec.input_path)
        output_rows = load_jsonl(spec.output_path)
        if len(input_rows) != len(output_rows):
            errors.append(
                f"{spec.output_path.name}: expected {len(input_rows)} rows, got {len(output_rows)}"
            )
            continue

        for idx, (input_row, output_row) in enumerate(zip(input_rows, output_rows, strict=True)):
            try:
                validate_enriched_row(input_row, output_row)
            except Exception as exc:  # noqa: BLE001 - report all validation failures uniformly.
                errors.append(f"{spec.output_path.name}:{idx}: {type(exc).__name__}: {exc}")
                if len(errors) >= 20:
                    break

    if errors:
        print("validation failed", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        return 1

    print("validation passed")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster Math500 reasoning units into logical steps with OpenAI."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--all", action="store_true", help="process all rows")
    mode_group.add_argument(
        "--pilot",
        nargs="?",
        type=int,
        const=50,
        default=None,
        help="process a stratified pilot sample; default size is 50",
    )
    mode_group.add_argument("--validate-only", action="store_true", help="validate final outputs only")

    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="maximum API attempts per row",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_INPUT_FILES),
        help="input JSONL filenames relative to --input-dir",
    )
    args = parser.parse_args(argv)

    if not args.all and args.pilot is None and not args.validate_only:
        args.pilot = 50

    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.max_retries <= 0:
        parser.error("--max-retries must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.pilot is not None and args.pilot <= 0:
        parser.error("--pilot must be positive")

    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    specs = make_file_specs(Path(args.input_dir), Path(args.output_dir), list(args.files))

    if args.validate_only:
        return validate_outputs(specs)
    if args.all:
        return asyncio.run(run_processing(args, specs, mode="all"))
    return asyncio.run(run_processing(args, specs, mode="pilot"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
