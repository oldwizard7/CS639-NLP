#!/usr/bin/env python3
"""Merge remapped CHTC MATH outputs and enforce shared-base metadata."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


METADATA_FILES = ["problem.json", "base_solution.json", "chunks.json"]
BASE_TYPES = ["correct", "incorrect"]
ROLLOUT_TYPES = ["default", "forced_answer"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if filecmp.cmp(src, dst, shallow=False):
            return False
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite conflicting file: {dst}")
    shutil.copy2(src, dst)
    return True


def copy_tree_contents(src_root: Path, dst_root: Path, overwrite: bool = False) -> int:
    copied = 0
    for src in src_root.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(src_root)
        if copy_file(src, dst_root / rel, overwrite=overwrite):
            copied += 1
    return copied


def find_remapped_math_outputs(outputs_root: Path) -> Iterable[Path]:
    if not outputs_root.exists():
        return []
    return sorted(
        path
        for path in outputs_root.glob("*/math_rollouts")
        if path.is_dir()
    )


def rollout_dir(
    math_root: Path,
    model_slug: str,
    base_type: str,
    rollout_type: str,
    suffix: str,
) -> Path:
    name = f"{base_type}_base_solution"
    if rollout_type == "forced_answer":
        name += "_forced_answer"
    name += f"_{suffix}"
    return math_root / model_slug / "temperature_0.6_top_p_0.95" / name


def sync_manifest_metadata(
    manifest_path: Path,
    math_root: Path,
    rollout_suffix: str,
    overwrite: bool = False,
) -> int:
    if not manifest_path.exists():
        return 0

    manifest = load_json(manifest_path)
    copied = 0
    for problem in manifest.get("problems", []):
        problem_name = f"problem_{problem['problem_id']}"
        for model_slug, model_entry in problem.get("models", {}).items():
            for base_type in BASE_TYPES:
                base_entry = model_entry["bases"][base_type]
                export_dir = Path(base_entry["export_dir"])
                for rollout_type in ROLLOUT_TYPES:
                    target_problem_dir = (
                        rollout_dir(
                            math_root,
                            model_slug,
                            base_type,
                            rollout_type,
                            rollout_suffix,
                        )
                        / problem_name
                    )
                    if not target_problem_dir.exists():
                        continue
                    for filename in METADATA_FILES:
                        if copy_file(
                            export_dir / filename,
                            target_problem_dir / filename,
                            overwrite=overwrite,
                        ):
                            copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="chtc/outputs/qwen_anchor")
    parser.add_argument("--math-root", default="math_rollouts")
    parser.add_argument("--manifest", default="analysis/matched_math/math_manifest.json")
    parser.add_argument("--rollout-suffix", default="paper10_matched_shared_apr28")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite conflicting copied files instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    math_root = Path(args.math_root)

    copied = 0
    remapped_outputs = list(find_remapped_math_outputs(outputs_root))
    for output_dir in remapped_outputs:
        copied += copy_tree_contents(output_dir, math_root, overwrite=args.overwrite)

    metadata_copied = sync_manifest_metadata(
        Path(args.manifest),
        math_root,
        args.rollout_suffix,
        overwrite=args.overwrite,
    )

    print(
        f"Merged {len(remapped_outputs)} remapped MATH outputs into {math_root}; "
        f"copied {copied} files."
    )
    print(f"Synced {metadata_copied} shared-base metadata files.")


if __name__ == "__main__":
    main()
