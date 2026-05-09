import json
import tempfile
import unittest
from pathlib import Path

from select_matched_math_problems import BASE_TYPES, MODELS, select_and_export


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SelectMatchedMathProblemsTest(unittest.TestCase):
    def test_selects_only_fully_matched_problem_and_exports_base_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "math_rollouts"
            output_root = root / "analysis" / "matched_math"
            suffix = "fake"

            for _, model_slug in MODELS:
                for base_type in BASE_TYPES:
                    problem_dir = (
                        source_root
                        / model_slug
                        / "temperature_0.6_top_p_0.95"
                        / f"{base_type}_base_solution_{suffix}"
                        / "problem_1"
                    )
                    write_json(problem_dir / "problem.json", {"problem": "p1"})
                    write_json(
                        problem_dir / "base_solution.json",
                        {
                            "solution": "s",
                            "full_cot": "s",
                            "answer": "1",
                            "is_correct": base_type == "correct",
                        },
                    )
                    write_json(
                        problem_dir / "chunks.json",
                        {"chunks": ["first chunk.", "second chunk."]},
                    )

            # problem_2 is incomplete, so it must not be selected.
            incomplete_dir = (
                source_root
                / MODELS[0][1]
                / "temperature_0.6_top_p_0.95"
                / f"correct_base_solution_{suffix}"
                / "problem_2"
            )
            write_json(incomplete_dir / "problem.json", {"problem": "p2"})

            manifest = select_and_export(
                selected_problem_ids=[2, 1],
                source_root=source_root,
                output_root=output_root,
                source_suffix=suffix,
                target_count=1,
                min_chunks=2,
                max_chunks=80,
            )

            self.assertEqual(manifest["selected_problem_ids"], [1])
            exported = (
                output_root
                / "base_traces"
                / MODELS[0][1]
                / f"correct_base_solution_{suffix}"
                / "problem_1"
                / "chunks.json"
            )
            self.assertTrue(exported.exists())


if __name__ == "__main__":
    unittest.main()
