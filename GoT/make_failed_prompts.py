import json
import os
import pandas as pd

prompts_file = os.path.expanduser("~/Desktop/semantic_edge_prompts_math500_base.jsonl")
matrices_file = os.path.expanduser("~/Desktop/semantic_edge_matrices_math500_base.jsonl")
failed_prompts_file = os.path.expanduser("~/Desktop/semantic_edge_prompts_failed_retry_base.jsonl")

all_failed_graph_ids = set()

with open(matrices_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        samples = row["samples"]

        valid_count = sum(
            1 for sample in samples
            if sample.get("matrix") is not None
        )

        if valid_count == 0:
            all_failed_graph_ids.add(row["graph_id"])

print("All-failed graph count:", len(all_failed_graph_ids))

prompts_df = pd.read_json(prompts_file, lines=True)

failed_rows = []

for _, row in prompts_df.iterrows():
    if row["graph_id"] in all_failed_graph_ids:
        row_dict = row.to_dict()

        row_dict["prompt"] = row_dict["prompt"] + """

IMPORTANT REMINDER:
Return exactly the required number of rows and columns.
Do not omit the final row, even if it is all zeros.
Every row must have exactly the required number of integers.
"""

        failed_rows.append(row_dict)

with open(failed_prompts_file, "w", encoding="utf-8") as f:
    for row in failed_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved failed retry prompts to:", failed_prompts_file)
print("Rows saved:", len(failed_rows))