import json
import os

original_file = os.path.expanduser("~/Desktop/semantic_edge_matrices_math500_base.jsonl")
retry_file = os.path.expanduser("~/Desktop/semantic_edge_matrices_failed_retry_base.jsonl")
merged_file = os.path.expanduser("~/Desktop/semantic_edge_matrices_math500_base_merged.jsonl")

rows_by_graph_id = {}

with open(original_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            rows_by_graph_id[row["graph_id"]] = row

with open(retry_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            rows_by_graph_id[row["graph_id"]] = row

with open(merged_file, "w", encoding="utf-8") as f:
    for row in rows_by_graph_id.values():
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved merged file to:", merged_file)
print("Total rows:", len(rows_by_graph_id))