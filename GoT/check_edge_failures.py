import json
import os

file_path = os.path.expanduser("~/Desktop/semantic_edge_matrices_math500_base.jsonl")

total_graphs = 0
all_failed = 0
partial_failed = 0
all_success = 0

failed_graph_ids = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        total_graphs += 1

        samples = row["samples"]
        valid_count = sum(1 for s in samples if s.get("matrix") is not None)

        if valid_count == 0:
            all_failed += 1
            failed_graph_ids.append(row["graph_id"])
        elif valid_count < len(samples):
            partial_failed += 1
        else:
            all_success += 1

print("Total processed graphs:", total_graphs)
print("All samples successful:", all_success)
print("Partially failed graphs:", partial_failed)
print("All samples failed:", all_failed)

print("\nFirst 20 all-failed graph IDs:")
for gid in failed_graph_ids[:20]:
    print(gid)