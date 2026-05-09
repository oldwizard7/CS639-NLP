import pandas as pd
import json
import os

# =========================
# File paths
# =========================

input_file = os.path.expanduser("~/Desktop/clustered_reasoning_units_math500_base.jsonl")

nodes_output_file = os.path.expanduser("~/Desktop/reasoning_graph_nodes_math500_base.jsonl")
prompts_output_file = os.path.expanduser("~/Desktop/semantic_edge_prompts_math500_base.jsonl")

# =========================
# Read clustered reasoning file
# =========================

df = pd.read_json(input_file, lines=True)


# =========================
# Convert logical_steps to graph nodes
# =========================

def logical_steps_to_nodes(logical_steps):
    """
    Convert logical_steps dictionary into ordered graph nodes.

    Example:
    logical_steps = {
        "s0": {"title": "...", "unit_ids": [...], "content": "..."},
        "s1": {"title": "...", "unit_ids": [...], "content": "..."}
    }

    Output:
    [
        {"node_id": 0, "step_key": "s0", "title": "...", "unit_ids": [...], "text": "..."},
        {"node_id": 1, "step_key": "s1", "title": "...", "unit_ids": [...], "text": "..."}
    ]
    """
    nodes = []

    sorted_items = sorted(
        logical_steps.items(),
        key=lambda item: int(item[0].replace("s", ""))
    )

    for step_key, step_info in sorted_items:
        node_id = int(step_key.replace("s", ""))

        node = {
            "node_id": node_id,
            "step_key": step_key,
            "title": step_info.get("title", ""),
            "unit_ids": step_info.get("unit_ids", []),
            "text": step_info.get("content", "")
        }

        nodes.append(node)

    return nodes


# =========================
# Build semantic edge prompt
# =========================

def build_semantic_edge_prompt(nodes):
    """
    Build prompt for LLM semantic dependency detection.

    The LLM should return a K x K adjacency matrix where:
    1 = support
    -1 = contradiction
    0 = independent
    """
    K = len(nodes)

    steps_text = ""

    for node in nodes:
        steps_text += f"Step {node['node_id']} - {node['title']}:\n"
        steps_text += node["text"]
        steps_text += "\n\n"

    # Create a dynamic example matrix with the correct K x K size
    example_matrix = [[0 for _ in range(K)] for _ in range(K)]

    # Add simple example support edges if possible
    if K >= 2:
        example_matrix[0][1] = 1
    if K >= 3:
        example_matrix[1][2] = 1

    example_json = json.dumps(
        {"matrix": example_matrix},
        indent=2
    )

    prompt = f"""
You are given an ordered list of reasoning steps from a language model's solution.

Your task is to infer semantic relationships between reasoning steps.

For every pair of steps (i, j) where i < j, classify the relationship from step i to step j as:

1 = step i supports step j
-1 = step i contradicts step j
0 = step i is independent of step j

Return ONLY valid JSON in this exact format:

{example_json}

Rules:
- The matrix must be {K} x {K}.
- The diagonal must be 0.
- Entries where i >= j must be 0.
- Entries where i < j must be one of -1, 0, or 1.
- Do not include explanation.
- Do not include markdown.
- Do not wrap the JSON in ```json or any code block.

Reasoning steps:

{steps_text}
"""
    return prompt.strip()


# =========================
# Main processing
# =========================

graph_rows = []
prompt_rows = []

for _, row in df.iterrows():
    nodes = logical_steps_to_nodes(row["logical_steps"])

    graph_id = f"{row['problem_id']}__sample_{row['sample_id']}"

    graph_row = {
        "graph_id": graph_id,
        "problem_id": row["problem_id"],
        "model_tag": row["model_tag"],
        "model_name": row["model_name"],
        "sample_id": int(row["sample_id"]),
        "finish_reason": row["finish_reason"],
        "nodes": nodes,
        "num_nodes": len(nodes),
        "edges": []
    }

    prompt_row = {
        "graph_id": graph_id,
        "problem_id": row["problem_id"],
        "model_tag": row["model_tag"],
        "model_name": row["model_name"],
        "sample_id": int(row["sample_id"]),
        "num_nodes": len(nodes),
        "prompt": build_semantic_edge_prompt(nodes)
    }

    graph_rows.append(graph_row)
    prompt_rows.append(prompt_row)


# =========================
# Save graph nodes JSONL
# =========================

with open(nodes_output_file, "w", encoding="utf-8") as f:
    for row in graph_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Save semantic edge prompts JSONL
# =========================

with open(prompts_output_file, "w", encoding="utf-8") as f:
    for row in prompt_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Print check
# =========================

print(f"Saved graph nodes to: {nodes_output_file}")
print(f"Saved semantic edge prompts to: {prompts_output_file}")
print(f"Total rows: {len(graph_rows)}")

print("\nExample graph nodes:")
print(json.dumps(graph_rows[0], indent=2, ensure_ascii=False))


print("\nExample prompt:")
print(prompt_rows[0]["prompt"])