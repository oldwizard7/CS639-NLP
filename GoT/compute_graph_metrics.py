import json
import os
import pandas as pd
import numpy as np


# =========================
# Input files
# =========================

GRAPH_FILES = {
    "base": os.path.expanduser("~/Desktop/reasoning_graph_final_math500_base_merged.jsonl"),
    "sft": os.path.expanduser("~/Desktop/reasoning_graph_final_math500_sft_merged.jsonl"),
    "drpo": os.path.expanduser("~/Desktop/reasoning_graph_final_math500_drpo.jsonl"),
}

PER_GRAPH_OUTPUT = os.path.expanduser("~/Desktop/graph_metrics_per_sample.csv")
SUMMARY_OUTPUT = os.path.expanduser("~/Desktop/graph_metrics_summary.csv")


def safe_divide(a, b):
    if b == 0:
        return 0.0
    return a / b


def compute_metrics_for_graph(row, model_tag_from_file):
    """
    Compute graph metrics for one reasoning graph.

    Nodes:
      V = reasoning steps

    Edges:
      E = support/contradiction semantic relationships
    """
    nodes = row.get("nodes", [])
    edges = row.get("edges", [])

    n = len(nodes)
    m = len(edges)

    # Initialize degree counts
    out_degree = {i: 0 for i in range(n)}
    in_degree = {i: 0 for i in range(n)}
    total_degree = {i: 0 for i in range(n)}

    support_edges = 0
    contradiction_edges = 0
    weights = []

    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        relation = edge.get("relation", "")
        weight = float(edge.get("weight", 1.0))

        if source not in out_degree:
            out_degree[source] = 0
            total_degree[source] = 0

        if target not in in_degree:
            in_degree[target] = 0
            total_degree[target] = 0

        out_degree[source] += 1
        in_degree[target] += 1
        total_degree[source] += 1
        total_degree[target] += 1

        weights.append(weight)

        if relation == "support":
            support_edges += 1
        elif relation == "contradiction":
            contradiction_edges += 1

    # =========================
    # Paper-style metrics
    # =========================

    # Exploration Density:
    # rho_E = |E| / (|V|(|V|-1))
    exploration_density = safe_divide(m, n * (n - 1))

    # Since your graph only allows forward edges i < j,
    # this is also useful:
    # max possible forward edges = n(n-1)/2
    forward_density = safe_divide(m, n * (n - 1) / 2)

    # Branching Ratio:
    # fraction of nodes with out-degree > 1
    branching_ratio = safe_divide(
        sum(1 for i in range(n) if out_degree.get(i, 0) > 1),
        n
    )

    # Convergence Ratio:
    # fraction of nodes with in-degree > 1
    convergence_ratio = safe_divide(
        sum(1 for i in range(n) if in_degree.get(i, 0) > 1),
        n
    )

    # Linearity:
    # paper formula:
    # l = 1 - |{s in V | d(s) > 2}| / |V|
    linearity = 1 - safe_divide(
        sum(1 for i in range(n) if total_degree.get(i, 0) > 2),
        n
    )

    # Extra helpful metrics
    mean_out_degree = safe_divide(sum(out_degree.values()), n)
    mean_in_degree = safe_divide(sum(in_degree.values()), n)
    mean_total_degree = safe_divide(sum(total_degree.values()), n)

    avg_edge_weight = float(np.mean(weights)) if weights else 0.0
    avg_abs_edge_weight = float(np.mean([abs(w) for w in weights])) if weights else 0.0

    max_out_degree = max(out_degree.values()) if out_degree else 0
    max_in_degree = max(in_degree.values()) if in_degree else 0
    max_total_degree = max(total_degree.values()) if total_degree else 0

    return {
        "graph_id": row.get("graph_id"),
        "problem_id": row.get("problem_id"),
        "sample_id": row.get("sample_id"),
        "model_tag": row.get("model_tag", model_tag_from_file),
        "model_name": row.get("model_name"),
        "finish_reason": row.get("finish_reason"),
        "edge_status": row.get("edge_status"),

        # Basic graph size
        "num_nodes": n,
        "num_edges": m,

        # Paper metrics
        "exploration_density": exploration_density,
        "forward_density": forward_density,
        "branching_ratio": branching_ratio,
        "convergence_ratio": convergence_ratio,
        "linearity": linearity,

        # Edge types
        "support_edges": support_edges,
        "contradiction_edges": contradiction_edges,
        "support_edge_ratio": safe_divide(support_edges, m),
        "contradiction_edge_ratio": safe_divide(contradiction_edges, m),

        # Weight/confidence metrics
        "avg_edge_weight": avg_edge_weight,
        "avg_abs_edge_weight": avg_abs_edge_weight,

        # Degree metrics
        "mean_out_degree": mean_out_degree,
        "mean_in_degree": mean_in_degree,
        "mean_total_degree": mean_total_degree,
        "max_out_degree": max_out_degree,
        "max_in_degree": max_in_degree,
        "max_total_degree": max_total_degree,
    }


def read_jsonl(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def main():
    all_metric_rows = []

    for model_tag, file_path in GRAPH_FILES.items():
        print(f"Reading {model_tag}: {file_path}")

        if not os.path.exists(file_path):
            print(f"WARNING: file not found: {file_path}")
            continue

        graph_rows = read_jsonl(file_path)

        print(f"  Loaded graphs: {len(graph_rows)}")

        for row in graph_rows:
            metric_row = compute_metrics_for_graph(row, model_tag)
            all_metric_rows.append(metric_row)

    metrics_df = pd.DataFrame(all_metric_rows)

    # Save per-sample metrics
    metrics_df.to_csv(PER_GRAPH_OUTPUT, index=False)

    # Summary by model
    summary_df = (
        metrics_df
        .groupby("model_tag")
        .agg(
            num_graphs=("graph_id", "count"),

            mean_nodes=("num_nodes", "mean"),
            sd_nodes=("num_nodes", "std"),

            mean_edges=("num_edges", "mean"),
            sd_edges=("num_edges", "std"),

            mean_exploration_density=("exploration_density", "mean"),
            sd_exploration_density=("exploration_density", "std"),

            mean_forward_density=("forward_density", "mean"),
            sd_forward_density=("forward_density", "std"),

            mean_branching_ratio=("branching_ratio", "mean"),
            sd_branching_ratio=("branching_ratio", "std"),

            mean_convergence_ratio=("convergence_ratio", "mean"),
            sd_convergence_ratio=("convergence_ratio", "std"),

            mean_linearity=("linearity", "mean"),
            sd_linearity=("linearity", "std"),

            mean_support_edges=("support_edges", "mean"),
            mean_contradiction_edges=("contradiction_edges", "mean"),

            mean_avg_abs_edge_weight=("avg_abs_edge_weight", "mean"),
            mean_total_degree=("mean_total_degree", "mean"),
        )
        .reset_index()
    )

    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    print("\nSaved per-sample metrics to:")
    print(PER_GRAPH_OUTPUT)

    print("\nSaved summary metrics to:")
    print(SUMMARY_OUTPUT)

    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()