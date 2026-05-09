import os
import json
import argparse
import pandas as pd
import numpy as np

# =========================
# Default file paths
# =========================

DEFAULT_NODES_FILE = os.path.expanduser("~/Desktop/reasoning_graph_nodes_math500_drpo.jsonl")
DEFAULT_MATRICES_FILE = os.path.expanduser("~/Desktop/semantic_edge_matrices_math500_drpo.jsonl")
DEFAULT_OUTPUT_FILE = os.path.expanduser("~/Desktop/reasoning_graph_final_math500_drpo.jsonl")


RELATION_LABELS = {
    1: "support",
    -1: "contradiction",
    0: "independent"
}


def aggregate_samples(samples, K):
    """
    Aggregate multiple sampled adjacency matrices.

    For each possible edge i -> j:
      p_support = fraction of samples predicting +1
      p_contradict = fraction of samples predicting -1
      p_independent = fraction of samples predicting 0

    Then compute:
      W = p_support - p_contradict

    W is the weighted signed confidence matrix.
    """
    valid_matrices = []

    for sample in samples:
        matrix = sample.get("matrix")

        if matrix is None:
            continue

        arr = np.array(matrix, dtype=int)

        if arr.shape != (K, K):
            continue

        # Force invalid backward/diagonal entries to 0
        for i in range(K):
            for j in range(K):
                if i >= j:
                    arr[i, j] = 0

        valid_matrices.append(arr)

    if len(valid_matrices) == 0:
        return None, None, None, None, 0

    stacked = np.stack(valid_matrices, axis=0)

    p_support = np.mean(stacked == 1, axis=0)
    p_contradict = np.mean(stacked == -1, axis=0)
    p_independent = np.mean(stacked == 0, axis=0)

    W = p_support - p_contradict

    return p_support, p_contradict, p_independent, W, len(valid_matrices)


def threshold_to_edges(W, tau_pos=0.5, tau_neg=0.5):
    """
    Convert weighted adjacency matrix W into final hard graph edges.

    If W[i][j] >= tau_pos:
        i supports j

    If W[i][j] <= -tau_neg:
        i contradicts j

    Otherwise:
        no edge
    """
    K = W.shape[0]
    A_final = np.zeros((K, K), dtype=int)
    edges = []

    for i in range(K):
        for j in range(i + 1, K):
            weight = float(W[i, j])

            if weight >= tau_pos:
                relation_value = 1
            elif weight <= -tau_neg:
                relation_value = -1
            else:
                relation_value = 0

            A_final[i, j] = relation_value

            if relation_value != 0:
                edges.append({
                    "source": i,
                    "target": j,
                    "relation": RELATION_LABELS[relation_value],
                    "weight": weight,
                    "confidence": abs(weight)
                })

    return A_final.tolist(), edges


def load_jsonl_as_dict(file_path, key="graph_id"):
    """
    Load a JSONL file and return dictionary:
      graph_id -> row
    """
    data = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                data[row[key]] = row

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes_file", default=DEFAULT_NODES_FILE)
    parser.add_argument("--matrices_file", default=DEFAULT_MATRICES_FILE)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)

    parser.add_argument("--tau_pos", type=float, default=0.5)
    parser.add_argument("--tau_neg", type=float, default=0.5)

    args = parser.parse_args()

    nodes_file = os.path.expanduser(args.nodes_file)
    matrices_file = os.path.expanduser(args.matrices_file)
    output_file = os.path.expanduser(args.output_file)

    print("Loading files...")
    print(f"Nodes file: {nodes_file}")
    print(f"Matrices file: {matrices_file}")

    nodes_df = pd.read_json(nodes_file, lines=True)
    matrices_by_graph_id = load_jsonl_as_dict(matrices_file, key="graph_id")

    final_rows = []

    ok_count = 0
    missing_count = 0
    no_valid_count = 0

    for _, graph_row in nodes_df.iterrows():
        graph_id = graph_row["graph_id"]
        K = int(graph_row["num_nodes"])

        graph_dict = graph_row.to_dict()

        if graph_id not in matrices_by_graph_id:
            graph_dict["edges"] = []
            graph_dict["num_edges"] = 0
            graph_dict["edge_status"] = "missing_edge_detection"
            final_rows.append(graph_dict)
            missing_count += 1
            continue

        matrix_row = matrices_by_graph_id[graph_id]
        samples = matrix_row["samples"]

        p_support, p_contradict, p_independent, W, valid_R = aggregate_samples(samples, K)

        if W is None:
            graph_dict["edges"] = []
            graph_dict["num_edges"] = 0
            graph_dict["edge_status"] = "no_valid_matrices"
            graph_dict["valid_R"] = valid_R
            final_rows.append(graph_dict)
            no_valid_count += 1
            continue

        A_final, edges = threshold_to_edges(
            W,
            tau_pos=args.tau_pos,
            tau_neg=args.tau_neg
        )

        graph_dict["edges"] = edges
        graph_dict["num_edges"] = len(edges)
        graph_dict["edge_status"] = "ok"
        graph_dict["valid_R"] = valid_R

        # Save matrices/probabilities for analysis
        graph_dict["A_final"] = A_final
        graph_dict["W"] = W.tolist()
        graph_dict["p_support"] = p_support.tolist()
        graph_dict["p_contradict"] = p_contradict.tolist()
        graph_dict["p_independent"] = p_independent.tolist()

        final_rows.append(graph_dict)
        ok_count += 1

    with open(output_file, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nDone.")
    print(f"Saved final graph file to: {output_file}")
    print(f"Total graphs: {len(final_rows)}")
    print(f"Graphs with processed edges: {ok_count}")
    print(f"Graphs missing edge detection: {missing_count}")
    print(f"Graphs with no valid matrices: {no_valid_count}")

    print("\nExample final graph:")
    print(json.dumps(final_rows[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()