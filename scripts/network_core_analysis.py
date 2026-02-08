# scripts/network_core_analysis.py

from pathlib import Path
import argparse
from networkx.exception import PowerIterationFailedConvergence


import numpy as np
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------
# Paths / constants (aligned with your existing project)
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

UNWANTED = [66, 67, 68, 69, 70, 71, 72]

MATRIX_FILES = {
    "1": "nk_All_060102final_56sites_Model.xlsx",
    "2": "nk_All_060103final_56sites_Model.xlsx",
}

# 23 sites that are common across ALL constant-P0 models in BOTH matrices
CORE_23 = [
    10, 12, 15, 16, 17,
    20, 26, 29, 31, 32,
    33, 36, 37, 40, 41,
    44, 47, 49, 51, 52,
    53, 54, 59,
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_connectivity(matrix_id: str):
    """
    Load the Excel connectivity matrix for the chosen matrix_id ("1" or "2"),
    drop sites 66-72, and return:

        labels: np.array of site IDs (e.g., [1,2,3,...,60 minus dropped])
        P:      2D np.array of internal connectivity (donor->receiver)

    We use the *raw* P (before scaling / surrogate), because we want the
    actual plumbing, not the surrogate objective.
    """
    xlsx_name = MATRIX_FILES[matrix_id]
    excel_path = DATA_DIR / xlsx_name

    arr = pd.read_excel(excel_path, header=None).values

    # First row (except col 0) = site labels
    labels = arr[0, 1:].astype(int)
    # Main matrix starts at row 1, col 1
    P_full = arr[1:, 1:].astype(float)

    # Drop unwanted sites (66-72)
    mask = ~np.isin(labels, UNWANTED)
    labels = labels[mask]
    P = P_full[np.ix_(mask, mask)]

    return labels, P


def build_graph(labels: np.ndarray, P: np.ndarray) -> nx.DiGraph:
    """
    Build a directed weighted graph from the connectivity matrix.

    Nodes are site IDs (e.g., 1..60 minus dropped).
    Edge weight = larval connectivity from i -> j.

    We ignore zero entries.
    """
    G = nx.DiGraph()

    # Add nodes explicitly for clarity
    for lab in labels:
        G.add_node(int(lab))

    n = len(labels)
    for i in range(n):
        src = int(labels[i])
        for j in range(n):
            w = float(P[i, j])
            if w > 0.0:
                dst = int(labels[j])
                G.add_edge(src, dst, weight=w)

    return G


def compute_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute network metrics on the full graph:

    - in_strength: sum of incoming weights
    - out_strength: sum of outgoing weights
    - betweenness_centrality (weighted)
    - eigenvector_centrality (weighted, power-iteration, works for disconnected)
    - pagerank (weighted)

    Returns a DataFrame with one row per site.
    """
    # Strength = weighted in/out degree
    in_strength = dict(G.in_degree(weight="weight"))
    out_strength = dict(G.out_degree(weight="weight"))

    # Betweenness centrality (weighted)
    betw = nx.betweenness_centrality(G, weight="weight", normalized=True)

    # Eigenvector centrality (weighted, power-method version)
    # This works even if the graph is disconnected.
    try:
        eig = nx.eigenvector_centrality(
            G,
            max_iter=1000,
            tol=1.0e-06,
            weight="weight",
        )
    except PowerIterationFailedConvergence:
        # If for some reason it still doesn't converge, just set all to 0
        eig = {n: 0.0 for n in G.nodes()}

    # PageRank (weighted)
    pr = nx.pagerank(G, weight="weight")

    # Assemble DataFrame
    nodes = sorted(G.nodes())
    df = pd.DataFrame({
        "site_id": nodes,
        "in_strength": [in_strength[n] for n in nodes],
        "out_strength": [out_strength[n] for n in nodes],
        "betweenness": [betw[n] for n in nodes],
        "eigenvector": [eig[n] for n in nodes],
        "pagerank": [pr[n] for n in nodes],
    })

    # Ranks (lower = more central)
    for col in ["in_strength", "out_strength", "betweenness", "eigenvector", "pagerank"]:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="min")

    # Simple average rank across metrics as a combined score
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    df["avg_rank"] = df[rank_cols].mean(axis=1)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        choices=MATRIX_FILES.keys(),
        default="1",
        help="Which connectivity matrix to use (1 or 2)",
    )
    args = parser.parse_args()

    RUNS_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"NETWORK ANALYSIS ON CORE SITES (matrix {args.matrix})")
    print("=" * 80)

    # 1) Load connectivity
    labels, P = load_connectivity(args.matrix)
    print(f"[load] Matrix {args.matrix}: {len(labels)} sites after dropping {UNWANTED}")

    # 2) Build graph
    G = build_graph(labels, P)
    print(f"[graph] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # 3) Compute metrics on full network
    df_all = compute_metrics(G)

    # 4) Filter to the 23-core sites
    core_set = set(CORE_23)
    df_core = df_all[df_all["site_id"].isin(core_set)].copy()
    df_core = df_core.sort_values("avg_rank")

    # 5) Pick top 7 "super-core" sites by avg_rank
    df_core7 = df_core.head(23).copy()

    # Save outputs
    out_all = RUNS_DIR / f"network_metrics_all_matrix{args.matrix}.csv"
    out_core = RUNS_DIR / f"network_metrics_core23_matrix{args.matrix}.csv"
    out_core7 = RUNS_DIR / f"network_core7_matrix{args.matrix}.csv"

    df_all.to_csv(out_all, index=False)
    df_core.to_csv(out_core, index=False)
    df_core7.to_csv(out_core7, index=False)

    print(f"[save] all-site metrics  -> {out_all}")
    print(f"[save] core-23 metrics   -> {out_core}")
    print(f"[save] top-7 core sites  -> {out_core7}")
    print("\nTop 7 core sites (by avg_rank across metrics):")
    print(df_core7[["site_id", "in_strength", "out_strength",
                    "betweenness", "eigenvector", "pagerank", "avg_rank"]])


if __name__ == "__main__":
    main()
