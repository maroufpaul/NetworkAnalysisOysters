# scripts/network_core_analysis.py
#
# Network-centrality analysis of the TSPS larval-connectivity matrices.
#
# What this does (and how it differs from the earlier version):
#   * Ranks ALL 49 candidate sites (66-72 dropped), not a 23-site subset.
#   * Treats the connectivity entry as a *strength* (higher = stronger link).
#     For betweenness this means edge DISTANCE = 1 / weight, so strong links
#     are short paths.  (The earlier version passed the strength directly as a
#     distance, which rewarded weak connections -- that was a bug.)
#   * Separates self-recruitment (the matrix diagonal) into its own column
#     instead of folding it into in/out strength.
#   * --self-loops {on,off} controls whether the diagonal is fed back into the
#     spectral metrics (pagerank, eigenvector).  Default "on", matching the
#     canonical model that keeps self-recruitment.  Betweenness and strength
#     always use the off-diagonal transport network.
#   * Flags the seven global-backbone sites and prints where each lands.

from pathlib import Path
import argparse
from networkx.exception import PowerIterationFailedConvergence

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

UNWANTED = [66, 67, 68, 69, 70, 71, 72]

MATRIX_FILES = {
    "1": "nk_All_060102final_56sites_Model.xlsx",
    "2": "nk_All_060103final_56sites_Model.xlsx",
}

# Seven sites selected by every optimized design (the global backbone).
BACKBONE_7 = [10, 31, 37, 40, 41, 49, 53]

# Metrics that feed the combined average rank.
RANK_METRICS = ["out_strength", "in_strength", "betweenness", "eigenvector", "pagerank"]


def load_connectivity(matrix_id: str):
    """Load raw connectivity for matrix_id, drop 66-72, return (labels, P)
    with P[i, j] = larval flow from donor i to receiver j."""
    excel_path = DATA_DIR / MATRIX_FILES[matrix_id]
    arr = pd.read_excel(excel_path, header=None).values

    labels = arr[0, 1:].astype(int)
    P_full = arr[1:, 1:].astype(float)

    mask = ~np.isin(labels, UNWANTED)
    labels = labels[mask]
    P = P_full[np.ix_(mask, mask)]
    return labels, P


def build_graph(labels: np.ndarray, P: np.ndarray, self_loops: bool) -> nx.DiGraph:
    """Directed weighted transport graph. Off-diagonal edges carry weight
    (strength) and distance = 1/weight. Diagonal added as self-loops only when
    self_loops is True, and self-loops carry no distance."""
    G = nx.DiGraph()
    for lab in labels:
        G.add_node(int(lab))

    n = len(labels)
    for i in range(n):
        src = int(labels[i])
        for j in range(n):
            w = float(P[i, j])
            if w <= 0.0:
                continue
            dst = int(labels[j])
            if i == j:
                if self_loops:
                    G.add_edge(src, dst, weight=w)
            else:
                G.add_edge(src, dst, weight=w, distance=1.0 / w)
    return G


def compute_metrics(labels: np.ndarray, P: np.ndarray, self_loops: bool) -> pd.DataFrame:
    n = len(labels)
    off = P.copy()
    diag = np.diag(off).copy()
    np.fill_diagonal(off, 0.0)

    out_strength = off.sum(axis=1)
    in_strength = off.sum(axis=0)

    G = build_graph(labels, P, self_loops=self_loops)

    betw = nx.betweenness_centrality(G, weight="distance", normalized=True)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=2000, tol=1.0e-06, weight="weight")
    except PowerIterationFailedConvergence:
        eig = {int(l): 0.0 for l in labels}
    pr = nx.pagerank(G, weight="weight")

    rows = []
    for i, lab in enumerate(labels):
        lab = int(lab)
        rows.append({
            "site_id": lab,
            "out_strength": out_strength[i],
            "in_strength": in_strength[i],
            "self_recruitment": diag[i],
            "betweenness": betw.get(lab, 0.0),
            "eigenvector": eig.get(lab, 0.0),
            "pagerank": pr.get(lab, 0.0),
            "backbone": lab in set(BACKBONE_7),
        })
    df = pd.DataFrame(rows)

    for col in RANK_METRICS:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="min").astype(int)
    df["avg_rank"] = df[[f"rank_{c}" for c in RANK_METRICS]].mean(axis=1)

    return df.sort_values("avg_rank").reset_index(drop=True)


def run_one(matrix_id: str, self_loops: bool):
    labels, P = load_connectivity(matrix_id)
    df = compute_metrics(labels, P, self_loops=self_loops)

    show = ["site_id", "backbone", "out_strength", "in_strength",
            "self_recruitment", "betweenness", "eigenvector", "pagerank", "avg_rank"]
    rank_show = ["site_id"] + [f"rank_{c}" for c in RANK_METRICS] + ["avg_rank"]
    bb = df[df["backbone"]].sort_values("avg_rank")
    r37 = df[df["site_id"] == 37].iloc[0]

    # Build the report as text so it goes to both the screen and a file, in order.
    lines = []
    lines.append("=" * 92)
    lines.append(f"NETWORK CENTRALITY, ALL {len(labels)} SITES  |  matrix {matrix_id}  "
                 f"|  self-loops {'ON' if self_loops else 'OFF'}")
    lines.append("=" * 92)
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        lines.append("\nFull ranking (most central first):")
        lines.append(df[show].to_string(index=False))
    with pd.option_context("display.width", 200,
                           "display.float_format", lambda v: f"{v:,.1f}"):
        lines.append("\nBackbone-7 placement (rank out of 49 per metric; 1 = most central):")
        lines.append(bb[rank_show].to_string(index=False))
    lines.append(f"\nSite 37: out-strength rank {r37['rank_out_strength']}/49, "
                 f"in-strength rank {r37['rank_in_strength']}/49, "
                 f"betweenness rank {r37['rank_betweenness']}/49, "
                 f"pagerank rank {r37['rank_pagerank']}/49, "
                 f"avg_rank {r37['avg_rank']:.1f}.")
    report = "\n".join(lines)
    print(report)

    RUNS_DIR.mkdir(exist_ok=True)
    tag = "selfloops_on" if self_loops else "selfloops_off"
    csv_out = RUNS_DIR / f"network_metrics_all49_matrix{matrix_id}_{tag}.csv"
    txt_out = RUNS_DIR / f"network_ranking_matrix{matrix_id}_{tag}.txt"
    df.to_csv(csv_out, index=False)              # rows already sorted by avg_rank
    txt_out.write_text(report + "\n")
    print(f"\n[save] ordered metrics CSV  -> {csv_out}")
    print(f"[save] readable ranking TXT -> {txt_out}\n")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    parser.add_argument("--self-loops", choices=["on", "off"], default="on",
                        help="Include the diagonal (self-recruitment) in spectral "
                             "metrics. Default on (canonical model).")
    args = parser.parse_args()

    self_loops = (args.self_loops == "on")
    matrices = ["1", "2"] if args.matrix == "both" else [args.matrix]
    for mid in matrices:
        run_one(mid, self_loops=self_loops)


if __name__ == "__main__":
    main()