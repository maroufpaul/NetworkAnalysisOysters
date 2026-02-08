#!/usr/bin/env python3
"""
Generate all presentation figures for the oyster reef site selection project.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
RUNS_DIR = ROOT / "runs"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------- color palette ----------
COLORS = {
    "Greedy+Local": "#2196F3",
    "Backward": "#FF9800",
    "MIQP": "#4CAF50",
    "MIQP+Comm": "#9C27B0",
    "MIQP+Comm+Size": "#F44336",
}

# ---------- site sets ----------
GREEDY_LOCAL = [4, 6, 10, 15, 19, 20, 21, 24, 27, 30, 31, 32, 36, 37, 38, 39, 40, 41, 47, 49, 51, 52, 53, 55, 60]
BACKWARD =     [4, 10, 15, 16, 17, 19, 20, 21, 26, 27, 31, 32, 33, 36, 37, 40, 41, 44, 47, 49, 51, 52, 53, 54, 59]
MIQP =         [10, 11, 12, 15, 16, 17, 20, 26, 27, 28, 29, 31, 32, 33, 36, 37, 40, 41, 44, 49, 51, 52, 53, 54, 59]
MIQP_COMM =    [10, 11, 12, 15, 16, 17, 21, 27, 28, 29, 31, 32, 33, 36, 37, 40, 41, 42, 47, 49, 51, 52, 53, 56, 59]
MIQP_COMM_SIZE = [10, 12, 15, 16, 17, 21, 26, 28, 29, 31, 32, 33, 37, 40, 41, 42, 44, 47, 48, 49, 51, 52, 53, 54, 59]

# ODE-validated scores (tmax=1000, realistic P0)
SCORES = {
    "Greedy+Local": 1.861968,
    "Backward": 1.819074,
    "MIQP": 1.713159,
    "MIQP+Comm": 1.383439,
}

ALL_METHODS = {
    "Greedy+Local": set(GREEDY_LOCAL),
    "Backward": set(BACKWARD),
    "MIQP": set(MIQP),
    "MIQP+Comm": set(MIQP_COMM),
    "MIQP+Comm+Size": set(MIQP_COMM_SIZE),
}

# All 49 candidate sites
ALL_CANDIDATES = [1,3,4,5,6,7,9,10,11,12,15,16,17,18,19,20,21,24,26,27,28,29,30,31,
                  32,33,35,36,37,38,39,40,41,42,44,47,48,49,50,51,52,53,54,55,56,57,58,59,60]


def fig1_score_comparison():
    """Bar chart comparing ODE-validated scores across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(SCORES.keys())
    scores = [SCORES[m] for m in methods]
    colors = [COLORS[m] for m in methods]

    bars = ax.bar(methods, scores, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

    # Add score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{score:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Total Adult Biomass (sum A at t=1000)", fontsize=13)
    ax.set_title("ODE-Validated Performance Comparison", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Add a baseline reference line
    best = max(scores)
    ax.axhline(y=best, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    fig.tight_layout()
    out = FIGURES_DIR / "fig1_score_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig2_site_overlap_heatmap():
    """Heatmap showing Jaccard similarity between methods."""
    method_names = list(ALL_METHODS.keys())
    n = len(method_names)
    jaccard = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = ALL_METHODS[method_names[i]]
            b = ALL_METHODS[method_names[j]]
            jaccard[i, j] = len(a & b) / len(a | b)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(jaccard, cmap="YlGnBu", vmin=0.3, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(method_names, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(method_names, fontsize=10)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            overlap_count = len(ALL_METHODS[method_names[i]] & ALL_METHODS[method_names[j]])
            text = f"{jaccard[i,j]:.2f}\n({overlap_count}/25)"
            color = "white" if jaccard[i, j] > 0.7 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

    ax.set_title("Site Selection Agreement Between Methods\n(Jaccard Similarity & Overlap Count)",
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Jaccard Similarity", fontsize=11)

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_site_overlap_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig3_site_selection_matrix():
    """Matrix showing which sites are selected by which methods."""
    method_names = ["Greedy+Local", "Backward", "MIQP", "MIQP+Comm", "MIQP+Comm+Size"]
    methods_sets = [ALL_METHODS[m] for m in method_names]

    # Get all sites that appear in at least one method
    all_selected = sorted(set().union(*methods_sets))

    matrix = np.zeros((len(method_names), len(all_selected)))
    for i, s in enumerate(methods_sets):
        for j, site in enumerate(all_selected):
            if site in s:
                matrix[i, j] = 1

    # Count how many methods select each site
    selection_count = matrix.sum(axis=0)

    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=10)
    ax.set_xticks(range(len(all_selected)))
    ax.set_xticklabels([str(s) for s in all_selected], rotation=90, fontsize=8)
    ax.set_xlabel("Site ID", fontsize=12)

    # Add count row at top
    for j, count in enumerate(selection_count):
        color = "darkgreen" if count == 5 else ("orange" if count >= 3 else "gray")
        ax.text(j, -0.7, f"{int(count)}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=color)
    ax.text(-1.5, -0.7, "Count", ha="right", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Site Selection Across Methods (blue = selected)", fontsize=14, fontweight="bold")

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_site_selection_matrix.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig4_consensus_sites():
    """Bar chart showing how many methods select each site (consensus core)."""
    method_names = list(ALL_METHODS.keys())
    site_counts = {}
    for site in ALL_CANDIDATES:
        count = sum(1 for m in method_names if site in ALL_METHODS[m])
        if count > 0:
            site_counts[site] = count

    # Sort by count (descending), then by site_id
    sorted_sites = sorted(site_counts.items(), key=lambda x: (-x[1], x[0]))
    sites = [s[0] for s in sorted_sites]
    counts = [s[1] for s in sorted_sites]

    fig, ax = plt.subplots(figsize=(14, 5))

    color_map = {5: "#2E7D32", 4: "#66BB6A", 3: "#FFC107", 2: "#FF9800", 1: "#E0E0E0"}
    bar_colors = [color_map[c] for c in counts]

    bars = ax.bar(range(len(sites)), counts, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels([str(s) for s in sites], rotation=90, fontsize=8)
    ax.set_ylabel("Number of Methods Selecting Site", fontsize=12)
    ax.set_xlabel("Site ID", fontsize=12)
    ax.set_title("Consensus Analysis: How Many Methods Select Each Site?", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color="red", linestyle="--", alpha=0.5, label="Majority threshold (3/5)")
    ax.legend(fontsize=10)

    # Custom legend for colors
    legend_elements = [
        mpatches.Patch(facecolor=color_map[5], edgecolor="black", label="All 5 methods"),
        mpatches.Patch(facecolor=color_map[4], edgecolor="black", label="4 methods"),
        mpatches.Patch(facecolor=color_map[3], edgecolor="black", label="3 methods"),
        mpatches.Patch(facecolor=color_map[2], edgecolor="black", label="2 methods"),
        mpatches.Patch(facecolor=color_map[1], edgecolor="black", label="1 method"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "fig4_consensus_sites.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig5_network_centrality():
    """Network centrality metrics for all sites, highlighting selected sites."""
    df = pd.read_csv(RUNS_DIR / "network_metrics_all_matrix1.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("in_strength", "In-Strength (Larval Inflow)"),
        ("out_strength", "Out-Strength (Larval Outflow)"),
        ("pagerank", "PageRank Centrality"),
    ]

    # Sites selected by MIQP (our benchmark)
    miqp_set = set(MIQP)

    for ax, (col, title) in zip(axes, metrics):
        selected_mask = df["site_id"].isin(miqp_set)

        # Not selected
        ax.bar(df.loc[~selected_mask, "site_id"].astype(str),
               df.loc[~selected_mask, col],
               color="#E0E0E0", edgecolor="gray", linewidth=0.3, label="Not selected")
        # Selected
        ax.bar(df.loc[selected_mask, "site_id"].astype(str),
               df.loc[selected_mask, col],
               color="#2196F3", edgecolor="navy", linewidth=0.5, label="MIQP selected")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.set_xlabel("Site ID", fontsize=9)

    axes[0].legend(fontsize=9)
    fig.suptitle("Network Centrality of Sites (Matrix 1)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "fig5_network_centrality.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig6_reef_sizes():
    """Reef size allocation from MIQP+Size and MIQP+Comm+Size."""
    df_size = pd.read_csv(RUNS_DIR / "miqp_size_sites_matrix1.csv")
    df_comm_size = pd.read_csv(RUNS_DIR / "miqp_comm_size_sites_matrix1.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # MIQP+Size
    ax = axes[0]
    colors_size = ["#4CAF50" if s >= 45 else "#FFC107" if s >= 15 else "#FF5722" for s in df_size["size"]]
    ax.bar(df_size["site_id"].astype(str), df_size["size"], color=colors_size, edgecolor="black", linewidth=0.5)
    ax.set_title("MIQP + Sizing", fontsize=12, fontweight="bold")
    ax.set_xlabel("Site ID", fontsize=11)
    ax.set_ylabel("Reef Size Allocation", fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)
    ax.text(0.5, 51, "Max (50)", fontsize=8, color="gray")

    # MIQP+Comm+Size
    ax = axes[1]
    colors_cs = ["#4CAF50" if s >= 45 else "#FFC107" if s >= 15 else "#FF5722" for s in df_comm_size["size"]]
    ax.bar(df_comm_size["site_id"].astype(str), df_comm_size["size"], color=colors_cs, edgecolor="black", linewidth=0.5)
    ax.set_title("MIQP + Communities + Sizing", fontsize=12, fontweight="bold")
    ax.set_xlabel("Site ID", fontsize=11)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", edgecolor="black", label="Full (>= 45)"),
        mpatches.Patch(facecolor="#FFC107", edgecolor="black", label="Medium (15-44)"),
        mpatches.Patch(facecolor="#FF5722", edgecolor="black", label="Small (< 15)"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=9, loc="upper right")

    fig.suptitle("Optimal Reef Size Allocation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "fig6_reef_sizes.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig7_greedy_selection_curve():
    """Show objective value growing as greedy adds sites."""
    # Re-run greedy incrementally to get the curve
    from src.model.jars_ode import load_connectivity
    from src.opt.evaluator import evaluate_subset

    connectivity, key_all = load_connectivity()

    # Greedy selection order
    greedy_order = [4, 6, 10, 15, 19, 20, 21, 24, 27, 30, 31, 32, 36, 37, 38, 39, 40, 41, 47, 49, 51, 52, 53, 55, 60]

    scores = []
    for k in range(1, len(greedy_order) + 1):
        subset = greedy_order[:k]
        score = evaluate_subset(subset, connectivity, key_all, tmax=1000, P1scaling=0.5, P0_mode="realistic")
        scores.append(score)
        print(f"  k={k:2d}: score={score:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 26), scores, "o-", color="#2196F3", markersize=6, linewidth=2, label="Greedy+Local Search")

    # Add backward score as reference
    ax.axhline(y=1.819074, color="#FF9800", linestyle="--", linewidth=1.5, label=f"Backward (1.8191)")
    ax.axhline(y=1.713159, color="#4CAF50", linestyle="--", linewidth=1.5, label=f"MIQP (1.7132)")

    ax.set_xlabel("Number of Sites Selected (K)", fontsize=13)
    ax.set_ylabel("Total Adult Biomass (sum A)", fontsize=13)
    ax.set_title("Greedy Site Selection: Marginal Returns Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(range(1, 26))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "fig7_greedy_curve.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig8_connectivity_heatmap():
    """Heatmap of the connectivity matrix for selected sites."""
    from src.model.jars_ode import load_connectivity, sitetoindex

    connectivity, key_all = load_connectivity()

    # Use MIQP sites as the showcase
    sites = sorted(MIQP)
    idx = sitetoindex(key_all, np.array(sites))
    sub_matrix = connectivity[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log10(sub_matrix + 1), cmap="YlOrRd", aspect="equal")
    ax.set_xticks(range(len(sites)))
    ax.set_yticks(range(len(sites)))
    ax.set_xticklabels([str(s) for s in sites], rotation=90, fontsize=8)
    ax.set_yticklabels([str(s) for s in sites], fontsize=8)
    ax.set_xlabel("Destination Site", fontsize=12)
    ax.set_ylabel("Source Site", fontsize=12)
    ax.set_title("Larval Connectivity Matrix (MIQP-Selected Sites)\nlog10(connectivity + 1)",
                 fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log10(Connectivity + 1)", fontsize=11)

    fig.tight_layout()
    out = FIGURES_DIR / "fig8_connectivity_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig9_method_timing():
    """Comparison of computational time across methods."""
    methods = ["Greedy+Local\n(tmax=1000)", "Backward\n(tmax=1000)", "MIQP\n(Gurobi)"]
    times = [1919.7, 279.1, 0.32]  # seconds from actual runs
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, times, color=colors, edgecolor="black", linewidth=0.8, width=0.5)

    for bar, t in zip(bars, times):
        if t > 10:
            label = f"{t:.0f}s\n({t/60:.1f} min)"
        else:
            label = f"{t:.2f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                label, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("Computational Time Comparison", fontsize=15, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 5000)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "fig9_timing.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def fig10_score_vs_gap():
    """Score vs optimality gap - shows how heuristics compare to MIQP surrogate."""
    # MIQP surrogate objective values
    surrogate_scores = {
        "MIQP": 13692.07,
        "MIQP+Comm": 12782.91,
    }

    ode_scores = {
        "Greedy+Local": 1.861968,
        "Backward": 1.819074,
        "MIQP": 1.713159,
        "MIQP+Comm": 1.383439,
    }

    best_ode = max(ode_scores.values())

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(ode_scores.keys())
    scores = [ode_scores[m] for m in methods]
    gaps = [(best_ode - s) / best_ode * 100 for s in scores]
    colors = [COLORS[m] for m in methods]

    bars = ax.barh(methods, gaps, color=colors, edgecolor="black", linewidth=0.8, height=0.5)

    for bar, gap, score in zip(bars, gaps, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{gap:.1f}%  (score: {score:.4f})",
                ha="left", va="center", fontsize=11)

    ax.set_xlabel("Gap from Best ODE Score (%)", fontsize=13)
    ax.set_title("Optimality Gap: Each Method vs Best (Greedy+Local = 1.862)",
                 fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(gaps) * 1.6)

    fig.tight_layout()
    out = FIGURES_DIR / "fig10_optimality_gap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating presentation figures...")
    print("=" * 60)

    fig1_score_comparison()
    fig2_site_overlap_heatmap()
    fig3_site_selection_matrix()
    fig4_consensus_sites()
    fig5_network_centrality()
    fig6_reef_sizes()
    fig7_greedy_selection_curve()
    fig8_connectivity_heatmap()
    fig9_method_timing()
    fig10_score_vs_gap()

    print("=" * 60)
    print("All figures saved to figures/")
    print("=" * 60)
