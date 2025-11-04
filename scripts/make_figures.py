from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

DATA_PREVIEW = RUNS / "oyster_data_preview.csv"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_preview():
    """
    Try to load oyster_data_preview.csv, which should have at least:
      site_id, Pe, maybe W_sum or internal, maybe community.
    Column names are normalized to lowercase.
    """
    if not DATA_PREVIEW.exists():
        print("[warn] No oyster_data_preview.csv found in runs/; "
              "some figures will be simpler.")
        return None

    df = pd.read_csv(DATA_PREVIEW)
    df.columns = [c.lower() for c in df.columns]

    # Normalize label name
    if "site_id" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "site_id"})

    return df


def load_sites_csv(path):
    """
    Read a generic '*_sites.csv' file and return (list_of_site_ids, dataframe).

    Works with files that have either 'site_id' or 'label' as the column name.
    """
    if not path.exists():
        print(f"[warn] {path.name} not found; skipping.")
        return [], None

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "site_id" in df.columns:
        key = "site_id"
    elif "label" in df.columns:
        key = "label"
    else:
        raise ValueError(f"{path} has no site_id/label column, got {df.columns}.")

    return df[key].tolist(), df


def safe_sum(series):
    """Return 0 if series is None."""
    if series is None:
        return 0.0
    return float(series.sum())


# -------------------------------------------------------------------
# Figure 1: Strategy comparison (counts + approximate larvae score)
# -------------------------------------------------------------------

def fig_strategy_comparison(preview):
    """
    Compare several strategies:
      - greedy (latest *_sites.csv)
      - backward
      - pure MIQP
      - MIQP + communities
      - MIQP + sizing

    For each method we compute:
      * number of sites
      * approximate larvae score = sum Pe[selected]
        (+ sum internal if we have a column like 'w_sum' or 'internal')
    """

    # --- 1) Load selection sets -------------------------------------
    # Greedy: use latest greedy_*_sites.csv
    greedy_files = sorted(RUNS.glob("greedy_*_sites.csv"))
    greedy_sites, _ = (load_sites_csv(greedy_files[-1])
                       if greedy_files else ([], None))

    backward_sites, _      = load_sites_csv(RUNS / "backward_sites.csv")
    miqp_sites, _          = load_sites_csv(RUNS / "miqp_sites.csv")
    miqp_comm_sites, _     = load_sites_csv(RUNS / "miqp_comm_sites.csv")
    miqp_size_sites, size_df = load_sites_csv(RUNS / "miqp_size_sites.csv")

    strategies = []
    site_sets = []

    if greedy_sites:
        strategies.append("Greedy")
        site_sets.append(greedy_sites)
    if backward_sites:
        strategies.append("Backward")
        site_sets.append(backward_sites)
    if miqp_sites:
        strategies.append("MIQP")
        site_sets.append(miqp_sites)
    if miqp_comm_sites:
        strategies.append("MIQP + comm")
        site_sets.append(miqp_comm_sites)
    if miqp_size_sites:
        strategies.append("MIQP + sizing")
        site_sets.append(miqp_size_sites)

    if not strategies:
        print("[fig1] No strategy CSVs found; skipping.")
        return

    # --- 2) Compute metrics -----------------------------------------
    counts = [len(s) for s in site_sets]

    # Build lookup from site_id -> Pe and maybe internal (W_sum)
    larvae_scores = []
    if preview is None or "pe" not in preview.columns:
        print("[fig1] No Pe information; using just counts.")
        larvae_scores = [0.0] * len(strategies)
    else:
        pe_map = dict(zip(preview["site_id"], preview["pe"]))

        if "w_sum" in preview.columns:
            int_map = dict(zip(preview["site_id"], preview["w_sum"]))
        elif "internal" in preview.columns:
            int_map = dict(zip(preview["site_id"], preview["internal"]))
        else:
            int_map = None

        for sites in site_sets:
            pe_total = sum(pe_map.get(s, 0.0) for s in sites)
            int_total = (sum(int_map.get(s, 0.0) for s in sites)
                         if int_map is not None else 0.0)
            larvae_scores.append(pe_total + int_total)

    # --- 3) Plot: two aligned bar charts ----------------------------
    x = range(len(strategies))

    plt.figure(figsize=(7, 4))
    width = 0.4

    # left y-axis: number of sites
    plt.bar([i - width/2 for i in x], counts, width=width, label="# sites")

    # right axis: larvae score (if available)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.bar([i + width/2 for i in x], larvae_scores, width=width,
            alpha=0.6, label="approx. larvae score")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(strategies, rotation=20)
    ax1.set_ylabel("# sites selected")
    ax2.set_ylabel("Approx. larvae score (Pe + internal)")
    plt.title("Comparison of optimization strategies")

    # Build a combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    out = FIGS / "fig_strategy_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[fig1] wrote {out}")


# -------------------------------------------------------------------
# Figure 2: Reef sizes from MIQP + sizing
# -------------------------------------------------------------------

def fig_reef_sizes():
    """
    Bar chart of reef sizes for the MIQP + sizing solution.
    Uses miqp_size_sites.csv which should have columns site_id, size.
    """
    path = RUNS / "miqp_size_sites.csv"
    if not path.exists():
        print("[fig2] miqp_size_sites.csv not found; skipping.")
        return

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    if "site_id" not in df.columns:
        df = df.rename(columns={"label": "site_id"})

    if "size" not in df.columns:
        print("[fig2] miqp_size_sites.csv has no 'size' column; skipping.")
        return

    df = df.sort_values("site_id")

    plt.figure(figsize=(8, 4))
    plt.bar(df["site_id"].astype(str), df["size"])
    plt.xlabel("Site ID")
    plt.ylabel("Reef size (s[i])")
    plt.title("Reef sizes chosen by MIQP + sizing model")
    plt.xticks(rotation=60, fontsize=7)
    plt.tight_layout()

    out = FIGS / "fig_reef_sizes_miqp_size.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[fig2] wrote {out}")


# -------------------------------------------------------------------
# Figure 3: External vs internal larvae, highlighting MIQP choice
# -------------------------------------------------------------------

def fig_external_vs_internal(preview):
    """
    Scatterplot: external larvae vs internal connectivity for all sites.
    Highlight which sites were picked by the MIQP model (no communities).

    Requires oyster_data_preview.csv to have:
      site_id, Pe, and some internal metric: w_sum or internal.
    """
    if preview is None:
        print("[fig3] No preview data; skipping scatter plot.")
        return

    cols = preview.columns
    if "pe" not in cols:
        print("[fig3] preview has no 'pe' column; skipping.")
        return

    # Internal column name
    if "w_sum" in cols:
        internal_col = "w_sum"
    elif "internal" in cols:
        internal_col = "internal"
    else:
        print("[fig3] preview has no internal connectivity column; "
              "using Pe only.")
        preview["internal_dummy"] = 0.0
        internal_col = "internal_dummy"

    miqp_sites, _ = load_sites_csv(RUNS / "miqp_sites.csv")
    selected = set(miqp_sites)

    preview["chosen"] = preview["site_id"].apply(lambda s: s in selected)

    plt.figure(figsize=(6, 5))
    not_chosen = preview[~preview["chosen"]]
    chosen = preview[preview["chosen"]]

    plt.scatter(not_chosen["pe"], not_chosen[internal_col],
                alpha=0.3, label="not selected")
    plt.scatter(chosen["pe"], chosen[internal_col],
                alpha=0.9, marker="o", edgecolor="k",
                label="MIQP selected")

    plt.xlabel("External larvae (Pe)")
    plt.ylabel("Internal connectivity (sum of W, approx.)")
    plt.title("How MIQP trades external vs internal larvae")
    plt.legend()
    plt.tight_layout()

    out = FIGS / "fig_external_vs_internal_miqp.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[fig3] wrote {out}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print("[make_figures] ROOT =", ROOT)
    preview = load_preview()

    fig_strategy_comparison(preview)
    fig_reef_sizes()
    fig_external_vs_internal(preview)

    print("[make_figures] done.")


if __name__ == "__main__":
    main()
