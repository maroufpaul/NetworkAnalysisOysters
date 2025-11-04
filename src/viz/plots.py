# src/viz/plots.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_site_order(csv_path: str | Path, out_path: str | Path | None = None):
    """
    Read a CSV with columns:
        order, site_id
    and make a simple line/marker plot:
        x = order
        y = site_id
    Saves a PNG.

    This works with the files we save from greedy/backward because we used
    the same format in src/utils/io_utils.py
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "order" not in df.columns or "site_id" not in df.columns:
        raise ValueError(f"{csv_path} does not have 'order' and 'site_id' columns")

    x = df["order"].values
    y = df["site_id"].values

    # if no output path given, make one next to the CSV
    if out_path is None:
        out_path = csv_path.with_suffix(".png")

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.title(f"Site selection order\n{csv_path.name}")
    plt.xlabel("selection order")
    plt.ylabel("site id")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[viz] wrote {out_path}")


def compare_two_site_sets(csv_path_1: str | Path,
                          csv_path_2: str | Path,
                          out_path: str | Path | None = None):
    """
    Quick visual to compare two runs:
      - load both CSVs
      - plot them on top of each other with different markers
    Very simple on purpose.
    """
    csv_path_1 = Path(csv_path_1)
    csv_path_2 = Path(csv_path_2)

    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    if out_path is None:
        out_path = csv_path_1.parent / f"compare_{csv_path_1.stem}__{csv_path_2.stem}.png"

    plt.figure(figsize=(6, 4))
    plt.plot(df1["order"], df1["site_id"], marker="o", label=csv_path_1.name)
    plt.plot(df2["order"], df2["site_id"], marker="x", label=csv_path_2.name)
    plt.title("Comparison of selected sites")
    plt.xlabel("selection order")
    plt.ylabel("site id")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[viz] wrote {out_path}")
