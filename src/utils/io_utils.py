# src/utils/io_utils.py

from pathlib import Path
import pandas as pd
from datetime import datetime

# base runs folder ( already created it at project root)
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def timestamp_str() -> str:
    """Return a simple timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_greedy_result(selected_sites, final_score, filename: str | None = None):
    """
    Save greedy output to runs/ as two files:
      - sites csv
      - summary txt
    """
    ts = timestamp_str()

    if filename is None:
        # e.g. greedy_20251031_153012
        base = f"greedy_{ts}"
    else:
        base = filename

    sites_path = RUNS_DIR / f"{base}_sites.csv"
    summary_path = RUNS_DIR / f"{base}_summary.txt"

    # save sites
    df = pd.DataFrame({
        "order": range(1, len(selected_sites) + 1),
        "site_id": selected_sites,
    })
    df.to_csv(sites_path, index=False)

    # save summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Greedy selection result\n")
        f.write("======================\n")
        f.write(f"total sites: {len(selected_sites)}\n")
        f.write(f"final score (sumA): {final_score:.6f}\n")
        f.write("sites (in order):\n")
        for i, s in enumerate(selected_sites, start=1):
            f.write(f"  {i:2d}. {s}\n")

    print(f"[save] wrote {sites_path}")
    print(f"[save] wrote {summary_path}")
