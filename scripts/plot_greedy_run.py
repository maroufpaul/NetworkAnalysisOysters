# scripts/plot_greedy_run.py

from pathlib import Path
from src.viz.plots import plot_site_order

def main():
    runs_dir = Path(__file__).resolve().parents[1] / "runs"
    csvs = sorted(runs_dir.glob("greedy_*_sites.csv"))
    if not csvs:
        print("[plot] no greedy_*_sites.csv files found in runs/")
        return

    latest = csvs[-1]
    print(f"[plot] plotting latest greedy run: {latest.name}")
    plot_site_order(latest)

if __name__ == "__main__":
    main()
