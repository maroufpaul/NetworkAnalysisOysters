# scripts/compare_greedy_backward.py

from pathlib import Path
from src.viz.plots import compare_two_site_sets

def main():
    runs_dir = Path(__file__).resolve().parents[1] / "runs"

    greedy_csvs = sorted(runs_dir.glob("greedy_*_sites.csv"))
    backward_csvs = sorted(runs_dir.glob("backward*_sites.csv"))

    if not greedy_csvs:
        print("[compare] no greedy runs found.")
        return
    if not backward_csvs:
        print("[compare] no backward runs found.")
        return

    latest_greedy = greedy_csvs[-1]
    latest_backward = backward_csvs[-1]

    print(f"[compare] greedy:   {latest_greedy.name}")
    print(f"[compare] backward: {latest_backward.name}")

    compare_two_site_sets(latest_greedy, latest_backward)

if __name__ == "__main__":
    main()
