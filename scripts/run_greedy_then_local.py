# scripts/run_greedy_then_local.py

import time
from src.opt.greedy import greedy_select_sites
from src.opt.local_search import local_swap_hillclimb

def main():
    start_time = time.time()  # <-- Start Timer

    # 1) run full greedy
    greedy_sites, greedy_score = greedy_select_sites(
        k=25,
        tmax=800,          # a bit lower to speed up greedy
        save=True,
        save_name=None,
    )
    print("\n[scripts] greedy finished")
    print(f"  sites: {greedy_sites}")
    print(f"  score: {greedy_score:.6f}")

    # 2) polish with local search, but use full tmax
    final_sites, final_score = local_swap_hillclimb(
        start_sites=greedy_sites,
        tmax=1000,
        improve_tol=1e-6,
        max_passes=50,
        save=True,
        save_name=None,
    )

    end_time = time.time()    # <-- End Timer
    elapsed = end_time - start_time

    print("\n[scripts] greedy + local finished")
    print(f"  final sites: {final_sites}")
    print(f"  final score: {final_score:.6f}")
    print(f"  total time: {elapsed:.4f} secs") # <-- Print result

if __name__ == "__main__":
    main()