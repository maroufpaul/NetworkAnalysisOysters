# scripts/run_greedy.py

import time
from src.opt.greedy import greedy_select_sites

def main():
    start_time = time.time()  # <-- Start Timer

    # this is the "real" run: full 25-site greedy, saved to runs/
    sites, score = greedy_select_sites(
        k=25,
        tmax=1000,          # your original horizon
        P1scaling=0.5,
        P0_mode="realistic",
        consP0=170.0,
        save=True,
        save_name=None,     # let it pick a timestamp
    )

    end_time = time.time()    # <-- End Timer
    elapsed = end_time - start_time

    print("\n[script] finished greedy 25")
    print(f"[script] sites: {sites}")
    print(f"[script] final score: {score:.6f}")
    print(f"[script] time taken: {elapsed:.4f} secs") # <-- Print result

if __name__ == "__main__":
    main()