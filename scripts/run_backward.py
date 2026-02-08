# scripts/run_backward.py

import time
from src.opt.backward import backward_greedy

def main():
    start_time = time.time()  # <-- Start Timer

    sites, score = backward_greedy(
        k=25,
        tmax=1000,
        P1scaling=0.5,
        P0_mode="realistic",
        consP0=170.0,
        save=True,
        save_name=None,
    )

    end_time = time.time()    # <-- End Timer
    elapsed = end_time - start_time

    print("\n[scripts] backward greedy finished")
    print(f"  sites: {sites}")
    print(f"  score: {score:.6f}")
    print(f"  time taken: {elapsed:.4f} secs") # <-- Print result

if __name__ == "__main__":
    main()