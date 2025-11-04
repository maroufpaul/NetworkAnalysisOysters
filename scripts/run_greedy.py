# scripts/run_greedy.py

from src.opt.greedy import greedy_select_sites

def main():
    # this is the "real" run: full 25-site greedy, saved to runs/
    sites, score = greedy_select_sites(
        k=25,
        tmax=1000,          # your original horizon
        P1scaling=0.5,
        P0_mode="constant",
        consP0=170.0,
        save=True,
        save_name=None,     # let it pick a timestamp
    )

    print("\n[script] finished greedy 25")
    print(f"[script] sites: {sites}")
    print(f"[script] final score: {score:.6f}")

if __name__ == "__main__":
    main()
