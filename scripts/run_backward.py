# scripts/run_backward.py

from src.opt.backward import backward_greedy

def main():
    sites, score = backward_greedy(
        k=25,
        tmax=1000,
        P1scaling=0.5,
        P0_mode="constant",
        consP0=170.0,
        save=True,
        save_name=None,
    )

    print("\n[scripts] backward greedy finished")
    print(f"  sites: {sites}")
    print(f"  score: {score:.6f}")

if __name__ == "__main__":
    main()
