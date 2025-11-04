# src/main.py

from src.opt.greedy import greedy_select_sites

def main():
    print("Oyster network project booting up...")

    # small demo run
    sites, score = greedy_select_sites(k=5, tmax=200, save=False)
    print("\n[main] demo greedy result:")
    print(f"  sites: {sites}")
    print(f"  score: {score:.6f}")

if __name__ == "__main__":
    main()
