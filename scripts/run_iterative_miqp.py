# scripts/run_iterative_miqp.py

"""
Run the iterative MIQP surrogate refinement and compare with baseline methods.
"""

import time
from src.opt.iterative_miqp import iterative_refinement
from src.model.jars_ode import load_connectivity
from src.opt.evaluator import evaluate_subset
from src.utils.io_utils import save_greedy_result


def main():
    print("=" * 70)
    print("ITERATIVE MIQP SURROGATE REFINEMENT")
    print("=" * 70)

    start = time.time()

    sites, ode_score, history = iterative_refinement(
        K=25,
        max_iter=10,
        tmax=1000,
        P1scaling=0.5,
        P0_mode="realistic",
        verbose=True,
    )

    elapsed = time.time() - start

    # Save result
    save_greedy_result(sites, ode_score, filename="iterative_miqp")

    # Compare with known baselines
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINE METHODS")
    print("=" * 70)

    connectivity, key_all = load_connectivity()

    baselines = {
        "Greedy+Local": [4, 6, 10, 15, 19, 20, 21, 24, 27, 30, 31, 32, 36,
                         37, 38, 39, 40, 41, 47, 49, 51, 52, 53, 55, 60],
        "Backward":     [4, 10, 15, 16, 17, 19, 20, 21, 26, 27, 31, 32, 33,
                         36, 37, 40, 41, 44, 47, 49, 51, 52, 53, 54, 59],
        "Old MIQP":     [10, 11, 12, 15, 16, 17, 20, 26, 27, 28, 29, 31, 32,
                         33, 36, 37, 40, 41, 44, 49, 51, 52, 53, 54, 59],
    }

    print(f"\n{'Method':<25s} {'ODE Score':>12s} {'vs Best':>10s}")
    print("-" * 50)

    all_scores = {"Iterative MIQP": ode_score}
    for name, s in baselines.items():
        score = evaluate_subset(s, connectivity, key_all, tmax=1000,
                                P1scaling=0.5, P0_mode="realistic")
        all_scores[name] = score

    best_score = max(all_scores.values())
    for name, score in all_scores.items():
        gap = (best_score - score) / best_score * 100
        marker = " <-- best" if score == best_score else ""
        print(f"{name:<25s} {score:>12.6f} {gap:>9.1f}%{marker}")

    # Overlap analysis
    iterative_set = set(sites)
    print(f"\nSite overlap with Iterative MIQP ({len(iterative_set)} sites):")
    for name, s in baselines.items():
        overlap = len(iterative_set & set(s))
        print(f"  vs {name:<20s}: {overlap}/25 shared")

    # Show convergence history
    print(f"\nConvergence history:")
    print(f"{'Iter':>4s} {'ODE Score':>12s} {'Changed':>8s} {'A_median':>10s} {'A_range':>24s}")
    for h in history:
        print(f"{h['iteration']:>4d} {h['ode_score']:>12.6f} {h['sites_changed']:>8d} "
              f"{h['A_median']:>10.5f} [{h['A_min']:.5f}, {h['A_max']:.5f}]")

    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
