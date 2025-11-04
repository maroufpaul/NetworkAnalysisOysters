# src/opt/backward.py

import time
import numpy as np
from src.model.jars_ode import load_connectivity, CANDIDATE_SITES
from src.opt.evaluator import evaluate_subset
from src.utils.io_utils import save_greedy_result  # we'll reuse the same saver


def backward_greedy(
    k: int = 25,
    tmax: int = 1000,
    P1scaling: float = 0.5,
    P0_mode: str = "constant",
    consP0: float = 170.0,
    save: bool = False,
    save_name: str | None = None,
):
    """
    'Stingy' backward greedy:
      1. start with ALL candidate sites (your 48)
      2. while len(S) > k:
            try removing each site
            pick the removal that hurts the objective the LEAST
            do that removal
      3. return remaining k sites
    """
    connectivity, key_all = load_connectivity()
    S = CANDIDATE_SITES.copy().tolist()   # current set
    current_score = evaluate_subset(
        site_labels=S,
        connectivity_data=connectivity,
        key_all=key_all,
        tmax=tmax,
        P1scaling=P1scaling,
        P0_mode=P0_mode,
        consP0=consP0,
    )
    print(f"[backward] start with {len(S)} sites → score={current_score:.6f}")

    round_no = 0
    while len(S) > k:
        round_no += 1
        print(f"[backward] round {round_no}: {len(S)} → {len(S) - 1}")
        start = time.time()

        best_site_to_remove = None
        best_new_score = None
        # removal that causes the smallest drop
        # i.e. we want to MAXIMIZE new_score
        for site in S:
            trial_set = [s for s in S if s != site]
            trial_score = evaluate_subset(
                site_labels=trial_set,
                connectivity_data=connectivity,
                key_all=key_all,
                tmax=tmax,
                P1scaling=P1scaling,
                P0_mode=P0_mode,
                consP0=consP0,
            )
            if (best_new_score is None) or (trial_score > best_new_score):
                best_new_score = trial_score
                best_site_to_remove = site

        # apply best removal
        S.remove(best_site_to_remove)
        elapsed = time.time() - start
        delta = best_new_score - current_score
        current_score = best_new_score
        sign = "+" if delta >= 0 else "-"
        print(f"  → removed {best_site_to_remove} | Δ={sign}{abs(delta):.6f} | score={current_score:.6f} | {elapsed:.1f}s")

    # optional save
    if save:
        final_sorted = sorted(S)
        # if no name is given, we'll let the saver pick a name,
        # but we can pass a prefix to distinguish
        prefix = save_name or "backward"
        save_greedy_result(final_sorted, current_score, filename=prefix)


    return sorted(S), current_score
