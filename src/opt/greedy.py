# src/opt/greedy.py

import time
import numpy as np
from src.model.jars_ode import load_connectivity, CANDIDATE_SITES
from src.opt.evaluator import evaluate_subset
from src.utils.io_utils import save_greedy_result  # NEW


def greedy_select_sites(
    k: int = 25,
    tmax: int = 200,
    P1scaling: float = 0.5,
    P0_mode: str = "constant",
    consP0: float = 170.0,
    save: bool = False,             # NEW
    save_name: str | None = None,   # NEW
):
    """
    Simple forward greedy.
    """
    connectivity, key_all = load_connectivity()
    all_sites = CANDIDATE_SITES.copy()

    selected = []
    remaining = all_sites.tolist()
    best_score = 0.0

    print(f"[greedy] starting greedy selection for k={k}")
    for step in range(1, k + 1):
        print(f"[greedy] step {step}/{k} — trying {len(remaining)} candidates...")
        start = time.time()

        step_best_site = None
        step_best_score = -1.0

        for candidate in remaining:
            trial_sites = selected + [candidate]
            score = evaluate_subset(
                site_labels=trial_sites,
                connectivity_data=connectivity,
                key_all=key_all,
                tmax=tmax,
                P1scaling=P1scaling,
                P0_mode=P0_mode,
                consP0=consP0,
            )
            if score > step_best_score:
                step_best_score = score
                step_best_site = candidate

        selected.append(step_best_site)
        remaining.remove(step_best_site)
        best_score = step_best_score

        elapsed = time.time() - start
        print(f"  → picked {step_best_site} | total adults = {best_score:.6f} | {elapsed:.1f}s")

    print(f"[greedy] done. selected {len(selected)} sites.")

    # NEW: save if requested
    if save:
        save_greedy_result(selected, best_score, filename=save_name)

    return selected, best_score
