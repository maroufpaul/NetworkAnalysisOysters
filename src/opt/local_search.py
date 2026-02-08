# src/opt/local_search.py

import time
import numpy as np
from src.model.jars_ode import load_connectivity, CANDIDATE_SITES
from src.opt.evaluator import evaluate_subset
from src.utils.io_utils import save_greedy_result


def local_swap_hillclimb(
    start_sites,
    tmax: int = 1000,
    P1scaling: float = 0.5,
    P0_mode: str = "realistic",
    consP0: float = 170.0,
    max_passes: int = 50,
    improve_tol: float = 1e-6,
    save: bool = False,
    save_name: str | None = None,
):
    """
    Classic 1-for-1 swap local search.

    start_sites: list of site labels (length K)
    We assume the universe is our 49 candidate sites.

    Algorithm:
      repeat:
        try swapping each selected site with each unselected candidate
        pick the swap with the biggest positive improvement
        if improvement > improve_tol: apply it
        else: stop
    """
    connectivity, key_all = load_connectivity()

    # make sure we work with numpy ints
    current = np.array(start_sites, dtype=int)
    K = len(current)

    universe = CANDIDATE_SITES.copy()
    # sites not currently in the set
    pool = np.setdiff1d(universe, current)

    # score the starting set
    current_score = evaluate_subset(
        site_labels=current,
        connectivity_data=connectivity,
        key_all=key_all,
        tmax=tmax,
        P1scaling=P1scaling,
        P0_mode=P0_mode,
        consP0=consP0,
    )
    print(f"[local] start score = {current_score:.6f} on {K} sites")

    pass_no = 0
    improved = True

    while improved and pass_no < max_passes:
        pass_no += 1
        improved = False
        best_delta = 0.0
        best_out = None
        best_in = None
        best_new_score = current_score

        start_t = time.time()

        # try all 1-swaps
        for out_site in current:
            for in_site in pool:
                trial = current.copy()
                # replace out_site with in_site
                idx = np.where(trial == out_site)[0][0]
                trial[idx] = in_site

                trial_score = evaluate_subset(
                    site_labels=trial,
                    connectivity_data=connectivity,
                    key_all=key_all,
                    tmax=tmax,
                    P1scaling=P1scaling,
                    P0_mode=P0_mode,
                    consP0=consP0,
                )
                delta = trial_score - current_score

                if delta > best_delta:
                    best_delta = delta
                    best_out = out_site
                    best_in = in_site
                    best_new_score = trial_score

        elapsed = time.time() - start_t

        if best_delta > improve_tol and best_out is not None and best_in is not None:
            # apply best swap
            idx = np.where(current == best_out)[0][0]
            current[idx] = best_in
            # rebuild pool
            pool = np.setdiff1d(universe, current)
            current_score = best_new_score
            improved = True
            print(f"[local] pass {pass_no}: {best_out} → {best_in} | Δ={best_delta:.6f} | score={current_score:.6f} | {elapsed:.1f}s")
        else:
            print(f"[local] pass {pass_no}: no improving swap (best Δ={best_delta:.3g}). stop.")
            break

    # optional save
    if save:
        #  can reuse the greedy saver, it's just "sites in order"
        # sort so it's nicer to read
        final_sites_sorted = sorted(current.tolist())
        save_greedy_result(final_sites_sorted, current_score, filename=save_name)

    # return sites (sorted for stability) and score
    return sorted(current.tolist()), current_score
