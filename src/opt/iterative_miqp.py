# src/opt/iterative_miqp.py

"""
Iterative surrogate refinement for the oyster site selection MIQP.

Instead of using a single global A* = 0.05675 for all sites, this module:
  1. Solves the MIQP with current surrogate weights
  2. Runs the full ODE on the MIQP solution to get per-site equilibrium A
  3. Updates the surrogate weights W_ij = P1_ij * A_j^1.72 using actual values
  4. Re-solves the MIQP with updated weights
  5. Repeats until the selected site set stabilizes

This closes the gap between the surrogate and true ODE while keeping the
speed advantage of MIQP (each iteration is ~0.3s MIQP + ~0.2s ODE).
"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from src.model.jars_ode import (
    load_connectivity,
    CANDIDATE_SITES,
    sitetoindex,
    setP0,
    odesys,
)


UNWANTED = [66, 67, 68, 69, 70, 71, 72]
ALPHA = 1.72


def _filter_candidates(connectivity, key_all):
    """Filter to candidate sites only (drop sites 66-72)."""
    mask = ~np.isin(key_all, UNWANTED)
    labels = key_all[mask]
    P = connectivity[np.ix_(mask, mask)]
    return P, labels


def run_ode_get_per_site_A(
    site_labels,
    connectivity,
    key_all,
    tmax=1000,
    P1scaling=0.5,
    P0_mode="realistic",
    consP0=170.0,
):
    """
    Run the full JARS ODE on a site subset and return per-site adult
    biomass at equilibrium (not just the sum).

    Returns
    -------
    A_final : np.ndarray of shape (n_sites,)
        Adult biomass at t=tmax for each site in site_labels.
    total : float
        Sum of A_final (same as evaluate_subset returns).
    """
    site_labels = np.array(site_labels, dtype=int)
    idx = sitetoindex(key_all, site_labels)
    if len(idx) == 0:
        return np.array([]), 0.0

    P1 = P1scaling * connectivity[np.ix_(idx, idx)]
    key_subset = key_all[idx]
    n = len(key_subset)

    if P0_mode == "constant":
        P0 = consP0 * np.ones(n)
    elif P0_mode == "realistic":
        P0 = setP0(key_subset)
    else:
        P0 = np.zeros(n)

    mu = 0.4 * np.ones(n)

    J0, A0, R0, S0 = 0.0, 0.2, 0.3, 0.0
    v0 = np.zeros(4 * n)
    v0[0:n] = J0
    v0[n:2*n] = A0
    v0[2*n:3*n] = R0
    v0[3*n:4*n] = S0

    sol = solve_ivp(
        lambda t, v: odesys(t, v, P0, P1, mu),
        [0, tmax],
        v0,
        method="RK45",
        rtol=1e-6,
    )

    v_final = sol.y[:, -1]
    A_final = v_final[n:2*n]
    return A_final, float(np.sum(A_final))


def build_surrogate_weights(P1_candidates, A_eq, candidate_labels, selected_labels):
    """
    Build the surrogate weight matrix W using per-site equilibrium A values.

    The true ODE internal term is: P1.T @ |A|^1.72
    So W_ij represents the contribution of site j to site i:
        W_ij = P1_ji * A_j^1.72

    For sites not in the current selection, we use a conservative estimate
    (median of selected sites' A values) so the MIQP can still explore.

    Parameters
    ----------
    P1_candidates : (n_cand, n_cand) connectivity among all candidates
    A_eq : dict mapping site_label -> equilibrium A value
    candidate_labels : all candidate site labels
    selected_labels : currently selected site labels

    Returns
    -------
    W : (n_cand, n_cand) surrogate weight matrix
    """
    n = len(candidate_labels)
    W = np.zeros((n, n))

    # Build A vector for all candidates
    known_A = [A_eq.get(s, None) for s in candidate_labels]
    known_values = [a for a in known_A if a is not None]
    fallback_A = np.median(known_values) if known_values else 0.05675

    A_vec = np.array([
        A_eq.get(s, fallback_A) for s in candidate_labels
    ])

    # Clip to avoid negative/zero values from ODE transients
    A_vec = np.clip(A_vec, 1e-6, None)

    # W_ij = P1_ji * A_j^alpha
    # The ODE term is P1.T @ |A|^alpha, so element (i,j) of P1.T is P1[j,i]
    # Contribution to site i from site j: P1[j,i] * A_j^alpha
    # In MIQP: sum_j W[i,j] * x_j = sum_j P1[j,i] * A_j^alpha * x_j
    # But the MIQP objective is sum_ij W[i,j]*x_i*x_j
    # So W[i,j] should capture the pairwise benefit when both i and j are selected
    # W[i,j] = P1[j,i] * A_j^alpha  (contribution of j to i's larval input)
    for j in range(n):
        a_j = A_vec[j] ** ALPHA
        for i in range(n):
            W[i, j] = P1_candidates[j, i] * a_j

    return W


def solve_miqp_python(Pe, W, K):
    """
    Solve the MIQP: maximize sum_i Pe_i*x_i + sum_ij W_ij*x_i*x_j
    subject to sum_i x_i = K, x_i in {0,1}

    Since scipy.optimize.milp only handles linear objectives, we use a
    greedy approach on the quadratic surrogate instead: iteratively pick
    the site that adds the most surrogate value. This is fast and works
    well for 48 choose 25.

    For exact MIQP, the user can swap in their AMPL/Gurobi solver.
    """
    n = len(Pe)
    selected = []
    remaining = list(range(n))

    for step in range(K):
        best_idx = None
        best_score = -np.inf

        for cand in remaining:
            trial = selected + [cand]
            # Evaluate surrogate objective
            score = sum(Pe[i] for i in trial)
            score += sum(W[i, j] for i in trial for j in trial)
            if score > best_score:
                best_score = score
                best_idx = cand

        selected.append(best_idx)
        remaining.remove(best_idx)

    surrogate_score = sum(Pe[i] for i in selected) + \
                      sum(W[i, j] for i in selected for j in selected)

    return selected, surrogate_score


def solve_miqp_python_fast(Pe, W, K):
    """
    Fast greedy on surrogate using marginal gain computation.
    Instead of recomputing the full objective each time, track
    the marginal gain of adding each candidate.
    """
    n = len(Pe)
    selected = []
    selected_set = set()
    remaining = list(range(n))

    current_score = 0.0

    for step in range(K):
        best_idx = None
        best_marginal = -np.inf

        for cand in remaining:
            # Marginal gain of adding cand:
            # Pe[cand] + W[cand,cand] + 2 * sum_{j in selected} W[cand,j]
            # (factor 2 because W[cand,j]*x_cand*x_j + W[j,cand]*x_j*x_cand)
            # Actually W is not symmetric, so:
            # gain = Pe[cand] + W[cand,cand] + sum_j_in_S (W[cand,j] + W[j,cand])
            gain = Pe[cand] + W[cand, cand]
            for j in selected:
                gain += W[cand, j] + W[j, cand]

            if gain > best_marginal:
                best_marginal = gain
                best_idx = cand

        selected.append(best_idx)
        selected_set.add(best_idx)
        remaining.remove(best_idx)
        current_score += best_marginal

    return selected, current_score


def iterative_refinement(
    K=25,
    max_iter=10,
    tmax=1000,
    P1scaling=0.5,
    P0_mode="realistic",
    consP0=170.0,
    tol=0,
    verbose=True,
):
    """
    Iterative surrogate refinement loop.

    1. Start with the old surrogate (global A* = 0.05675)
    2. Solve MIQP-surrogate greedy
    3. Run ODE on the solution, get per-site A values
    4. Update W using actual per-site A values
    5. Re-solve, repeat until site set stabilizes

    Parameters
    ----------
    K : int
        Number of sites to select.
    max_iter : int
        Maximum refinement iterations.
    tmax : int
        ODE integration horizon.
    P1scaling : float
        Internal connectivity scaling.
    P0_mode : str
        External larvae mode for ODE evaluation.
    consP0 : float
        Constant P0 value (used if P0_mode == "constant").
    tol : int
        Stop if site set changes by <= tol sites between iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    best_sites : list of int
        Selected site labels.
    best_ode_score : float
        ODE-validated total adult biomass.
    history : list of dict
        Per-iteration details.
    """
    connectivity, key_all = load_connectivity()

    # Filter to candidates (drop 66-72)
    P_cand, cand_labels = _filter_candidates(connectivity, key_all)
    P1_cand = P1scaling * P_cand
    n = len(cand_labels)

    # Build Pe vector matching the ODE's P0 mode
    if P0_mode == "realistic":
        Pe = setP0(cand_labels).astype(float)
    elif P0_mode == "constant":
        Pe = consP0 * np.ones(n)
    else:
        Pe = np.zeros(n)

    # --- Iteration 0: use old global A* ---
    A_STAR_GLOBAL = 0.05675
    A_eq = {s: A_STAR_GLOBAL for s in cand_labels}

    history = []
    prev_site_set = set()
    best_ode_score = -1.0
    best_sites = None

    total_start = time.time()

    for iteration in range(max_iter):
        iter_start = time.time()

        # Build surrogate weights from current A estimates
        W = build_surrogate_weights(P1_cand, A_eq, cand_labels, list(prev_site_set))

        # Solve MIQP (greedy on surrogate)
        selected_idx, surrogate_score = solve_miqp_python_fast(Pe, W, K)
        selected_labels = [int(cand_labels[i]) for i in selected_idx]
        selected_set = set(selected_labels)

        # Run full ODE to validate and get per-site A
        A_per_site, ode_score = run_ode_get_per_site_A(
            selected_labels, connectivity, key_all,
            tmax=tmax, P1scaling=P1scaling, P0_mode=P0_mode, consP0=consP0,
        )

        # Update A_eq with per-site values from ODE
        for label, a_val in zip(selected_labels, A_per_site):
            A_eq[label] = float(a_val)

        # Track best
        if ode_score > best_ode_score:
            best_ode_score = ode_score
            best_sites = sorted(selected_labels)

        # Check convergence
        n_changed = len(selected_set ^ prev_site_set)
        elapsed = time.time() - iter_start

        record = {
            "iteration": iteration,
            "sites": sorted(selected_labels),
            "surrogate_score": surrogate_score,
            "ode_score": ode_score,
            "sites_changed": n_changed,
            "elapsed": elapsed,
            "A_median": float(np.median(A_per_site)),
            "A_min": float(np.min(A_per_site)),
            "A_max": float(np.max(A_per_site)),
        }
        history.append(record)

        if verbose:
            print(f"[iter {iteration}] ODE={ode_score:.6f}  surrogate={surrogate_score:.1f}  "
                  f"changed={n_changed}  A_med={record['A_median']:.5f}  "
                  f"A_range=[{record['A_min']:.5f}, {record['A_max']:.5f}]  "
                  f"({elapsed:.2f}s)")

        if n_changed <= tol and iteration > 0:
            if verbose:
                print(f"[iter {iteration}] converged (changed <= {tol})")
            break

        prev_site_set = selected_set

    total_elapsed = time.time() - total_start
    if verbose:
        print(f"\n[done] best ODE score = {best_ode_score:.6f}  "
              f"sites = {best_sites}  "
              f"total time = {total_elapsed:.1f}s  "
              f"iterations = {len(history)}")

    return best_sites, best_ode_score, history
