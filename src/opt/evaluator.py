# src/opt/evaluator.py

from typing import Sequence
import numpy as np
import config
from src.model.jars_ode import (
    odesys,
    sitetoindex,
    setP0,
)
from scipy.integrate import solve_ivp


def evaluate_subset(
    site_labels: Sequence[int],
    connectivity_data: np.ndarray,
    key_all: np.ndarray,
    tmax: int = config.TMAX,
    P1scaling: float = config.P1SCALING,
    P0_mode: str = "realistic",
    consP0: float = config.CONST_P0,
    return_densities: bool = False,
) -> float:
    """
    Core operation: run the JARS ODE on *just* the given site labels and
    return total adult biomass (sum of A at final time).

    Parameters
    ----------
    site_labels : list/array of ints
        Site IDs like [10, 40, 41].
    connectivity_data : np.ndarray
        Full connectivity matrix (numeric part) for all sites.
    key_all : np.ndarray
        Site labels corresponding to rows/cols in connectivity_data.
    tmax : int
        Integration horizon.
    P1scaling : float
        Multiply connectivity by this.
    P0_mode : str
        "constant" → use consP0
        "realistic" → use setP0(...)
        "zero" → no external larvae
    consP0 : float
        Value used when P0_mode == "constant".

    return_densities : bool
        If True, also return {site_label: equilibrium adult density}.

    Returns
    -------
    float : total adults at t = tmax
    (float, dict) if return_densities.
    """
    # turn into numpy array
    site_labels = np.array(site_labels, dtype=int)

    # map labels into indices in key_all
    idx = sitetoindex(key_all, site_labels)
    if len(idx) == 0:
        return 0.0

    # restrict connectivity to those indices
    P1 = P1scaling * connectivity_data[np.ix_(idx, idx)]
    key_subset = key_all[idx]
    n = len(key_subset)

    # external larvae
    if P0_mode == "constant":
        P0 = consP0 * np.ones(n)
    elif P0_mode == "realistic":
        P0 = setP0(key_subset)
    else:
        P0 = np.zeros(n)

    mu = config.MU * np.ones(n)

    # initial conditions (same as your original)
    J0, A0, R0, S0 = config.IC["J"], config.IC["A"], config.IC["R"], config.IC["S"]
    v0 = np.zeros(4 * n)
    v0[0:n] = J0
    v0[n:2*n] = A0
    v0[2*n:3*n] = R0
    v0[3*n:4*n] = S0

    # integrate
    sol = solve_ivp(
        lambda t, v: odesys(t, v, P0, P1, mu),
        [0, tmax],
        v0,
        method="RK45",
        rtol=1e-6,
    )

    v_final = sol.y[:, -1]
    A_final = v_final[n:2*n]

    # A site that goes extinct lands on ~ -5e-10 instead of exactly 0 -- that's
    # just integrator noise, and negative oysters aren't a thing. Clip it here so
    # nobody downstream ends up doing (-5e-10) ** 1.72 and getting NaN. (The ODE
    # itself never trips on this because odesys uses np.abs(A) ** ALPHA
    # internally, so the problem only shows up once you take these densities out
    # and build surrogate weights with them.)
    A_final = np.maximum(A_final, 0.0)
    if return_densities:
        return (float(np.sum(A_final)),
                {int(key_subset[i]): float(A_final[i]) for i in range(n)})
    return float(np.sum(A_final))