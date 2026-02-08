# src/opt/evaluator.py

from typing import Sequence
import numpy as np
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
    tmax: int = 1000,
    P1scaling: float = 0.5,
    P0_mode: str = "realistic",
    consP0: float = 170.0,
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

    Returns
    -------
    float : total adults at t = tmax
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

    mu = 0.4 * np.ones(n)

    # initial conditions (same as your original)
    J0, A0, R0, S0 = 0.0, 0.2, 0.3, 0.0
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
    return float(np.sum(A_final))
