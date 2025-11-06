# src/model/jars_ode.py

#Dr. Leah shaw's ODE models JARS ODE model implementation

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp

# candidate sites (professor shaw) removed the following sites: 8, 13, 14, 22, 23, 25, 34, 43, 45, 46
CANDIDATE_SITES = np.array([
    1, 3, 4, 5, 6, 7,  
    9, 10, 11, 12,
    15, 16, 17, 18, 19, 20, 21,
    24, 26, 27, 28, 29, 30, 31, 32, 33,
    35, 36, 37, 38, 39, 40, 41, 42,
    44, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60
], dtype=int)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CONNECTIVITY_FILE = DATA_DIR / "nk_All_060102final_56sites_Model.xlsx"


def load_connectivity(path: str | Path | None = None):
    """
    Load the big 56-site connectivity Excel and return:
      connectivity (numpy array)
      key_all (site labels as ints)
    """
    if path is None:
        path = DEFAULT_CONNECTIVITY_FILE

    arr = pd.read_excel(path, header=None).values
    key_all = arr[0, 1:].astype(int)
    connectivity = arr[1:, 1:].astype(float)
    return connectivity, key_all


def sitetoindex(key: np.ndarray, site_set: np.ndarray) -> np.ndarray:
    """
    Given key (list of all site labels in the matrix)
    and a site_set (subset of labels),
    return the indices in key that correspond to site_set.
    """
    index = []
    for site in site_set:
        j = np.where(key == site)[0]
        if len(j) > 0:
            index.append(j[0])
    return np.array(index, dtype=int)


def setP0(sites: np.ndarray) -> np.ndarray:
    """
    Realistic external larvae vector (copied from  original code).
     just index into it using the site labels.
    """
    allP0 = np.array([
        400, 400, 400, 400, 400, 400, 200, 200, 200, 100, 100, 100,
        100, 100, 100, 100, 100, 400, 400, 200, 200, 400, 400, 400,
        200, 100, 200, 100, 100, 200, 200, 100, 100, 100, 400, 200,
        200, 400, 400, 100, 100, 400, 400, 100, 400, 400, 200, 400,
        200, 100, 100, 100, 200, 100, 400, 400, 400, 100, 100, 400,
        100, 400, 400, 400, 100, 400, 400, 400, 400, 400, 200, 400
    ])
    # sites are 1-based in the original data
    P0 = allP0[sites - 1]
    return P0


def odesys(t, v, P0, P1, mu):
    """
    The JARS ODE system .
    v is a flat vector with blocks: J, A, R, S
    """
    Npatch = len(P0)
    x = v.reshape((4, Npatch)).T  # shape (Npatch, 4)
    J = x[:, 0]
    A = x[:, 1]
    R = x[:, 2]
    S = x[:, 3]

    # parameters (same as your file)
    aj = 0.5
    aa = 0.9
    h = 20

    psi = 0.3135
    theta = 2.43
    zeta = 0.8068

    eta = 3.33

    F0 = 1
    y0 = 0.02

    m = 1
    KJ = 1500

    alpha = 3.529e-5
    phi = 0.6469
    KA = 0.1055
    epsilon = 0.94

    gamma = 0.1155

    beta = 0.01
    C = 0.02

    # functions from your original
    dj = aj * A + R - S
    da = aa * A + R - S
    fj = 1.0 / (1 + np.exp(-h * dj))
    fa = 1.0 / (1 + np.exp(-h * da))
    L = (A/psi + (R/psi)**2) / (zeta + (np.abs(A)/psi)**theta + (R/psi)**2)
    g = np.exp(-eta * (A + R))
    Cg = C * g
    F = F0 * Cg / y0 * np.exp((y0 - Cg) / y0)

    # larval input (P1 is internal)
    larvalinput = (P0 + P1.T @ (np.abs(A)**1.72)) * L * fj

    dJ = np.minimum(larvalinput, m * KJ * np.ones(Npatch)) - m * J
    dA = (m * alpha * J + phi * A * fa * (1 - A / KA) -
          mu * A * fa - epsilon * A * (1 - fa))
    dR = mu * A * fa + epsilon * A * (1 - fa) - gamma * R
    dS = -beta * S + Cg * np.exp(-F * A / Cg)

    dv = np.concatenate([dJ, dA, dR, dS])
    return dv


def run_full_jars_on_subset(site_labels: list[int] | np.ndarray,
                            tmax: int = 1000,
                            P1scaling: float = 0.5,
                            P0_mode: str = "constant",
                            consP0: float = 170.0) -> float:
    """
    Convenience helper: load connectivity, restrict to given site_labels,
    run the ODE to tmax, return total adult biomass.
    This will be useful for quick tests and for notebooks.
    """
    connectivity, key_all = load_connectivity()
    site_labels = np.array(site_labels, dtype=int)
    idx = sitetoindex(key_all, site_labels)
    if len(idx) == 0:
        return 0.0

    P1 = P1scaling * connectivity[np.ix_(idx, idx)]
    key_subset = key_all[idx]
    Npatch = len(key_subset)

    # external larvae
    if P0_mode == "constant":
        P0 = consP0 * np.ones(Npatch)
    elif P0_mode == "realistic":
        P0 = setP0(key_subset)
    else:
        P0 = np.zeros(Npatch)

    mu = 0.4 * np.ones(Npatch)

    # initial conditions
    J0, A0, R0, S0 = 0, 0.2, 0.3, 0
    v0 = np.zeros(4 * Npatch)
    v0[0:Npatch] = J0
    v0[Npatch:2*Npatch] = A0
    v0[2*Npatch:3*Npatch] = R0
    v0[3*Npatch:4*Npatch] = S0

    sol = solve_ivp(lambda t, v: odesys(t, v, P0, P1, mu),
                    [0, tmax], v0, method="RK45", rtol=1e-6)

    v_final = sol.y[:, -1]
    A_final = v_final[Npatch:2*Npatch]
    return float(np.sum(A_final))
