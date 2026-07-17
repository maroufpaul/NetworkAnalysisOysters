# src/model/jars_ode.py
#
# Python implementation of the JARS (Juvenile, Adult, Reef, Sediment)
# oyster population model.
#
# The single-reef JARS model and all demographic parameters below are from:
#   Lipcius, R.N., Y. Zhang, J. Zhou, L.B. Shaw, and J. Shi (2021),
#   "Modeling oyster reef restoration: larval supply and reef geometry
#   jointly determine population resilience and performance,"
#   Frontiers in Marine Science, 8:677640.
#   https://doi.org/10.3389/fmars.2021.677640
#   (Parameter values match Table 2; model time t is in years.)
#
# JARS itself revises the earlier three-equation reef/sediment model of
#   Jordan-Cooley et al. (2011), J. Theoretical Biology 289:1-11.
#
# The metapopulation extension implemented here -- larval coupling across
# sites via the connectivity matrix (the P1 term) -- follows the TSPS study:
#   Lipcius, Shen, Shaw, and Shi (2024), USACE Chesapeake Watershed CESU
#   Final Report W912HZ-23-02-0015.
#
# Original MATLAB by Prof. Leah B. Shaw (William & Mary); Python port by Marouf Paul.
#
# NOTE: data (the candidate list, the realistic-P0 vector) and the run knobs
# (P1SCALING, CONST_P0, MU, TMAX, initial conditions, default connectivity file)
# now live in config.py. They are re-exported here so existing imports such as
# `from src.model.jars_ode import CANDIDATE_SITES` keep working unchanged.

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import config

# Re-exported for backward compatibility.
CANDIDATE_SITES = config.CANDIDATE_SITES
DATA_DIR = config.DATA_DIR
DEFAULT_CONNECTIVITY_FILE = config.MATRICES[config.DEFAULT_MATRIX]


def load_connectivity(path=None):
    """
    Load a 56-site connectivity Excel and return:
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
    Given key (all site labels in the matrix) and a site_set (subset of labels),
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
    Realistic external-larvae vector, indexed by site label (1-based).
    The vector itself lives in config.P0_REALISTIC.
    """
    return config.P0_REALISTIC[sites - 1]


def odesys(t, v, P0, P1, mu):
    """
    The JARS ODE system. v is a flat vector with blocks: J, A, R, S.
    """
    Npatch = len(P0)
    x = v.reshape((4, Npatch)).T  # shape (Npatch, 4)
    J = x[:, 0]
    A = x[:, 1]
    R = x[:, 2]
    S = x[:, 3]

    # demographic parameters (Lipcius et al. 2021, Table 2)
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

    dj = aj * A + R - S
    da = aa * A + R - S
    fj = 1.0 / (1 + np.exp(-h * dj))
    fa = 1.0 / (1 + np.exp(-h * da))
    L = (A/psi + (R/psi)**2) / (zeta + (np.abs(A)/psi)**theta + (R/psi)**2)
    g = np.exp(-eta * (A + R))
    Cg = C * g
    F = F0 * Cg / y0 * np.exp((y0 - Cg) / y0)

    # larval input (P1 is internal connectivity)
    larvalinput = (P0 + P1.T @ (np.abs(A)**config.ALPHA)) * L * fj

    dJ = np.minimum(larvalinput, m * KJ * np.ones(Npatch)) - m * J
    dA = (m * alpha * J + phi * A * fa * (1 - A / KA) -
          mu * A * fa - epsilon * A * (1 - fa))
    dR = mu * A * fa + epsilon * A * (1 - fa) - gamma * R
    dS = -beta * S + Cg * np.exp(-F * A / Cg)

    return np.concatenate([dJ, dA, dR, dS])


# run_full_jars_on_subset() removed: dead code (nothing called it) that
# duplicated evaluator.evaluate_subset with a DIFFERENT default (P0_mode=
# "constant" vs "realistic"). Use src.opt.evaluator.evaluate_subset.