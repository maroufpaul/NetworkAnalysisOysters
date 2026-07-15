# scripts/surrogate_fidelity.py
#
# Surrogate-fidelity experiments for the ORSSP paper.
#
#   1. CONSTANT Pe: score the MIQP-selected set under the true JARS ODE F(S),
#      compare to the best ODE heuristic.
#
#   2. REALISTIC (site-specific) Pe: three ways to weight the source term
#      W[l,k] = P[l,k] * A_l^1.72:
#         (a) frozen single A_*                        -- current surrogate
#         (b) site-specific A_l, ISOLATED candidate    -- knowable a priori
#         (c) site-specific A_l, ALL-49 metapopulation -- CIRCULAR (uses the ODE
#                                                         answer); upper bound only
#
#   3. A_* sensitivity: does the MIQP selection change with A_*?
#
#   4. ITERATED surrogate (RECOMMENDED for realistic Pe):
#         A_l <- isolated densities
#         repeat: solve MIQP -> run ODE on THAT selected set -> read the actual
#                 A_l for the selected sites -> re-weight -> re-solve
#      Non-circular: each A_l comes from a set the surrogate itself chose, not
#      from a pre-computed all-49 solve. Converges in 1-2 passes.
#
# Run from the repo root:
#   python -m scripts.surrogate_fidelity

import numpy as np
import config
from src.model.jars_ode import load_connectivity, sitetoindex, odesys, setP0
from scipy.integrate import solve_ivp
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

BEST_HEUR_CONST = {
    "M1": [10,12,15,16,17,20,21,26,28,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59],
    "M2": [10,11,12,15,16,17,19,20,26,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59],
}
BEST_HEUR_REAL = {
    "M1": [4,6,10,15,19,20,21,24,27,30,31,32,36,37,38,39,40,41,47,49,51,52,53,55,60],
    "M2": [1,4,6,10,18,19,20,21,24,30,31,35,36,37,38,39,40,41,42,47,49,53,55,57,60],
}


def _prep(mid):
    conn, key = load_connectivity(config.MATRICES[mid])
    m = ~np.isin(key, config.UNWANTED)
    return key[m], conn[np.ix_(m, m)] * config.P1SCALING


def solve_miqp(mid, source_weight, Pe, K=25):
    lab, P = _prep(mid); n = len(lab)
    W = P * source_weight[:, None]
    lin = Pe + np.diag(W)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j and W[i, j] != 0]
    nv = n + len(pairs); c = np.zeros(nv); c[:n] = -lin
    for p, (i, j) in enumerate(pairs):
        c[n + p] = -W[i, j]
    cons = [LinearConstraint(lil_matrix(np.r_[np.ones(n), np.zeros(len(pairs))].reshape(1, -1)), K, K)]
    Ay = lil_matrix((2 * len(pairs), nv))
    for p, (i, j) in enumerate(pairs):
        Ay[2*p, n+p] = 1; Ay[2*p, i] = -1
        Ay[2*p+1, n+p] = 1; Ay[2*p+1, j] = -1
    cons.append(LinearConstraint(Ay, -np.inf, 0))
    ig = np.zeros(nv); ig[:n] = 1
    r = milp(c=c, constraints=cons, integrality=ig,
             bounds=Bounds(np.zeros(nv), np.ones(nv)), options={"time_limit": 180})
    return sorted(int(lab[i]) for i in range(n) if r.x[i] > 0.5)


def ode_on(sites, mid, pe_mode, tmax=1000):
    """Return (F(S), {site: equilibrium adult density}) for the selected set."""
    conn, key = load_connectivity(config.MATRICES[mid])
    idx = sitetoindex(key, np.array(sites, int)); n = len(idx)
    P1 = config.P1SCALING * conn[np.ix_(idx, idx)]
    P0 = config.CONST_P0 * np.ones(n) if pe_mode == "constant" else setP0(key[idx])
    v0 = np.zeros(4*n); v0[n:2*n] = config.IC["A"]; v0[2*n:3*n] = config.IC["R"]
    sol = solve_ivp(lambda t, v: odesys(t, v, P0, P1, config.MU*np.ones(n)),
                    [0, tmax], v0, method="RK45", rtol=1e-6)
    Af = np.maximum(sol.y[n:2*n, -1], 0.0)
    return float(Af.sum()), {int(key[idx][i]): float(Af[i]) for i in range(n)}


def isolated_A(mid, pe_mode):
    """Each candidate's equilibrium adult density run ALONE (self-recruitment only)."""
    lab, P = _prep(mid); vals = []
    for i, s in enumerate(lab):
        P1 = np.array([[P[i, i]]])
        P0 = np.array([config.CONST_P0]) if pe_mode == "constant" else setP0(np.array([s])).astype(float)
        v0 = np.array([0, config.IC["A"], config.IC["R"], 0], float)
        sol = solve_ivp(lambda t, v: odesys(t, v, P0, P1, config.MU*np.ones(1)),
                        [0, 1000], v0, method="RK45", rtol=1e-6)
        vals.append(max(float(sol.y[1, -1]), 0.0))
    return lab, np.array(vals)


def metapop_A(mid, pe_mode):
    """All-49 coupled equilibrium densities. CIRCULAR: uses the ODE answer."""
    lab, P = _prep(mid); n = len(lab)
    P0 = config.CONST_P0*np.ones(n) if pe_mode == "constant" else setP0(lab)
    v0 = np.zeros(4*n); v0[n:2*n] = config.IC["A"]; v0[2*n:3*n] = config.IC["R"]
    sol = solve_ivp(lambda t, v: odesys(t, v, P0, P, config.MU*np.ones(n)),
                    [0, 1000], v0, method="RK45", rtol=1e-6)
    return lab, np.maximum(sol.y[n:2*n, -1], 0.0)


def iterated_surrogate(mid, max_iter=5, verbose=True):
    """Fixed-point: isolated A_l -> solve -> ODE on that set -> re-weight -> re-solve."""
    lab, P = _prep(mid); Pe = setP0(lab).astype(float)
    l2i = {int(l): i for i, l in enumerate(lab)}
    f_best, _ = ode_on(BEST_HEUR_REAL[mid], mid, "realistic")
    _, Al = isolated_A(mid, "realistic")
    prev = None
    for it in range(max_iter):
        pick = solve_miqp(mid, Al**config.ALPHA, Pe)
        F, dens = ode_on(pick, mid, "realistic")
        if verbose:
            tag = "  (converged)" if prev == tuple(pick) else ""
            print(f"     iter {it}: F={F:.4f}  gap={100-100*F/f_best:5.2f}%  "
                  f"|set∩best|={len(set(pick)&set(BEST_HEUR_REAL[mid]))}/25{tag}")
        if prev == tuple(pick):
            return pick, F, f_best
        prev = tuple(pick)
        for s, d in dens.items():
            Al[l2i[s]] = d          # selected sites get their ACTUAL density
    return pick, F, f_best


def main():
    A = config.A_STAR; ALPHA = config.ALPHA

    print("="*72); print("1) SURROGATE FIDELITY under CONSTANT Pe"); print("="*72)
    for mid in ("M1", "M2"):
        lab, _ = _prep(mid)
        pick = solve_miqp(mid, np.full(len(lab), A**ALPHA), config.CONST_P0*np.ones(len(lab)))
        f_m, _ = ode_on(pick, mid, "constant"); f_h, _ = ode_on(BEST_HEUR_CONST[mid], mid, "constant")
        print(f"  {mid}: F(MIQP)={f_m:.4f}  F(best heur)={f_h:.4f}  gap={100-100*f_m/f_h:.2f}%")

    print("\n"+"="*72); print("2) REALISTIC Pe -- three source-weightings"); print("="*72)
    for mid in ("M1", "M2"):
        lab, _ = _prep(mid); Pe = setP0(lab).astype(float)
        _, Al_iso = isolated_A(mid, "realistic")
        _, Al_meta = metapop_A(mid, "realistic")
        f_best, _ = ode_on(BEST_HEUR_REAL[mid], mid, "realistic")
        print(f"\n  {mid}: best heuristic F={f_best:.4f}  "
              f"({int((Al_iso>1e-6).sum())}/49 establish in isolation)")
        for name, w in [
            ("frozen A_*                          ", np.full(len(lab), A**ALPHA)),
            ("site-specific A_l (isolated)        ", Al_iso**ALPHA),
            ("site-specific A_l (all-49, CIRCULAR)", Al_meta**ALPHA),
        ]:
            pick = solve_miqp(mid, w, Pe); f, _ = ode_on(pick, mid, "realistic")
            print(f"     {name}: F={f:.4f}  gap={100-100*f/f_best:6.2f}%")

    print("\n"+"="*72); print("3) A_* SENSITIVITY"); print("="*72)
    for pe_mode in ("constant", "realistic"):
        for mid in ("M1", "M2"):
            lab, _ = _prep(mid)
            Pe = config.CONST_P0*np.ones(len(lab)) if pe_mode == "constant" else setP0(lab).astype(float)
            picks = {a: tuple(solve_miqp(mid, np.full(len(lab), a**ALPHA), Pe))
                     for a in (0.02, 0.0435, 0.05675, 0.2, 0.5)}
            u = len(set(picks.values()))
            print(f"  {pe_mode:9s} {mid}: {u} distinct selection(s) -> "
                  f"{'SENSITIVE' if u>1 else 'invariant'}")

    print("\n"+"="*72); print("4) ITERATED SURROGATE under REALISTIC Pe (non-circular)"); print("="*72)
    for mid in ("M1", "M2"):
        print(f"\n  {mid}:")
        pick, F, f_best = iterated_surrogate(mid)
        print(f"     converged: F={F:.4f}  gap={100-100*F/f_best:.2f}%  ({len(pick)} sites)")


if __name__ == "__main__":
    main()
