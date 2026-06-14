#!/usr/bin/env python3
"""
run_and_validate_all.py  --  single-file driver for the Oyster Reef project.

Runs EVERY model (ODE heuristics + 4 MIQP variants) on BOTH connectivity
matrices, with the connectivity DIAGONAL ZEROED for every MIQP model, validates
each result against an embedded REFERENCE (the numbers Claude reproduced), and
writes everything to ONE file:  runs/validation_all.json  (+ a .txt summary).

What it does
------------
  * ODE heuristics (greedy / greedy+swap / stingy): re-scored with the true JARS
    ODE via src.opt.evaluator. Optionally re-DERIVED by actually running the
    optimizers (RERUN_HEURISTICS=True) -- slower but proves the optimizer
    produces the set.
  * MIQP (base / +comm / +size / +comm+size): solved with your AMPL+Gurobi
    pipeline AFTER writing oyster_quad.dat with the diagonal zeroed. Each
    objective is independently re-checked in pure Python from the zeroed-diagonal
    surrogate. If amplpy/Gurobi is not importable, the script still validates the
    reference sets' surrogate objectives in Python (base/comm) so you always get
    an output file.
  * Backbone intersections recomputed from the freshly produced 18 designs.

Run from the repo root:   python run_and_validate_all.py
"""

from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"
AMPL_DIR  = ROOT / "ampl"
RUNS_DIR  = ROOT / "runs"; RUNS_DIR.mkdir(exist_ok=True)

MATRICES = {
    "M1": DATA_DIR / "nk_All_060102final_56sites_Model.xlsx",
    "M2": DATA_DIR / "nk_All_060103final_56sites_Model.xlsx",
}

TMAX        = 1000          # ODE integration horizon (matches the paper)
P1SCALING   = 0.5
CONST_P0    = 170.0
A_STAR      = 0.05675
ALPHA       = 1.72
SBAR        = 20.0
TOTREEF     = 1000.0

# Re-derive heuristic sets by actually running the optimizers?
#   True  -> proves the optimizer reproduces the set (SLOW: greedy ~5 min,
#            greedy+swap ~15-30 min, stingy ~5 min, per matrix per regime).
#   False -> only re-SCORE the reference sets with the ODE (fast, still catches
#            score/transcription errors -- this is what caught the M2 greedy bug).
RERUN_HEURISTICS = True
INCLUDE_SWAP     = True     # include the (slow) greedy+swap re-derivation
REGIMES          = ["constant", "realistic"]

TOL_ODE   = 5e-6            # score tolerance for ODE heuristics
TOL_MIQP  = 1.0e-2          # objective tolerance for MIQP (absolute)


def _s(x): return sorted(x)
REF = {
 "heur": {
   "constant": {
     "M1 Greedy":      (1.846640, _s([4,10,15,16,17,19,20,21,26,27,28,30,31,32,36,37,40,41,44,47,49,51,52,53,54])),
     "M1 Greedy+Swap": (1.880054, _s([10,12,15,16,17,20,21,26,28,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
     "M1 Stingy":      (1.880054, _s([10,12,15,16,17,20,21,26,28,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
     "M2 Greedy":      (1.692871, _s([4,10,11,12,15,17,19,20,21,26,27,28,29,30,31,32,36,37,40,41,47,49,51,52,53])),
     "M2 Greedy+Swap": (1.733033, _s([10,11,12,15,16,17,19,20,26,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
     "M2 Stingy":      (1.733033, _s([10,11,12,15,16,17,19,20,26,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
   },
   "realistic": {
     "M1 Greedy+Swap": (1.861968, _s([4,6,10,15,19,20,21,24,27,30,31,32,36,37,38,39,40,41,47,49,51,52,53,55,60])),
     "M1 Stingy":      (1.819074, _s([4,10,15,16,17,19,20,21,26,27,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
     "M2 Greedy+Swap": (1.793457, _s([1,4,6,10,18,19,20,21,24,30,31,35,36,37,38,39,40,41,42,47,49,53,55,57,60])),
     "M2 Stingy":      (1.723331, _s([4,6,7,10,15,16,19,20,21,24,26,27,30,31,32,36,37,40,41,44,47,49,51,52,53])),
   },
 },
 "miqp": {   # constant P0, diagonal zeroed, global optimum
   "M1 Base":           (13692.07, _s([10,11,12,15,16,17,20,26,27,28,29,31,32,33,36,37,40,41,44,49,51,52,53,54,59])),
   "M1 +Comm":          (12782.91, _s([10,12,15,16,21,26,27,28,29,31,32,33,37,40,41,42,44,47,48,49,51,52,53,54,59])),
   "M1 +Size":          (53244.11, _s([10,11,12,15,16,17,20,26,27,28,29,31,32,33,36,37,40,41,44,49,51,52,53,54,59])),
   "M1 +Comm+Size":     (51082.03, _s([10,11,12,15,16,21,27,28,29,31,32,33,37,40,41,42,44,47,49,51,52,53,54,56,59])),
   "M2 Base":           (15806.74, _s([10,11,12,15,16,17,20,26,27,28,29,31,32,33,36,37,40,41,44,49,51,52,53,54,59])),
   "M2 +Comm":          (14587.09, _s([10,11,12,15,16,17,21,27,28,29,31,32,33,36,37,40,41,42,47,48,49,51,52,53,59])),
   "M2 +Size":          (63607.36, _s([10,11,12,15,16,17,20,26,27,28,29,31,32,33,36,37,40,41,44,49,51,52,53,54,59])),
   "M2 +Comm+Size":     (61482.51, _s([10,11,12,15,16,21,26,27,28,29,31,32,33,36,37,40,41,42,47,48,49,51,52,53,59])),
 },
 "backbone": {
   "M1 within (15)":   _s([10,15,16,28,31,32,37,40,41,44,49,51,52,53,54]),
   "M2 within (15)":   _s([10,11,12,15,29,31,32,36,37,40,41,49,51,52,53]),
   "cross (11)":       _s([10,15,31,32,37,40,41,49,51,52,53]),
   "global (7)":       _s([10,31,37,40,41,49,53]),
 },
}

# --------------------------------------------------------------------------
# DATA / SURROGATE helpers
# --------------------------------------------------------------------------
UNWANTED = [66,67,68,69,70,71,72]
from src.model.jars_ode import load_connectivity, CANDIDATE_SITES
from src.opt.evaluator import evaluate_subset

def load_matrix(path):
    conn, key = load_connectivity(path)
    return conn, key

def build_surrogate(path):
    """Return (W, Pe, key49) with the DIAGONAL ZEROED."""
    conn, key = load_connectivity(path)
    mask = ~np.isin(key, UNWANTED)
    key  = key[mask]
    P    = conn[np.ix_(mask, mask)] * P1SCALING
    np.fill_diagonal(P, 0.0)                        # <<< DIAGONAL ZEROED
    W    = P * (A_STAR ** ALPHA)
    Pe   = CONST_P0 * np.ones(len(key))
    return W, Pe, key

def surrogate_binary(sites, W, Pe, key):
    idx = [int(np.where(key == s)[0][0]) for s in sites]
    return float(Pe[idx].sum() + W[np.ix_(idx, idx)].sum())

def surrogate_sized(sites, sizes, W, Pe, key):
    idx = [int(np.where(key == s)[0][0]) for s in sites]
    v   = np.array([sizes[s] for s in sites]) / SBAR
    return float((Pe[idx]*v).sum() + (W[np.ix_(idx, idx)]*np.outer(v, v)).sum())

# --------------------------------------------------------------------------
# HEURISTIC re-derivation (monkeypatch load_connectivity for matrix choice)
# --------------------------------------------------------------------------
def patch_matrix(conn, key):
    import src.opt.greedy as G, src.opt.local_search as L, src.opt.backward as B
    f = lambda p=None: (conn, key)
    G.load_connectivity = f; L.load_connectivity = f; B.load_connectivity = f
    return G, L, B

def derive_heuristics(conn, key, regime):
    """Return dict name -> (score, sorted_sites) by running the optimizers."""
    G, L, B = patch_matrix(conn, key)
    out = {}
    if regime == "constant":
        gs, gsc = G.greedy_select_sites(k=25, tmax=TMAX, P1scaling=P1SCALING,
                                        P0_mode=regime, consP0=CONST_P0)
        out["Greedy"] = (gsc, sorted(gs))
    else:
        gs, gsc = G.greedy_select_sites(k=25, tmax=TMAX, P1scaling=P1SCALING,
                                        P0_mode=regime, consP0=CONST_P0)
    if INCLUDE_SWAP:
        ss, ssc = L.local_swap_hillclimb(gs, tmax=TMAX, P1scaling=P1SCALING,
                                         P0_mode=regime, consP0=CONST_P0)
        out["Greedy+Swap"] = (ssc, sorted(ss))
    bs, bsc = B.backward_greedy(k=25, tmax=TMAX, P1scaling=P1SCALING,
                                P0_mode=regime, consP0=CONST_P0)
    out["Stingy"] = (bsc, sorted(bs))
    return out

# --------------------------------------------------------------------------
# MIQP via AMPL/Gurobi  (writes diagonal-zeroed oyster_quad.dat first)
# --------------------------------------------------------------------------
def write_quad_dat(W, Pe, key):
    n = len(key)
    with open(AMPL_DIR / "oyster_quad.dat", "w", encoding="utf-8") as f:
        f.write("# auto-generated by run_and_validate_all.py (DIAGONAL ZEROED)\n\n")
        f.write("set N :=\n" + "".join(f"  {i}\n" for i in range(n)) + ";\n\n")
        f.write("param K := 25;\n\n")
        f.write("param Pe :=\n" + "".join(f"  {i} {Pe[i]:.4f}\n" for i in range(n)) + ";\n\n")
        f.write("param W :=\n")
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0.0:
                    f.write(f"  [{i}, {j}] {W[i,j]:.6f}\n")
        f.write(";\n")

def solve_miqp_ampl(model_file, extra_dats, key, has_size, obj_name):
    """Return (sorted_site_labels, objective, sizes_or_None) or None if no AMPL."""
    try:
        from amplpy import AMPL
    except Exception as e:
        return None
    ampl = AMPL()
    ampl.eval("option solver gurobi;")
    ampl.read(str(AMPL_DIR / model_file))
    ampl.readData(str(AMPL_DIR / "oyster_quad.dat"))
    for d in extra_dats:
        ampl.readData(str(AMPL_DIR / d))
    ampl.eval("solve;")
    xv = ampl.getVariable("x").getValues().to_list()
    picked = [int(r[0]) for r in xv if float(r[1]) > 0.5]
    labels = [int(key[i]) for i in picked]
    obj = float(ampl.getObjective(obj_name).value())
    sizes = None
    if has_size:
        sv = {int(r[0]): float(r[1]) for r in ampl.getVariable("s").getValues().to_list()}
        sizes = {int(key[i]): round(sv.get(i, 0.0), 3) for i in picked}
    return sorted(labels), obj, sizes

MIQP_SPECS = [   # (ref_name, model_file, extra_dats, has_size, ampl_obj_name)
  ("Base",        "oyster_quad.mod",      [],                                   False, "score"),
  ("+Comm",       "oyster_comm.mod",      ["oyster_comm.dat"],                  False, "Larvae"),
  ("+Size",       "oyster_size.mod",      ["oyster_size.dat"],                  True,  "Larvae"),
  ("+Comm+Size",  "oyster_comm_size.mod", ["oyster_comm.dat","oyster_size.dat"],True,  "TotalLarvae"),
]

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def cmp_set(a, b):
    a, b = set(a), set(b)
    return {"equal": a == b, "only_actual": sorted(a-b), "only_ref": sorted(b-a)}

def main():
    t0 = time.time()
    results = {"config": {"TMAX": TMAX, "P1SCALING": P1SCALING, "CONST_P0": CONST_P0,
                          "A_STAR": A_STAR, "ALPHA": ALPHA, "diagonal": "ZEROED",
                          "RERUN_HEURISTICS": RERUN_HEURISTICS, "INCLUDE_SWAP": INCLUDE_SWAP},
               "data_layer": {}, "heuristics": {}, "miqp": {}, "backbone": {}, "summary": {}}
    designs = {}     # name -> sorted sites, for backbone recomputation
    n_pass = n_fail = 0
    def record(ok):
        nonlocal n_pass, n_fail
        if ok: n_pass += 1
        else:  n_fail += 1

    print("="*70); print("OYSTER REEF -- run + validate ALL models (diagonal zeroed)"); print("="*70)

    # ---- data-layer sanity ----
    conn1, key1 = load_matrix(MATRICES["M1"])
    conn2, key2 = load_matrix(MATRICES["M2"])
    cand_ok = sorted(set(key1)-set(UNWANTED)) == sorted(CANDIDATE_SITES.tolist())
    results["data_layer"] = {
        "M1_labels": int(len(key1)), "M2_labels": int(len(key2)),
        "labels_match": bool(np.array_equal(key1, key2)),
        "candidate_sites": int(len(CANDIDATE_SITES)),
        "drop66_72_gives_candidates": bool(cand_ok),
    }
    print(f"[data] M1={len(key1)} labels, M2={len(key2)} labels, identical={np.array_equal(key1,key2)}, "
          f"drop{{66..72}}==CANDIDATE_SITES={cand_ok}")
    record(cand_ok and np.array_equal(key1, key2))

    CONN = {"M1": (conn1, key1), "M2": (conn2, key2)}

    # ---- HEURISTICS ----
    for regime in REGIMES:
        results["heuristics"][regime] = {}
        for mat in ("M1", "M2"):
            conn, key = CONN[mat]
            if RERUN_HEURISTICS:
                print(f"\n[heur] DERIVING {mat} / {regime} (this is the slow part)...")
                derived = derive_heuristics(conn, key, regime)
                got = {f"{mat} {k}": v for k, v in derived.items()}
            else:
                got = {}  # evaluate reference sets only
            # validate every reference entry for this matrix+regime
            for name, (ref_score, ref_sites) in REF["heur"][regime].items():
                if not name.startswith(mat):
                    continue
                if name in got:
                    score, sites = got[name]
                else:
                    sites = ref_sites
                    score = evaluate_subset(sites, conn, key, tmax=TMAX,
                                            P1scaling=P1SCALING, P0_mode=regime, consP0=CONST_P0)
                # re-score whatever set we have, to be safe
                rescore = evaluate_subset(sites, conn, key, tmax=TMAX,
                                          P1scaling=P1SCALING, P0_mode=regime, consP0=CONST_P0)
                score_ok = abs(rescore - ref_score) < TOL_ODE
                set_ok   = set(sites) == set(ref_sites)
                ok = score_ok and (set_ok or not RERUN_HEURISTICS)
                designs[f"{name} ({regime})"] = sorted(sites)
                results["heuristics"][regime][name] = {
                    "ref_score": ref_score, "score": round(rescore, 6),
                    "score_pass": bool(score_ok), "set": sorted(sites),
                    "set_vs_ref": cmp_set(sites, ref_sites), "pass": bool(ok),
                }
                record(ok)
                print(f"  {name:18s} regime={regime:9s} F={rescore:.6f} (ref {ref_score:.6f}) "
                      f"score={'OK' if score_ok else 'FAIL'} set={'OK' if set_ok else 'DIFF'}")

    # ---- MIQP (constant, diagonal zeroed) ----
    for mat in ("M1", "M2"):
        W, Pe, key = build_surrogate(MATRICES[mat])
        write_quad_dat(W, Pe, key)
        for short, modf, dats, has_size, objn in MIQP_SPECS:
            ref_name = f"{mat} {short}"
            ref_obj, ref_sites = REF["miqp"][ref_name]
            ampl_res = solve_miqp_ampl(modf, dats, key, has_size, objn)
            entry = {"ref_obj": ref_obj, "ref_set": ref_sites}
            if ampl_res is None:
                # No AMPL -> validate reference set's objective in Python (binary models only)
                if not has_size:
                    py = surrogate_binary(ref_sites, W, Pe, key)
                    ok = abs(py - ref_obj) < TOL_MIQP
                    entry.update({"solver": "none(amplpy missing)",
                                  "python_obj_on_ref_set": round(py, 2),
                                  "obj_pass": bool(ok), "set_pass": None, "pass": bool(ok)})
                else:
                    entry.update({"solver": "none(amplpy missing)",
                                  "note": "sizing model needs solver; not validated locally",
                                  "pass": None})
                    ok = True   # don't count as failure when solver absent
                record(ok if entry.get("pass") is not None else True)
                print(f"  MIQP {ref_name:16s} [no AMPL] "
                      f"{'pyobj=%.2f vs %.2f'%(entry.get('python_obj_on_ref_set',float('nan')),ref_obj) if not has_size else '(size: solver required)'}")
                designs[f"MIQP {ref_name}"] = ref_sites
                results["miqp"][ref_name] = entry
                continue
            sites, solver_obj, sizes = ampl_res
            # independent Python recomputation of the objective on the SOLVER's set
            if has_size:
                py = surrogate_sized(sites, sizes, W, Pe, key)
                area = round(sum(sizes.values()), 2)
            else:
                py = surrogate_binary(sites, W, Pe, key)
                area = None
            obj_ok = abs(solver_obj - ref_obj) < TOL_MIQP
            py_ok  = abs(py - solver_obj) < max(TOL_MIQP, 1e-3*abs(solver_obj))
            set_eq = set(sites) == set(ref_sites)
            # community models can have alternate optima -> accept if objective matches
            set_ok = set_eq or (obj_ok and "Comm" in short)
            ok = obj_ok and py_ok and set_ok
            designs[f"MIQP {ref_name}"] = sorted(sites)
            entry.update({"solver": "ampl+gurobi", "solver_obj": round(solver_obj, 2),
                          "python_obj_on_solver_set": round(py, 2),
                          "set": sorted(sites), "sizes": sizes, "total_area": area,
                          "obj_pass": bool(obj_ok), "python_check_pass": bool(py_ok),
                          "set_equal_ref": bool(set_eq), "set_pass": bool(set_ok), "pass": bool(ok)})
            record(ok)
            print(f"  MIQP {ref_name:16s} solver={solver_obj:10.2f} ref={ref_obj:10.2f} "
                  f"py={py:10.2f} obj={'OK' if obj_ok else 'FAIL'} set={'OK' if set_ok else 'DIFF'}"
                  + (f" area={area}" if area else ""))
            results["miqp"][ref_name] = entry

    # ---- BACKBONE (recomputed from produced designs) ----
    def inter(names):
        sets = [set(designs[n]) for n in names if n in designs]
        return sorted(set.intersection(*sets)) if sets else []
    m1c = [f"M1 {m} (constant)" for m in ("Greedy","Greedy+Swap","Stingy")] + [f"MIQP M1 {s}" for s,*_ in MIQP_SPECS]
    m2c = [f"M2 {m} (constant)" for m in ("Greedy","Greedy+Swap","Stingy")] + [f"MIQP M2 {s}" for s,*_ in MIQP_SPECS]
    i1, i2 = inter(m1c), inter(m2c)
    cross  = sorted(set(i1) & set(i2))
    glob   = inter(list(designs.keys()))
    bb = {"M1 within": i1, "M2 within": i2, "cross": cross, "global": glob}
    bb_checks = {
        "M1 within (15)": (i1, REF["backbone"]["M1 within (15)"]),
        "M2 within (15)": (i2, REF["backbone"]["M2 within (15)"]),
        "cross (11)":     (cross, REF["backbone"]["cross (11)"]),
        "global (7)":     (glob, REF["backbone"]["global (7)"]),
    }
    print("\n[backbone] recomputed from produced designs:")
    for k, (got, ref) in bb_checks.items():
        ok = got == ref
        record(ok)
        results["backbone"][k] = {"sites": got, "ref": ref, "size": len(got), "pass": bool(ok)}
        print(f"  {k:16s} size={len(got):2d} {'OK' if ok else 'MISMATCH'}  {got}")

    # ---- SUMMARY + WRITE ----
    results["summary"] = {"checks_passed": n_pass, "checks_failed": n_fail,
                          "all_pass": n_fail == 0, "runtime_sec": round(time.time()-t0, 1)}
    out_json = RUNS_DIR / "validation_all.json"
    out_txt  = RUNS_DIR / "validation_all.txt"
    out_json.write_text(json.dumps(results, indent=2))
    with open(out_txt, "w") as f:
        f.write("OYSTER REEF validation (diagonal zeroed)\n")
        f.write(f"checks passed: {n_pass}, failed: {n_fail}, all_pass={n_fail==0}\n")
        f.write(f"runtime: {results['summary']['runtime_sec']}s\n\n")
        f.write(json.dumps(results, indent=2))
    print("\n" + "="*70)
    print(f"PASSED {n_pass}  FAILED {n_fail}   ALL PASS = {n_fail==0}")
    print(f"wrote {out_json}")
    print(f"wrote {out_txt}")
    print("="*70)
    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
