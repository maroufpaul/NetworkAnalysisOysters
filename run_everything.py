#!/usr/bin/env python3
"""
run_everything.py  --  run ALL models (ODE heuristics + integer programs),
parallelized, and validate every output against references.json.

    python run_everything.py
      --skip-miqp        only run the heuristics
      --skip-heur        only run the integer programs
      --workers N        CPU workers for the heuristics (default: all cores)
      --update-refs      write this run's numbers back into references.json
                         (use after a fresh Gurobi solve to record/refresh the
                         golden values, e.g. the bilinear +Size objectives)

All knobs come from config.py. All expected values come from references.json,
so "proving" a result is just: re-run and check every line says OK. Anything
without a reference prints "-- (no ref)" instead of passing or failing.

Self-recruitment is intrinsic to the ODE and kept in the MIQP (the connectivity
diagonal is retained by scripts/prepare_data.py), so every model is canonical.

Heuristic functions are re-implemented here ONLY to add parallelism; the
algorithm and first-max tie-breaking exactly match src/opt/{greedy,local_search,
backward}.py. Writes runs/run_everything.json.
"""
from __future__ import annotations
import sys, os, json, time, argparse, subprocess
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.model.jars_ode import CANDIDATE_SITES

AMPL_DIR = config.AMPL_DIR
RUNS_DIR = config.RUNS_DIR; RUNS_DIR.mkdir(exist_ok=True)
MATRICES = config.MATRICES
TMAX, P1, CP0 = config.TMAX, config.P1SCALING, config.CONST_P0
REFS_FILE = ROOT / "references.json"

# ---------- references (golden values) ----------
def load_refs():
    if REFS_FILE.exists():
        return json.loads(REFS_FILE.read_text())
    print(f"  (no references.json found at {REFS_FILE}; everything runs unvalidated)")
    return {"heuristics": {}, "miqp": {}, "backbone": {}}

REFS = load_refs()

# ---------- parallel ODE worker (top-level for 'spawn') ----------
_CONN = None; _KEY = None
def _init(path):
    global _CONN, _KEY
    from src.model.jars_ode import load_connectivity
    _CONN, _KEY = load_connectivity(path)
def _score(task):
    from src.opt.evaluator import evaluate_subset
    sites, tmax, mode = task
    return evaluate_subset(list(sites), _CONN, _KEY, tmax=tmax, P1scaling=P1, P0_mode=mode, consP0=CP0)

# ---------- heuristics (parallel; tie-breaking matches src/opt) ----------
def _one(ex, sites, tmax, mode):
    return list(ex.map(_score, [(tuple(sites), tmax, mode)]))[0]

def greedy(ex, k, tmax, mode):
    selected, remaining = [], CANDIDATE_SITES.tolist()
    for _ in range(k):
        scores = list(ex.map(_score, [(tuple(selected+[c]), tmax, mode) for c in remaining]))
        bi = int(np.argmax(scores))
        selected.append(remaining[bi]); best = scores[bi]; remaining.pop(bi)
    return selected, best

def stingy(ex, k, tmax, mode):
    S = CANDIDATE_SITES.tolist(); cur = _one(ex, S, tmax, mode)
    while len(S) > k:
        scores = list(ex.map(_score, [(tuple(s for s in S if s != site), tmax, mode) for site in S]))
        bi = int(np.argmax(scores)); S.pop(bi); cur = scores[bi]
    return sorted(S), cur

def swap(ex, start, tmax, mode, max_passes=50, tol=1e-6):
    current = np.array(start, dtype=int); universe = CANDIDATE_SITES
    poolarr = np.setdiff1d(universe, current); cur = _one(ex, current, tmax, mode)
    for _ in range(max_passes):
        pairs = [(o, i) for o in current.tolist() for i in poolarr.tolist()]
        tasks = []
        for (o, i) in pairs:
            t = current.copy(); t[np.where(t == o)[0][0]] = i; tasks.append((tuple(t), tmax, mode))
        scores = np.array(list(ex.map(_score, tasks))); deltas = scores - cur
        bi = int(np.argmax(deltas))
        if deltas[bi] > tol:
            o, i = pairs[bi]; current[np.where(current == o)[0][0]] = i
            poolarr = np.setdiff1d(universe, current); cur = float(scores[bi])
        else:
            break
    return sorted(current.tolist()), cur

# ---------- MIQP via AMPL/Gurobi ----------
def prepare(matrix_id):
    r = subprocess.run([sys.executable, "-m", "scripts.prepare_data", "--matrix", matrix_id],
                       cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise RuntimeError("prepare_data failed")
    dat = (AMPL_DIR/"oyster_quad.dat").read_text()
    return sum(1 for ln in dat.splitlines() if ln.strip().startswith("[")
               and ln.split("[")[1].split(",")[0].strip() == ln.split(",")[1].split("]")[0].strip())

def solve_miqp(model, dats, objname, has_size, matrix_id):
    from amplpy import AMPL
    import pandas as pd
    a = AMPL(); a.eval("option solver gurobi;")
    a.eval(f"option gurobi_options '{config.GUROBI_OPTIONS}';")
    a.read(str(AMPL_DIR/model)); a.readData(str(AMPL_DIR/"oyster_quad.dat"))
    for d in dats: a.readData(str(AMPL_DIR/d))
    a.eval("solve;")
    lab = pd.read_csv(RUNS_DIR/f"oyster_index_mapping_matrix{matrix_id}.csv")["site_id"].tolist()
    picked = [int(r[0]) for r in a.getVariable("x").getValues().to_list() if float(r[1]) > 0.5]
    sites = sorted(lab[i] for i in picked); obj = float(a.getObjective(objname).value())
    area = None
    if has_size:
        sv = {int(r[0]): float(r[1]) for r in a.getVariable("s").getValues().to_list()}
        area = round(sum(sv.get(i, 0.0) for i in picked), 2)
    return sites, obj, area

MIQP_SPECS = [("Base","oyster_quad.mod",[],"score",False),
              ("+Comm","oyster_comm.mod",["oyster_comm.dat"],"Larvae",False),
              ("+Size","oyster_size.mod",["oyster_size.dat"],"Larvae",True),
              ("+Comm+Size","oyster_comm_size.mod",["oyster_comm.dat","oyster_size.dat"],"TotalLarvae",True)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-miqp", action="store_true")
    ap.add_argument("--skip-heur", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--update-refs", action="store_true",
                    help="write this run's numbers back into references.json")
    args = ap.parse_args()

    t0 = time.time()
    out = {"config": {"TMAX":TMAX,"P1SCALING":P1,"CONST_P0":CP0,"K":config.K,
                      "A_STAR":config.A_STAR,"workers":args.workers},
           "heuristics": {}, "miqp": {}, "backbone": {}, "summary": {}}
    npass = nfail = 0
    def rec(ok):
        nonlocal npass, nfail
        if ok is True: npass += 1
        elif ok is False: nfail += 1
    designs = {}
    new_refs = {"heuristics": dict(REFS.get("heuristics", {})),
                "miqp": dict(REFS.get("miqp", {})),
                "backbone": dict(REFS.get("backbone", {}))}

    # ---------------- heuristics ----------------
    if not args.skip_heur:
        for mat in ("M1", "M2"):
            print(f"\n===== heuristics {mat} ({args.workers} workers) =====")
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                     initargs=(str(MATRICES[mat]),)) as ex:
                for regime in ("constant", "realistic"):
                    t = time.time()
                    g_sites, g_sc = greedy(ex, config.K, TMAX, regime)
                    sw_sites, sw_sc = swap(ex, g_sites, TMAX, regime)
                    st_sites, st_sc = stingy(ex, config.K, TMAX, regime)
                    for meth, (sites, sc) in [("Greedy",(g_sites,g_sc)),("Swap",(sw_sites,sw_sc)),
                                              ("Stingy",(st_sites,st_sc))]:
                        designs[f"{mat} {meth} ({regime[0]})"] = sorted(sites)
                        ref = REFS.get("heuristics", {}).get(f"{mat} {regime} {meth}")
                        ok = None
                        if ref is not None:
                            ok = abs(sc-ref["score"]) < 5e-6 and sorted(sites) == sorted(ref["set"]); rec(ok)
                        out["heuristics"][f"{mat} {meth} {regime}"] = {
                            "score": round(sc,6), "set": sorted(sites),
                            "ref": (ref["score"] if ref else None), "pass": ok}
                        new_refs["heuristics"][f"{mat} {regime} {meth}"] = {"score": round(sc,6), "set": sorted(sites)}
                        flag = "OK" if ok else ("-- (no ref)" if ok is None else "*** MISMATCH")
                        print(f"  {mat} {regime:9s} {meth:7s} F={sc:.6f} {flag}")
                    print(f"  ({mat} {regime} done in {time.time()-t:.0f}s)")

    # ---------------- MIQP ----------------
    if not args.skip_miqp:
        for matrix_id in ("1", "2"):
            mat = f"M{matrix_id}"
            diag = prepare(matrix_id)
            print(f"\n===== MIQP {mat}: oyster_quad.dat has {diag} diagonal entries "
                  f"({'self-recruitment KEPT' if diag>0 else 'WARNING empty diagonal'}) =====")
            for short, model, dats, objn, has_size in MIQP_SPECS:
                sites, obj, area = solve_miqp(model, dats, objn, has_size, matrix_id)
                ref = REFS.get("miqp", {}).get(f"{mat} {short}")
                ok = None
                if ref is not None:
                    ok = abs(obj-ref) < max(1e-2, 1e-4*abs(ref)); rec(ok)
                designs[f"MIQP {mat} {short}"] = sorted(sites)
                out["miqp"][f"{mat} {short}"] = {"objective": round(obj,4), "ref": ref,
                                                 "pass": ok, "n_sites": len(sites),
                                                 "sites": sorted(sites), "total_area": area}
                new_refs["miqp"][f"{mat} {short}"] = round(obj,2)
                flag = "OK" if ok else ("-- (no ref)" if ok is None else "*** MISMATCH")
                print(f"  {mat} {short:11s} obj={obj:12.4f} {flag}" + (f" area={area}" if area else ""))

    # ---------------- backbone ----------------
    if not args.skip_heur and not args.skip_miqp:
        inter = lambda ns: sorted(set.intersection(*[set(designs[n]) for n in ns if n in designs]))
        # Within-matrix backbones use every CONSTANT-forcing run, greedy seed
        # included (paper Table 5).
        m1c = [f"M1 {m} (c)" for m in ("Greedy","Swap","Stingy")] + [f"MIQP M1 {s}" for s,*_ in MIQP_SPECS]
        m2c = [f"M2 {m} (c)" for m in ("Greedy","Swap","Stingy")] + [f"MIQP M2 {s}" for s,*_ in MIQP_SPECS]
        i1, i2 = inter(m1c), inter(m2c); cross = sorted(set(i1)&set(i2))
        # Global backbone = the 16 POST-SEARCH and EXACT designs only (4 swap +
        # 4 stingy across both matrices and both forcing regimes, + 8 MIQP).
        # Standalone greedy is excluded: it merely seeds the swap, and the M1
        # realistic greedy seed omits site 37, so intersecting over greedy too
        # would drop 37 and give a spurious 6-site core.
        post_exact = [n for n in designs if "Greedy" not in n]
        glob = inter(post_exact)
        for k, got in [("M1 within",i1),("M2 within",i2),("cross",cross),("global",glob)]:
            ref = REFS.get("backbone", {}).get(k)
            ok = None
            if ref is not None:
                ok = got == sorted(ref); rec(ok)
            out["backbone"][k] = {"sites": got, "ref": ref, "size": len(got), "pass": ok}
            new_refs["backbone"][k] = got
            flag = "OK" if ok else ("-- (no ref)" if ok is None else "*** MISMATCH")
            print(f"  backbone {k:10s} size={len(got):2d} {flag} {got}")

    out["summary"] = {"passed": npass, "failed": nfail, "all_pass": nfail == 0,
                      "runtime_sec": round(time.time()-t0,1)}
    (RUNS_DIR/"run_everything.json").write_text(json.dumps(out, indent=2))
    print(f"\nPASSED {npass}  FAILED {nfail}  ALL_PASS={nfail==0}  ({out['summary']['runtime_sec']}s)")
    print(f"wrote {RUNS_DIR/'run_everything.json'}")

    if args.update_refs:
        REFS_FILE.write_text(json.dumps(new_refs, indent=2))
        print(f"updated {REFS_FILE} with this run's values")

    return 0 if nfail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
