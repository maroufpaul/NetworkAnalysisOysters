#!/usr/bin/env python3
"""
run_everything.py  --  run ALL models (ODE heuristics + integer programs) with
self-recruitment KEPT (canonical), parallelized, and validate every output.

python run_everything.py
  optional flags:
    --skip-miqp        only run the heuristics
    --skip-heur        only run the integer programs
    --workers N        CPU workers for the heuristics (default: all cores)

Speed: the heuristics spend all their time evaluating the JARS ODE on candidate
sets; those evaluations are independent, so this script fans them out across CPU
cores. The parallel evaluation is bit-identical to the serial library functions
(same tie-breaking), so results are unchanged -- only faster. Self-recruitment is
intrinsic to the ODE and kept in the MIQP (diagonal retained), so every model is
canonical.

Heuristic functions are re-implemented here ONLY to add parallelism + matrix
selection; the algorithm and tie-breaking exactly match src/opt/{greedy,
local_search,backward}.py. Each result is validated against the canonical
reference; a mismatch is reported loudly, never silently accepted.

Writes runs/run_everything.json.
"""
from __future__ import annotations
import sys, os, json, time, argparse, subprocess
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent
AMPL_DIR = ROOT / "ampl"
RUNS_DIR = ROOT / "runs"; RUNS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.model.jars_ode import CANDIDATE_SITES

MATRICES = {"M1": ROOT/"data"/"nk_All_060102final_56sites_Model.xlsx",
            "M2": ROOT/"data"/"nk_All_060103final_56sites_Model.xlsx"}
TMAX, P1, CP0 = 1000, 0.5, 170.0

# ---------- canonical reference (self-recruitment kept) ----------
def _s(x): return sorted(x)
REF_HEUR = {
 ("M1","constant","Greedy"):(1.846640,_s([4,10,15,16,17,19,20,21,26,27,28,30,31,32,36,37,40,41,44,47,49,51,52,53,54])),
 ("M1","constant","Swap"):(1.880054,_s([10,12,15,16,17,20,21,26,28,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
 ("M1","constant","Stingy"):(1.880054,_s([10,12,15,16,17,20,21,26,28,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
 ("M2","constant","Greedy"):(1.692871,_s([4,10,11,12,15,17,19,20,21,26,27,28,29,30,31,32,36,37,40,41,47,49,51,52,53])),
 ("M2","constant","Swap"):(1.733033,_s([10,11,12,15,16,17,19,20,26,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
 ("M2","constant","Stingy"):(1.733033,_s([10,11,12,15,16,17,19,20,26,29,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
 ("M1","realistic","Swap"):(1.861968,_s([4,6,10,15,19,20,21,24,27,30,31,32,36,37,38,39,40,41,47,49,51,52,53,55,60])),
 ("M1","realistic","Stingy"):(1.819074,_s([4,10,15,16,17,19,20,21,26,27,31,32,33,36,37,40,41,44,47,49,51,52,53,54,59])),
 ("M2","realistic","Swap"):(1.793457,_s([1,4,6,10,18,19,20,21,24,30,31,35,36,37,38,39,40,41,42,47,49,53,55,57,60])),
 ("M2","realistic","Stingy"):(1.723331,_s([4,6,7,10,15,16,19,20,21,24,26,27,30,31,32,36,37,40,41,44,47,49,51,52,53])),
}
REF_MIQP = {("1","Base"):14785.03,("1","+Comm"):13844.54,("1","+Size"):58564.19,("1","+Comm+Size"):56443.65,
            ("2","Base"):16446.59,("2","+Comm"):15218.16,("2","+Size"):66552.23,("2","+Comm+Size"):64448.16}
REF_BACKBONE = {"M1 within":_s([10,15,16,17,26,28,31,32,37,40,41,44,49,51,52,53,54]),
                "M2 within":_s([10,11,12,15,29,31,32,36,37,40,41,49,51,52,53]),
                "cross":_s([10,15,31,32,37,40,41,49,51,52,53]),
                "global":_s([10,31,37,40,41,49,53])}

# ---------- parallel ODE worker (top-level for Windows 'spawn') ----------
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
        bi = int(np.argmax(scores))                       # first-max wins (== src greedy)
        selected.append(remaining[bi]); best = scores[bi]; remaining.pop(bi)
    return selected, best

def stingy(ex, k, tmax, mode):
    S = CANDIDATE_SITES.tolist(); cur = _one(ex, S, tmax, mode)
    while len(S) > k:
        scores = list(ex.map(_score, [(tuple(s for s in S if s != site), tmax, mode) for site in S]))
        bi = int(np.argmax(scores)); S.pop(bi); cur = scores[bi]   # remove least-harmful, first-wins
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
        bi = int(np.argmax(deltas))                       # best improving swap, first-wins
        if deltas[bi] > tol:
            o, i = pairs[bi]; current[np.where(current == o)[0][0]] = i
            poolarr = np.setdiff1d(universe, current); cur = float(scores[bi])
        else:
            break
    return sorted(current.tolist()), cur

# ---------- MIQP via AMPL/Gurobi (loud; mirrors your run_miqp_*.py) ----------
def prepare(matrix_id):
    r = subprocess.run([sys.executable, "-m", "scripts.prepare_miqp_data", "--matrix", matrix_id],
                       cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise RuntimeError("prepare_miqp_data failed")
    dat = (AMPL_DIR/"oyster_quad.dat").read_text()
    return sum(1 for ln in dat.splitlines() if ln.strip().startswith("[")
               and ln.split("[")[1].split(",")[0].strip() == ln.split(",")[1].split("]")[0].strip())

def solve_miqp(model, dats, objname, has_size, matrix_id):
    from amplpy import AMPL
    import pandas as pd
    a = AMPL(); a.eval("option solver gurobi;")
    a.eval("option gurobi_options 'nonconvex=2 mipgap=1e-9';")
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
    args = ap.parse_args()

    t0 = time.time()
    out = {"model": "CANONICAL (self-recruitment KEPT)", "workers": args.workers,
           "heuristics": {}, "miqp": {}, "backbone": {}, "summary": {}}
    npass = nfail = 0
    def rec(ok):
        nonlocal npass, nfail
        if ok is True: npass += 1
        elif ok is False: nfail += 1
    designs = {}

    # ---------------- heuristics ----------------
    if not args.skip_heur:
        for mat in ("M1", "M2"):
            print(f"\n===== heuristics {mat} ({args.workers} workers) =====")
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                     initargs=(str(MATRICES[mat]),)) as ex:
                for regime in ("constant", "realistic"):
                    t = time.time()
                    g_sites, g_sc = greedy(ex, 25, TMAX, regime)
                    sw_sites, sw_sc = swap(ex, g_sites, TMAX, regime)
                    st_sites, st_sc = stingy(ex, 25, TMAX, regime)
                    for meth, (sites, sc) in [("Greedy",(g_sites,g_sc)),("Swap",(sw_sites,sw_sc)),
                                              ("Stingy",(st_sites,st_sc))]:
                        designs[f"{mat} {meth} ({regime[0]})"] = sorted(sites)
                        ref = REF_HEUR.get((mat, regime, meth))
                        ok = None
                        if ref is not None:
                            ok = abs(sc-ref[0]) < 5e-6 and sorted(sites) == ref[1]; rec(ok)
                        out["heuristics"][f"{mat} {meth} {regime}"] = {
                            "score": round(sc,6), "ref": (ref[0] if ref else None),
                            "set": sorted(sites), "set_ok": (sorted(sites)==ref[1] if ref else None),
                            "pass": ok}
                        flag = "OK" if ok else ("--" if ok is None else "*** MISMATCH")
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
                ref = REF_MIQP[(matrix_id, short)]
                ok = abs(obj-ref) < max(1e-2, 1e-4*abs(ref)); rec(ok)
                designs[f"MIQP {mat} {short}"] = sorted(sites)
                out["miqp"][f"{mat} {short}"] = {"objective": round(obj,4), "ref": ref,
                                                 "pass": bool(ok), "n_sites": len(sites),
                                                 "sites": sorted(sites), "total_area": area}
                print(f"  {mat} {short:11s} obj={obj:12.4f} ref={ref:12.2f} "
                      f"{'OK' if ok else '*** MISMATCH'}" + (f" area={area}" if area else ""))

    # ---------------- backbone (only if both heur+miqp ran) ----------------
    if not args.skip_heur and not args.skip_miqp:
        inter = lambda ns: sorted(set.intersection(*[set(designs[n]) for n in ns if n in designs]))
        m1c = [f"M1 {m} (c)" for m in ("Greedy","Swap","Stingy")] + [f"MIQP M1 {s}" for s,*_ in MIQP_SPECS]
        m2c = [f"M2 {m} (c)" for m in ("Greedy","Swap","Stingy")] + [f"MIQP M2 {s}" for s,*_ in MIQP_SPECS]
        i1, i2 = inter(m1c), inter(m2c); cross = sorted(set(i1)&set(i2)); glob = inter(list(designs.keys()))
        for k, got in [("M1 within",i1),("M2 within",i2),("cross",cross),("global",glob)]:
            ok = got == REF_BACKBONE[k]; rec(ok)
            out["backbone"][k] = {"sites": got, "ref": REF_BACKBONE[k], "size": len(got), "pass": ok}
            print(f"  backbone {k:10s} size={len(got):2d} {'OK' if ok else '*** MISMATCH'} {got}")

    out["summary"] = {"passed": npass, "failed": nfail, "all_pass": nfail == 0,
                      "runtime_sec": round(time.time()-t0,1)}
    (RUNS_DIR/"run_everything.json").write_text(json.dumps(out, indent=2))
    print(f"\nPASSED {npass}  FAILED {nfail}  ALL_PASS={nfail==0}  ({out['summary']['runtime_sec']}s)")
    print(f"wrote {RUNS_DIR/'run_everything.json'}")
    return 0 if nfail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())