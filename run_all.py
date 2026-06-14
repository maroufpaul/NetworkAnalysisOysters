#!/usr/bin/env python3
"""
run_all_miqp.py  --  run ALL four integer-programming models for BOTH matrices
through AMPL + Gurobi, with self-recruitment KEPT (canonical), and validate each
objective against the canonical reference.

Drop at the REPO ROOT and run:   python run_all_miqp.py

This mirrors your existing scripts (run_miqp.py / run_miqp_comm.py /
run_miqp_size.py / run_miqp_comm_size.py) exactly:
  * option solver gurobi;  (+ nonconvex=2 mipgap=1e-9 for the bilinear models)
  * read .mod, readData oyster_quad.dat (+ comm/size .dat as needed)
  * getVariable("x") / getVariable("s"); objective names score/Larvae/TotalLarvae
  * labels via runs/oyster_index_mapping_matrix{1,2}.csv

It DELIBERATELY does not catch solver errors -- if AMPL/Gurobi fails, you see the
real traceback instead of a silent skip. It regenerates oyster_quad.dat WITH the
diagonal (self-recruitment) for each matrix before solving, by calling your
prepare_miqp_data.py. Writes runs/all_miqp_results.json.
"""
import sys, json, subprocess, time
from pathlib import Path
from amplpy import AMPL
import pandas as pd

ROOT = Path(__file__).resolve().parent
AMPL_DIR = ROOT / "ampl"
RUNS_DIR = ROOT / "runs"; RUNS_DIR.mkdir(exist_ok=True)

# canonical reference (diagonal kept), objective values from the SCIP cross-check
REF_OBJ = {
 ("1","Base"):14785.03, ("1","+Comm"):13844.54, ("1","+Size"):58564.19, ("1","+Comm+Size"):56443.65,
 ("2","Base"):16446.59, ("2","+Comm"):15218.16, ("2","+Size"):66552.23, ("2","+Comm+Size"):64448.16,
}

def labels(matrix_id):
    df = pd.read_csv(RUNS_DIR / f"oyster_index_mapping_matrix{matrix_id}.csv")
    return df["site_id"].tolist()

def prepare(matrix_id):
    """Regenerate oyster_quad.dat WITH diagonal for this matrix (your script)."""
    r = subprocess.run([sys.executable, "-m", "scripts.prepare_miqp_data", "--matrix", matrix_id],
                       cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); raise RuntimeError("prepare_miqp_data failed")
    # sanity: confirm the freshly written .dat is NOT diagonal-zeroed
    dat = (AMPL_DIR / "oyster_quad.dat").read_text()
    diag_entries = sum(1 for ln in dat.splitlines()
                       if ln.strip().startswith("[") and "," in ln
                       and ln.split("[")[1].split(",")[0].strip() == ln.split(",")[1].split("]")[0].strip())
    return diag_entries

def new_ampl():
    a = AMPL()
    a.eval("option solver gurobi;")
    a.eval("option gurobi_options 'nonconvex=2 mipgap=1e-9';")
    return a

def solve(model, dats, objname, has_size, matrix_id):
    a = new_ampl()
    a.read(str(AMPL_DIR / model))
    a.readData(str(AMPL_DIR / "oyster_quad.dat"))
    for d in dats:
        a.readData(str(AMPL_DIR / d))
    a.eval("solve;")
    lab = labels(matrix_id)
    xv = a.getVariable("x").getValues().to_list()
    picked = [int(r[0]) for r in xv if float(r[1]) > 0.5]
    sites = sorted(lab[i] for i in picked)
    obj = float(a.getObjective(objname).value())
    sizes = None; area = None
    if has_size:
        sv = {int(r[0]): float(r[1]) for r in a.getVariable("s").getValues().to_list()}
        sizes = {lab[i]: round(sv.get(i, 0.0), 3) for i in picked}
        area = round(sum(sizes.values()), 2)
    return sites, obj, sizes, area

SPECS = [  # (short, model, extra_dats, objname, has_size)
  ("Base",       "oyster_quad.mod",      [],                                    "score",       False),
  ("+Comm",      "oyster_comm.mod",      ["oyster_comm.dat"],                   "Larvae",      False),
  ("+Size",      "oyster_size.mod",      ["oyster_size.dat"],                   "Larvae",      True),
  ("+Comm+Size", "oyster_comm_size.mod", ["oyster_comm.dat","oyster_size.dat"], "TotalLarvae", True),
]

def main():
    t0 = time.time()
    out = {"model": "CANONICAL (diagonal KEPT)", "runs": {}}
    npass = nfail = 0
    for matrix_id in ("1", "2"):
        diag = prepare(matrix_id)
        print(f"\n=== Matrix {matrix_id}: oyster_quad.dat regenerated, "
              f"{diag} diagonal entries present "
              f"({'OK self-recruitment kept' if diag > 0 else 'WARNING: diagonal empty!'}) ===")
        for short, model, dats, objname, has_size in SPECS:
            sites, obj, sizes, area = solve(model, dats, objname, has_size, matrix_id)
            ref = REF_OBJ[(matrix_id, short)]
            ok = abs(obj - ref) < max(1e-2, 1e-4 * abs(ref))
            npass += ok; nfail += (not ok)
            key = f"M{matrix_id} {short}"
            out["runs"][key] = {"objective": round(obj, 4), "ref": ref,
                                "match": bool(ok), "n_sites": len(sites),
                                "sites": sites, "total_area": area, "sizes": sizes}
            tag = "OK" if ok else "*** MISMATCH"
            print(f"  {key:14s} obj={obj:12.4f}  ref={ref:12.2f}  {tag}"
                  + (f"  area={area}" if area else ""))
    out["summary"] = {"passed": int(npass), "failed": int(nfail),
                      "all_pass": nfail == 0, "runtime_sec": round(time.time() - t0, 1)}
    (RUNS_DIR / "all_miqp_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nPASSED {npass}  FAILED {nfail}  ALL_PASS={nfail == 0}  ({out['summary']['runtime_sec']}s)")
    print(f"wrote {RUNS_DIR / 'all_miqp_results.json'}")
    return 0 if nfail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())