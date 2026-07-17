# scripts/run_miqp.py
#
# All four MIQP variants. Replaces run_miqp.py + run_miqp_comm.py +
# run_miqp_size.py + run_miqp_comm_size.py, which were the same 60 lines four
# times, none of which set gurobi_options.
#
#     python -m scripts.run_miqp                          # all models, both matrices
#     python -m scripts.run_miqp --matrix 1 --model base
#     python -m scripts.run_miqp --matrix 2 --model comm+size
#     python -m scripts.run_miqp --selfcheck              # verify vs paper Table 4
#
# Models come from config.MIQP_MODELS. AMPL call lives in src/opt/miqp.py.
# Results -> runs/miqp_results.csv + one summary .txt per model/matrix.

import argparse
import time
from collections import defaultdict

import pandas as pd

import config
from src.opt.miqp import solve, selfcheck

# paper Table 4, for the OK / MISMATCH column
REF = {("base", "1"): 14785.03, ("comm", "1"): 13863.50,
       ("size", "1"): 32620.49, ("comm+size", "1"): 30517.81,
       ("base", "2"): 16446.59, ("comm", "2"): 15294.18,
       ("size", "2"): 36695.67, ("comm+size", "2"): 34226.75}


def run_one(model, matrix_id):
    mid = config.matrix_num(matrix_id)
    t0 = time.time()
    r = solve(model, mid)
    r["seconds"] = round(time.time() - t0, 3)

    ref = REF.get((model, mid))
    r["ref"] = ref
    r["pass"] = None if ref is None else abs(r["obj"] - ref) < max(1e-2, 1e-4 * abs(ref))
    flag = "OK" if r["pass"] else ("-- (no ref)" if r["pass"] is None else "*** MISMATCH")

    print(f"  M{mid} {model:10s} obj={r['obj']:12.2f}  paper={ref}  {flag}  ({r['seconds']}s)")
    print(f"     sites: {r['sites']}")
    if r["sizes"]:
        grp = defaultdict(list)
        for lab, a in r["sizes"].items():
            grp[a].append(lab)
        print(f"     area {r['total_area']} of {config.SIZE['TotReefSize']} acres -> "
              + " | ".join(f"{a:g}ac: {sorted(v)}" for a, v in sorted(grp.items(), reverse=True)))

    config.RUNS_DIR.mkdir(exist_ok=True)
    out = config.RUNS_DIR / f"miqp_{model.replace('+', '_')}_matrix{mid}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"model: {model}  ({config.MIQP_MODELS[model][0]})\n")
        f.write(f"matrix: M{mid}   data: {r['dat']}\n")
        f.write(f"gurobi_options: {config.GUROBI_OPTIONS}\n")
        f.write(f"objective: {r['obj']:.6f}   paper: {ref}   {flag}\n")
        f.write(f"time: {r['seconds']}s\npicked {len(r['sites'])} sites\n")
        for s in r["sites"]:
            f.write(f"  {s:>3}" + (f"  {r['sizes'][s]:8.3f} ac\n" if r["sizes"] else "\n"))
        if r["total_area"] is not None:
            f.write(f"total area: {r['total_area']} of {config.SIZE['TotReefSize']}\n")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", choices=["1", "2", "both"], default="both")
    ap.add_argument("--model", choices=list(config.MIQP_MODELS) + ["all"], default="all")
    ap.add_argument("--selfcheck", action="store_true",
                    help="only verify the constant-Pe baseline against Table 4")
    args = ap.parse_args()

    mats = ["1", "2"] if args.matrix == "both" else [args.matrix]
    for m in mats:
        if not config.quad_dat(m).exists():
            raise SystemExit(f"{config.quad_dat(m)} missing. Run:\n"
                             f"    python -m scripts.prepare_data --matrix {m}")

    if args.selfcheck:
        return 0 if all(selfcheck(m) for m in mats) else 1

    models = list(config.MIQP_MODELS) if args.model == "all" else [args.model]
    rows, npass, nfail = [], 0, 0
    for m in mats:
        print(f"\n===== MIQP matrix {m}  ({config.quad_dat(m).name}) =====")
        for model in models:
            r = run_one(model, m)
            if r["pass"] is True:
                npass += 1
            elif r["pass"] is False:
                nfail += 1
            rows.append({"matrix": r["matrix"], "model": model,
                         "objective": round(r["obj"], 4), "paper_ref": r["ref"],
                         "pass": r["pass"], "n_sites": len(r["sites"]),
                         "total_area": r["total_area"], "seconds": r["seconds"],
                         "sites": " ".join(map(str, r["sites"]))})

    out = config.RUNS_DIR / "miqp_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nPASSED {npass}  FAILED {nfail}  ALL_PASS={nfail == 0}")
    print(f"wrote {out}")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())