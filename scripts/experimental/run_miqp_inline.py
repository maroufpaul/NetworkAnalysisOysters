# scripts/run_miqp_inline.py
"""
Run the oyster MIQP with AMPL + Gurobi,
but define the model inline so we don't depend on the .mod file on disk.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from amplpy import AMPL, DataFrame

UNWANTED = [66, 67, 68, 69, 70, 71, 72]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

EXCEL_PATH = DATA_DIR / "nk_All_060102final_56sites_Model.xlsx"

# this is the SAME model we’ve been trying to use
MODEL_TEXT = r"""
set N;
param k integer > 0;
param Pe{N} >= 0 default 0;
param P1{N, N} >= 0 default 0;

var x{N} binary;

maximize score:
    sum {i in N} Pe[i] * x[i]
  + sum {i in N, j in N} P1[i,j] * x[i] * x[j];

subject to choose_k:
    sum {i in N} x[i] = k;
"""


def main():
    print(">>> inline MIQP start")

    # 1) load Excel
    raw = pd.read_excel(EXCEL_PATH, header=None).values
    labels = raw[0, 1:].astype(int)
    P = raw[1:, 1:].astype(float)
    print(f"[inline] raw labels ({len(labels)}): {labels.tolist()}")

    # 2) drop unwanted
    mask = ~np.isin(labels, UNWANTED)
    labels = labels[mask]
    P = P[np.ix_(mask, mask)]
    print(f"[inline] after drop ({len(labels)}): {labels.tolist()}")

    # 3) scale
    P *= 0.5

    # 4) Pe
    Pe = 170.0 * np.ones(len(labels), dtype=float)

    # 5) no self loops
    np.fill_diagonal(P, 0.0)

    # 6) start AMPL + set solver
    ampl = AMPL()
    ampl.option["solver"] = "gurobi"
    ampl.option["gurobi_options"] = "outlev=1"
    print("[inline] AMPL made, solver=gurobi")

    # 7) define model inline
    ampl.eval(MODEL_TEXT)
    print("[inline] model defined inline")

    # 8) send set N
    dfN = DataFrame(1, "site")
    dfN.setValues(labels.tolist())
    ampl.setData(dfN, "N")
    print("[inline] sent N")

    # 9) k
    ampl.param["k"] = 25
    print("[inline] set k=25")

    # 10) Pe
    dfPe = DataFrame(("site", "Pe"), list(zip(labels, Pe)))
    ampl.setData(dfPe, "Pe")
    print("[inline] sent Pe")

    # 11) P1 (sparse)
    rows = []
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            val = float(P[i, j])
            if val != 0.0:
                rows.append((int(li), int(lj), val))
    dfP1 = DataFrame(("i", "j", "P1"), rows)
    ampl.setData(dfP1, "P1")
    print(f"[inline] sent P1 with {len(rows)} entries")

    # 12) solve
    print("[inline] calling solve() ...")
    ampl.solve()

    # 13) print AMPL solve info
    try:
        out = ampl.get_output("display _solve_status, _solve_message, _solve_time;")
        print("[inline] AMPL said:")
        print(out)
    except Exception as e:
        print("[inline] could not get AMPL output:", e)

    # 14) get x
    x_df = ampl.getVariable("x").getValues()
    x_dict = x_df.toDict()
    print("[inline] raw x:", x_dict)

    picked = []
    for k, v in x_dict.items():
        if float(v) > 0.5:
            if isinstance(k, tuple):
                picked.append(int(k[0]))
            else:
                picked.append(int(k))
    picked = sorted(picked)
    print("[inline] picked sites:", picked)

    # 15) save
    out_csv = RUNS_DIR / "miqp_sites.csv"
    pd.DataFrame({"site_id": picked}).to_csv(out_csv, index=False)
    print("[inline] wrote", out_csv)

    out_txt = RUNS_DIR / "miqp_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("MIQP (inline, gurobi) solution\n")
        f.write("==============================\n")
        f.write(f"sites ({len(picked)}):\n")
        for s in picked:
            f.write(f"  {s}\n")
    print("[inline] wrote", out_txt)

    print(">>> inline MIQP done")


if __name__ == "__main__":
    main()
