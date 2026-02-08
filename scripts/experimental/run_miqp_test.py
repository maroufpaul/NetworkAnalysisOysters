# scripts/run_miqp_test.py
"""
Very loud MIQP runner.
- loads Excel
- drops 66-72
- scales by 0.5
- sends data to AMPL
- solves with GUROBI
- prints everything
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from amplpy import AMPL, DataFrame

UNWANTED = [66, 67, 68, 69, 70, 71, 72]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AMPL_DIR = ROOT / "ampl"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

EXCEL_PATH = DATA_DIR / "nk_All_060102final_56sites_Model.xlsx"
MODEL_PATH = AMPL_DIR / "oyster_quad.mod"


def main():
    print(">>> [run_miqp] start", flush=True)
    print(f">>> [run_miqp] Excel: {EXCEL_PATH}", flush=True)
    print(f">>> [run_miqp] Model: {MODEL_PATH}", flush=True)

    # 1) read excel
    raw = pd.read_excel(EXCEL_PATH, header=None).values
    labels = raw[0, 1:].astype(int)
    P = raw[1:, 1:].astype(float)
    print(f">>> [run_miqp] raw labels ({len(labels)}): {labels.tolist()}", flush=True)

    # 2) drop unwanted
    mask = ~np.isin(labels, UNWANTED)
    labels = labels[mask]
    P = P[np.ix_(mask, mask)]
    print(f">>> [run_miqp] after drop ({len(labels)}): {labels.tolist()}", flush=True)

    # 3) scale
    P1scaling = 0.5
    P = P * P1scaling
    print(">>> [run_miqp] scaled P by 0.5", flush=True)

    # 4) external Pe
    Pe = 170.0 * np.ones(len(labels), dtype=float)
    print(">>> [run_miqp] built Pe (all 170)", flush=True)

    # 5) zero diag
    np.fill_diagonal(P, 0.0)
    print(">>> [run_miqp] zeroed diagonal", flush=True)

    # 6) AMPL
    ampl = AMPL()
    ampl.option["solver"] = "gurobi"  #  want gurobi
    # show gurobi output too
    ampl.option["gurobi_options"] = "outlev=1"
    print(">>> [run_miqp] AMPL created, solver=gurobi", flush=True)

    # 7) read model
    ampl.read(str(MODEL_PATH))
    print(">>> [run_miqp] model read", flush=True)

    # 8) send set N
    dfN = DataFrame(1, "site")
    dfN.setValues(labels.tolist())
    ampl.setData(dfN, "N")
    print(">>> [run_miqp] sent set N", flush=True)

    # 9) k
    ampl.param["k"] = 25
    print(">>> [run_miqp] set k=25", flush=True)

    # 10) Pe
    dfPe = DataFrame(("site", "Pe"), list(zip(labels, Pe)))
    ampl.setData(dfPe, "Pe")
    print(">>> [run_miqp] sent Pe", flush=True)

    # 11) P1 sparse
    rows = []
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            v = float(P[i, j])
            if v != 0.0:
                rows.append((int(li), int(lj), v))
    dfP1 = DataFrame(("i", "j", "P1"), rows)
    ampl.setData(dfP1, "P1")
    print(f">>> [run_miqp] sent P1 with {len(rows)} nonzeros", flush=True)

    # 12) solve
    print(">>> [run_miqp] calling solve() ...", flush=True)
    ampl.solve()

    # 13) print AMPL output
    # ampl.get_output() in this version needs a command
    print(">>> [run_miqp] AMPL solve info:", flush=True)
    try:
        txt = ampl.get_output("display _solve_status, _solve_message, _solve_time;")
        print(txt, flush=True)
    except Exception as e:
        print(">>> [run_miqp] could not get AMPL output:", e, flush=True)

    # 14) get x
    x_df = ampl.getVariable("x").getValues()
    x_dict = x_df.toDict()
    print(">>> [run_miqp] raw x dict:", x_dict, flush=True)

    picked = []
    for k, v in x_dict.items():
        if float(v) > 0.5:
            if isinstance(k, tuple):
                picked.append(int(k[0]))
            else:
                picked.append(int(k))
    picked = sorted(picked)
    print(">>> [run_miqp] picked sites:", picked, flush=True)

    # 15) save
    out_csv = RUNS_DIR / "miqp_sites.csv"
    pd.DataFrame({"site_id": picked}).to_csv(out_csv, index=False)
    print(f">>> [run_miqp] wrote {out_csv}", flush=True)

    out_txt = RUNS_DIR / "miqp_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("MIQP (AMPL, gurobi) solution\n")
        f.write("============================\n")
        f.write(f"sites ({len(picked)}):\n")
        for s in picked:
            f.write(f"  {s}\n")
    print(f">>> [run_miqp] wrote {out_txt}", flush=True)

    print(">>> [run_miqp] done", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(">>> [run_miqp] ERROR:", e, flush=True)
        traceback.print_exc()
        sys.exit(1)
