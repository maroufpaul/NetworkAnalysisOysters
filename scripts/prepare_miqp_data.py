# scripts/prepare_miqp_data.py

from pathlib import Path
import numpy as np
import pandas as pd

UNWANTED = [66, 67, 68, 69, 70, 71, 72]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AMPL_DIR = ROOT / "ampl"
RUNS_DIR = ROOT / "runs"

EXCEL_PATH = DATA_DIR / "nk_All_060102final_56sites_Model.xlsx"
OUT_DAT = AMPL_DIR / "oyster_data.dat"
OUT_CSV = RUNS_DIR / "oyster_data_preview.csv"


def main():
    # load original excel
    arr = pd.read_excel(EXCEL_PATH, header=None).values
    labels = arr[0, 1:].astype(int)
    P = arr[1:, 1:].astype(float)

    # drop labels 66–72
    mask = ~np.isin(labels, UNWANTED)
    labels = labels[mask]
    P = P[np.ix_(mask, mask)]

    # scale internal
    P1scaling = 0.5
    P = P * P1scaling

    # constant external larvae
    Pe = 170.0 * np.ones(len(labels), dtype=float)

    # no self-loops
    np.fill_diagonal(P, 0.0)

    # 1) write CSV for us to look at
    df = pd.DataFrame(P, index=labels, columns=labels)
    df.to_csv(OUT_CSV, index=True)
    print(f"[prepare] wrote {OUT_CSV}")

    # 2) write AMPL .dat in **sparse** format
    #    this is the important part
    with open(OUT_DAT, "w", encoding="utf-8") as f:
        f.write("# auto-generated MIQP data for oyster problem\n\n")

        # set of sites
        f.write("set N :=\n")
        for lab in labels:
            f.write(f"  {lab}\n")
        f.write(";\n\n")

        # how many to pick
        f.write("param k := 25;\n\n")

        # external larvae
        f.write("param Pe :=\n")
        for lab, pe_val in zip(labels, Pe):
            f.write(f"  {lab} {pe_val:.4f}\n")
        f.write(";\n\n")

        # internal matrix as sparse triples
        f.write("param P1 :=\n")
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                val = float(P[i, j])
                if val != 0.0:
                    # AMPL sparse entry: [row, col] value
                    f.write(f"  [{li}, {lj}] {val:.6f}\n")
        f.write(";\n")

    print(f"[prepare] wrote {OUT_DAT}")


if __name__ == "__main__":
    main()
