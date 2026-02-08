# scripts/prepare_miqp_data.py

from pathlib import Path
import numpy as np
import pandas as pd
import argparse

# Site labels to drop (same as before)
UNWANTED = [66, 67, 68, 69, 70, 71, 72]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
AMPL_DIR = ROOT / "ampl"
RUNS_DIR = ROOT / "runs"

# Available connectivity matrices
MATRIX_FILES = {
    "1": "nk_All_060102final_56sites_Model.xlsx",
    "2": "nk_All_060103final_56sites_Model.xlsx",
}

# Only this file will be regenerated
OUT_DAT_QUAD = AMPL_DIR / "oyster_quad.dat"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        choices=MATRIX_FILES.keys(),
        default="1",
        help="Which connectivity matrix to use (1 or 2)",
    )
    args = parser.parse_args()

    xlsx_name = MATRIX_FILES[args.matrix]
    excel_path = DATA_DIR / xlsx_name

    RUNS_DIR.mkdir(exist_ok=True)
    AMPL_DIR.mkdir(exist_ok=True)

    print(f"[prepare] Using matrix {args.matrix}: {xlsx_name}")

    # ------------------------------------------------------------------
    # 1. Load Excel and drop unwanted labels
    # ------------------------------------------------------------------
    arr = pd.read_excel(excel_path, header=None).values
    labels = arr[0, 1:].astype(int)         # first row (except first col)
    P_full = arr[1:, 1:].astype(float)      # main matrix

    # Drop sites 66–72
    mask = ~np.isin(labels, UNWANTED)
    labels = labels[mask]
    P = P_full[np.ix_(mask, mask)]
    n = len(labels)
    print(f"[prepare] kept {n} sites after dropping {UNWANTED}")

    # ------------------------------------------------------------------
    # 2. Build surrogate weights W
    # ------------------------------------------------------------------
    # scale internal connectivity
    P1scaling = 0.5
    P = P * P1scaling

    # remove self-loops (no reward for staying on same reef)
    #np.fill_diagonal(P, 0.0)

    # constant external larvae
    Pe = 170.0 * np.ones(n, dtype=float)

    # surrogate step: fold in median A* via |A*|^alpha
    A_STAR = 0.05675   # median equilibrium adults for a single reef
    ALPHA = 1.72
    W = P * (A_STAR ** ALPHA)

    # ------------------------------------------------------------------
    # 3. Preview CSVs (for sanity checks)
    # ------------------------------------------------------------------
    preview_path = RUNS_DIR / f"oyster_data_preview_matrix{args.matrix}.csv"
    pd.DataFrame(W, index=labels, columns=labels).to_csv(preview_path)
    print(f"[prepare] wrote {preview_path}")

    mapping_path = RUNS_DIR / f"oyster_index_mapping_matrix{args.matrix}.csv"
    pd.DataFrame({"index": np.arange(n, dtype=int), "site_id": labels}).to_csv(
        mapping_path, index=False
    )
    print(f"[prepare] wrote {mapping_path}")

    # ------------------------------------------------------------------
    # 4. Write AMPL data (ONLY oyster_quad.dat)
    #    Indices are 0..n-1; labels are stored separately in CSV mapping.
    # ------------------------------------------------------------------
    with open(OUT_DAT_QUAD, "w", encoding="utf-8") as f:
        f.write("# auto-generated MIQP data for oyster problem\n")
        f.write(f"# matrix version: {args.matrix}\n\n")

        # set of sites
        f.write("set N :=\n")
        for i in range(n):
            f.write(f"  {i}\n")
        f.write(";\n\n")

        # how many to pick
        f.write("param K := 25;\n\n")

        # external larvae Pe[i]
        f.write("param Pe :=\n")
        for i in range(n):
            f.write(f"  {i} {Pe[i]:.4f}\n")
        f.write(";\n\n")

        # internal surrogate weights W[i,j] in sparse format
        f.write("param W :=\n")
        for i in range(n):
            for j in range(n):
                val = float(W[i, j])
                if val != 0.0:
                    f.write(f"  [{i}, {j}] {val:.6f}\n")
        f.write(";\n")

    print(f"[prepare] wrote {OUT_DAT_QUAD}")
    print("[prepare] done.")
    print("NOTE: oyster_size.dat and oyster_comm.dat were NOT modified.")


if __name__ == "__main__":
    main()
