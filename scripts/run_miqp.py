# scripts/run_miqp.py

import time
import argparse
from pathlib import Path
from amplpy import AMPL
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AMPL_DIR = ROOT / "ampl"
RUNS_DIR = ROOT / "runs"


def load_labels_from_mapping(matrix_id: str):
    """
    Read index -> site_id mapping written by prepare_miqp_data.py
    """
    mapping_path = RUNS_DIR / f"oyster_index_mapping_matrix{matrix_id}.csv"
    df = pd.read_csv(mapping_path)
    # index column is 0..n-1 in order; site_id is the real label
    return df["site_id"].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        choices=["1", "2"],
        default="1",
        help="Which connectivity matrix version to interpret indices with",
    )
    args = parser.parse_args()
    matrix_id = args.matrix

    RUNS_DIR.mkdir(exist_ok=True)

    start_time = time.time()

    ampl = AMPL()
    ampl.eval("option solver gurobi;")

    # Uses N, K, Pe, W from oyster_quad.dat
    ampl.read(str(AMPL_DIR / "oyster_quad.mod"))
    ampl.readData(str(AMPL_DIR / "oyster_quad.dat"))

    ampl.eval("solve;")

    x_vals = ampl.getVariable("x").getValues()
    picked_idx = [int(row[0]) for row in x_vals.to_list() if float(row[1]) > 0.5]

    labels = load_labels_from_mapping(matrix_id)
    rows = [{"site_id": labels[i]} for i in picked_idx]

    elapsed = time.time() - start_time

    out_csv = RUNS_DIR / f"miqp_sites_matrix{matrix_id}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_txt = RUNS_DIR / f"miqp_summary_matrix{matrix_id}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Plain MIQP (no communities, no sizing)\n")
        f.write(f"matrix version: {matrix_id}\n")
        f.write(f"picked {len(picked_idx)} sites\n")
        f.write(f"time taken: {elapsed:.4f} secs\n")
        for i in picked_idx:
            f.write(f"  label {labels[i]:>3}\n")

    print("[miqp] matrix", matrix_id)
    print("[miqp] picked:", [labels[i] for i in picked_idx])
    print("[miqp] wrote:", out_csv)
    print("[miqp] wrote:", out_txt)
    print(f"[miqp] time taken: {elapsed:.4f} secs")


if __name__ == "__main__":
    main()
