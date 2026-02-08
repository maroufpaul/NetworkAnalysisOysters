# scripts/run_miqp_comm.py

import argparse
from pathlib import Path
from amplpy import AMPL
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AMPL_DIR = ROOT / "ampl"
RUNS_DIR = ROOT / "runs"


def load_labels_from_mapping(matrix_id: str):
    mapping_path = RUNS_DIR / f"oyster_index_mapping_matrix{matrix_id}.csv"
    df = pd.read_csv(mapping_path)
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

    ampl = AMPL()
    ampl.eval("option solver gurobi;")

    ampl.read(str(AMPL_DIR / "oyster_comm.mod"))      # N, C1..C5
    ampl.readData(str(AMPL_DIR / "oyster_quad.dat"))  # N, K, Pe, W
    ampl.readData(str(AMPL_DIR / "oyster_comm.dat"))  # C1..C5 sets

    ampl.eval("solve;")

    x_vals = ampl.getVariable("x").getValues()
    picked_idx = [int(row[0]) for row in x_vals.to_list() if float(row[1]) > 0.5]

    labels = load_labels_from_mapping(matrix_id)
    rows = [{"site_id": labels[i]} for i in picked_idx]

    out_csv = RUNS_DIR / f"miqp_comm_sites_matrix{matrix_id}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_txt = RUNS_DIR / f"miqp_comm_summary_matrix{matrix_id}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("MIQP with community minimums (no sizing)\n")
        f.write(f"matrix version: {matrix_id}\n")
        f.write(f"picked {len(picked_idx)} sites\n")
        for i in picked_idx:
            f.write(f"  label {labels[i]:>3}\n")

    print("[comm] matrix", matrix_id)
    print("[comm] picked:", [labels[i] for i in picked_idx])
    print("[comm] wrote:", out_csv)
    print("[comm] wrote:", out_txt)


if __name__ == "__main__":
    main()
