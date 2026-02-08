# scripts/run_miqp_size.py

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

    ampl.read(str(AMPL_DIR / "oyster_size.mod"))      # N, sizing model
    ampl.readData(str(AMPL_DIR / "oyster_quad.dat"))  # N, K, Pe, W
    ampl.readData(str(AMPL_DIR / "oyster_size.dat"))  # L, U, TotReefSize, Sbar

    ampl.eval("solve;")

    # === NEW: grab objective value (Larvae) ===
    obj = ampl.getObjective("Larvae").value()

    x_vals = ampl.getVariable("x").getValues()
    s_vals = ampl.getVariable("s").getValues()

    picked_idx = [int(row[0]) for row in x_vals.to_list() if float(row[1]) > 0.5]
    size_map = {int(row[0]): float(row[1]) for row in s_vals.to_list()}

    labels = load_labels_from_mapping(matrix_id)

    rows = [{"site_id": labels[i], "size": size_map.get(i, 0.0)} for i in picked_idx]

    out_csv = RUNS_DIR / f"miqp_size_sites_matrix{matrix_id}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_txt = RUNS_DIR / f"miqp_size_summary_matrix{matrix_id}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("MIQP with variable reef sizing\n")
        f.write(f"matrix version: {matrix_id}\n")
        f.write(f"objective (Larvae) = {obj:.6f}\n")   # <-- NEW LINE
        f.write(f"picked {len(picked_idx)} sites (x=1)\n")
        for i in picked_idx:
            f.write(
                f"  label {labels[i]:>3}  size={size_map.get(i, 0.0):8.3f}\n"
            )

    print("[size] matrix", matrix_id)
    print("[size] objective (Larvae):", obj)           # <-- NEW PRINT
    print("[size] picked:", [labels[i] for i in picked_idx])
    print("[size] wrote:", out_csv)
    print("[size] wrote:", out_txt)


if __name__ == "__main__":
    main()
