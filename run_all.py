import subprocess
from pathlib import Path
import sys

PYTHON = sys.executable

MODULES = [
    "scripts.run_greedy_then_local",
    "scripts.run_greedy",
    "scripts.run_backward",
    # add others back in when you’re ready
]

def main():
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")
    print(f"Using Python: {PYTHON}")

    processes = []

    for mod in MODULES:
        print(f"🚀 Starting {mod}")
        p = subprocess.Popen(
            [PYTHON, "-m", mod],
            cwd=project_root,
        )
        processes.append((mod, p))

    # Just wait for them to finish (no captured logs)
    for mod, p in processes:
        p.wait()
        print(f"\n===== {mod} finished with exit code {p.returncode} =====")

if __name__ == "__main__":
    main()
