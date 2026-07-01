# Ensure the repository root is importable so `import config` works from any
# submodule, regardless of how the process was launched.
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
