"""Executes the notebook's NON-TensorFlow cells (used for sandbox verification).

Cells whose first line is '# [needs-tensorflow]' or '# [needs-model]' are skipped.
Usage: DR_LOCAL_WORKDIR=/tmp/w python tools/run_notebook_cpu.py
"""
import json, os, sys, traceback, types
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
root = Path(__file__).resolve().parents[1]
nb = json.loads((root / "notebook" / "diabetic_retinopathy_image_classification.ipynb").read_text())
os.chdir(root / "notebook")
ns = {"__name__": "__main__", "_display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x)}
ran = skipped = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if src.lstrip().startswith(("# [needs-tensorflow]", "# [needs-model]")):
        skipped += 1; print(f"--- cell {i}: SKIPPED (needs TensorFlow/model)"); continue
    src = src.replace("from IPython.display import display", "display = _display")   # no IPython in the sandbox
    print(f"--- cell {i}: running")
    try:
        exec(compile(src, f"cell_{i}", "exec"), ns); ran += 1
    except Exception:
        traceback.print_exc(); print(f"!!! cell {i} FAILED"); sys.exit(1)
print(f"\nOK: {ran} code cells executed, {skipped} skipped")
