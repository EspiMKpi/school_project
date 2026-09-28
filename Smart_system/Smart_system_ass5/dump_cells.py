import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("03_keras_experiments.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(len(nb["cells"])):
    cell = nb["cells"][idx]
    src = "".join(cell.get("source", []))
    print(f"\n--- CELL {idx} ({cell['cell_type']}) ---")
    print(src)
