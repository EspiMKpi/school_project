import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("03_keras_experiments.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell_idx in [5, 8]:
    cell = nb["cells"][cell_idx]
    print(f"\n==================== CELL {cell_idx} SOURCE ====================")
    print("".join(cell.get("source", [])))
    print(f"==================== CELL {cell_idx} OUTPUT TYPES ====================")
    out_types = {}
    for o in cell.get("outputs", []):
        t = o.get("output_type")
        out_types[t] = out_types.get(t, 0) + 1
    print("Output types:", out_types)
