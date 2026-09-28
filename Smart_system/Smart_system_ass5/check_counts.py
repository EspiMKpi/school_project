import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("03_keras_experiments.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    outputs = cell.get("outputs", [])
    src = "".join(cell.get("source", []))[:60].replace("\n", " ")
    print(f"Cell {idx:02d}: {cell.get('cell_type')} | {len(outputs)} outputs | src: {src}")
