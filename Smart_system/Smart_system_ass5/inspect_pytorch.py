import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("04_pytorch_experiments.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    outputs = cell.get("outputs", [])
    src = "".join(cell.get("source", []))[:60].replace("\n", " ")
    print(f"Cell {idx:02d}: {cell.get('cell_type')} | {len(outputs)} outputs | src: {src}")
    for o in outputs:
        if o.get("output_type") == "stream":
            text = "".join(o.get("text", []))[:200].replace("\n", " \\n ")
            print(f"    stream: {text}")
