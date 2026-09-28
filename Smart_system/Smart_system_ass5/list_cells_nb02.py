import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("02_data_preparation_and_augmentation.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))[:120].replace("\n", " ")
        print(f"Cell {idx:02d} (code): {src}")
    else:
        src = "".join(cell.get("source", []))[:120].replace("\n", " ")
        print(f"Cell {idx:02d} (md)  : {src}")
