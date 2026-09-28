import json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("02_data_preparation_and_augmentation.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [9, 10, 11]:
    cell = nb["cells"][idx]
    print(f"\n--- Cell {idx} ({cell['cell_type']}) ---")
    print("".join(cell.get("source", [])))
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            print("STREAM OUTPUT:", "".join(out.get("text", [])))
