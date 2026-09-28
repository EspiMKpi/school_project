import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

nb_path = "03_keras_experiments.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb["cells"]):
    cell_type = cell.get("cell_type")
    outputs = cell.get("outputs", [])
    src = "".join(cell.get("source", []))[:80].replace("\n", " ")
    print(f"Cell {idx:02d}: type={cell_type:8s} outputs={len(outputs):4d} | src: {src}")
    for o_idx, out in enumerate(outputs):
        out_type = out.get("output_type")
        if out_type == "stream":
            text = "".join(out.get("text", []))
            lines = len(text.splitlines())
            has_bs = '\x08' in text or '\r' in text
            print(f"    Out {o_idx}: stream lines={lines}, length={len(text)}, has_control_chars={has_bs}")
        elif out_type == "display_data":
            data_keys = list(out.get("data", {}).keys())
            print(f"    Out {o_idx}: display_data keys={data_keys}")
        elif out_type == "execute_result":
            data_keys = list(out.get("data", {}).keys())
            print(f"    Out {o_idx}: execute_result keys={data_keys}")
        else:
            print(f"    Out {o_idx}: {out_type}")
