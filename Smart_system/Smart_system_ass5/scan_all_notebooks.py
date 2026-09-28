import json
import glob
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for nb_file in sorted(glob.glob("*.ipynb")):
    with open(nb_file, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\nScanning {nb_file}:")
    for idx, cell in enumerate(nb["cells"]):
        outputs = cell.get("outputs", [])
        ctrl_chars = 0
        total_len = 0
        for out in outputs:
            text = "".join(out.get("text", []))
            total_len += len(text)
            ctrl_chars += text.count('\x08') + text.count('\r')
        if ctrl_chars > 0 or len(outputs) > 20:
            print(f"  Cell {idx}: outputs={len(outputs)}, total_chars={total_len}, ctrl_chars={ctrl_chars}")
    print("Done scanning", nb_file)
