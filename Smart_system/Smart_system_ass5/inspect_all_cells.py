import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("03_keras_experiments.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    print(f"\n==================== CELL {idx} ({cell['cell_type']}) ====================")
    src = "".join(cell.get("source", []))
    print(src[:200] + ("..." if len(src) > 200 else ""))
    for o_idx, out in enumerate(cell.get("outputs", [])):
        otype = out.get("output_type")
        if otype == "stream":
            lines = out.get("text", [])
            print(f"  Output {o_idx} (stream): {len(lines)} chunks, total chars={sum(len(x) for x in lines)}")
            first_line = "".join(lines)[:100].replace("\n", " \\n ")
            print(f"    preview: {first_line}")
        elif otype == "display_data":
            print(f"  Output {o_idx} (display_data): {list(out.get('data', {}).keys())}")
        elif otype == "execute_result":
            print(f"  Output {o_idx} (execute_result): {list(out.get('data', {}).keys())}")
        else:
            print(f"  Output {o_idx}: {otype}")
