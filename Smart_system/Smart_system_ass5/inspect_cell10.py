import json

with open("03_keras_experiments.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell10 = nb["cells"][10]
for o in cell10.get("outputs", []):
    data = o.get("data", {})
    if "text/plain" in data:
        print("text/plain:\n", "".join(data["text/plain"]))
