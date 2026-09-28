import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for dataset in ["01_fashion_mnist", "02_svhn", "03_diabetes"]:
    dpath = os.path.join("models_keras", dataset)
    print(f"\n==================== {dataset} ====================")
    eval_path = os.path.join(dpath, "evaluation_metrics.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            print("Evaluation metrics:", json.dumps(json.load(f), indent=2))
    
    for hfile in ["history_3layer.json", "history_5layer.json"]:
        hpath = os.path.join(dpath, hfile)
        if os.path.exists(hpath):
            with open(hpath, "r", encoding="utf-8") as f:
                h = json.load(f)
                print(f"{hfile} keys: {list(h.keys())}, epochs: {len(h.get('loss', []))}")
                for k in h.keys():
                    vals = h[k]
                    print(f"  {k}: first={vals[0]:.4f}, last={vals[-1]:.4f}")
