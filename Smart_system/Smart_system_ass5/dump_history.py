import json

def show_history(dataset, name):
    path = f"models_keras/{dataset}/{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"--- {dataset} {name} ---")
    n = len(data["loss"])
    for i in range(n):
        loss = data["loss"][i]
        acc = data["accuracy"][i]
        vloss = data["val_loss"][i]
        vacc = data["val_accuracy"][i]
        print(f"Epoch {i+1:2d}/{n}: loss={loss:.4f}, acc={acc*100:.2f}% | val_loss={vloss:.4f}, val_acc={vacc*100:.2f}%")

show_history("01_fashion_mnist", "history_3layer")
show_history("01_fashion_mnist", "history_5layer")
show_history("02_svhn", "history_3layer")
show_history("02_svhn", "history_5layer")
