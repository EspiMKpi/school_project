"""Sinh các notebook .ipynb cho dự án Assignment 03 dựa trên mã nguồn đã trích xuất từ báo cáo."""

from __future__ import annotations

import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)


def create_nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "language_info": {"name": "python", "version": "3.11"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }
    return nb


def build_nb04():
    cells = [
        nbf.v4.new_markdown_cell("""# Ứng dụng 4: Deep Learning từ đầu với NumPy
Notebook này dựng lại toàn bộ một mạng nơ-ron nhiều lớp **chỉ bằng NumPy**: lan truyền xuôi, hàm mất mát, lan truyền ngược và gradient descent đều viết tay. Không dùng TensorFlow, PyTorch, Keras hay scikit-learn cho phần học máy.

Bốn mô hình được huấn luyện trên cùng một tập dữ liệu và cùng một quy trình, chỉ khác nhau ở kiến trúc:
| Mã | Kiến trúc | Kích hoạt lớp ẩn | Ý nghĩa |
|---|---|---|---|
| M1 | 8 - 16 - 8 - 1 | ReLU | Kiến trúc gốc của bài giảng |
| M2 | 8 - 32 - 16 - 1 | ReLU | Rộng gấp đôi |
| M3 | 8 - 16 - 1 | ReLU | Bỏ bớt một lớp ẩn |
| M4 | 8 - 16 - 8 - 1 | tuyến tính | Bỏ ReLU, kiểm chứng vai trò của phi tuyến |"""),
        nbf.v4.new_code_cell("""import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

print("thư mục gốc :", ROOT)
print("numpy :", np.__version__)
print("pandas :", pd.__version__)"""),
        nbf.v4.new_markdown_cell("""## 1. Nạp dữ liệu và chọn 8 đặc trưng
Bộ BRFSS 2015 ở đây có 21 biến nên cần chọn ra 8 theo trị tuyệt đối hệ số tương quan với nhãn nhị phân."""),
        nbf.v4.new_code_cell("""DIABETES_CSV = DATA / "diabetes" / "diabetes.csv"
TARGET = "Diabetes_012"
SPLIT_SEED = 42
TEST_SHARE = 0.20

raw = pd.read_csv(DIABETES_CSV)
frame = raw.drop_duplicates().reset_index(drop=True)
y_all = (frame[TARGET] > 0).astype(int)

ranking = (
    frame.drop(columns=[TARGET])
    .apply(lambda s: np.corrcoef(s, y_all)[0, 1])
    .abs()
    .sort_values(ascending=False)
)

FEATURES = tuple(ranking.index[:8])

print(f"Dòng thô : {len(raw):,}")
print(f"Sau khi bỏ trùng : {len(frame):,} (bỏ {len(raw) - len(frame):,})")
print(f"Tỉ lệ lớp dương : {y_all.mean():.4f}")
print("\\nXếp hạng |tương quan| với nhãn nhị phân:")
print(ranking.round(4).to_string())
print("\\n8 đặc trưng được chọn:", list(FEATURES))"""),
        nbf.v4.new_code_cell("""X_all = frame[list(FEATURES)].to_numpy(dtype=np.float64)
y_all_arr = y_all.to_numpy(dtype=np.float64).reshape(-1, 1)

rng_split = np.random.default_rng(SPLIT_SEED)
order = rng_split.permutation(len(X_all))
cut = int((1.0 - TEST_SHARE) * len(X_all))
train_idx, test_idx = order[:cut], order[cut:]

X_train_raw, X_test_raw = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all_arr[train_idx], y_all_arr[test_idx]

mean = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0) + 1e-8
X_train = (X_train_raw - mean) / std
X_test = (X_test_raw - mean) / std

print(f"train {len(X_train):,} dòng test {len(X_test):,} dòng")
print(f"tỉ lệ dương train {y_train.mean():.4f} test {y_test.mean():.4f}")
print("\\nChuẩn hoá bằng mean/std của tập train, áp lại cho test:")
for name, m, s in zip(FEATURES, mean, std):
    print(f" {name:22s} mean {m:8.4f} std {s:8.4f}")
print("\\nSau chuẩn hoá, X_train mean ~ 0 và std ~ 1:")
print(" mean:", np.round(X_train.mean(axis=0), 6))
print(" std :", np.round(X_train.std(axis=0), 6))"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))

counts = frame[TARGET].value_counts().sort_index()
axes[0].bar([str(int(k)) for k in counts.index], counts.values,
            color="white", edgecolor="black", hatch="///")
axes[0].set_title("Ba lớp gốc")
axes[0].set_xlabel("Diabetes_012")
axes[0].set_ylabel("Số dòng")

binary_counts = y_all.value_counts().sort_index()
axes[1].bar(["0 (âm)", "1 (dương)"], binary_counts.values, color="white", edgecolor="black")
axes[1].set_title(f"Nhãn nhị phân, tỉ lệ dương {y_all.mean():.3f}")
axes[1].set_ylabel("Số dòng")

pos = ranking.iloc[:8][::-1]
axes[2].barh(range(len(pos)), pos.values, color="white", edgecolor="black")
axes[2].set_yticks(range(len(pos)))
axes[2].set_yticklabels(pos.index, fontsize=7)
axes[2].set_title("|tương quan| với nhãn")

for axis in axes:
    axis.grid(True, axis="y", linewidth=0.3, alpha=0.4)
fig.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 2. Mạng nơ-ron viết tay thuần NumPy"""),
        nbf.v4.new_code_cell("""def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def identity(x):
    return x


def identity_derivative(x):
    return np.ones_like(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


ACTIVATIONS = {"relu": (relu, relu_derivative), "linear": (identity, identity_derivative)}


def init_params(sizes, seed):
    rng = np.random.default_rng(seed)
    return tuple(
        (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in),
         np.zeros((1, fan_out)))
        for fan_in, fan_out in zip(sizes, sizes[1:])
    )


def forward(params, X, hidden_activation):
    act, _ = ACTIVATIONS[hidden_activation]
    pre, activations = [], [X]
    h = X
    last = len(params) - 1
    for index, (W, b) in enumerate(params):
        z = h @ W + b
        h = sigmoid(z) if index == last else act(z)
        pre.append(z)
        activations.append(h)
    return h, {"z": pre, "h": activations}


def binary_cross_entropy(y, y_hat):
    p = np.clip(y_hat, 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def backward(params, y, cache, hidden_activation):
    _, act_derivative = ACTIVATIONS[hidden_activation]
    z, h = cache["z"], cache["h"]
    grads = []
    delta = (h[-1] - y) / len(y)
    for index in range(len(params) - 1, -1, -1):
        grads.append((h[index].T @ delta, np.sum(delta, axis=0, keepdims=True)))
        if index:
            delta = (delta @ params[index][0].T) * act_derivative(z[index - 1])
    return tuple(reversed(grads))


def update(params, grads, learning_rate):
    return tuple(
        (W - learning_rate * dW, b - learning_rate * db)
        for (W, b), (dW, db) in zip(params, grads)
    )


demo = init_params((8, 16, 8, 1), 42)
demo_out, demo_cache = forward(demo, X_train[:5], "relu")
print("Kiểm tra kích thước ma trận trên 5 dòng đầu:")
for index, (W, b) in enumerate(demo, start=1):
    print(f" W{index} {W.shape} b{index} {b.shape}")
for index, h in enumerate(demo_cache["h"][1:-1], start=1):
    print(f" H{index}.shape = {h.shape}")
print(" y_hat.shape =", demo_out.shape)"""),
        nbf.v4.new_markdown_cell("""## 3. Chỉ số đánh giá"""),
        nbf.v4.new_code_cell("""EPS = 1e-8


def confusion(y_true, y_pred):
    positive_true = y_true == 1
    positive_pred = y_pred == 1
    return {
        "tp": int(np.sum(positive_pred & positive_true)),
        "tn": int(np.sum(~positive_pred & ~positive_true)),
        "fp": int(np.sum(positive_pred & ~positive_true)),
        "fn": int(np.sum(~positive_pred & positive_true)),
    }


def score(y_true, y_prob, threshold=0.5):
    counts = confusion(y_true, (y_prob >= threshold).astype(int))
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + EPS),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + EPS),
        "balanced_accuracy": 0.5 * (recall + tn / (tn + fp + EPS)),
        **counts,
    }


def roc_auc(y_true, y_prob):
    scores = y_prob.ravel()
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    ordered = scores[order]
    start = 0
    for end in range(1, len(ordered) + 1):
        if end == len(ordered) or ordered[end] != ordered[start]:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    labels = y_true.ravel()
    n_pos = float(np.sum(labels == 1))
    n_neg = float(len(labels) - n_pos)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


print("Kiểm tra nhanh trên nhãn dựng sẵn:")
probe_y = np.array([[1], [0], [1], [0]], dtype=float)
probe_p = np.array([[0.9], [0.2], [0.4], [0.1]], dtype=float)
print(" ", {k: round(v, 4) if isinstance(v, float) else v
            for k, v in score(probe_y, probe_p).items()})
print(" ROC-AUC =", round(roc_auc(probe_y, probe_p), 4), "(kỳ vọng 0.75)")"""),
        nbf.v4.new_markdown_cell("""## 4. Huấn luyện bốn mô hình"""),
        nbf.v4.new_code_cell("""LEARNING_RATE = 0.05
EPOCHS = 60
BATCH_SIZE = 512
INIT_SEED = 42
SHUFFLE_SEED = 7
THRESHOLDS = (0.3, 0.5, 0.7)

ARCHITECTURES = (
    {"key": "M1", "label": "Baseline", "sizes": (8, 16, 8, 1), "activation": "relu"},
    {"key": "M2", "label": "Wider", "sizes": (8, 32, 16, 1), "activation": "relu"},
    {"key": "M3", "label": "Shallow", "sizes": (8, 16, 1), "activation": "relu"},
    {"key": "M4", "label": "No ReLU", "sizes": (8, 16, 8, 1), "activation": "linear"},
)


def n_parameters(sizes):
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def fit(arch, X_tr, y_tr, X_te, y_te, epochs, learning_rate, batch_size):
    params = init_params(arch["sizes"], INIT_SEED)
    rng = np.random.default_rng(SHUFFLE_SEED)
    n = len(X_tr)
    step = batch_size or n
    history = []
    started = time.perf_counter()

    for _ in range(epochs):
        order = rng.permutation(n) if batch_size else np.arange(n)
        epoch_loss = None
        for begin in range(0, n, step):
            rows = order[begin:begin + step]
            y_hat, cache = forward(params, X_tr[rows], arch["activation"])
            if batch_size is None:
                epoch_loss = binary_cross_entropy(y_tr, y_hat)
            grads = backward(params, y_tr[rows], cache, arch["activation"])
            params = update(params, grads, learning_rate)
        if batch_size is None:
            history.append(epoch_loss)
        else:
            probe, _ = forward(params, X_tr, arch["activation"])
            history.append(binary_cross_entropy(y_tr, probe))

    elapsed = time.perf_counter() - started
    train_prob, cache_tr = forward(params, X_tr, arch["activation"])
    test_prob, _ = forward(params, X_te, arch["activation"])

    def rounded(row):
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}

    return params, {
        "key": arch["key"],
        "label": arch["label"],
        "shape": " - ".join(str(s) for s in arch["sizes"]),
        "hidden_activation": arch["activation"],
        "n_params": n_parameters(arch["sizes"]),
        "epochs": epochs,
        "batch_size": step,
        "updates": epochs * int(np.ceil(n / step)),
        "learning_rate": learning_rate,
        "seconds": round(elapsed, 2),
        "loss_first": round(history[0], 6),
        "loss_final": round(history[-1], 6),
        "loss_test_end": round(binary_cross_entropy(y_te, test_prob), 6),
        "loss_history": [round(v, 6) for v in history],
        "roc_auc_test": round(roc_auc(y_te, test_prob), 6),
        "train": rounded(score(y_tr, train_prob, 0.5)),
        "test_by_threshold": {str(t): rounded(score(y_te, test_prob, t)) for t in THRESHOLDS},
        "hidden_shapes": [list(h.shape) for h in cache_tr["h"][1:-1]],
    }"""),
        nbf.v4.new_code_cell("""trained = {}
models = []
for arch in ARCHITECTURES:
    params, row = fit(arch, X_train, y_train, X_test, y_test,
                      EPOCHS, LEARNING_RATE, BATCH_SIZE)
    trained[arch["key"]] = params
    models.append(row)
    t = row["test_by_threshold"]["0.5"]
    print(f"{row['key']} {row['label']:9s} {row['shape']:18s} "
          f"loss {row['loss_final']:.4f} acc {t['accuracy']:.4f} "
          f"prec {t['precision']:.4f} rec {t['recall']:.4f} f1 {t['f1']:.4f} "
          f"auc {row['roc_auc_test']:.4f} ({row['seconds']:.1f}s)")"""),
        nbf.v4.new_code_cell("""table = pd.DataFrame([
    {
        "Mô hình": f"{m['key']} {m['label']}",
        "Kiến trúc": m["shape"],
        "Kích hoạt": m["hidden_activation"],
        "Tham số": m["n_params"],
        "Loss train": m["loss_final"],
        "Loss test": m["loss_test_end"],
        "Accuracy": m["test_by_threshold"]["0.5"]["accuracy"],
        "Precision": m["test_by_threshold"]["0.5"]["precision"],
        "Recall": m["test_by_threshold"]["0.5"]["recall"],
        "F1": m["test_by_threshold"]["0.5"]["f1"],
        "ROC-AUC": m["roc_auc_test"],
        "Giây": m["seconds"],
    }
    for m in models
])
print("Kết quả trên tập test, ngưỡng 0.5")
print(table.to_string(index=False))

print("\\nMa trận nhầm lẫn tại ngưỡng 0.5")
for m in models:
    t = m["test_by_threshold"]["0.5"]
    print(f" {m['key']} {m['label']:9s} TP {t['tp']:6,} FN {t['fn']:6,} "
          f"FP {t['fp']:6,} TN {t['tn']:6,}")

print("\\nẢnh hưởng của ngưỡng quyết định")
for m in models:
    parts = []
    for th in THRESHOLDS:
        t = m["test_by_threshold"][str(th)]
        parts.append(f"{th}: prec {t['precision']:.4f} rec {t['recall']:.4f} f1 {t['f1']:.4f}")
    print(f" {m['key']} {m['label']:9s} " + " | ".join(parts))"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
styles = ["-", "--", "-.", ":"]

for axis, start in zip(axes, (0, 10)):
    for m, style in zip(models, styles):
        history = m["loss_history"][start:]
        axis.plot(range(start, start + len(history)), history, style, color="black",
                  linewidth=1.2, label=f"{m['key']} {m['label']}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary cross-entropy (train)")
    axis.grid(True, linewidth=0.3, alpha=0.4)
axes[0].set_title("Toàn bộ quá trình huấn luyện")
axes[1].set_title("Phóng to từ epoch 10")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
markers = ["o", "s", "^", "D"]

for m, marker in zip(models, markers):
    rows = [m["test_by_threshold"][str(t)] for t in THRESHOLDS]
    axes[0].plot([r["recall"] for r in rows], [r["precision"] for r in rows], marker + "-",
                 color="black", markerfacecolor="white", linewidth=1.0,
                 label=f"{m['key']} {m['label']}")
    for threshold, r in zip(THRESHOLDS, rows):
        axes[0].annotate(f"{threshold}", (r["recall"], r["precision"]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Đánh đổi khi đổi ngưỡng quyết định")
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(True, linewidth=0.3, alpha=0.4)

positions = np.arange(len(models))
width = 0.26
for offset, (metric, hatch) in enumerate(zip(("accuracy", "f1"), ("", "///"))):
    values = [m["test_by_threshold"]["0.5"][metric] for m in models]
    axes[1].bar(positions + offset * width, values, width, color="white",
                edgecolor="black", hatch=hatch, label=metric)
axes[1].bar(positions + 2 * width, [m["roc_auc_test"] for m in models], width,
            color="white", edgecolor="black", hatch="...", label="roc_auc")
axes[1].set_xticks(positions + width)
axes[1].set_xticklabels([f"{m['key']}\\n{m['label']}" for m in models], fontsize=8)
axes[1].set_ylim(0, 1.16)
axes[1].set_title("So sánh bốn mô hình trên tập test")
axes[1].legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
axes[1].grid(True, axis="y", linewidth=0.3, alpha=0.4)

fig.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 5. Học suất và công thức gốc của bài giảng (Full-batch GD 1000 epochs)"""),
        nbf.v4.new_code_cell("""SPEC_EPOCHS = 1000
SPEC_RATES = (0.001, 0.01, 0.1)

lr_study = []
lr_curves = {}
for rate in SPEC_RATES:
    _, row = fit(ARCHITECTURES[0], X_train, y_train, X_test, y_test, SPEC_EPOCHS, rate, None)
    t = row["test_by_threshold"]["0.5"]
    lr_curves[rate] = row["loss_history"]
    lr_study.append({
        "learning_rate": rate,
        "epochs": SPEC_EPOCHS,
        "batch_size": row["batch_size"],
        "updates": row["updates"],
        "loss_first": row["loss_first"],
        "loss_final": row["loss_final"],
        "accuracy": t["accuracy"],
        "precision": t["precision"],
        "recall": t["recall"],
        "f1": t["f1"],
        "roc_auc_test": row["roc_auc_test"],
        "seconds": row["seconds"],
    })
    print(f"lr {rate:<6} loss {row['loss_first']:.4f} -> {row['loss_final']:.4f} "
          f"acc {t['accuracy']:.4f} rec {t['recall']:.4f} f1 {t['f1']:.4f} "
          f"auc {row['roc_auc_test']:.4f} ({row['seconds']:.1f}s)")

m1 = models[0]
t1 = m1["test_by_threshold"]["0.5"]
print("\\nĐối chiếu: M1 lô nhỏ 512, 60 epoch")
print(f"lr {m1['learning_rate']:<6} loss {m1['loss_first']:.4f} -> {m1['loss_final']:.4f} "
      f"acc {t1['accuracy']:.4f} rec {t1['recall']:.4f} f1 {t1['f1']:.4f} "
      f"auc {m1['roc_auc_test']:.4f} ({m1['seconds']:.1f}s, {m1['updates']:,} lần cập nhật)")"""),
        nbf.v4.new_code_cell("""fig, axis = plt.subplots(figsize=(6.6, 3.4))
for rate, style in zip(SPEC_RATES, ["-", "--", "-."]):
    axis.plot(lr_curves[rate], style, color="black", linewidth=1.2,
              label=f"toàn tập, lr = {rate}")
axis.axhline(models[0]["loss_final"], color="black", linewidth=0.8, alpha=0.45)
axis.annotate(f"M1 lô nhỏ đạt {models[0]['loss_final']:.4f}",
              (SPEC_EPOCHS * 0.40, models[0]["loss_final"]), fontsize=8,
              xytext=(0, 6), textcoords="offset points")
axis.set_xlabel("Epoch")
axis.set_ylabel("Binary cross-entropy (train)")
axis.set_title("Gradient descent toàn tập theo công thức gốc")
axis.legend(frameon=False, fontsize=8)
axis.grid(True, linewidth=0.3, alpha=0.4)
fig.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 6. Ghi kết quả ra file"""),
        nbf.v4.new_code_cell("""payload = {
    "data_file": DIABETES_CSV.relative_to(ROOT).as_posix(),
    "features": list(FEATURES),
    "feature_ranking": {k: round(float(v), 6) for k, v in ranking.items()},
    "rows_raw": int(len(raw)),
    "rows_after_dedup": int(len(frame)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "positive_rate_train": round(float(y_train.mean()), 6),
    "positive_rate_test": round(float(y_test.mean()), 6),
    "split_seed": SPLIT_SEED,
    "init_seed": INIT_SEED,
    "shuffle_seed": SHUFFLE_SEED,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "thresholds": list(THRESHOLDS),
    "models": models,
    "learning_rate_study": lr_study,
}

out = REPORTS / "deeplearning_results.json"
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print("đã ghi", out.relative_to(ROOT).as_posix(), f"({out.stat().st_size / 1024:.1f} KB)")
print("số mô hình:", len(payload["models"]))
print("mô hình tốt nhất theo ROC-AUC:",
      max(payload["models"], key=lambda m: m["roc_auc_test"])["key"])"""),
    ]
    nb = create_nb(cells)
    nbf.write(nb, NOTEBOOKS / "04_deep_learning.ipynb")
    print("Created notebooks/04_deep_learning.ipynb")


def build_nb05():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 05 - K-Nearest Neighbors trên ba category
Notebook này bổ sung một họ thuật toán chưa từng xuất hiện trong notebook 01, 02, 03: học dựa trên thực thể (instance based learning)."""),
        nbf.v4.new_code_cell("""import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors,
)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)

print("thư mục gốc :", ROOT)
print("numpy :", np.__version__)
print("pandas :", pd.__version__)
print("seed chung :", ml.RANDOM_SEED)"""),
        nbf.v4.new_markdown_cell("""## 2. Nạp bốn bài toán"""),
        nbf.v4.new_code_cell("""t0 = time.perf_counter()
TASKS = {key: loader() for key, loader in ml.LOADERS.items()}
print(f"Nạp bốn bài toán trong {time.perf_counter() - t0:.1f}s\\n")

for task in TASKS.values():
    print(task.summary())"""),
        nbf.v4.new_code_cell("""pd.DataFrame([
    {
        "bài toán": task.title,
        "kiểu": task.kind,
        "train": f"{task.n_train:,}",
        "val": f"{task.n_val:,}",
        "test": f"{task.n_test:,}",
        "mốc so sánh": task.baseline_name,
    }
    for task in TASKS.values()
])"""),
        nbf.v4.new_markdown_cell("""## 3. Chọn $k$ bằng một lần tìm hàng xóm duy nhất"""),
        nbf.v4.new_code_cell("""def sweep_k(task, rep, k_grid, metric="euclidean"):
    '''Quét lưới k bằng MỘT lần tìm hàng xóm, trả về bảng kết quả trên tập validation.'''
    prep = task.prep(rep)
    Z_train = prep.fit_transform(task.X_train, task.y_train)
    Z_val = prep.transform(task.X_val)
    y_train = np.asarray(task.y_train)

    searcher = NearestNeighbors(
        n_neighbors=max(k_grid), metric=metric, algorithm="brute", n_jobs=-1
    ).fit(Z_train)

    t0 = time.perf_counter()
    _, index = searcher.kneighbors(Z_val)
    search_seconds = time.perf_counter() - t0
    neighbour_labels = y_train[index]

    rows = []
    for k in k_grid:
        window = neighbour_labels[:, :k]
        if task.kind == "regression":
            prediction = window.mean(axis=1)
            m = ml.regression_metrics(task.y_val, prediction)
            rows.append({
                "k": k, "R2_log": f"{m['r2_log']:.4f}",
                "MAE_USD": f"{m['mae_usd']:,.0f}", "MAPE_%": f"{m['mape_percent']:.1f}",
                "_score": m["r2_log"],
            })
        elif task.kind == "binary":
            proba = window.mean(axis=1)
            m = ml.classification_metrics(task.y_val, (proba >= 0.5).astype(int), proba)
            rows.append({
                "k": k, "accuracy": f"{m['accuracy']:.4f}", "recall": f"{m['recall']:.4f}",
                "f1": f"{m['f1']:.4f}", "f1_macro": f"{m['f1_macro']:.4f}",
                "roc_auc": f"{m['roc_auc']:.4f}", "_score": m["f1_macro"],
            })
        else:
            prediction = np.array([Counter(row).most_common(1)[0][0] for row in window])
            m = ml.classification_metrics(task.y_val, prediction, average="macro")
            rows.append({
                "k": k, "accuracy": f"{m['accuracy']:.4f}",
                "f1_macro": f"{m['f1_macro']:.4f}", "f1_weighted": f"{m['f1_weighted']:.4f}",
                "_score": m["f1_macro"],
            })

    best = max(rows, key=lambda r: r["_score"])
    print(f"Một lần tìm {max(k_grid)} hàng xóm cho {Z_val.shape[0]:,} điểm validation: "
          f"{search_seconds:.1f}s (d = {Z_train.shape[1]:,}, metric = {metric})")
    return rows, int(best["k"])


def show(rows, best_k):
    frame = pd.DataFrame(rows).drop(columns="_score")
    print(frame.to_string(index=False))
    print(f"\\n-> k tốt nhất trên validation: {best_k}")"""),
        nbf.v4.new_markdown_cell("""## 4. Bài toán 1 - Tiểu đường (phân loại nhị phân)"""),
        nbf.v4.new_code_cell("""task = TASKS["diabetes"]
rows_dia, k_dia = sweep_k(task, "native", [5, 11, 25, 51, 101])
show(rows_dia, k_dia)"""),
        nbf.v4.new_code_cell("""results_dia = [
    ml.run_model(task, f"KNN (k={k_dia})", "native",
                 KNeighborsClassifier(n_neighbors=k_dia, algorithm="brute", n_jobs=-1)),
    ml.run_model(task, task.baseline_name + " (mốc)", task.baseline_rep,
                 task.baseline_estimator()),
]
ml.print_table([ml.table_row(task, r) for r in results_dia])"""),
        nbf.v4.new_code_cell("""test_dia = {r["name"]: ml.evaluate_on_test(task, r["pipe"]) for r in results_dia}
for name, m in test_dia.items():
    print(f"TEST {name:<34} accuracy={m['accuracy']:.4f} recall={m['recall']:.4f} "
          f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")"""),
        nbf.v4.new_markdown_cell("""## 5. Bài toán 2 - Giá nhà (hồi quy)"""),
        nbf.v4.new_code_cell("""task = TASKS["house_price"]
rows_house, k_house = sweep_k(task, "target_enc_scaled", [5, 11, 25, 51])
show(rows_house, k_house)"""),
        nbf.v4.new_code_cell("""results_house = [
    ml.run_model(task, f"KNN (k={k_house})", "target_enc_scaled",
                 KNeighborsRegressor(n_neighbors=k_house, algorithm="brute", n_jobs=-1)),
    ml.run_model(task, task.baseline_name + " (mốc)", task.baseline_rep,
                 task.baseline_estimator()),
]
ml.print_table([ml.table_row(task, r) for r in results_house])"""),
        nbf.v4.new_code_cell("""test_house = {r["name"]: ml.evaluate_on_test(task, r["pipe"]) for r in results_house}
for name, m in test_house.items():
    print(f"TEST {name:<34} R2_log={m['r2_log']:.4f} MAE={m['mae_usd']:,.0f} USD "
          f"MAPE={m['mape_percent']:.1f}%")"""),
        nbf.v4.new_markdown_cell("""## 6. Bài toán 3 - E-commerce, mức hài lòng (bảng + TF-IDF)"""),
        nbf.v4.new_code_cell("""task = TASKS["ecommerce_satisfaction"]
rows_sat, k_sat = sweep_k(task, "tabular_text", [5, 11, 25, 51], metric="cosine")
show(rows_sat, k_sat)"""),
        nbf.v4.new_code_cell("""results_sat = [
    ml.run_model(task, f"KNN cosin (k={k_sat})", "tabular_text",
                 KNeighborsClassifier(n_neighbors=k_sat, metric="cosine",
                                      algorithm="brute", n_jobs=-1)),
    ml.run_model(task, task.baseline_name + " (mốc)", task.baseline_rep,
                 task.baseline_estimator()),
]
ml.print_table([ml.table_row(task, r) for r in results_sat])"""),
        nbf.v4.new_code_cell("""test_sat = {r["name"]: ml.evaluate_on_test(task, r["pipe"]) for r in results_sat}
for name, m in test_sat.items():
    print(f"TEST {name:<34} accuracy={m['accuracy']:.4f} recall={m['recall']:.4f} "
          f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")"""),
        nbf.v4.new_markdown_cell("""## 7. Bài toán 4 - E-commerce, nhóm sản phẩm quan tâm (9 lớp)"""),
        nbf.v4.new_code_cell("""task = TASKS["ecommerce_interest"]
print("Phân bố lớp trên tập train:")
print(pd.Series(task.y_train).value_counts().to_string())
print()
rows_int, k_int = sweep_k(task, "tabular_text", [5, 11, 25, 51], metric="cosine")
show(rows_int, k_int)"""),
        nbf.v4.new_code_cell("""results_int = [
    ml.run_model(task, f"KNN cosin (k={k_int})", "tabular_text",
                 KNeighborsClassifier(n_neighbors=k_int, metric="cosine",
                                      algorithm="brute", n_jobs=-1)),
    ml.run_model(task, task.baseline_name + " (mốc)", task.baseline_rep,
                 task.baseline_estimator()),
]
ml.print_table([ml.table_row(task, r) for r in results_int])"""),
        nbf.v4.new_code_cell("""test_int = {r["name"]: ml.evaluate_on_test(task, r["pipe"]) for r in results_int}
for name, m in test_int.items():
    print(f"TEST {name:<34} accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"f1_weighted={m['f1_weighted']:.4f}")"""),
        nbf.v4.new_markdown_cell("""## 8. Tổng hợp và ghi kết quả"""),
        nbf.v4.new_code_cell("""def block(task, results, test_metrics, sweep_rows, best_k, metric):
    return {
        "task": task.key,
        "title": task.title,
        "kind": task.kind,
        "n_train": task.n_train,
        "n_val": task.n_val,
        "n_test": task.n_test,
        "audit": task.audit,
        "distance": metric,
        "k_grid": [int(r["k"]) for r in sweep_rows],
        "k_selected": best_k,
        "sweep": [{k: v for k, v in r.items() if k != "_score"} for r in sweep_rows],
        "validation_table": [ml.table_row(task, r) for r in results],
        "test_metrics": {name: m for name, m in test_metrics.items()},
        "model_names": [r["name"] for r in results],
    }


payload = {
    "algorithm": "K-Nearest Neighbors",
    "library": "scikit-learn",
    "seed": ml.RANDOM_SEED,
    "note": (
        "Mô hình mốc được huấn luyện lại trên bộ dữ liệu hiện tại trong data/, "
        "không chép lại số của notebook 01-03."
    ),
    "tasks": [
        block(TASKS["diabetes"], results_dia, test_dia, rows_dia, k_dia, "euclidean"),
        block(TASKS["house_price"], results_house, test_house, rows_house, k_house, "euclidean"),
        block(TASKS["ecommerce_satisfaction"], results_sat, test_sat, rows_sat, k_sat, "cosine"),
        block(TASKS["ecommerce_interest"], results_int, test_int, rows_int, k_int, "cosine"),
    ],
}

path = ml.save_results("knn", payload)
print("Đã ghi", path.relative_to(ROOT), f"({path.stat().st_size / 1024:.1f} KB)")"""),
    ]
    nb = create_nb(cells)
    nbf.write(nb, NOTEBOOKS / "05_knn.ipynb")
    print("Created notebooks/05_knn.ipynb")


def build_nb06():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 06 - AdaBoost trên ba category
Notebook này bổ sung AdaBoost để so sánh boosting theo trọng số mẫu với gradient boosting."""),
        nbf.v4.new_code_cell("""import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)

print("thư mục gốc :", ROOT)
print("numpy :", np.__version__)
print("pandas :", pd.__version__)
print("seed chung :", ml.RANDOM_SEED)"""),
        nbf.v4.new_markdown_cell("""## 2. Nạp bốn bài toán"""),
        nbf.v4.new_code_cell("""t0 = time.perf_counter()
TASKS = {key: loader() for key, loader in ml.LOADERS.items()}
print(f"Nạp bốn bài toán trong {time.perf_counter() - t0:.1f}s\\n")
for task in TASKS.values():
    print(task.summary())"""),
        nbf.v4.new_markdown_cell("""## 3. Quét số vòng boosting bằng một lần huấn luyện duy nhất"""),
        nbf.v4.new_code_cell("""def sweep_stages(task, rep, estimator, stage_grid, sample_weight=None):
    '''Huấn luyện MỘT lần với số vòng lớn nhất, đọc kết quả tại từng mốc trong lưới.'''
    prep = task.prep(rep)
    Z_train = prep.fit_transform(task.X_train, task.y_train)
    Z_val = prep.transform(task.X_val)

    t0 = time.perf_counter()
    estimator.fit(Z_train, task.y_train, sample_weight=sample_weight)
    fit_seconds = time.perf_counter() - t0
    n_fitted = len(estimator.estimators_)

    wanted = {s for s in stage_grid if s <= n_fitted}
    rows = []

    t0 = time.perf_counter()
    if task.kind == "regression":
        for stage, prediction in enumerate(estimator.staged_predict(Z_val), start=1):
            if stage not in wanted:
                continue
            m = ml.regression_metrics(task.y_val, prediction)
            rows.append({
                "vòng": stage, "R2_log": f"{m['r2_log']:.4f}",
                "MAE_USD": f"{m['mae_usd']:,.0f}", "MAPE_%": f"{m['mape_percent']:.1f}",
                "_score": m["r2_log"],
            })
    else:
        classes = estimator.classes_
        for stage, proba in enumerate(estimator.staged_predict_proba(Z_val), start=1):
            if stage not in wanted:
                continue
            prediction = classes[proba.argmax(axis=1)]
            score = proba[:, 1] if task.average == "binary" else None
            m = ml.classification_metrics(
                task.y_val, prediction, score, average=task.average
            )
            row = {
                "vòng": stage, "accuracy": f"{m['accuracy']:.4f}",
                "recall": f"{m['recall']:.4f}", "f1_macro": f"{m['f1_macro']:.4f}",
                "_score": m["f1_macro"],
            }
            if m["roc_auc"] is not None:
                row["roc_auc"] = f"{m['roc_auc']:.4f}"
            rows.append(row)
    stage_seconds = time.perf_counter() - t0

    best = max(rows, key=lambda r: r["_score"])
    print(f"Huấn luyện {n_fitted} vòng trong {fit_seconds:.1f}s, "
          f"đọc {len(rows)} mốc trong {stage_seconds:.1f}s (d = {Z_train.shape[1]:,})")
    print(pd.DataFrame(rows).drop(columns="_score").to_string(index=False))
    print(f"\\n-> số vòng tốt nhất trên validation: {best['vòng']}")
    return rows, int(best["vòng"])"""),
        nbf.v4.new_markdown_cell("""## 4. Bài toán 1 - Tiểu đường (phân loại nhị phân)"""),
        nbf.v4.new_code_cell("""task = TASKS["diabetes"]
ada_dia = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
    n_estimators=300, learning_rate=0.5, random_state=ml.RANDOM_SEED,
)
weight_dia = compute_sample_weight("balanced", task.y_train)
rows_dia, m_dia = sweep_stages(task, "native", ada_dia, [25, 50, 100, 200, 300], weight_dia)"""),
        nbf.v4.new_code_cell("""results_dia = [
    ml.run_model(task, f"AdaBoost ({m_dia} vòng, cây sâu 3)", "native",
                 AdaBoostClassifier(
                     estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
                     n_estimators=m_dia, learning_rate=0.5, random_state=ml.RANDOM_SEED),
                 sample_weight=weight_dia),
    ml.run_model(task, task.baseline_name + " (mốc)", task.baseline_rep,
                 task.baseline_estimator()),
]
ml.print_table([ml.table_row(task, r) for r in results_dia])"""),
        nbf.v4.new_code_cell("""test_dia = {r["name"]: ml.evaluate_on_test(task, r["pipe"]) for r in results_dia}
for name, m in test_dia.items():
    print(f"TEST {name:<36} accuracy={m['accuracy']:.4f} recall={m['recall']:.4f} "
          f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")"""),
        nbf.v4.new_markdown_cell("""## 5. Bài toán 2 - Giá nhà (hồi quy, AdaBoost.R2)"""),
        nbf.v4.new_code_cell("""task = TASKS["house_price"]
ada_house = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=8, random_state=ml.RANDOM_SEED),
    n_estimators=100, learning_rate=0.5, random_state=ml.RANDOM_SEED,
)
rows_house, m_house = sweep_stages(task, "target_enc", ada_house, [10, 25, 50, 75, 100])"""),
        nbf.v4.new_code_cell("""results_house = [
    ml.run_model(task, f"AdaBoost.R2 ({m_house} vòng, cây sâu 8)", "target_enc",
                 AdaBoostRegressor(
                     estimator=DecisionTreeRegressor(max_depth=8, random_state=ml.RANDOM_SEED),
                     n_estimators=m_house, learning_rate=0.5, random_state=ml.RANDOM_SEED)),
    ml.run_model(task, task.baseline_name + " (mốc)", task.baseline_rep,
                 task.baseline_estimator()),
]
ml.print_table([ml.table_row(task, r) for r in results_house])"""),
        nbf.v4.new_code_cell("""test_house = {r["name"]: ml.evaluate_on_test(task, r["pipe"]) for r in results_house}
for name, m in test_house.items():
    print(f"TEST {name:<36} R2_log={m['r2_log']:.4f} MAE={m['mae_usd']:,.0f} USD "
          f"MAPE={m['mape_percent']:.1f}%")"""),
        nbf.v4.new_markdown_cell("""## 6. Bài toán 3 - E-commerce, mức hài lòng (bảng + TF-IDF)"""),
        nbf.v4.new_code_cell("""task = TASKS["ecommerce_satisfaction"]
ada_sat = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
    n_estimators=300, learning_rate=0.5, random_state=ml.RANDOM_SEED,
)
weight_sat = compute_sample_weight("balanced", task.y_train)
rows_sat, m_sat = sweep_stages(task, "tabular_text", ada_sat, [25, 50, 100, 200, 300], weight_sat)"""),
        nbf.v4.new_code_cell("""results_sat = [
    ml.run_model(task, f"AdaBoost ({m_sat} vòng, cây sâu 3)", "tabular_text",
                 AdaBoostClassifier(
                     estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
                     n_estimators=m_sat, learning_rate=0.5, random_state=ml.RANDOM_SEED),
                 sample_weight=weight_sat),
    ml.run_model(task, task.baseline_name + " (mốc)", task.baseline_rep,
                 task.baseline_estimator()),
]
ml.print_table([ml.table_row(task, r) for r in results_sat])"""),
        nbf.v4.new_code_cell("""test_sat = {r["name"]: ml.evaluate_on_test(task, r["pipe"]) for r in results_sat}
for name, m in test_sat.items():
    print(f"TEST {name:<36} accuracy={m['accuracy']:.4f} recall={m['recall']:.4f} "
          f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")"""),
        nbf.v4.new_markdown_cell("""## 7. Bài toán 4 - E-commerce, nhóm sản phẩm quan tâm (9 lớp)"""),
        nbf.v4.new_code_cell("""task = TASKS["ecommerce_interest"]
ada_int = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
    n_estimators=300, learning_rate=0.5, random_state=ml.RANDOM_SEED,
)
weight_int = compute_sample_weight("balanced", task.y_train)
rows_int, m_int = sweep_stages(task, "tabular_text", ada_int, [25, 50, 100, 200, 300], weight_int)"""),
        nbf.v4.new_code_cell("""results_int = [
    ml.run_model(task, f"AdaBoost ({m_int} vòng, cây sâu 3)", "tabular_text",
                 AdaBoostClassifier(
                     estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
                     n_estimators=m_int, learning_rate=0.5, random_state=ml.RANDOM_SEED),
                 sample_weight=weight_int),
    ml.run_model(task, task.baseline_name + " (mốc)", task.baseline_rep,
                 task.baseline_estimator()),
]
ml.print_table([ml.table_row(task, r) for r in results_int])"""),
        nbf.v4.new_code_cell("""test_int = {r["name"]: ml.evaluate_on_test(task, r["pipe"]) for r in results_int}
for name, m in test_int.items():
    print(f"TEST {name:<36} accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"f1_weighted={m['f1_weighted']:.4f}")"""),
        nbf.v4.new_markdown_cell("""## 8. Tổng hợp và ghi kết quả"""),
        nbf.v4.new_code_cell("""def block(task, results, test_metrics, sweep_rows, best_stage, weak_learner):
    return {
        "task": task.key,
        "title": task.title,
        "kind": task.kind,
        "n_train": task.n_train,
        "n_val": task.n_val,
        "n_test": task.n_test,
        "audit": task.audit,
        "weak_learner": weak_learner,
        "stage_grid": [int(r["vòng"]) for r in sweep_rows],
        "stage_selected": best_stage,
        "sweep": [{k: v for k, v in r.items() if k != "_score"} for r in sweep_rows],
        "validation_table": [ml.table_row(task, r) for r in results],
        "test_metrics": dict(test_metrics),
        "model_names": [r["name"] for r in results],
    }


payload = {
    "algorithm": "AdaBoost (SAMME cho phân loại, AdaBoost.R2 cho hồi quy)",
    "library": "scikit-learn",
    "seed": ml.RANDOM_SEED,
    "learning_rate": 0.5,
    "note": (
        "Mô hình mốc được huấn luyện lại trên bộ dữ liệu hiện tại trong data/, "
        "không chép lại số của notebook 01-03."
    ),
    "tasks": [
        block(TASKS["diabetes"], results_dia, test_dia, rows_dia, m_dia,
              "DecisionTreeClassifier(max_depth=3) + sample_weight cân bằng lớp"),
        block(TASKS["house_price"], results_house, test_house, rows_house, m_house,
              "DecisionTreeRegressor(max_depth=8)"),
        block(TASKS["ecommerce_satisfaction"], results_sat, test_sat, rows_sat, m_sat,
              "DecisionTreeClassifier(max_depth=3) + sample_weight cân bằng lớp"),
        block(TASKS["ecommerce_interest"], results_int, test_int, rows_int, m_int,
              "DecisionTreeClassifier(max_depth=3) + sample_weight cân bằng lớp"),
    ],
}

path = ml.save_results("adaboost", payload)
print("Đã ghi", path.relative_to(ROOT), f"({path.stat().st_size / 1024:.1f} KB)")"""),
    ]
    nb = create_nb(cells)
    nbf.write(nb, NOTEBOOKS / "06_adaboost.ipynb")
    print("Created notebooks/06_adaboost.ipynb")


def build_nb07():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 07 - Mạng nơ-ron hồi quy viết tay cho bài toán giá nhà
Toàn bộ mạng vẫn viết tay: lan truyền xuôi, hàm mất mát, lan truyền ngược theo quy tắc chuỗi và gradient descent đều là NumPy thuần."""),
        nbf.v4.new_code_cell("""import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

print("thư mục gốc :", ROOT)
print("numpy :", np.__version__)"""),
        nbf.v4.new_markdown_cell("""## 1. Dữ liệu và biểu diễn đầu vào"""),
        nbf.v4.new_code_cell("""task = ml.load_house_price()
prep = task.prep("target_enc_scaled")

X_train = prep.fit_transform(task.X_train, task.y_train)
X_val = prep.transform(task.X_val)
X_test = prep.transform(task.X_test)

print(f"X_train {X_train.shape} X_val {X_val.shape} X_test {X_test.shape}")
print(f"trung bình mỗi chiều sau chuẩn hoá: {np.round(X_train.mean(axis=0), 3)}")
print(f"độ lệch chuẩn mỗi chiều : {np.round(X_train.std(axis=0), 3)}")"""),
        nbf.v4.new_code_cell("""y_mean = float(task.y_train.mean())
y_std = float(task.y_train.std())

z_train = ((task.y_train - y_mean) / y_std).reshape(-1, 1)
z_val = ((task.y_val - y_mean) / y_std).reshape(-1, 1)

def to_log_price(z):
    '''Quy dự đoán đã chuẩn hoá về lại log(price) để tính chỉ số nghiệp vụ.'''
    return z.ravel() * y_std + y_mean

print(f"log(price): trung bình = {y_mean:.4f}, độ lệch chuẩn = {y_std:.4f}")
print(f"sau chuẩn hoá: trung bình = {z_train.mean():.6f}, độ lệch chuẩn = {z_train.std():.6f}")"""),
        nbf.v4.new_markdown_cell("""## 2. Mạng nơ-ron viết tay"""),
        nbf.v4.new_code_cell("""def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def identity(x):
    return x


def identity_derivative(x):
    return np.ones_like(x)


ACTIVATIONS = {"relu": (relu, relu_derivative), "linear": (identity, identity_derivative)}


@dataclass(frozen=True)
class Architecture:
    key: str
    label: str
    sizes: tuple
    hidden_activation: str = "relu"

    @property
    def shape(self):
        return " - ".join(str(s) for s in self.sizes)

    @property
    def n_params(self):
        return sum(a * b + b for a, b in zip(self.sizes, self.sizes[1:]))


def init_params(arch, seed):
    rng = np.random.default_rng(seed)
    layers = []
    for fan_in, fan_out in zip(arch.sizes, arch.sizes[1:]):
        W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        layers.append((W, np.zeros((1, fan_out))))
    return tuple(layers)


def forward(params, X, hidden_activation):
    '''Lớp cuối KHÔNG có hàm kích hoạt: đầu ra hồi quy là số thực bất kỳ.'''
    act, _ = ACTIVATIONS[hidden_activation]
    pre_activations, activations = [], [X]
    h = X
    last = len(params) - 1
    for index, (W, b) in enumerate(params):
        z = h @ W + b
        h = z if index == last else act(z)
        pre_activations.append(z)
        activations.append(h)
    return h, {"z": pre_activations, "h": activations}"""),
        nbf.v4.new_markdown_cell("""## 3. Hàm mất mát và lan truyền ngược"""),
        nbf.v4.new_code_cell("""def mean_squared_error(z_true, z_pred):
    return float(0.5 * np.mean((z_pred - z_true) ** 2))


def backward(params, z_true, cache, hidden_activation):
    _, act_derivative = ACTIVATIONS[hidden_activation]
    z, h = cache["z"], cache["h"]
    n = len(z_true)
    grads = []
    delta = (h[-1] - z_true) / n
    for index in range(len(params) - 1, -1, -1):
        grads.append((h[index].T @ delta, np.sum(delta, axis=0, keepdims=True)))
        if index:
            delta = (delta @ params[index][0].T) * act_derivative(z[index - 1])
    return tuple(reversed(grads))


def update(params, grads, learning_rate):
    return tuple(
        (W - learning_rate * dW, b - learning_rate * db)
        for (W, b), (dW, db) in zip(params, grads)
    )"""),
        nbf.v4.new_markdown_cell("""## 4. Huấn luyện bốn kiến trúc"""),
        nbf.v4.new_code_cell("""LEARNING_RATE = 0.05
EPOCHS = 40
BATCH_SIZE = 512
INIT_SEED = 42
SHUFFLE_SEED = 7

ARCHITECTURES = (
    Architecture("H1", "Baseline", (8, 32, 16, 1), "relu"),
    Architecture("H2", "Wider", (8, 64, 32, 1), "relu"),
    Architecture("H3", "Shallow", (8, 32, 1), "relu"),
    Architecture("H4", "No ReLU", (8, 32, 16, 1), "linear"),
)


def fit(arch):
    params = init_params(arch, INIT_SEED)
    rng = np.random.default_rng(SHUFFLE_SEED)
    n = len(X_train)
    history_train, history_val = [], []
    started = time.perf_counter()

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for begin in range(0, n, BATCH_SIZE):
            rows = order[begin : begin + BATCH_SIZE]
            _, cache = forward(params, X_train[rows], arch.hidden_activation)
            grads = backward(params, z_train[rows], cache, arch.hidden_activation)
            params = update(params, grads, LEARNING_RATE)
        history_train.append(mean_squared_error(z_train, forward(params, X_train, arch.hidden_activation)[0]))
        history_val.append(mean_squared_error(z_val, forward(params, X_val, arch.hidden_activation)[0]))

    seconds = time.perf_counter() - started
    val_log = to_log_price(forward(params, X_val, arch.hidden_activation)[0])
    metrics = ml.regression_metrics(task.y_val, val_log)
    return {
        "arch": arch, "params": params, "seconds": seconds,
        "history_train": history_train, "history_val": history_val,
        "metrics": metrics,
    }


trained = {}
for arch in ARCHITECTURES:
    trained[arch.key] = fit(arch)
    m = trained[arch.key]["metrics"]
    print(f"[{trained[arch.key]['seconds']:6.1f}s] {arch.key} {arch.label:<9} {arch.shape:<14} "
          f"{arch.n_params:>5d} tham số R2_log={m['r2_log']:.4f} MAE={m['mae_usd']:,.0f} USD")"""),
        nbf.v4.new_code_cell("""pd.DataFrame([
    {
        "mô hình": f"{r['arch'].key} {r['arch'].label}",
        "kiến trúc": r["arch"].shape,
        "kích hoạt": r["arch"].hidden_activation,
        "tham số": r["arch"].n_params,
        "loss cuối (train)": f"{r['history_train'][-1]:.5f}",
        "loss cuối (val)": f"{r['history_val'][-1]:.5f}",
        "R2_log": f"{r['metrics']['r2_log']:.4f}",
        "MAE_USD": f"{r['metrics']['mae_usd']:,.0f}",
        "MAPE_%": f"{r['metrics']['mape_percent']:.1f}",
        "giây": f"{r['seconds']:.1f}",
    }
    for r in trained.values()
])"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
styles = ["-", "--", "-.", ":"]

for (key, result), style in zip(trained.items(), styles):
    label = f"{key} {result['arch'].label}"
    axes[0].plot(result["history_train"], style, color="black", linewidth=1.1, label=label)
    axes[1].plot(result["history_val"], style, color="black", linewidth=1.1, label=label)

axes[0].set_title("Mất mát trên tập train")
axes[1].set_title("Mất mát trên tập validation")
for axis in axes:
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE (thang đã chuẩn hoá)")
    axis.grid(True, linewidth=0.3, alpha=0.4)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 5. So sánh với mô hình machine learning tốt nhất"""),
        nbf.v4.new_code_cell("""best_key = max(trained, key=lambda k: trained[k]["metrics"]["r2_log"])
best = trained[best_key]
best_arch = best["arch"]
print(f"Kiến trúc tốt nhất trên validation: {best_key} {best_arch.label} ({best_arch.shape})")

baseline = ml.run_model(task, task.baseline_name + " (mốc ML)", task.baseline_rep,
                        task.baseline_estimator())
ml.print_table([ml.table_row(task, baseline)])"""),
        nbf.v4.new_code_cell("""test_log = to_log_price(forward(best["params"], X_test, best_arch.hidden_activation)[0])
dl_test = ml.regression_metrics(task.y_test, test_log)
ml_test = ml.evaluate_on_test(task, baseline["pipe"])

pd.DataFrame([
    {"mô hình": f"Mạng nơ-ron {best_key} ({best_arch.shape})", **{
        "R2_log": f"{dl_test['r2_log']:.4f}", "MAE_USD": f"{dl_test['mae_usd']:,.0f}",
        "RMSE_USD": f"{dl_test['rmse_usd']:,.0f}", "MAPE_%": f"{dl_test['mape_percent']:.1f}"}},
    {"mô hình": task.baseline_name + " (mốc ML)", **{
        "R2_log": f"{ml_test['r2_log']:.4f}", "MAE_USD": f"{ml_test['mae_usd']:,.0f}",
        "RMSE_USD": f"{ml_test['rmse_usd']:,.0f}", "MAPE_%": f"{ml_test['mape_percent']:.1f}"}},
])"""),
        nbf.v4.new_code_cell("""actual_usd = np.exp(task.y_test)
predicted_usd = np.exp(test_log)
residual = predicted_usd - actual_usd

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
sample = np.random.default_rng(INIT_SEED).choice(len(actual_usd), size=6000, replace=False)

axes[0].scatter(actual_usd[sample], predicted_usd[sample], s=3, alpha=0.25, color="black")
limits = [actual_usd.min(), actual_usd.max()]
axes[0].plot(limits, limits, "-", color="black", linewidth=0.9)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Giá thật (USD, thang log)")
axes[0].set_ylabel("Giá dự đoán (USD, thang log)")
axes[0].set_title("Dự đoán so với thực tế")

axes[1].scatter(actual_usd[sample], residual[sample], s=3, alpha=0.25, color="black")
axes[1].axhline(0, color="black", linewidth=0.9)
axes[1].set_xscale("log")
axes[1].set_xlabel("Giá thật (USD, thang log)")
axes[1].set_ylabel("Sai số dự đoán (USD)")
axes[1].set_title("Phân bố sai số theo phân khúc giá")

for axis in axes:
    axis.grid(True, linewidth=0.3, alpha=0.4)
fig.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 6. Ghi kết quả"""),
        nbf.v4.new_code_cell("""payload = {
    "task": task.key,
    "title": task.title,
    "representation": "target_enc_scaled",
    "d_input": int(X_train.shape[1]),
    "n_train": task.n_train,
    "n_val": task.n_val,
    "n_test": task.n_test,
    "audit": task.audit,
    "target_transform": {"log_price_mean": y_mean, "log_price_std": y_std},
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "init_seed": INIT_SEED,
    "shuffle_seed": SHUFFLE_SEED,
    "models": [
        {
            "key": r["arch"].key,
            "label": r["arch"].label,
            "shape": r["arch"].shape,
            "hidden_activation": r["arch"].hidden_activation,
            "n_params": r["arch"].n_params,
            "seconds": round(r["seconds"], 2),
            "loss_train_final": round(r["history_train"][-1], 6),
            "loss_val_final": round(r["history_val"][-1], 6),
            "loss_history_train": [round(v, 6) for v in r["history_train"]],
            "loss_history_val": [round(v, 6) for v in r["history_val"]],
            "validation": {k: round(v, 6) for k, v in r["metrics"].items()},
        }
        for r in trained.values()
    ],
    "best_model": best_key,
    "test_metrics_dl": {k: round(v, 6) for k, v in dl_test.items()},
    "test_metrics_ml_baseline": {k: round(v, 6) for k, v in ml_test.items()},
    "ml_baseline_name": task.baseline_name,
}

path = ml.save_results("deeplearning_house", payload)
print("Đã ghi", path.relative_to(ROOT), f"({path.stat().st_size / 1024:.1f} KB)")"""),
    ]
    nb = create_nb(cells)
    nbf.write(nb, NOTEBOOKS / "07_deep_learning_house.ipynb")
    print("Created notebooks/07_deep_learning_house.ipynb")


def build_nb08():
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 08 - Mạng nơ-ron có lớp nhúng học được cho bài toán e-commerce
Notebook này dựng ma trận nhúng từ $E$ thành tham số của mạng và được học bằng lan truyền ngược, kết hợp gộp trung bình có che (`np.add.at`)."""),
        nbf.v4.new_code_cell("""import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

print("thư mục gốc :", ROOT)
print("numpy :", np.__version__)"""),
        nbf.v4.new_markdown_cell("""## 1. Dữ liệu"""),
        nbf.v4.new_code_cell("""task = ml.load_ecommerce_satisfaction()
prep_tab = task.prep("tabular")

TAB_TRAIN = np.asarray(prep_tab.fit_transform(task.X_train, task.y_train).toarray())
TAB_VAL = np.asarray(prep_tab.transform(task.X_val).toarray())
TAB_TEST = np.asarray(prep_tab.transform(task.X_test).toarray())

y_train = np.asarray(task.y_train).reshape(-1, 1).astype(float)
y_val = np.asarray(task.y_val).reshape(-1, 1).astype(float)
y_test = np.asarray(task.y_test).reshape(-1, 1).astype(float)

print(f"bảng: train {TAB_TRAIN.shape} val {TAB_VAL.shape} test {TAB_TEST.shape}")
print(f"tỉ lệ hài lòng: train={y_train.mean():.4f} val={y_val.mean():.4f} test={y_test.mean():.4f}")"""),
        nbf.v4.new_markdown_cell("""## 2. Văn bản thành số: tokenize, từ điển, chuỗi id"""),
        nbf.v4.new_code_cell("""VOCAB_SIZE = 5000
T_PAD = 32
D_EMBED = 32

counter = Counter()
for text in task.X_train[ml.datasets.ECOM_TEXT]:
    counter.update(ml.tokenize(str(text)))

vocab = {word: index for index, (word, _) in enumerate(counter.most_common(VOCAB_SIZE), start=1)}
print(f"Token khác nhau trong tập train : {len(counter):,}")
print(f"Giữ lại trong từ điển : {len(vocab):,} (id 1..{len(vocab)}, id 0 = PAD)")
print("10 từ phổ biến nhất :", [w for w, _ in counter.most_common(10)])


def texts_to_ids(texts):
    out = np.zeros((len(texts), T_PAD), dtype=np.int64)
    for row, text in enumerate(texts):
        ids = [vocab[w] for w in ml.tokenize(str(text)) if w in vocab][:T_PAD]
        out[row, : len(ids)] = ids
    return out


IDS_TRAIN = texts_to_ids(task.X_train[ml.datasets.ECOM_TEXT].values)
IDS_VAL = texts_to_ids(task.X_val[ml.datasets.ECOM_TEXT].values)
IDS_TEST = texts_to_ids(task.X_test[ml.datasets.ECOM_TEXT].values)

print(f"\\nIDS_TRAIN {IDS_TRAIN.shape} (B x T)")
print("Ví dụ một bình luận:")
print(" nguyên văn :", repr(str(task.X_train[ml.datasets.ECOM_TEXT].values[0])[:90]))
print(" chuỗi id :", IDS_TRAIN[0][:16], "...")
print(f" số token thật: {int((IDS_TRAIN[0] > 0).sum())} / {T_PAD}")"""),
        nbf.v4.new_markdown_cell("""## 3. Lớp nhúng và phép gộp trung bình có che"""),
        nbf.v4.new_code_cell("""def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def init_dense(sizes, seed):
    rng = np.random.default_rng(seed)
    layers = []
    for fan_in, fan_out in zip(sizes, sizes[1:]):
        W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        layers.append((W, np.zeros((1, fan_out))))
    return tuple(layers)


def init_embedding(seed):
    '''Hàng 0 là token đệm, luôn giữ bằng 0 để nó không đóng góp gì vào phép gộp.'''
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((VOCAB_SIZE + 1, D_EMBED)) * np.sqrt(1.0 / D_EMBED)
    E[0] = 0.0
    return E


def pool(E, ids):
    mask = (ids > 0).astype(float)
    counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    pooled = (E[ids] * mask[:, :, None]).sum(axis=1) / counts
    return pooled, mask, counts


def scatter_embedding_grad(d_pooled, ids, mask, counts):
    '''Đưa gradient của vector gộp về đúng các hàng từ vựng đã tham gia.'''
    dE = np.zeros((VOCAB_SIZE + 1, D_EMBED))
    rows, columns = np.nonzero(mask)
    np.add.at(dE, ids[rows, columns], d_pooled[rows] / counts[rows])
    dE[0] = 0.0
    return dE"""),
        nbf.v4.new_markdown_cell("""## 4. Mạng và hàm mất mát có trọng số lớp"""),
        nbf.v4.new_code_cell("""def class_weights(y):
    n = len(y)
    positive = float(y.sum())
    return {1.0: n / (2.0 * positive), 0.0: n / (2.0 * (n - positive))}


WEIGHTS = class_weights(y_train)
W_TRAIN = np.where(y_train == 1.0, WEIGHTS[1.0], WEIGHTS[0.0])
print(f"trọng số lớp: hài lòng = {WEIGHTS[1.0]:.4f}, không hài lòng = {WEIGHTS[0.0]:.4f}")


def weighted_bce(y, y_hat, weight):
    eps = 1e-8
    p = np.clip(y_hat, eps, 1.0 - eps)
    return float(-np.mean(weight * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def dense_forward(layers, X):
    pre_activations, activations = [], [X]
    h = X
    last = len(layers) - 1
    for index, (W, b) in enumerate(layers):
        z = h @ W + b
        h = sigmoid(z) if index == last else relu(z)
        pre_activations.append(z)
        activations.append(h)
    return h, {"z": pre_activations, "h": activations}


def dense_backward(layers, y, weight, cache):
    '''Trả về gradient của từng lớp VÀ gradient theo đầu vào, để đẩy tiếp vào lớp nhúng.'''
    z, h = cache["z"], cache["h"]
    n = len(y)
    grads = []
    delta = weight * (h[-1] - y) / n
    for index in range(len(layers) - 1, -1, -1):
        grads.append((h[index].T @ delta, np.sum(delta, axis=0, keepdims=True)))
        delta = delta @ layers[index][0].T
        if index:
            delta = delta * relu_derivative(z[index - 1])
    return tuple(reversed(grads)), delta"""),
        nbf.v4.new_markdown_cell("""## 5. Huấn luyện bốn mô hình"""),
        nbf.v4.new_code_cell("""LEARNING_RATE = 0.1
EPOCHS = 40
BATCH_SIZE = 256
INIT_SEED = 42
SHUFFLE_SEED = 7
HIDDEN = (32, 16)


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    use_tabular: bool
    use_text: bool
    train_embedding: bool


MODELS = (
    Model("N1", "Chỉ bảng", True, False, False),
    Model("N2", "Bảng + nhúng học được", True, True, True),
    Model("N3", "Bảng + nhúng đóng băng", True, True, False),
    Model("N4", "Chỉ văn bản", False, True, True),
)


def assemble(model, tab, ids, E):
    '''Ghép đầu vào của mạng theo đúng cấu hình của từng mô hình.'''
    if not model.use_text:
        return tab, None, None, None
    pooled, mask, counts = pool(E, ids)
    X = np.hstack([tab, pooled]) if model.use_tabular else pooled
    return X, mask, counts, pooled


def fit(model):
    d_tab = TAB_TRAIN.shape[1]
    d_in = (d_tab if model.use_tabular else 0) + (D_EMBED if model.use_text else 0)
    layers = init_dense((d_in, *HIDDEN, 1), INIT_SEED)
    E = init_embedding(INIT_SEED) if model.use_text else None

    rng = np.random.default_rng(SHUFFLE_SEED)
    n = len(TAB_TRAIN)
    history_train, history_val = [], []
    started = time.perf_counter()

    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for begin in range(0, n, BATCH_SIZE):
            rows = order[begin : begin + BATCH_SIZE]
            X, mask, counts, _ = assemble(model, TAB_TRAIN[rows], IDS_TRAIN[rows], E)
            _, cache = dense_forward(layers, X)
            grads, d_input = dense_backward(layers, y_train[rows], W_TRAIN[rows], cache)
            layers = tuple(
                (W - LEARNING_RATE * dW, b - LEARNING_RATE * db)
                for (W, b), (dW, db) in zip(layers, grads)
            )
            if model.use_text and model.train_embedding:
                d_pooled = d_input[:, -D_EMBED:] if model.use_tabular else d_input
                E = E - LEARNING_RATE * scatter_embedding_grad(
                    d_pooled, IDS_TRAIN[rows], mask, counts
                )

        history_train.append(weighted_bce(
            y_train, predict_proba(model, layers, E, TAB_TRAIN, IDS_TRAIN), W_TRAIN))
        history_val.append(weighted_bce(
            y_val, predict_proba(model, layers, E, TAB_VAL, IDS_VAL),
            np.where(y_val == 1.0, WEIGHTS[1.0], WEIGHTS[0.0])))

    seconds = time.perf_counter() - started
    proba_val = predict_proba(model, layers, E, TAB_VAL, IDS_VAL)
    metrics = ml.classification_metrics(
        task.y_val, (proba_val.ravel() >= 0.5).astype(int), proba_val.ravel()
    )
    n_params = sum(W.size + b.size for W, b in layers) + (E.size if model.use_text else 0)
    return {
        "model": model, "layers": layers, "E": E, "seconds": seconds,
        "history_train": history_train, "history_val": history_val,
        "metrics": metrics, "d_in": d_in, "n_params": int(n_params),
    }


def predict_proba(model, layers, E, tab, ids):
    X, _, _, _ = assemble(model, tab, ids, E)
    return dense_forward(layers, X)[0]"""),
        nbf.v4.new_code_cell("""trained = {}
for model in MODELS:
    trained[model.key] = fit(model)
    r = trained[model.key]
    m = r["metrics"]
    print(f"[{r['seconds']:6.1f}s] {model.key} {model.label:<24} d_in={r['d_in']:<4d} "
          f"tham số={r['n_params']:>7,d} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")"""),
        nbf.v4.new_code_cell("""pd.DataFrame([
    {
        "mô hình": f"{r['model'].key} {r['model'].label}",
        "d đầu vào": r["d_in"],
        "tham số": f"{r['n_params']:,}",
        "loss cuối (val)": f"{r['history_val'][-1]:.4f}",
        "accuracy": f"{r['metrics']['accuracy']:.4f}",
        "recall": f"{r['metrics']['recall']:.4f}",
        "f1_macro": f"{r['metrics']['f1_macro']:.4f}",
        "roc_auc": f"{r['metrics']['roc_auc']:.4f}",
        "giây": f"{r['seconds']:.1f}",
    }
    for r in trained.values()
])"""),
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
styles = ["-", "--", "-.", ":"]

for (key, result), style in zip(trained.items(), styles):
    label = f"{key} {result['model'].label}"
    axes[0].plot(result["history_train"], style, color="black", linewidth=1.1, label=label)
    axes[1].plot(result["history_val"], style, color="black", linewidth=1.1, label=label)

axes[0].set_title("Mất mát trên tập train")
axes[1].set_title("Mất mát trên tập validation")
for axis in axes:
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary cross entropy có trọng số lớp")
    axis.grid(True, linewidth=0.3, alpha=0.4)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("""## 6. Văn bản đóng góp bao nhiêu, và học nhúng đóng góp bao nhiêu"""),
        nbf.v4.new_code_cell("""gain_text = trained["N2"]["metrics"]["f1_macro"] - trained["N1"]["metrics"]["f1_macro"]
gain_learn = trained["N2"]["metrics"]["f1_macro"] - trained["N3"]["metrics"]["f1_macro"]

print(f"N1 chỉ bảng F1-macro = {trained['N1']['metrics']['f1_macro']:.4f}")
print(f"N3 bảng + nhúng đóng băng F1-macro = {trained['N3']['metrics']['f1_macro']:.4f}")
print(f"N2 bảng + nhúng học được F1-macro = {trained['N2']['metrics']['f1_macro']:.4f}")
print(f"N4 chỉ văn bản F1-macro = {trained['N4']['metrics']['f1_macro']:.4f}")
print()
print(f"Thêm văn bản (N2 - N1) : {gain_text:+.4f} F1-macro")
print(f"Học ma trận nhúng (N2 - N3) : {gain_learn:+.4f} F1-macro")
print(f"Notebook 03, TF-IDF (tham chiếu): +0.1431 F1-macro trên bộ dữ liệu cũ")"""),
        nbf.v4.new_markdown_cell("""## 7. Ma trận nhúng đã học được gì"""),
        nbf.v4.new_code_cell("""E_learned = trained["N2"]["E"]
index_to_word = {index: word for word, index in vocab.items()}
norms = np.linalg.norm(E_learned, axis=1, keepdims=True)
E_unit = E_learned / np.maximum(norms, 1e-8)


def nearest(word, top=6):
    if word not in vocab:
        return f"{word}: không có trong từ điển"
    similarity = E_unit @ E_unit[vocab[word]]
    similarity[0] = -np.inf
    similarity[vocab[word]] = -np.inf
    best = np.argsort(similarity)[::-1][:top]
    return f"{word:<12} -> " + ", ".join(
        f"{index_to_word[i]} ({similarity[i]:.2f})" for i in best if i in index_to_word
    )


for word in ["good", "bad", "worst", "thanks", "refund", "delay"]:
    print(nearest(word))"""),
        nbf.v4.new_markdown_cell("""## 8. So sánh với mô hình machine learning tốt nhất"""),
        nbf.v4.new_code_cell("""best_key = max(trained, key=lambda k: trained[k]["metrics"]["f1_macro"])
best = trained[best_key]
print(f"Mô hình deep learning tốt nhất trên validation: {best_key} {best['model'].label}")

baseline = ml.run_model(task, task.baseline_name + " (mốc ML)", task.baseline_rep,
                        task.baseline_estimator())
ml.print_table([ml.table_row(task, baseline)])"""),
        nbf.v4.new_code_cell("""proba_test = predict_proba(best["model"], best["layers"], best["E"], TAB_TEST, IDS_TEST).ravel()
dl_test = ml.classification_metrics(
    task.y_test, (proba_test >= 0.5).astype(int), proba_test
)
ml_test = ml.evaluate_on_test(task, baseline["pipe"])

pd.DataFrame([
    {"mô hình": f"Mạng nơ-ron {best_key} ({best['model'].label})",
     "accuracy": f"{dl_test['accuracy']:.4f}", "recall": f"{dl_test['recall']:.4f}",
     "f1": f"{dl_test['f1']:.4f}", "f1_macro": f"{dl_test['f1_macro']:.4f}",
     "roc_auc": f"{dl_test['roc_auc']:.4f}"},
    {"mô hình": task.baseline_name + " (mốc ML)",
     "accuracy": f"{ml_test['accuracy']:.4f}", "recall": f"{ml_test['recall']:.4f}",
     "f1": f"{ml_test['f1']:.4f}", "f1_macro": f"{ml_test['f1_macro']:.4f}",
     "roc_auc": f"{ml_test['roc_auc']:.4f}"},
])"""),
        nbf.v4.new_markdown_cell("""## 9. Ghi kết quả"""),
        nbf.v4.new_code_cell("""payload = {
    "task": task.key,
    "title": task.title,
    "n_train": task.n_train,
    "n_val": task.n_val,
    "n_test": task.n_test,
    "audit": task.audit,
    "vocab_size": len(vocab),
    "T_pad": T_PAD,
    "d_embed": D_EMBED,
    "hidden": list(HIDDEN),
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "init_seed": INIT_SEED,
    "shuffle_seed": SHUFFLE_SEED,
    "class_weights": {"positive": WEIGHTS[1.0], "negative": WEIGHTS[0.0]},
    "models": [
        {
            "key": r["model"].key,
            "label": r["model"].label,
            "use_tabular": r["model"].use_tabular,
            "use_text": r["model"].use_text,
            "train_embedding": r["model"].train_embedding,
            "d_in": r["d_in"],
            "n_params": r["n_params"],
            "seconds": round(r["seconds"], 2),
            "loss_train_final": round(r["history_train"][-1], 6),
            "loss_val_final": round(r["history_val"][-1], 6),
            "loss_history_train": [round(v, 6) for v in r["history_train"]],
            "loss_history_val": [round(v, 6) for v in r["history_val"]],
            "validation": {k: (round(v, 6) if v is not None else None)
                           for k, v in r["metrics"].items()},
        }
        for r in trained.values()
    ],
    "best_model": best_key,
    "gain_text_f1_macro": round(gain_text, 6),
    "gain_learned_embedding_f1_macro": round(gain_learn, 6),
    "test_metrics_dl": {k: (round(v, 6) if v is not None else None) for k, v in dl_test.items()},
    "test_metrics_ml_baseline": {k: (round(v, 6) if v is not None else None)
                                 for k, v in ml_test.items()},
    "ml_baseline_name": task.baseline_name,
}

path = ml.save_results("deeplearning_ecommerce", payload)
print("Đã ghi", path.relative_to(ROOT), f"({path.stat().st_size / 1024:.1f} KB)")"""),
    ]
    nb = create_nb(cells)
    nbf.write(nb, NOTEBOOKS / "08_deep_learning_ecommerce.ipynb")
    print("Created notebooks/08_deep_learning_ecommerce.ipynb")


if __name__ == "__main__":
    build_nb04()
    build_nb05()
    build_nb06()
    build_nb07()
    build_nb08()
    print("All 5 notebooks created successfully!")
