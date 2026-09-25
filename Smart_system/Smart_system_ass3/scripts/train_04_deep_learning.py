"""Notebook 04 / Script 04: Deep Learning từ đầu với NumPy (Diabetes Binary Classification)
Tái lập toàn bộ logic từ Phụ lục A của báo cáo A3_02_HungNguyenBa_120.pdf.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Handle Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name in ("scripts", "notebooks") else Path.cwd()
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figs"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

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


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def identity(x):
    return x


def identity_derivative(x):
    return np.ones_like(x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "linear": (identity, identity_derivative),
}


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


EPS = 1e-8


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
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def n_parameters(sizes):
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def fit(arch, X_tr, y_tr, X_te, y_te, epochs, learning_rate, batch_size, init_seed=42, shuffle_seed=7, thresholds=(0.3, 0.5, 0.7)):
    params = init_params(arch["sizes"], init_seed)
    rng = np.random.default_rng(shuffle_seed)
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
        "test_by_threshold": {str(t): rounded(score(y_te, test_prob, t)) for t in thresholds},
        "hidden_shapes": [list(h.shape) for h in cache_tr["h"][1:-1]],
    }


def main():
    parser = argparse.ArgumentParser(description="Train Handmade NumPy DL for Diabetes")
    parser.add_argument("--skip-lr-study", action="store_true", help="Skip the slow full-batch learning rate study (1000 epochs)")
    args = parser.parse_args()

    print("=== Bước 1: Nạp dữ liệu và chọn 8 đặc trưng ===")
    DIABETES_CSV = DATA / "diabetes" / "diabetes.csv"
    if not DIABETES_CSV.exists():
        print(f"Lỗi: Không tìm thấy tệp {DIABETES_CSV}")
        return

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
    print("\nXếp hạng |tương quan| với nhãn nhị phân:")
    print(ranking.round(4).to_string())
    print("\n8 đặc trưng được chọn:", list(FEATURES))

    X_all = frame[list(FEATURES)].to_numpy(dtype=np.float64)
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

    # Plot Figure 1 (nb04_0.png)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    counts = frame[TARGET].value_counts().sort_index()
    axes[0].bar([str(int(k)) for k in counts.index], counts.values, color="white", edgecolor="black", hatch="///")
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
    fig.savefig(FIGS / "nb04_0.png", dpi=300)
    plt.close(fig)

    print("\n=== Bước 2: Huấn luyện bốn kiến trúc ===")
    LEARNING_RATE = 0.05
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

    trained = {}
    models = []
    for arch in ARCHITECTURES:
        params, row = fit(arch, X_train, y_train, X_test, y_test,
                          EPOCHS, LEARNING_RATE, BATCH_SIZE, INIT_SEED, SHUFFLE_SEED, THRESHOLDS)
        trained[arch["key"]] = params
        models.append(row)
        t = row["test_by_threshold"]["0.5"]
        print(f"{row['key']} {row['label']:9s} {row['shape']:18s} "
              f"loss {row['loss_final']:.4f} acc {t['accuracy']:.4f} "
              f"prec {t['precision']:.4f} rec {t['recall']:.4f} f1 {t['f1']:.4f} "
              f"auc {row['roc_auc_test']:.4f} ({row['seconds']:.1f}s)")

    table = pd.DataFrame([
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
    print("\nKết quả trên tập test, ngưỡng 0.5:")
    print(table.to_string(index=False))

    # Figure nb04_1.png (loss history)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
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
    fig.savefig(FIGS / "nb04_1.png", dpi=300)
    plt.close(fig)

    # Figure nb04_2.png (precision-recall curve and metric comparison)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    markers = ["o", "s", "^", "D"]
    for m, marker in zip(models, markers):
        rows = [m["test_by_threshold"][str(t)] for t in THRESHOLDS]
        axes[0].plot([r["recall"] for r in rows], [r["precision"] for r in rows], marker + "-",
                     color="black", markerfacecolor="white", linewidth=1.0, label=f"{m['key']} {m['label']}")
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
    axes[1].set_xticklabels([f"{m['key']}\n{m['label']}" for m in models], fontsize=8)
    axes[1].set_ylim(0, 1.16)
    axes[1].set_title("So sánh bốn mô hình trên tập test")
    axes[1].legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    axes[1].grid(True, axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "nb04_2.png", dpi=300)
    plt.close(fig)

    lr_study = []
    if not args.skip_lr_study:
        print("\n=== Bước 3: Nghiên cứu học suất GD toàn tập (1000 epoch) ===")
        SPEC_EPOCHS = 1000
        SPEC_RATES = (0.001, 0.01, 0.1)
        lr_curves = {}
        for rate in SPEC_RATES:
            _, row = fit(ARCHITECTURES[0], X_train, y_train, X_test, y_test, SPEC_EPOCHS, rate, None, INIT_SEED, SHUFFLE_SEED, THRESHOLDS)
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

        fig, axis = plt.subplots(figsize=(6.6, 3.4))
        for rate, style in zip(SPEC_RATES, ["-", "--", "-."]):
            axis.plot(lr_curves[rate], style, color="black", linewidth=1.2, label=f"toàn tập, lr = {rate}")
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
        fig.savefig(FIGS / "nb04_3.png", dpi=300)
        plt.close(fig)

    payload = {
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
    print(f"\nĐã ghi kết quả ra {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.1f} KB)")
    print("Mô hình tốt nhất theo ROC-AUC:", max(payload["models"], key=lambda m: m["roc_auc_test"])["key"])


if __name__ == "__main__":
    main()
