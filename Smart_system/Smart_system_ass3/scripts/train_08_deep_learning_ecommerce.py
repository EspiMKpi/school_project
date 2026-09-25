"""Notebook 08 / Script 08: Mạng nơ-ron có lớp nhúng học được cho bài toán e-commerce
Tái lập toàn bộ logic từ Phụ lục E của báo cáo A3_02_HungNguyenBa_120.pdf.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Handle Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name in ("scripts", "notebooks") else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml

warnings.filterwarnings("ignore")
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

VOCAB_SIZE = 5000
T_PAD = 32
D_EMBED = 32


def sigmoid(x):
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
    """Hàng 0 là token đệm, luôn giữ bằng 0 để nó không đóng góp gì vào phép gộp."""
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
    """Đưa gradient của vector gộp về đúng các hàng từ vựng đã tham gia."""
    dE = np.zeros((VOCAB_SIZE + 1, D_EMBED))
    rows, columns = np.nonzero(mask)
    np.add.at(dE, ids[rows, columns], d_pooled[rows] / counts[rows])
    dE[0] = 0.0
    return dE


def class_weights(y):
    n = len(y)
    positive = float(y.sum())
    return {1.0: n / (2.0 * positive), 0.0: n / (2.0 * (n - positive))}


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
    """Trả về gradient của từng lớp VÀ gradient theo đầu vào, để đẩy tiếp vào lớp nhúng."""
    z, h = cache["z"], cache["h"]
    n = len(y)
    grads = []
    delta = weight * (h[-1] - y) / n
    for index in range(len(layers) - 1, -1, -1):
        grads.append((h[index].T @ delta, np.sum(delta, axis=0, keepdims=True)))
        delta = delta @ layers[index][0].T
        if index:
            delta = delta * relu_derivative(z[index - 1])
    return tuple(reversed(grads)), delta


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
    """Ghép đầu vào của mạng theo đúng cấu hình của từng mô hình."""
    if not model.use_text:
        return tab, None, None, None
    pooled, mask, counts = pool(E, ids)
    X = np.hstack([tab, pooled]) if model.use_tabular else pooled
    return X, mask, counts, pooled


def predict_proba(model, layers, E, tab, ids):
    X, _, _, _ = assemble(model, tab, ids, E)
    return dense_forward(layers, X)[0]


def main():
    print("=== Bước 1: Chuẩn bị dữ liệu và tiền xử lý ===")
    task = ml.load_ecommerce_satisfaction()
    prep_tab = task.prep("tabular")

    TAB_TRAIN = np.asarray(prep_tab.fit_transform(task.X_train, task.y_train).toarray())
    TAB_VAL = np.asarray(prep_tab.transform(task.X_val).toarray())
    TAB_TEST = np.asarray(prep_tab.transform(task.X_test).toarray())

    y_train = np.asarray(task.y_train).reshape(-1, 1).astype(float)
    y_val = np.asarray(task.y_val).reshape(-1, 1).astype(float)
    y_test = np.asarray(task.y_test).reshape(-1, 1).astype(float)

    print(f"bảng: train {TAB_TRAIN.shape} val {TAB_VAL.shape} test {TAB_TEST.shape}")
    print(f"tỉ lệ hài lòng: train={y_train.mean():.4f} val={y_val.mean():.4f} test={y_test.mean():.4f}")

    print("\n=== Bước 2: Xây từ điển và chuỗi id ===")
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

    WEIGHTS = class_weights(y_train)
    W_TRAIN = np.where(y_train == 1.0, WEIGHTS[1.0], WEIGHTS[0.0])
    print(f"trọng số lớp: hài lòng = {WEIGHTS[1.0]:.4f}, không hài lòng = {WEIGHTS[0.0]:.4f}")

    print("\n=== Bước 3: Huấn luyện bốn cấu hình ===")
    LEARNING_RATE = 0.1
    EPOCHS = 40
    BATCH_SIZE = 256
    INIT_SEED = 42
    SHUFFLE_SEED = 7
    HIDDEN = (32, 16)

    def fit_model(model: Model):
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

    trained = {}
    for model in MODELS:
        trained[model.key] = fit_model(model)
        r = trained[model.key]
        m = r["metrics"]
        print(f"[{r['seconds']:6.1f}s] {model.key} {model.label:<24} d_in={r['d_in']:<4d} "
              f"tham số={r['n_params']:>7,d} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")

    res_df = pd.DataFrame([
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
    ])
    print("\nTổng hợp validation:")
    print(res_df.to_string(index=False))

    # Figure nb08_0.png
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
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
    fig.savefig(FIGS / "nb08_0.png", dpi=300)
    plt.close(fig)

    gain_text = trained["N2"]["metrics"]["f1_macro"] - trained["N1"]["metrics"]["f1_macro"]
    gain_learn = trained["N2"]["metrics"]["f1_macro"] - trained["N3"]["metrics"]["f1_macro"]

    print(f"\nThêm văn bản (N2 - N1) : {gain_text:+.4f} F1-macro")
    print(f"Học ma trận nhúng (N2 - N3) : {gain_learn:+.4f} F1-macro")

    # Semantic word neighbors inspection
    print("\n=== Bước 4: Kiểm tra ngữ nghĩa ma trận nhúng đã học ===")
    E_learned = trained["N2"]["E"]
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
        print(nearest(word))

    # Baseline comparison
    print("\n=== Bước 5: So sánh với mô hình ML mốc ===")
    best_key = max(trained, key=lambda k: trained[k]["metrics"]["f1_macro"])
    best = trained[best_key]
    print(f"Mô hình deep learning tốt nhất trên validation: {best_key} {best['model'].label}")

    baseline = ml.run_model(task, task.baseline_name + " (mốc ML)", task.baseline_rep,
                            task.baseline_estimator())
    ml.print_table([ml.table_row(task, baseline)])

    proba_test = predict_proba(best["model"], best["layers"], best["E"], TAB_TEST, IDS_TEST).ravel()
    dl_test = ml.classification_metrics(
        task.y_test, (proba_test >= 0.5).astype(int), proba_test
    )
    ml_test = ml.evaluate_on_test(task, baseline["pipe"])

    comp_df = pd.DataFrame([
        {"mô hình": f"Mạng nơ-ron {best_key} ({best['model'].label})",
         "accuracy": f"{dl_test['accuracy']:.4f}", "recall": f"{dl_test['recall']:.4f}",
         "f1": f"{dl_test['f1']:.4f}", "f1_macro": f"{dl_test['f1_macro']:.4f}",
         "roc_auc": f"{dl_test['roc_auc']:.4f}"},
        {"mô hình": task.baseline_name + " (mốc ML)",
         "accuracy": f"{ml_test['accuracy']:.4f}", "recall": f"{ml_test['recall']:.4f}",
         "f1": f"{ml_test['f1']:.4f}", "f1_macro": f"{ml_test['f1_macro']:.4f}",
         "roc_auc": f"{ml_test['roc_auc']:.4f}"},
    ])
    print("\nSo sánh trên tập test:")
    print(comp_df.to_string(index=False))

    payload = {
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
                "validation": {k: (round(v, 6) if v is not None else None) for k, v in r["metrics"].items()},
            }
            for r in trained.values()
        ],
        "best_model": best_key,
        "gain_text_f1_macro": round(gain_text, 6),
        "gain_learned_embedding_f1_macro": round(gain_learn, 6),
        "test_metrics_dl": {k: (round(v, 6) if v is not None else None) for k, v in dl_test.items()},
        "test_metrics_ml_baseline": {k: (round(v, 6) if v is not None else None) for k, v in ml_test.items()},
        "ml_baseline_name": task.baseline_name,
    }

    path = ml.save_results("deeplearning_ecommerce", payload)
    print("\nĐã ghi", path.relative_to(ROOT), f"({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
