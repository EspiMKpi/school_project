"""Notebook 07 / Script 07: Mạng nơ-ron hồi quy viết tay cho bài toán giá nhà
Tái lập toàn bộ logic từ Phụ lục D của báo cáo A3_02_HungNguyenBa_120.pdf.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
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


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def identity(x):
    return x


def identity_derivative(x):
    return np.ones_like(x)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "linear": (identity, identity_derivative),
}


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
    """Lớp cuối KHÔNG có hàm kích hoạt: đầu ra hồi quy là số thực bất kỳ."""
    act, _ = ACTIVATIONS[hidden_activation]
    pre_activations, activations = [], [X]
    h = X
    last = len(params) - 1
    for index, (W, b) in enumerate(params):
        z = h @ W + b
        h = z if index == last else act(z)
        pre_activations.append(z)
        activations.append(h)
    return h, {"z": pre_activations, "h": activations}


def mean_squared_error(z_true, z_pred):
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
    )


def main():
    print("=== Bước 1: Dữ liệu và biểu diễn đầu vào ===")
    task = ml.load_house_price()
    prep = task.prep("target_enc_scaled")

    X_train = prep.fit_transform(task.X_train, task.y_train)
    X_val = prep.transform(task.X_val)
    X_test = prep.transform(task.X_test)

    print(f"X_train {X_train.shape} X_val {X_val.shape} X_test {X_test.shape}")
    print(f"trung bình mỗi chiều sau chuẩn hoá: {np.round(X_train.mean(axis=0), 3)}")
    print(f"độ lệch chuẩn mỗi chiều : {np.round(X_train.std(axis=0), 3)}")

    y_mean = float(task.y_train.mean())
    y_std = float(task.y_train.std())

    z_train = ((task.y_train - y_mean) / y_std).reshape(-1, 1)
    z_val = ((task.y_val - y_mean) / y_std).reshape(-1, 1)

    def to_log_price(z):
        """Quy dự đoán đã chuẩn hoá về lại log(price) để tính chỉ số nghiệp vụ."""
        return z.ravel() * y_std + y_mean

    print(f"\nlog(price): trung bình = {y_mean:.4f}, độ lệch chuẩn = {y_std:.4f}")
    print(f"sau chuẩn hoá: trung bình = {z_train.mean():.6f}, độ lệch chuẩn = {z_train.std():.6f}")

    print("\n=== Bước 2: Huấn luyện bốn kiến trúc ===")
    LEARNING_RATE = 0.05
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

    def fit_nn(arch):
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
        trained[arch.key] = fit_nn(arch)
        m = trained[arch.key]["metrics"]
        print(f"[{trained[arch.key]['seconds']:6.1f}s] {arch.key} {arch.label:<9} {arch.shape:<14} "
              f"{arch.n_params:>5d} tham số R2_log={m['r2_log']:.4f} MAE={m['mae_usd']:,.0f} USD")

    res_df = pd.DataFrame([
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
    ])
    print("\nBảng tổng hợp validation:")
    print(res_df.to_string(index=False))

    # Figure nb07_0.png
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
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
    fig.savefig(FIGS / "nb07_0.png", dpi=300)
    plt.close(fig)

    print("\n=== Bước 3: So sánh với mô hình ML mốc (HistGradientBoosting) ===")
    best_key = max(trained, key=lambda k: trained[k]["metrics"]["r2_log"])
    best = trained[best_key]
    best_arch = best["arch"]
    print(f"Kiến trúc tốt nhất trên validation: {best_key} {best_arch.label} ({best_arch.shape})")

    baseline = ml.run_model(task, task.baseline_name + " (mốc ML)", task.baseline_rep,
                            task.baseline_estimator())
    ml.print_table([ml.table_row(task, baseline)])

    test_log = to_log_price(forward(best["params"], X_test, best_arch.hidden_activation)[0])
    dl_test = ml.regression_metrics(task.y_test, test_log)
    ml_test = ml.evaluate_on_test(task, baseline["pipe"])

    comp_df = pd.DataFrame([
        {"mô hình": f"Mạng nơ-ron {best_key} ({best_arch.shape})", **{
            "R2_log": f"{dl_test['r2_log']:.4f}", "MAE_USD": f"{dl_test['mae_usd']:,.0f}",
            "RMSE_USD": f"{dl_test['rmse_usd']:,.0f}", "MAPE_%": f"{dl_test['mape_percent']:.1f}"}},
        {"mô hình": task.baseline_name + " (mốc ML)", **{
            "R2_log": f"{ml_test['r2_log']:.4f}", "MAE_USD": f"{ml_test['mae_usd']:,.0f}",
            "RMSE_USD": f"{ml_test['rmse_usd']:,.0f}", "MAPE_%": f"{ml_test['mape_percent']:.1f}"}},
    ])
    print("\nSo sánh trên tập test:")
    print(comp_df.to_string(index=False))

    # Figure nb07_1.png
    actual_usd = np.exp(task.y_test)
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
    fig.savefig(FIGS / "nb07_1.png", dpi=300)
    plt.close(fig)

    # Save payload
    payload = {
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
    print("\nĐã ghi", path.relative_to(ROOT), f"({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
