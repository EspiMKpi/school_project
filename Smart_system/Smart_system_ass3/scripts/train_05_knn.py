"""Notebook 05 / Script 05: K-Nearest Neighbors trên ba category (bốn bài toán)
Tái lập toàn bộ logic từ Phụ lục B của báo cáo A3_02_HungNguyenBa_120.pdf.
"""

from __future__ import annotations

import sys
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

# Handle Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name in ("scripts", "notebooks") else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)


def sweep_k(task, rep, k_grid, metric="euclidean"):
    """Quét lưới k bằng MỘT lần tìm hàng xóm, trả về bảng kết quả trên tập validation."""
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
    print(f"\n-> k tốt nhất trên validation: {best_k}\n")


def block(task, results, test_metrics, sweep_rows, best_k, metric):
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


def main():
    print("=== Nạp bốn bài toán ===")
    t0 = time.perf_counter()
    TASKS = {key: loader() for key, loader in ml.LOADERS.items()}
    print(f"Nạp bốn bài toán trong {time.perf_counter() - t0:.1f}s\n")
    for task in TASKS.values():
        print(task.summary())

    # --- Task 1: Diabetes ---
    print("\n--- Bài toán 1: Tiểu đường (phân loại nhị phân) ---")
    task_dia = TASKS["diabetes"]
    rows_dia, k_dia = sweep_k(task_dia, "native", [5, 11, 25, 51, 101])
    show(rows_dia, k_dia)

    results_dia = [
        ml.run_model(task_dia, f"KNN (k={k_dia})", "native",
                     KNeighborsClassifier(n_neighbors=k_dia, algorithm="brute", n_jobs=-1)),
        ml.run_model(task_dia, task_dia.baseline_name + " (mốc)", task_dia.baseline_rep,
                     task_dia.baseline_estimator()),
    ]
    ml.print_table([ml.table_row(task_dia, r) for r in results_dia])
    test_dia = {r["name"]: ml.evaluate_on_test(task_dia, r["pipe"]) for r in results_dia}
    for name, m in test_dia.items():
        print(f"TEST {name:<34} accuracy={m['accuracy']:.4f} recall={m['recall']:.4f} "
              f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")

    # --- Task 2: House Price ---
    print("\n--- Bài toán 2: Giá nhà (hồi quy) ---")
    task_house = TASKS["house_price"]
    rows_house, k_house = sweep_k(task_house, "target_enc_scaled", [5, 11, 25, 51])
    show(rows_house, k_house)

    results_house = [
        ml.run_model(task_house, f"KNN (k={k_house})", "target_enc_scaled",
                     KNeighborsRegressor(n_neighbors=k_house, algorithm="brute", n_jobs=-1)),
        ml.run_model(task_house, task_house.baseline_name + " (mốc)", task_house.baseline_rep,
                     task_house.baseline_estimator()),
    ]
    ml.print_table([ml.table_row(task_house, r) for r in results_house])
    test_house = {r["name"]: ml.evaluate_on_test(task_house, r["pipe"]) for r in results_house}
    for name, m in test_house.items():
        print(f"TEST {name:<34} R2_log={m['r2_log']:.4f} MAE={m['mae_usd']:,.0f} USD "
              f"MAPE={m['mape_percent']:.1f}%")

    # --- Task 3: Ecommerce Satisfaction ---
    print("\n--- Bài toán 3: E-commerce mức hài lòng ---")
    task_sat = TASKS["ecommerce_satisfaction"]
    rows_sat, k_sat = sweep_k(task_sat, "tabular_text", [5, 11, 25, 51], metric="cosine")
    show(rows_sat, k_sat)

    results_sat = [
        ml.run_model(task_sat, f"KNN cosin (k={k_sat})", "tabular_text",
                     KNeighborsClassifier(n_neighbors=k_sat, metric="cosine", algorithm="brute", n_jobs=-1)),
        ml.run_model(task_sat, task_sat.baseline_name + " (mốc)", task_sat.baseline_rep,
                     task_sat.baseline_estimator()),
    ]
    ml.print_table([ml.table_row(task_sat, r) for r in results_sat])
    test_sat = {r["name"]: ml.evaluate_on_test(task_sat, r["pipe"]) for r in results_sat}
    for name, m in test_sat.items():
        print(f"TEST {name:<34} accuracy={m['accuracy']:.4f} recall={m['recall']:.4f} "
              f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")

    # --- Task 4: Ecommerce Interest ---
    print("\n--- Bài toán 4: E-commerce nhóm sản phẩm quan tâm (9 lớp) ---")
    task_int = TASKS["ecommerce_interest"]
    rows_int, k_int = sweep_k(task_int, "tabular_text", [5, 11, 25, 51], metric="cosine")
    show(rows_int, k_int)

    results_int = [
        ml.run_model(task_int, f"KNN cosin (k={k_int})", "tabular_text",
                     KNeighborsClassifier(n_neighbors=k_int, metric="cosine", algorithm="brute", n_jobs=-1)),
        ml.run_model(task_int, task_int.baseline_name + " (mốc)", task_int.baseline_rep,
                     task_int.baseline_estimator()),
    ]
    ml.print_table([ml.table_row(task_int, r) for r in results_int])
    test_int = {r["name"]: ml.evaluate_on_test(task_int, r["pipe"]) for r in results_int}
    for name, m in test_int.items():
        print(f"TEST {name:<34} accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
              f"f1_weighted={m['f1_weighted']:.4f}")

    # --- Save Payload ---
    payload = {
        "algorithm": "K-Nearest Neighbors",
        "library": "scikit-learn",
        "seed": ml.RANDOM_SEED,
        "note": "Mô hình mốc được huấn luyện lại trên bộ dữ liệu hiện tại trong data/, không chép lại số của notebook 01-03.",
        "tasks": [
            block(task_dia, results_dia, test_dia, rows_dia, k_dia, "euclidean"),
            block(task_house, results_house, test_house, rows_house, k_house, "euclidean"),
            block(task_sat, results_sat, test_sat, rows_sat, k_sat, "cosine"),
            block(task_int, results_int, test_int, rows_int, k_int, "cosine"),
        ],
    }
    path = ml.save_results("knn", payload)
    print("\nĐã ghi", path.relative_to(ROOT), f"({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
