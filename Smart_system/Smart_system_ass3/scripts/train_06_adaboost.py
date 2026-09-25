"""Notebook 06 / Script 06: AdaBoost trên ba category (bốn bài toán)
Tái lập toàn bộ logic từ Phụ lục C của báo cáo A3_02_HungNguyenBa_120.pdf.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_sample_weight

# Handle Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name in ("scripts", "notebooks") else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ml

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)


def sweep_stages(task, rep, estimator, stage_grid, sample_weight=None):
    """Huấn luyện MỘT lần với số vòng lớn nhất, đọc kết quả tại từng mốc trong lưới."""
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
    print(f"\n-> số vòng tốt nhất trên validation: {best['vòng']}\n")
    return rows, int(best["vòng"])


def block(task, results, test_metrics, sweep_rows, best_stage, weak_learner):
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
    ada_dia = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
        n_estimators=300, learning_rate=0.5, random_state=ml.RANDOM_SEED,
    )
    weight_dia = compute_sample_weight("balanced", task_dia.y_train)
    rows_dia, m_dia = sweep_stages(task_dia, "native", ada_dia, [25, 50, 100, 200, 300], weight_dia)

    results_dia = [
        ml.run_model(task_dia, f"AdaBoost ({m_dia} vòng, cây sâu 3)", "native",
                     AdaBoostClassifier(
                         estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
                         n_estimators=m_dia, learning_rate=0.5, random_state=ml.RANDOM_SEED),
                     sample_weight=weight_dia),
        ml.run_model(task_dia, task_dia.baseline_name + " (mốc)", task_dia.baseline_rep,
                     task_dia.baseline_estimator()),
    ]
    ml.print_table([ml.table_row(task_dia, r) for r in results_dia])
    test_dia = {r["name"]: ml.evaluate_on_test(task_dia, r["pipe"]) for r in results_dia}
    for name, m in test_dia.items():
        print(f"TEST {name:<36} accuracy={m['accuracy']:.4f} recall={m['recall']:.4f} "
              f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")

    # --- Task 2: House Price ---
    print("\n--- Bài toán 2: Giá nhà (hồi quy, AdaBoost.R2) ---")
    task_house = TASKS["house_price"]
    ada_house = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=8, random_state=ml.RANDOM_SEED),
        n_estimators=100, learning_rate=0.5, random_state=ml.RANDOM_SEED,
    )
    rows_house, m_house = sweep_stages(task_house, "target_enc", ada_house, [10, 25, 50, 75, 100])

    results_house = [
        ml.run_model(task_house, f"AdaBoost.R2 ({m_house} vòng, cây sâu 8)", "target_enc",
                     AdaBoostRegressor(
                         estimator=DecisionTreeRegressor(max_depth=8, random_state=ml.RANDOM_SEED),
                         n_estimators=m_house, learning_rate=0.5, random_state=ml.RANDOM_SEED)),
        ml.run_model(task_house, task_house.baseline_name + " (mốc)", task_house.baseline_rep,
                     task_house.baseline_estimator()),
    ]
    ml.print_table([ml.table_row(task_house, r) for r in results_house])
    test_house = {r["name"]: ml.evaluate_on_test(task_house, r["pipe"]) for r in results_house}
    for name, m in test_house.items():
        print(f"TEST {name:<36} R2_log={m['r2_log']:.4f} MAE={m['mae_usd']:,.0f} USD "
              f"MAPE={m['mape_percent']:.1f}%")

    # --- Task 3: Ecommerce Satisfaction ---
    print("\n--- Bài toán 3: E-commerce mức hài lòng ---")
    task_sat = TASKS["ecommerce_satisfaction"]
    ada_sat = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
        n_estimators=300, learning_rate=0.5, random_state=ml.RANDOM_SEED,
    )
    weight_sat = compute_sample_weight("balanced", task_sat.y_train)
    rows_sat, m_sat = sweep_stages(task_sat, "tabular_text", ada_sat, [25, 50, 100, 200, 300], weight_sat)

    results_sat = [
        ml.run_model(task_sat, f"AdaBoost ({m_sat} vòng, cây sâu 3)", "tabular_text",
                     AdaBoostClassifier(
                         estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
                         n_estimators=m_sat, learning_rate=0.5, random_state=ml.RANDOM_SEED),
                     sample_weight=weight_sat),
        ml.run_model(task_sat, task_sat.baseline_name + " (mốc)", task_sat.baseline_rep,
                     task_sat.baseline_estimator()),
    ]
    ml.print_table([ml.table_row(task_sat, r) for r in results_sat])
    test_sat = {r["name"]: ml.evaluate_on_test(task_sat, r["pipe"]) for r in results_sat}
    for name, m in test_sat.items():
        print(f"TEST {name:<36} accuracy={m['accuracy']:.4f} recall={m['recall']:.4f} "
              f"f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f} auc={m['roc_auc']:.4f}")

    # --- Task 4: Ecommerce Interest ---
    print("\n--- Bài toán 4: E-commerce nhóm sản phẩm quan tâm (9 lớp) ---")
    task_int = TASKS["ecommerce_interest"]
    ada_int = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
        n_estimators=300, learning_rate=0.5, random_state=ml.RANDOM_SEED,
    )
    weight_int = compute_sample_weight("balanced", task_int.y_train)
    rows_int, m_int = sweep_stages(task_int, "tabular_text", ada_int, [25, 50, 100, 200, 300], weight_int)

    results_int = [
        ml.run_model(task_int, f"AdaBoost ({m_int} vòng, cây sâu 3)", "tabular_text",
                     AdaBoostClassifier(
                         estimator=DecisionTreeClassifier(max_depth=3, random_state=ml.RANDOM_SEED),
                         n_estimators=m_int, learning_rate=0.5, random_state=ml.RANDOM_SEED),
                     sample_weight=weight_int),
        ml.run_model(task_int, task_int.baseline_name + " (mốc)", task_int.baseline_rep,
                     task_int.baseline_estimator()),
    ]
    ml.print_table([ml.table_row(task_int, r) for r in results_int])
    test_int = {r["name"]: ml.evaluate_on_test(task_int, r["pipe"]) for r in results_int}
    for name, m in test_int.items():
        print(f"TEST {name:<36} accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
              f"f1_weighted={m['f1_weighted']:.4f}")

    # --- Save Payload ---
    payload = {
        "algorithm": "AdaBoost (SAMME cho phân loại, AdaBoost.R2 cho hồi quy)",
        "library": "scikit-learn",
        "seed": ml.RANDOM_SEED,
        "learning_rate": 0.5,
        "note": "Mô hình mốc được huấn luyện lại trên bộ dữ liệu hiện tại trong data/, không chép lại số của notebook 01-03.",
        "tasks": [
            block(task_dia, results_dia, test_dia, rows_dia, m_dia,
                  "DecisionTreeClassifier(max_depth=3) + sample_weight cân bằng lớp"),
            block(task_house, results_house, test_house, rows_house, m_house,
                  "DecisionTreeRegressor(max_depth=8)"),
            block(task_sat, results_sat, test_sat, rows_sat, m_sat,
                  "DecisionTreeClassifier(max_depth=3) + sample_weight cân bằng lớp"),
            block(task_int, results_int, test_int, rows_int, m_int,
                  "DecisionTreeClassifier(max_depth=3) + sample_weight cân bằng lớp"),
        ],
    }
    path = ml.save_results("adaboost", payload)
    print("\nĐã ghi", path.relative_to(ROOT), f"({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
