"""Chạy một mô hình trên một `Task` rồi gom kết quả về dạng thống nhất.

Cả notebook 05 và 06 đều gọi đúng hàm này, nên thời gian đo, số chiều d và tên
khoá metric của hai notebook so sánh được với nhau mà không cần đối chiếu tay.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from .datasets import ROOT, Task
from .metrics import classification_metrics, regression_metrics

REPORTS = ROOT / "reports"


def run_model(
    task: Task, name: str, rep: str, estimator: Any, *, sample_weight: Any = None
) -> dict[str, Any]:
    """Fit trên tập train, đo trên tập validation. Tập test không bị chạm tới ở đây.

    `sample_weight` dành cho các mô hình không có tham số `class_weight` riêng, ví dụ
    AdaBoost: trọng số cân bằng lớp được truyền thẳng vào lời gọi `fit`.
    """
    pipe = Pipeline([("prep", task.prep(rep)), ("model", estimator)])
    fit_params = {} if sample_weight is None else {"model__sample_weight": sample_weight}

    started = time.perf_counter()
    pipe.fit(task.X_train, task.y_train, **fit_params)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prediction = pipe.predict(task.X_val)
    predict_seconds = time.perf_counter() - started

    d = int(pipe.named_steps["prep"].transform(task.X_train.head(50)).shape[1])

    if task.kind == "regression":
        metrics = regression_metrics(task.y_val, prediction)
    else:
        score = None
        if task.average == "binary" and hasattr(pipe.named_steps["model"], "predict_proba"):
            score = pipe.predict_proba(task.X_val)[:, 1]
        metrics = classification_metrics(
            task.y_val, prediction, score, average=task.average
        )

    return {
        "name": name,
        "rep": rep,
        "d": d,
        "metrics": metrics,
        "fit_seconds": round(fit_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
        "pipe": pipe,
    }


def evaluate_on_test(task: Task, pipe: Pipeline) -> dict[str, Any]:
    prediction = pipe.predict(task.X_test)
    if task.kind == "regression":
        return regression_metrics(task.y_test, prediction)

    score = None
    if task.average == "binary" and hasattr(pipe.named_steps["model"], "predict_proba"):
        score = pipe.predict_proba(task.X_test)[:, 1]
    return classification_metrics(task.y_test, prediction, score, average=task.average)


def classification_row(result: dict[str, Any]) -> dict[str, str]:
    m = result["metrics"]
    auc = m.get("roc_auc")
    return {
        "model": result["name"],
        "d": f"{result['d']:,}",
        "accuracy": f"{m['accuracy']:.4f}",
        "precision": f"{m['precision']:.4f}",
        "recall": f"{m['recall']:.4f}",
        "f1_macro": f"{m['f1_macro']:.4f}",
        "roc_auc": f"{auc:.4f}" if auc is not None else "-",
        "fit_s": f"{result['fit_seconds']:.1f}",
        "predict_s": f"{result['predict_seconds']:.1f}",
    }


def regression_row(result: dict[str, Any]) -> dict[str, str]:
    m = result["metrics"]
    return {
        "model": result["name"],
        "d": f"{result['d']:,}",
        "R2_log": f"{m['r2_log']:.4f}",
        "MAE_USD": f"{m['mae_usd']:,.0f}",
        "RMSE_USD": f"{m['rmse_usd']:,.0f}",
        "R2_USD": f"{m['r2_usd']:.4f}",
        "MAPE_%": f"{m['mape_percent']:.1f}",
        "fit_s": f"{result['fit_seconds']:.1f}",
        "predict_s": f"{result['predict_seconds']:.1f}",
    }


def table_row(task: Task, result: dict[str, Any]) -> dict[str, str]:
    return regression_row(result) if task.kind == "regression" else classification_row(result)


def print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    line = " ".join(c.ljust(widths[c]) for c in columns)
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" ".join(str(row[c]).ljust(widths[c]) for c in columns))


def save_results(name: str, payload: dict[str, Any]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"{name}_results.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
