"""Chỉ số đánh giá cho bài toán phân loại và hồi quy.

Dùng scikit-learn cho phần tính toán, lớp bọc ở đây chỉ gom kết quả về một dict
phẳng để bảng so sánh và tệp JSON kết quả dùng chung một tên khoá.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true, y_pred, y_score=None, *, average: str = "binary"
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None,
    }
    if y_score is not None and average == "binary":
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def regression_metrics(y_true_log, y_pred_log) -> dict[str, float]:
    """Đo trên cả hai không gian: log(price) để so mô hình, USD để đọc nghiệp vụ."""
    true_usd = np.exp(y_true_log)
    pred_usd = np.exp(y_pred_log)
    return {
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "mae_usd": float(mean_absolute_error(true_usd, pred_usd)),
        "rmse_usd": float(np.sqrt(mean_squared_error(true_usd, pred_usd))),
        "r2_usd": float(r2_score(true_usd, pred_usd)),
        "mape_percent": float(np.mean(np.abs(pred_usd - true_usd) / true_usd) * 100.0),
    }
