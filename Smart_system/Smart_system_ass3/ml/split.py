"""Chia 70/15/15 dùng chung cho cả ba bài toán.

Công thức giữ nguyên như quy ước đã dùng ở notebook 01-03: cắt 15% làm test
trước, phần còn lại cắt tiếp 15/85 làm validation. Cùng một seed và cùng một
thứ tự cắt thì cùng một dataset luôn cho ra cùng một tập chỉ số.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TEST_SHARE = 0.15
VAL_SHARE_OF_REST = TEST_SHARE / (1.0 - TEST_SHARE)


def split_70_15_15(
    X: Any, y: Any, *, stratify: bool = True, seed: int = RANDOM_SEED
) -> tuple[Any, Any, Any, Any, Any, Any]:
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SHARE, random_state=seed, stratify=y if stratify else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=VAL_SHARE_OF_REST,
        random_state=seed,
        stratify=y_rest if stratify else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
