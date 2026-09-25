"""Nạp và làm sạch bốn bài toán ML, tái lập đúng quy trình của notebook 01-03.

Mỗi hàm `load_*` trả về một `Task` đã chia sẵn 70/15/15, kèm các nhà máy sinh bộ
tiền xử lý và mô hình mốc (mô hình đã được chọn ở notebook 01-03). Nhờ vậy hai
notebook 05 và 06 dùng chung đúng một cách làm sạch, một cách chia tập và một
mốc so sánh, khác biệt giữa hai notebook chỉ nằm ở thuật toán được thêm vào.

Dữ liệu đọc từ `data/` ở trạng thái hiện tại của kho, không phải bản chụp cũ mà
notebook 01-03 đã chạy, nên các con số ở đây không trùng với báo cáo Assignment 02.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

from .split import RANDOM_SEED, split_70_15_15

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIABETES_CSV = DATA / "diabetes" / "diabetes.csv"
HOUSE_CSV = DATA / "house_price" / "house_price.csv"
ECOM_CSV = DATA / "ecommerce" / "ecommerce.csv"

HOUSE_SAMPLE_SIZE = 250_000


@dataclass(frozen=True)
class Task:
    key: str
    title: str
    kind: str
    average: str
    X_train: Any
    X_val: Any
    X_test: Any
    y_train: Any
    y_val: Any
    y_test: Any
    preprocessors: dict[str, Callable[[], Any]]
    baseline_name: str
    baseline_rep: str
    baseline_estimator: Callable[[], Any]
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_val(self) -> int:
        return len(self.X_val)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    def prep(self, rep: str):
        return self.preprocessors[rep]()

    def summary(self) -> str:
        return (
            f"{self.title}: train={self.n_train:,} val={self.n_val:,} test={self.n_test:,} "
            f"| mốc so sánh: {self.baseline_name} ({self.baseline_rep})"
        )


# --------------------------------------------------------------------------- #
# Bài toán 1 — tiểu đường, phân loại nhị phân
# --------------------------------------------------------------------------- #
DIA_BINARY = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "Sex",
]
DIA_NUMERIC = ["BMI", "MentHlth", "PhysHlth"]
DIA_ORDINAL = ["GenHlth", "Age", "Education", "Income"]
DIA_FEATURES = DIA_BINARY + DIA_NUMERIC + DIA_ORDINAL


def prep_diabetes_native() -> ColumnTransformer:
    """Giữ nguyên 21 cột, chỉ chuẩn hoá phần liên tục và thứ bậc. d = 21.

    Cột nhị phân đã nằm sẵn trong {0, 1} nên không cần đụng tới; hai nhóm còn lại
    được đưa về cùng thang đo, điều kiện bắt buộc với mọi mô hình dựa trên khoảng cách.
    """
    return ColumnTransformer([
        ("bin", "passthrough", DIA_BINARY),
        ("num", StandardScaler(), DIA_NUMERIC),
        ("ord", StandardScaler(), DIA_ORDINAL),
    ])


def load_diabetes(path: Path = DIABETES_CSV) -> Task:
    raw = pd.read_csv(path)
    frame = raw.drop_duplicates().reset_index(drop=True)

    X = frame[DIA_FEATURES].copy()
    y = (frame["Diabetes_012"] > 0).astype(int)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_70_15_15(X, y, stratify=True)

    return Task(
        key="diabetes",
        title="Tiểu đường (phân loại nhị phân)",
        kind="binary",
        average="binary",
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_train=y_tr, y_val=y_va, y_test=y_te,
        preprocessors={"native": prep_diabetes_native},
        baseline_name="Random Forest",
        baseline_rep="native",
        baseline_estimator=lambda: RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED,
        ),
        audit={
            "rows_raw": int(len(raw)),
            "rows_after_dedup": int(len(frame)),
            "duplicates_removed": int(len(raw) - len(frame)),
            "positive_rate": float(y.mean()),
            "n_features": len(DIA_FEATURES),
        },
    )


# --------------------------------------------------------------------------- #
# Bài toán 2 — giá nhà, hồi quy trên log(price)
# --------------------------------------------------------------------------- #
HOUSE_USE_COLS = ["status", "price", "bed", "bath", "acre_lot", "city", "state", "house_size"]
HOUSE_NUM = ["bed", "bath", "acre_lot", "house_size", "size_per_bed"]
HOUSE_CAT_LOW = ["state", "status"]
HOUSE_CAT_HIGH = ["city"]
HOUSE_FEATURES = HOUSE_NUM + HOUSE_CAT_LOW + HOUSE_CAT_HIGH


def prep_house_target_enc() -> ColumnTransformer:
    """Target encoding: `city` nén về một chiều duy nhất. d = 8."""
    return ColumnTransformer([
        ("num", "passthrough", HOUSE_NUM),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), HOUSE_CAT_LOW),
        ("city", TargetEncoder(smooth=20.0, cv=KFold(5, shuffle=True, random_state=RANDOM_SEED)),
         HOUSE_CAT_HIGH),
    ])


def prep_house_target_enc_scaled() -> Pipeline:
    """Bản target encoding đã đưa tám chiều về cùng thang đo.

    Cây quyết định cắt theo từng chiều nên không quan tâm thang đo, nhưng khoảng
    cách Euclid thì có: `house_size` chạy tới 10,000 còn `bed` chỉ tới 10, để
    nguyên thì một chiều nuốt trọn bảy chiều còn lại.
    """
    return Pipeline([("encode", prep_house_target_enc()), ("scale", StandardScaler())])


def load_house_price(path: Path = HOUSE_CSV, sample_size: int = HOUSE_SAMPLE_SIZE) -> Task:
    df = pd.read_csv(path, usecols=HOUSE_USE_COLS)
    rows_raw = len(df)

    h = df.dropna(subset=["price", "bed", "bath", "house_size", "city", "state"])
    h = h[h["price"].between(50_000, 2_000_000)]
    h = h[h["house_size"].between(200, 10_000)]
    h = h[h["bed"].between(1, 10)]
    h = h[h["bath"].between(1, 10)].copy()
    h["acre_lot"] = h["acre_lot"].fillna(h["acre_lot"].median())
    h = h[h["acre_lot"].between(0, 50)]
    h = h.drop_duplicates(
        subset=["price", "bed", "bath", "acre_lot", "city", "state", "house_size"]
    )
    h["city"] = h["city"].astype(str).str.strip().str.lower()
    h["state"] = h["state"].astype(str).str.strip()
    rows_clean = len(h)

    h = h.sample(n=min(sample_size, len(h)), random_state=RANDOM_SEED).reset_index(drop=True)
    h["size_per_bed"] = h["house_size"] / h["bed"].clip(lower=1)

    X = h[HOUSE_FEATURES].copy()
    y = np.log(h["price"].to_numpy(dtype=float))
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_70_15_15(X, y, stratify=False)

    return Task(
        key="house_price",
        title="Giá nhà (hồi quy trên log giá)",
        kind="regression",
        average="",
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_train=y_tr, y_val=y_va, y_test=y_te,
        preprocessors={
            "target_enc": prep_house_target_enc,
            "target_enc_scaled": prep_house_target_enc_scaled,
        },
        baseline_name="HistGradientBoosting",
        baseline_rep="target_enc",
        baseline_estimator=lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, random_state=RANDOM_SEED
        ),
        audit={
            "rows_raw": int(rows_raw),
            "rows_clean": int(rows_clean),
            "kept_percent": round(rows_clean / rows_raw * 100, 2),
            "sample_size": int(len(h)),
            "n_city": int(h["city"].nunique()),
            "n_state": int(h["state"].nunique()),
            "skew_log_price": float(pd.Series(y).skew()),
        },
    )


# --------------------------------------------------------------------------- #
# Bài toán 3 và 4 — e-commerce, bảng ghép văn bản
# --------------------------------------------------------------------------- #
ECOM_NUM = ["item_price", "resp_min", "hour", "dow", "rem_len", "rem_words", "has_rem"]
ECOM_CAT = ["channel_name", "category", "Sub-category", "Tenure Bucket", "Agent Shift"]
ECOM_TEXT = "remark"
ECOM_FEATURES = ECOM_NUM + ECOM_CAT + [ECOM_TEXT]

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        max_features=20000, stop_words="english",
    )


def prep_ecom_tabular() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         ECOM_NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), ECOM_CAT),
    ])


def prep_ecom_tabular_text() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         ECOM_NUM),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), ECOM_CAT),
        ("text", _tfidf(), ECOM_TEXT),
    ])


def build_ecommerce_frame(path: Path = ECOM_CSV) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dựng bảng feature từ tệp thô, đúng các quyết định làm sạch của notebook 03."""
    raw = pd.read_csv(path)

    reported = pd.to_datetime(raw["Issue_reported at"], format="%d/%m/%Y %H:%M", errors="coerce")
    responded = pd.to_datetime(raw["issue_responded"], format="%d/%m/%Y %H:%M", errors="coerce")
    resp_min_raw = (responded - reported).dt.total_seconds() / 60.0

    out = pd.DataFrame(index=raw.index)
    out["item_price"] = pd.to_numeric(raw["Item_price"], errors="coerce")
    out["resp_min"] = resp_min_raw.where(resp_min_raw >= 0)
    out["hour"] = reported.dt.hour
    out["dow"] = reported.dt.dayofweek

    remark = raw["Customer Remarks"].fillna("").astype(str)
    out["rem_len"] = remark.str.len()
    out["rem_words"] = remark.str.split().str.len().fillna(0)
    out["has_rem"] = (remark.str.strip() != "").astype(int)
    for column in ECOM_CAT:
        out[column] = raw[column].fillna("UNKNOWN").astype(str)
    out[ECOM_TEXT] = remark

    out["y_sat"] = (raw["CSAT Score"] >= 4).astype(int)
    out["y_interest"] = raw["Product_category"]

    audit = {
        "rows_raw": int(len(raw)),
        "rows_with_remark": int(out["has_rem"].sum()),
        "negative_response_time": int((resp_min_raw < 0).sum()),
        "sat_rate_overall": float(out["y_sat"].mean()),
    }
    return out, audit


def load_ecommerce_satisfaction(path: Path = ECOM_CSV) -> Task:
    frame, audit = build_ecommerce_frame(path)
    sub = frame[frame["has_rem"] == 1].reset_index(drop=True)

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_70_15_15(
        sub[ECOM_FEATURES], sub["y_sat"], stratify=True
    )
    return Task(
        key="ecommerce_satisfaction",
        title="E-commerce: mức hài lòng (bảng + văn bản)",
        kind="binary",
        average="binary",
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_train=y_tr, y_val=y_va, y_test=y_te,
        preprocessors={
            "tabular": prep_ecom_tabular,
            "tabular_text": prep_ecom_tabular_text,
        },
        baseline_name="LogReg bảng + TF-IDF",
        baseline_rep="tabular_text",
        baseline_estimator=lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        audit={**audit, "rows_used": int(len(sub)), "positive_rate": float(sub["y_sat"].mean())},
    )


def load_ecommerce_interest(path: Path = ECOM_CSV) -> Task:
    frame, audit = build_ecommerce_frame(path)
    labelled = frame[frame["y_interest"].notna()].reset_index(drop=True)
    sub = labelled[labelled["has_rem"] == 1].reset_index(drop=True)

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_70_15_15(
        sub[ECOM_FEATURES], sub["y_interest"], stratify=False
    )
    return Task(
        key="ecommerce_interest",
        title="E-commerce: nhóm sản phẩm quan tâm (9 lớp)",
        kind="multiclass",
        average="macro",
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_train=y_tr, y_val=y_va, y_test=y_te,
        preprocessors={
            "tabular": prep_ecom_tabular,
            "tabular_text": prep_ecom_tabular_text,
        },
        baseline_name="LogReg bảng + TF-IDF",
        baseline_rep="tabular_text",
        baseline_estimator=lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        audit={
            **audit,
            "rows_labelled": int(len(labelled)),
            "rows_used": int(len(sub)),
            "n_classes": int(sub["y_interest"].nunique()),
        },
    )


LOADERS: dict[str, Callable[[], Task]] = {
    "diabetes": load_diabetes,
    "house_price": load_house_price,
    "ecommerce_satisfaction": load_ecommerce_satisfaction,
    "ecommerce_interest": load_ecommerce_interest,
}
