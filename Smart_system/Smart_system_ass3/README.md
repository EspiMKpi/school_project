# Assignment 03: Deep Learning Viết Tay Với NumPy Và Ba Mô Hình ML Cho Mỗi Category

> Môn học: **Intelligent System Development (Phát triển hệ thống thông minh)**  
> Sinh viên: **Nguyễn Đại Dũng** — Mã sinh viên: **B23DCVT103** — Lớp: **E23VT01** — Nhóm: **02**  
> Giảng viên hướng dẫn: **Dinh Que Tran, Ph.D., Assoc. Prof.**  
> Báo cáo hoàn chỉnh: [`A3_02_NguyenDaiDung_103_Master.pdf`](A3_02_NguyenDaiDung_103_Master.pdf) (97 trang)  
> Cẩm nang thuyết trình: [`HUONG_DAN_THUYET_TRINH_VA_HIEU_HE_THONG.md`](HUONG_DAN_THUYET_TRINH_VA_HIEU_HE_THONG.md)

---

## 1. Tổng quan đề tài & Kiến trúc hệ thống

Dự án Assignment 03 giải quyết 2 bài toán cốt lõi:
1. **Dựng lại mạng nơ-ron nhiều lớp (Deep Learning) hoàn toàn từ đầu chỉ dùng NumPy**:
   - Tự viết hàm lan truyền xuôi (`forward`), hàm mất mát (`loss`), lan truyền ngược (`backward`) theo quy tắc chuỗi và tối ưu hoá gradient descent theo lô nhỏ (mini-batch SGD).
   - Tuyệt đối **không** dùng framework học sâu (TensorFlow, PyTorch, Keras) hay thư viện ML (scikit-learn) cho phần neural network.
   - So sánh 4 kiến trúc mạng trên bài toán phân loại nhị phân nguy cơ tiểu đường (M1 Baseline, M2 Wider, M3 Shallow, M4 No ReLU/Linear).
   - Chứng minh về mặt toán học tính chất của **cặp hàm liên kết chuẩn (canonical link)**: đạo hàm của hàm kích hoạt triệt tiêu gọn với mẫu số của hàm mất mát:
     $$\frac{\partial L}{\partial z^{(L)}} = \frac{\hat{y} - y}{n}$$
     áp dụng được cho cả cặp `Sigmoid + Binary Cross-Entropy` lẫn `Linear Output + Mean Squared Error`.

2. **Mở rộng phạm vi ra cả 3 category của học kỳ, phủ 4 bài toán**:
   - Mỗi category có đủ **3 mô hình Machine Learning** và **1 mô hình Deep Learning**:
     - **Category 1: Tiểu đường (Phân loại nhị phân)**: Random Forest (Mốc), KNN ($k=5$), AdaBoost ($50$ vòng, cây sâu 3), Mạng nơ-ron M2/M1 (NumPy).
     - **Category 2: Giá nhà (Hồi quy log giá)**: HistGradientBoosting (Mốc), KNN ($k=25$), AdaBoost.R2 ($25$ vòng, cây sâu 8), Mạng nơ-ron hồi quy H2 (NumPy).
     - **Category 3a: E-commerce mức hài lòng (Phân loại nhị phân bảng + văn bản)**: Logistic Regression + TF-IDF (Mốc), KNN cosin ($k=5$), AdaBoost ($300$ vòng, cây sâu 3), Mạng nơ-ron N2 có ma trận nhúng học được (NumPy).
     - **Category 3b: E-commerce nhóm sản phẩm (Phân loại 9 lớp)**: Logistic Regression + TF-IDF (Mốc), KNN cosin ($k=5$), AdaBoost ($200$ vòng, cây sâu 3).
   - Đặc biệt ở Category 3: Xây dựng lớp nhúng từ (Word Embedding $E \in \mathbb{R}^{5001 \times 32}$) học trực tiếp bằng lan truyền ngược kết hợp masked average pooling và `np.add.at`, vượt qua mô hình Logistic Regression + TF-IDF mà chỉ cần số chiều ít hơn 100 lần (113 chiều so với 11,835 chiều).

---

## 2. Cấu trúc thư mục mã nguồn

```
ass3/
├── A3_02_HungNguyenBa_120.pdf       # Báo cáo gốc Assignment 03
├── README.md                       # Tài liệu hướng dẫn & giải trình hệ thống
├── run_all.py                      # Kịch bản CLI chạy toàn bộ / từng phần
├── data/                           # Thư mục chứa dữ liệu huấn luyện
│   ├── diabetes/
│   │   └── diabetes.csv            # Bộ BRFSS 2015 Diabetes Health Indicators
│   ├── house_price/
│   │   └── house_price.csv         # Bộ USA Real Estate Dataset
│   └── ecommerce/
│       └── ecommerce.csv           # Bộ Customer Support Ticket & Remarks
├── ml/                             # Gói mã nguồn Python dùng chung
│   ├── __init__.py                 # Export giao diện API dùng chung
│   ├── split.py                    # Chia tập 70/15/15 nhất quán
│   ├── metrics.py                  # Đo đạc chỉ số (F1-macro, ROC-AUC, R2, MAE, MAPE, ...)
│   ├── datasets.py                 # Nạp, tiền xử lý và sinh Pipeline cho 4 bài toán
│   └── benchmark.py                # Wrapper chạy mô hình, đo thời gian và lưu kết quả
├── notebooks/                      # 5 Jupyter Notebooks chuẩn theo phụ lục báo cáo
│   ├── 04_deep_learning.ipynb      # Phụ lục A: Deep Learning từ đầu với NumPy (Tiểu đường)
│   ├── 05_knn.ipynb                # Phụ lục B: K-Nearest Neighbors trên 3 category
│   ├── 06_adaboost.ipynb           # Phụ lục C: AdaBoost trên 3 category
│   ├── 07_deep_learning_house.ipynb# Phụ lục D: Mạng nơ-ron hồi quy cho giá nhà
│   └── 08_deep_learning_ecommerce.ipynb # Phụ lục E: Mạng nơ-ron có lớp nhúng cho e-commerce
├── scripts/                        # Các file script chạy độc lập từ CLI
│   ├── train_04_deep_learning.py
│   ├── train_05_knn.py
│   ├── train_06_adaboost.py
│   ├── train_07_deep_learning_house.py
│   ├── train_08_deep_learning_ecommerce.py
│   └── generate_notebooks.py
└── reports/                        # Kết quả thí nghiệm và biểu đồ
    ├── deeplearning_results.json
    ├── knn_results.json
    ├── adaboost_results.json
    ├── deeplearning_house_results.json
    ├── deeplearning_ecommerce_results.json
    ├── figs/                       # Các biểu đồ xuất ra
    └── build/                      # Kịch bản dựng HTML/PDF báo cáo
        ├── content03.py
        └── content03_extra.py
```

---

## 3. Chuẩn bị Dữ liệu Huấn luyện (Data Preparation)

Dự án yêu cầu 3 tập dữ liệu đặt đúng cấu trúc thư mục sau:

### 1) Dataset Tiểu đường (`data/diabetes/diabetes.csv`)
* **Nguồn**: Bộ dữ liệu khảo sát **CDC BRFSS 2015 Diabetes Health Indicators** (trên Kaggle: `alexteboul/diabetes-health-indicators-dataset` file `diabetes_012_health_indicators_BRFSS2015.csv`).
* **Kích thước**: ~253,680 dòng, 22 cột.
* **Cột nhãn (`target`)**: `Diabetes_012` (0 = không bệnh, 1 = tiền tiểu đường, 2 = tiểu đường).
* **Quy ước nhị phân**: `y = (Diabetes_012 > 0).astype(int)` (gộp tiền tiểu đường và tiểu đường thành lớp 1).
* **8 đặc trưng có tương quan cao nhất được chọn**: `GenHlth`, `HighBP`, `BMI`, `DiffWalk`, `HighChol`, `Age`, `HeartDiseaseorAttack`, `PhysHlth`.

### 2) Dataset Giá nhà (`data/house_price/house_price.csv`)
* **Nguồn**: Bộ dữ liệu **USA Real Estate Dataset** (trên Kaggle: `ahmedshahriarsakib/usa-real-estate-dataset` file `realtor-data.zip.csv`).
* **Kích thước**: ~2.2 triệu dòng thô, lấy mẫu ngẫu nhiên `sample_size = 250,000` dòng sạch.
* **Các cột sử dụng**: `status`, `price`, `bed`, `bath`, `acre_lot`, `city`, `state`, `house_size`.
* **Biến phái sinh**: `size_per_bed = house_size / clip(bed, 1)`.
* **Biến mục tiêu**: `y = log(price)`, chuẩn hoá $z = (y - \mu) / \sigma$ khi huấn luyện mạng nơ-ron.
* **Tiền xử lý**: `city` dùng TargetEncoder kết hợp KFold(5) mượt mà, sau đó chuẩn hoá `StandardScaler()`.

### 3) Dataset E-commerce (`data/ecommerce/ecommerce.csv`)
* **Nguồn**: Bộ dữ liệu **Customer Support Ticket Data** (`Customer_support_data.csv`).
* **Kích thước**: ~85,907 dòng (28,742 dòng có bình luận khách hàng `Customer Remarks`).
* **Các cột bảng**: `item_price`, `resp_min`, `hour`, `dow`, `rem_len`, `rem_words`, `has_rem`, `channel_name`, `category`, `Sub-category`, `Tenure Bucket`, `Agent Shift`.
* **Cột văn bản**: `Customer Remarks` (chuỗi phản hồi của khách hàng).
* **Hai bài toán**:
  - `y_sat`: Mức hài lòng nhị phân (`CSAT Score >= 4` là 1, ngược lại là 0).
  - `y_interest`: Nhóm sản phẩm quan tâm (`Product_category` gồm 9 lớp).

> **Lưu ý**: Cả 3 tập dữ liệu trên đã được chuẩn bị đầy đủ và nằm sẵn trong thư mục `data/` của dự án!

---

## 4. Hướng dẫn Chạy Thí nghiệm trên Local

### 4.1 Môi trường yêu cầu
- **Python**: 3.10, 3.11 hoặc 3.12 (khuyên dùng Python 3.11).
- **Thư viện**:
  ```bash
  pip install numpy pandas scikit-learn matplotlib
  ```

### 4.2 Chạy bằng dòng lệnh (CLI Scripts)
Bạn có thể dùng kịch bản tổng thể `run_all.py` ở thư mục gốc:

```bash
# 1. Chạy Deep Learning tiểu đường (NumPy thuần) - nhanh bỏ qua quét GD toàn tập:
python run_all.py --step 4 --skip-lr-study

# 2. Chạy KNN trên cả 3 category (4 bài toán):
python run_all.py --step 5

# 3. Chạy AdaBoost trên cả 3 category (4 bài toán):
python run_all.py --step 6

# 4. Chạy Deep Learning hồi quy giá nhà (NumPy thuần):
python run_all.py --step 7

# 5. Chạy Deep Learning có lớp nhúng từ E-commerce (NumPy thuần):
python run_all.py --step 8

# Hoặc chạy toàn bộ 5 bước liên tiếp:
python run_all.py --all --skip-lr-study
```

### 4.3 Chạy qua Jupyter Notebook
Tất cả các file notebook nằm trong thư mục `notebooks/`:
1. `notebooks/04_deep_learning.ipynb`
2. `notebooks/05_knn.ipynb`
3. `notebooks/06_adaboost.ipynb`
4. `notebooks/07_deep_learning_house.ipynb`
5. `notebooks/08_deep_learning_ecommerce.ipynb`

Mở bằng VS Code hoặc chạy lệnh:
```bash
jupyter notebook notebooks/
```

---

## 5. Giải thích Bản chất Toán học & Thuật toán

### 5.1 Cặp hàm liên kết chuẩn (Canonical Link)
Tại sao hàm lan truyền ngược `backward` ở Notebook 04 (phân loại nhị phân) và Notebook 07 (hồi quy) có cùng biểu thức đạo hàm lớp cuối:
$$\delta^{(L)} = \frac{\hat{y} - y}{n}$$
- Ở phân loại nhị phân: Hàm kích hoạt $\sigma(z) = \frac{1}{1 + e^{-z}}$, hàm mất mát Binary Cross-Entropy $L = -\frac{1}{n}\sum [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$.  
  Đạo hàm $\frac{\partial L}{\partial \hat{y}} = \frac{1}{n}\frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$, nhân với $\frac{d\hat{y}}{dz} = \hat{y}(1-\hat{y})$, mẫu số triệt tiêu hoàn hảo cho ra $(\hat{y} - y)/n$.
- Ở hồi quy: Hàm kích hoạt đồng nhất $\hat{y} = z$, hàm mất mát Mean Squared Error $L = \frac{1}{2n}\sum (\hat{y} - y)^2$.  
  Đạo hàm $\frac{\partial L}{\partial z} = \frac{\hat{y} - y}{n} \cdot 1 = \frac{\hat{y} - y}{n}$.

### 5.2 Đạo hàm lan truyền ngược qua lớp nhúng từ (Notebook 08)
- Phép gộp trung bình có che: $p_b = \frac{1}{c_b} \sum_t m_{bt} E[\text{ids}_{bt}]$ với $c_b$ là số từ thật (bỏ padding).
- Khi lùi gradient: Một từ vựng $v$ có thể xuất hiện nhiều lần trong câu hoặc nhiều câu trong batch, do đó gradient của hàng $E[v]$ là tổng tích luỹ của các đóng góp:
  $$\frac{\partial L}{\partial E[v]} = \sum_{(b,t): \text{ids}_{bt} = v} \frac{1}{c_b} \frac{\partial L}{\partial p_b}$$
  Biểu thức này được cài đặt chỉ bằng 1 dòng NumPy vector hoá hiệu năng cao:
  ```python
  np.add.at(dE, ids[rows, columns], d_pooled[rows] / counts[rows])
  ```
  giúp ma trận nhúng tự học các cụm từ đồng nghĩa cảm xúc (`good` gần `nice`, `bad` gần `worst`) chỉ sau 40 epoch mà không cần nạp bất kỳ từ điển cảm xúc nào trước!
