# CẨM NANG TOÀN DIỆN: THẤU HIỂU HỆ THỐNG & BẢO VỆ ASSIGNMENT 03 TRƯỚC HỘI ĐỒNG / GIẢNG VIÊN

**Học phần:** Phát triển Hệ thống Thông minh (Intelligent System Development)  
**Đề tài:** Assignment 03 – Deep Learning viết tay với NumPy & Hoàn thiện hệ thống so sánh (3 ML + 1 DL cho mỗi category)  
**Sinh viên thực hiện:** Nguyễn Đại Dũng  
**Mã sinh viên:** B23DCVT103  
**Lớp:** E23VT01  
**Nhóm:** 02  
**Giảng viên hướng dẫn:** PGS. TS. Trần Đình Quế (Dinh Que Tran, Ph.D., Assoc. Prof.)  
**Học kỳ:** I.2026  

---

## MỤC LỤC

1. [TỔNG QUAN BÀI TOÁN & Ý NGHĨA CỐT LÕI CỦA ASSIGNMENT 03](#1-tổng-quan-bài-toán--ý-nghĩa-cốt-lõi-của-assignment-03)
2. [BẢN ĐỒ KIẾN TRÚC MÃ NGUỒN VÀ LUỒNG DỮ LIỆU TỰ ĐỘNG](#2-bản-đồ-kiến-trúc-mã-nguồn-và-luồng-dữ-liệu-tự-động)
3. [TRỌNG TÂM KỸ THUẬT 1: MẠNG NƠ-RON VIẾT TAY BẰNG NUMPY (NOTEBOOK 04)](#3-trọng-tâm-kỹ-thuật-1-mạng-nơ-ron-viết-tay-bằng-numpy-notebook-04)
   - 3.1. Lan truyền xuôi (Forward Propagation) & Chống tràn số
   - 3.2. Hàm mất mát Binary Cross-Entropy (BCE)
   - 3.3. Lan truyền ngược (Backpropagation) & Quy tắc đạo hàm chuỗi (Chain Rule)
   - 3.4. Bốn biến thể kiến trúc (M1, M2, M3, M4)
   - 3.5. Khảo sát tốc độ học (Learning Rate Study)
   - 3.6. Tối ưu ngưỡng phân loại (Classification Threshold Tuning)
4. [TRỌNG TÂM KỸ THUẬT 2: BA BÀI TOÁN & MA TRẬN SO SÁNH (3 ML + 1 DL)](#4-trọng-tâm-kỹ-thuật-2-ba-bài-toán--ma-trận-so-sánh-3-ml--1-dl)
   - 4.1. Category 1: Dự đoán nguy cơ tiểu đường (Dữ liệu bảng số)
   - 4.2. Category 2: Đánh giá cảm nhận khách hàng E-commerce (Văn bản NLP + Embedding Layer)
   - 4.3. Category 3: Định giá bất động sản (Hồi quy bảng số)
5. [KỊCH BẢN THUYẾT TRÌNH 7-10 PHÚT TRƯỚC GIẢNG VIÊN](#5-kịch-bản-thuyết-trình-7-10-phút-trước-giảng-viên)
6. [BỘ 15 CÂU HỎI VẤN ĐÁP THƯỜNG GẶP CỦA THẦY CÔ & CÁCH ĐÁP HOÀN HẢO](#6-bộ-15-câu-hỏi-vấn-đáp-thường-gặp-của-thầy-cô--cách-đáp-hoàn-hảo)
7. [HƯỚNG DẪN DEMO MÃ NGUỒN TRỰC TIẾP TẠI BÀN](#7-hướng-dẫn-demo-mã-nguồn-trực-tiếp-tại-bàn)

---

# 1. TỔNG QUAN BÀI TOÁN & Ý NGHĨA CỐT LÕI CỦA ASSIGNMENT 03

### Thầy cô thực sự muốn kiểm tra điều gì ở Assignment 03?
1. **Bóc tách chiếc "hộp đen" (Black Box) Deep Learning:**
   - Hầu hết sinh viên làm Deep Learning chỉ biết gọi `torch.nn` hoặc `tf.keras`, dùng `.fit()` hay `.backward()`. Giảng viên muốn bạn chứng minh rằng bạn **thực sự hiểu bản chất toán học**: Đạo hàm riêng $\frac{\partial \mathcal{L}}{\partial W}$ được tính ra sao? Vector hóa ma trận (Matrix Vectorization) hoạt động thế nào trên CPU mà không cần GPU?
2. **Khả năng so sánh thuật toán đa chiều (Multi-paradigm Comparison):**
   - Assignment 02 bạn đã có các mô hình Tuyến tính (Logistic/Linear Regression), Cây quyết định và Tổ hợp cây (Random Forest, Gradient Boosting).
   - Assignment 03 bổ sung thêm 2 mô hình máy học cổ điển: **K-Nearest Neighbors (KNN)** (học dựa trên cá thể - Instance-based) và **AdaBoost** (tổ hợp boosting trọng số mẫu), cùng với **Deep Learning** cho cả 3 category: Phân loại nhị phân (Tiểu đường), Phân loại đa lớp văn bản (E-commerce), và Hồi quy (Giá nhà).
3. **Kỹ năng thực nghiệm chuẩn mực:**
   - Dữ liệu chia đồng nhất `70/15/15` phân tầng (Stratified Split).
   - Chuẩn hoá chỉ fit trên tập Train, không bị rò rỉ dữ liệu (Data Leakage).
   - Đánh giá trung thực bằng F1-macro, ROC-AUC, Balanced Accuracy, R2, MAE chứ không bị "đánh lừa" bởi Accuracy thông thường.

---

# 2. BẢN ĐỒ KIẾN TRÚC MÃ NGUỒN VÀ LUỒNG DỮ LIỆU TỰ ĐỘNG

Dự án được cấu trúc theo chuẩn công nghiệp, phân tách rành mạch giữa mã nguồn cốt lõi (`ml/`), các kịch bản thực thi (`scripts/`), sổ tay phân tích (`notebooks/`), và hệ thống báo cáo tự động (`reports/`):

```
ass3/
├── data/                                 # 3 tập dữ liệu thực tế lớn
│   ├── diabetes/diabetes.csv             # 253,680 dòng (BRFSS 2015)
│   ├── house_price/house_price.csv       # 178 MB (USA Real Estate)
│   └── ecommerce/ecommerce.csv           # 85,907 dòng (Customer Support Dialogue)
│
├── ml/                                   # Module máy học tự viết dùng chung
│   ├── __init__.py                       # Package exports
│   ├── datasets.py                       # Nạp & tiền xử lý chuẩn hoá (Imputation, Encoding, TF-IDF)
│   ├── split.py                          # Chia tập 70/15/15 Stratified Split cố định SEED=42
│   ├── metrics.py                        # Tính toán F1-macro, ROC-AUC, Confusion Matrix, R2, MAE
│   └── benchmark.py                      # Bộ đo thời gian huấn luyện, suy luận và bộ nhớ
│
├── notebooks/                            # 5 Sổ tay Jupyter chính theo đề bài
│   ├── 04_deep_learning.ipynb            # Dựng MLP từ đầu bằng NumPy cho dữ liệu Tiểu đường
│   ├── 05_knn.ipynb                      # Khảo sát KNN (Euclidean / Cosine)
│   ├── 06_adaboost.ipynb                 # Khảo sát AdaBoost (Cây nông / sâu, số vòng lặp)
│   ├── 07_deep_learning_house.ipynb      # Mạng nơ-ron Hồi quy định giá bất động sản
│   └── 08_deep_learning_ecommerce.ipynb  # Mạng nơ-ron Text Classification có Embedding Layer
│
├── scripts/                              # Kịch bản thực thi độc lập (Headless CLI)
│   ├── train_04_deep_learning.py         # Chạy huấn luyện M1-M4 & khảo sát lr -> deeplearning_results.json
│   ├── train_05_knn.py                   # Chạy KNN -> knn_results.json
│   ├── train_06_adaboost.py              # Chạy AdaBoost -> adaboost_results.json
│   ├── train_07_house.py                 # Chạy DL House -> deeplearning_house_results.json
│   ├── train_08_ecommerce.py             # Chạy DL E-commerce -> deeplearning_ecommerce_results.json
│   └── build_master_pdf.py               # Biên soạn Báo cáo Master 97 trang (Thân + Phụ lục A-F)
│
├── reports/                              # Kết quả đầu ra và hình ảnh trực quan
│   ├── deeplearning_results.json         # Lưu toàn bộ số liệu M1-M4, learning rates, thresholds
│   ├── knn_results.json, ...             # Các kết quả JSON khác
│   ├── figs/                             # Các đồ thị đã xuất (loss, ROC curve, LR curves, Confusion matrix)
│   └── Assignment03_NguyenDaiDung_Full.pdf # Báo cáo PDF master 97 trang hoàn chỉnh
│
├── run_all.py                            # Kịch bản 1 click: Chạy toàn bộ pipeline từ A đến Z
├── A3_02_NguyenDaiDung_103_Master.pdf    # Bản PDF báo cáo chính thức nộp bài (97 trang)
└── HUONG_DAN_THUYET_TRINH_VA_HIEU_HE_THONG.md  # Chính là tài liệu này
```

---

# 3. TRỌNG TÂM KỸ THUẬT 1: MẠNG NƠ-RON VIẾT TAY BẰNG NUMPY (NOTEBOOK 04)

Đây là **phần quan trọng nhất** của Assignment 03. Khi giảng viên hỏi về thuật toán, 90% sẽ tập trung vào phần này.

### 3.1. Lan truyền xuôi (Forward Propagation) & Chống tràn số

Mạng nơ-ron của chúng ta gồm 3 lớp biến đổi affine liên tiếp:
$$\hat{y} = f_3(f_2(f_1(X))) = \sigma\Big(\text{ReLU}\big(\text{ReLU}(XW_1 + b_1)W_2 + b_2\big)W_3 + b_3\Big)$$

- **Lớp 1:**
  $$Z_1 = X W_1 + b_1 \quad \in \mathbb{R}^{N \times H_1}$$
  $$H_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$$
- **Lớp 2:**
  $$Z_2 = H_1 W_2 + b_2 \quad \in \mathbb{R}^{N \times H_2}$$
  $$H_2 = \text{ReLU}(Z_2) = \max(0, Z_2)$$
- **Lớp 3 (Lớp đầu ra):**
  $$Z_3 = H_2 W_3 + b_3 \quad \in \mathbb{R}^{N \times 1}$$
  $$\hat{y} = \sigma(Z_3) = \frac{1}{1 + e^{-Z_3}}$$

#### Kỹ thuật chống tràn số (Numerical Stability):
Hàm $e^{-z}$ sẽ bị tràn số (`overflow` ra `inf`) nếu $z < -709$. Để ổn định toán học trong NumPy:
```python
def sigmoid(z):
    # Kẹp giá trị z vào khoảng [-500, 500] để ngăn np.exp tràn số
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))
```

---

### 3.2. Hàm mất mát Binary Cross-Entropy (BCE)

Với $N$ mẫu huấn luyện, hàm mất mát trung bình là:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \Big[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big]$$

Trong code, để tránh $\log(0) = -\infty$, ta kẹp $\hat{y}$ trong khoảng $[\epsilon, 1 - \epsilon]$ với $\epsilon = 10^{-15}$:
```python
def bce_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
```

---

### 3.3. Lan truyền ngược (Backpropagation) & Quy tắc đạo hàm chuỗi (Chain Rule)

Đây là điểm ăn điểm tuyệt đối nếu bạn viết được công thức lên bảng hoặc giải thích cặn kẽ:

#### Bước 1: Đạo hàm mất mát theo đầu vào lớp cuối ($Z_3$)
Kết hợp tuyệt đẹp giữa BCE và hàm Sigmoid:
$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}$$
$$\frac{\partial \hat{y}}{\partial Z_3} = \sigma(Z_3)(1 - \sigma(Z_3)) = \hat{y}(1 - \hat{y})$$
Theo quy tắc dây chuyền:
$$dZ_3 = \frac{\partial \mathcal{L}}{\partial Z_3} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial Z_3} = \hat{y} - y$$
*(Hai thành phần mẫu số tự triệt tiêu lẫn nhau! Đây là lý do vì sao Sigmoid + BCE là một cặp bài trùng kinh điển).*

#### Bước 2: Gradient của $W_3$ và $b_3$
$$dW_3 = \frac{1}{N} H_2^T \cdot dZ_3$$
$$db_3 = \frac{1}{N} \sum_{i=1}^N dZ_{3}^{(i)} \quad (\text{axis}=0)$$

#### Bước 3: Lan truyền ngược qua hàm kích hoạt ẩn (ReLU)
Đạo hàm của hàm ReLU rất đơn giản: bằng 1 nếu $Z > 0$, và bằng 0 nếu $Z \le 0$:
$$dH_2 = dZ_3 \cdot W_3^T$$
$$dZ_2 = dH_2 \odot \mathbb{I}(Z_2 > 0)$$
*(Trong đó $\odot$ là phép nhân từng phần tử Hadamard, $\mathbb{I}$ là hàm chỉ thị).*

Tương tự cho lớp 2:
$$dW_2 = \frac{1}{N} H_1^T \cdot dZ_2, \quad db_2 = \frac{1}{N} \sum dZ_2$$
$$dH_1 = dZ_2 \cdot W_2^T, \quad dZ_1 = dH_1 \odot \mathbb{I}(Z_1 > 0)$$
$$dW_1 = \frac{1}{N} X^T \cdot dZ_1, \quad db_1 = \frac{1}{N} \sum dZ_1$$

#### Bước 4: Cập nhật trọng số bằng Gradient Descent
$$W \leftarrow W - \eta \cdot dW, \quad b \leftarrow b - \eta \cdot db$$

Toàn bộ quy trình này được viết thuần bằng NumPy, xử lý ma trận `(200000, 8)` trong **chưa đầy 10 mili-giây mỗi bước lặp**!

---

### 3.4. Bốn biến thể kiến trúc (M1, M2, M3, M4)

Để kiểm chứng xem mạng sâu hơn, rộng hơn, hay hẹp hơn thì tốt hơn, đề bài thiết kế 4 biến thể:

| Mã | Tên gọi | Cấu hình lớp ẩn | Số tham số | Loss test | ROC-AUC | Thời gian (s) | Nhận xét chuyên môn |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **M1** | Baseline | 8 $\rightarrow$ 16 $\rightarrow$ 8 $\rightarrow$ 1 | 297 | 0.3807 | **0.8042** | ~2.5s | Kiến trúc chuẩn, cân đối |
| **M2** | Wider | 8 $\rightarrow$ 32 $\rightarrow$ 16 $\rightarrow$ 1 | 849 | 0.3804 | **0.8043** | ~3.8s | Đạt ROC-AUC cao nhất (rộng hơn) |
| **M3** | Slim | 8 $\rightarrow$ 8 $\rightarrow$ 4 $\rightarrow$ 1 | 117 | 0.3809 | **0.8038** | **~1.8s** | **Tối ưu nhất**: giảm 60% tham số, nhanh gấp rưỡi, AUC chỉ kém 0.0005 |
| **M4** | Bottleneck | 8 $\rightarrow$ 16 $\rightarrow$ 4 $\rightarrow$ 8 $\rightarrow$ 1 | 265 | 0.3813 | 0.8028 | ~3.2s | Nén qua nút cổ chai 4 nơ-ron làm mất một phần thông tin |

> **Bài học rút ra cho câu hỏi vấn đáp:** Thêm nhiều nơ-ron (như M2) chỉ tăng ROC-AUC thêm một lượng rất nhỏ (từ 0.8042 lên 0.8043) nhưng tăng gần gấp 3 số tham số. Mạng nơ-ron đã chạm **"trần thông tin" (Information Capacity Ceiling)** của 8 biến đầu vào. Muốn tăng chất lượng, cần thêm biến đặc trưng hoặc biến tương tác, chứ không phải tăng kích thước mạng.

---

### 3.5. Khảo sát tốc độ học (Learning Rate Study)

Huấn luyện mô hình M1 với 4 giá trị $\eta \in \{0.001, 0.01, 0.1, 0.5\}$ trong cùng 500 epochs:

- **$\eta = 0.001$:** Loss giảm rất chậm (0.5284 $\rightarrow$ 0.4431), ROC-AUC chỉ 0.6920. Tại ngưỡng 0.5, mô hình đoán toàn bộ là 0 (Recall = 0.0000). Mạng chưa hội tụ vì bước nhảy quá ngắn.
- **$\eta = 0.01$:** Khá hơn nhưng vẫn còn chậm (Loss 0.3904, ROC-AUC 0.7746).
- **$\eta = 0.1$:** Điểm bùng phát tối ưu (Loss giảm về 0.3757, ROC-AUC đạt 0.8004). Mạng đã học được biên phân loại.
- **$\eta = 0.5$:** Hội tụ nhanh nhất (Loss về 0.3644, ROC-AUC đạt 0.8042). Không bị phân kỳ (divergence) vì bài toán được chuẩn hoá tốt và bề mặt hàm mất mát lồi xấp xỉ trong lân cận điểm bắt đầu.

---

### 3.6. Tối ưu ngưỡng phân loại (Classification Threshold Tuning)

Trên tập dữ liệu tiểu đường, tỉ lệ người có nguy cơ/mắc bệnh chỉ chiếm **~17.3%** (tập dữ liệu mất cân bằng - Imbalanced Class).

Nếu dùng ngưỡng mặc định $p \ge 0.5$:
- Mạng dự đoán an toàn: gán hầu hết là 0.
- Độ chính xác (Accuracy) đạt cao: **83.1%**, nhưng Recall chỉ đạt **~15.5%** (bỏ sót hơn 84% số bệnh nhân thực tế)!
- Trong y tế, **bỏ sót bệnh nhân (False Negative)** nguy hiểm hơn nhiều so với chẩn đoán nhầm người khoẻ (False Positive).

Khi quét ngưỡng từ 0.1 đến 0.9:
- Tại ngưỡng **0.25 – 0.30**: Recall tăng vọt lên **65% – 70%**, F1-score đạt đỉnh **0.46**, Balanced Accuracy đạt **73.5%**.
- Đây là minh chứng rõ ràng nhất cho thấy: **Accuracy là thước đo đánh lừa trên dữ liệu mất cân bằng**, và điều chỉnh ngưỡng phân loại là bước bắt buộc khi đưa mô hình vào ứng dụng thực tiễn.

---

# 4. TRỌNG TÂM KỸ THUẬT 2: BA BÀI TOÁN & MA TRẬN SO SÁNH (3 ML + 1 DL)

Assignment 03 hoàn thiện bức tranh học máy bằng việc đưa mỗi category lên đủ **3 mô hình ML + 1 mô hình DL**:

### 4.1. Category 1: Dự đoán nguy cơ tiểu đường (Dữ liệu bảng số)
- **3 ML:** Logistic Regression, Random Forest, AdaBoost (hoặc KNN).
- **1 DL:** Mạng M2 Wider (NumPy tự viết).
- **Kết quả:**
  + AdaBoost (50 vòng, cây sâu 3): ROC-AUC = **0.8078**, F1-macro = **0.6446** (Đứng đầu).
  + DL M2 Wider (NumPy): ROC-AUC = **0.8043** (Rất sát AdaBoost).
  + KNN ($k=5$): ROC-AUC = **0.7159** (Xếp cuối vì suy luận chậm, bị ảnh hưởng bởi chiều dữ liệu).
- **Ý nghĩa:** Trên dữ liệu bảng có cấu trúc (Tabular Data), các mô hình cây Boosting (AdaBoost, Gradient Boosting) vẫn là "vua", mạng nơ-ron đuổi rất sát nhưng không áp đảo được vì dữ liệu bảng thiếu tính cấu trúc cục bộ (local connectivity) như ảnh hay chữ.

### 4.2. Category 2: Đánh giá cảm nhận khách hàng E-commerce (Văn bản NLP)
- **3 ML:** Logistic Regression (TF-IDF), Random Forest, AdaBoost/KNN.
- **1 DL:** Mạng nơ-ron sâu có **Embedding Layer** (tự học biểu diễn vector từ vựng).
- **Kết quả:**
  + **Deep Learning (với Embedding): F1-macro = 0.7918** $\rightarrow$ **VƯỢT LÊN DẪN ĐẦU**, đánh bại mốc Logistic Regression (0.7900).
  + Logistic Regression + TF-IDF: F1-macro = 0.7900.
  + AdaBoost: F1-macro = 0.6974.
  + KNN Cosine ($k=11$): F1-macro = 0.5880.
- **Ý nghĩa:** Đây là bài toán duy nhất có đầu vào phi cấu trúc (văn bản thô). Mạng nơ-ron phát huy sức mạnh vượt trội nhờ khả năng **học biểu diễn ngữ nghĩa (Representation Learning)** thông qua ma trận nhúng (Embedding), nắm bắt được tương quan ngữ cảnh giữa các từ mà Bag-of-words hay TF-IDF không thể có.

### 4.3. Category 3: Định giá bất động sản (Hồi quy bảng số)
- **3 ML:** Ridge/Linear Regression, Random Forest Regressor, AdaBoost Regressor.
- **1 DL:** Mạng nơ-ron hồi quy nhiều lớp (MLP Regressor với loss MSE/MAE).
- **Kết quả:**
  + Random Forest: $R^2 \approx 0.74$, MAE thấp nhất.
  + DL MLP Regressor: $R^2 \approx 0.68 - 0.71$.
  + Linear Regression: $R^2 \approx 0.61$.
- **Ý nghĩa:** Mạng nơ-ron học được mối quan hệ phi tuyến tốt hơn hồi quy tuyến tính cổ điển, nhưng trên bảng số thực tế nhiều nhiễu, Random Forest vẫn chiếm ưu thế nhờ khả năng phân chia siêu phẳng không gian trực giao rất mạnh.

---

# 5. KỊCH BẢN THUYẾT TRÌNH 7-10 PHÚT TRƯỚC GIẢNG VIÊN

Bạn hãy dùng dàn ý này khi đứng trước thầy cô:

### Phút 1: Mở đầu & Giới thiệu mục tiêu (Tự tin, đi thẳng vào vấn đề)
> *"Em chào thầy/cô. Em là Nguyễn Đại Dũng, mã sinh viên B23DCVT103, thuộc nhóm 02. Hôm nay em xin phép báo cáo kết quả thực hiện Assignment 03 môn Phát triển Hệ thống Thông minh.*  
> *Mục tiêu cốt lõi của Assignment 03 gồm 2 nhiệm vụ chính:*  
> *1. Xây dựng trọn vẹn một mạng nơ-ron nhiều lớp hoàn toàn từ con số không (From Scratch) bằng thư viện NumPy thuần, không dùng bất kỳ framework cấp cao nào như PyTorch hay TensorFlow.*  
> *2. Bổ sung các mô hình KNN, AdaBoost và Deep Learning vào cả 3 bài toán nghiệp vụ của dự án để thiết lập ma trận so sánh toàn diện: 3 mô hình Machine Learning cổ điển và 1 mô hình Deep Learning cho mỗi category."*

### Phút 2-4: Trình bày phần kỹ thuật Deep Learning viết tay (Trọng tâm)
> *"Về phần mạng nơ-ron viết tay trên dữ liệu chẩn đoán tiểu đường (hơn 200,000 dòng):*  
> *- Em đã tự thiết kế thuật toán lan truyền xuôi với 3 lớp affine, dùng hàm kích hoạt ẩn ReLU và hàm đầu ra Sigmoid.*  
> *- Toàn bộ thuật toán lan truyền ngược (Backpropagation) được lập trình thuần bằng các phép nhân ma trận của NumPy. Cặp hàm mất mát Binary Cross-Entropy và Sigmoid cho đạo hàm đẹp là $dZ_3 = \hat{y} - y$. Đạo hàm qua lớp ReLU được giải quyết bằng phép nhân mặt nạ logic $Z > 0$.*  
> *- Để chương trình chạy mượt trên CPU, em đã vector hoá 100% các phép tính, áp dụng kỹ thuật kẹp giá trị `np.clip` để triệt tiêu hiện tượng tràn số (numerical overflow).*  
> *- Kết quả: Mỗi mô hình huấn luyện 500 epochs trên hơn 200,000 dòng chỉ mất từ 2 đến 4 giây trên CPU thông thường."*

### Phút 5-6: Kết quả thực nghiệm và các phát hiện then chốt
> *"Em đã tiến hành 3 thí nghiệm chuyên sâu:*  
> *1. **So sánh 4 kiến trúc M1 đến M4:** Kết quả cho thấy mô hình M2 Wider rộng hơn đạt ROC-AUC cao nhất là 0.8043. Tuy nhiên, mô hình M3 Slim cắt giảm 60% số tham số vẫn đạt 0.8038. Điều này chứng minh dữ liệu 8 biến đã chạm ngưỡng thông tin, việc tăng mạng sâu hơn không đem lại nhiều giá trị.*  
> *2. **Khảo sát tốc độ học (Learning Rate):** Tốc độ học nhỏ như 0.001 khiến mạng không hội tụ kịp sau 500 bước; trong khi $\eta = 0.5$ mang lại tốc độ hội tụ nhanh và tối ưu nhất.*  
> *3. **Phân tích ngưỡng phân loại:** Do dữ liệu dương tính chỉ chiếm 17%, nếu giữ ngưỡng 0.5 thì Recall chỉ đạt 15.5%. Khi em hạ ngưỡng về 0.25 - 0.30, Recall tăng lên 65-70% với F1 đạt đỉnh 0.46, rất có ý nghĩa trong bài toán sàng lọc y tế."*

### Phút 7-8: So sánh đa category (3 ML + 1 DL) & Kết luận
> *"Mở rộng sang 3 category:*  
> *- Ở bài toán tiểu đường và giá nhà (dữ liệu dạng bảng số), các mô hình cây như AdaBoost và Random Forest vẫn duy trì vị trí dẫn đầu, mạng nơ-ron bám rất sát ở vị trí số 2.*  
> *- Nhưng ở bài toán E-commerce (phân loại cảm nhận văn bản), mô hình Deep Learning có lớp nhúng (Embedding) đã vươn lên dẫn đầu với F1-macro đạt 0.7918, vượt qua mô hình Logistic Regression TF-IDF.*  
> *Bài học rút ra: Thuật toán không có cái nào mạnh tuyệt đối, mà sức mạnh phụ thuộc vào việc cấu trúc mô hình có khớp với bản chất của dữ liệu hay không. Mạng nơ-ron mạnh nhất khi tự học biểu diễn trên dữ liệu phi cấu trúc.*  
> *Em xin kết thúc phần trình bày và sẵn sàng nhận câu hỏi từ thầy/cô!"*

---

# 6. BỘ 15 CÂU HỎI VẤN ĐÁP THƯỜNG GẶP CỦA THẦY CÔ & CÁCH ĐÁP HOÀN HẢO

### Câu 1: Tại sao em lại tự viết Deep Learning bằng NumPy mà không dùng PyTorch hay TensorFlow?
- **Trả lời:** *"Dạ thưa thầy/cô, việc tự viết bằng NumPy là yêu cầu cốt lõi của đề tài nhằm hiểu thấu đáo bản chất toán học bên dưới của mạng nơ-ron: từ đạo hàm chuỗi (chain rule), lan truyền ngược, tới cập nhật trọng số. Các framework như PyTorch đóng gói quá nhiều trong cơ chế AutoGrad; tự viết giúp em kiểm soát được kích thước ma trận, quản lý bộ nhớ và nhận biết được các vấn đề tính toán như tràn số (numerical overflow) hay chết nơ-ron (dying ReLU)."*

### Câu 2: Em hãy giải thích vì sao đạo hàm của hàm mất mát BCE kết hợp với Sigmoid lại bằng $(\hat{y} - y)$?
- **Trả lời:** *"Dạ thưa thầy/cô, hàm mất mát BCE là $\mathcal{L} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$, đạo hàm của nó theo $\hat{y}$ là $\frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$. Còn hàm Sigmoid $\hat{y} = \sigma(z)$ có đạo hàm theo $z$ là $\hat{y}(1-\hat{y})$. Khi áp dụng quy tắc chuỗi $\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$, thành phần $\hat{y}(1-\hat{y})$ ở tử số và mẫu số triệt tiêu nhau hoàn toàn, chỉ còn lại hiệu số trực tiếp $(\hat{y} - y)$. Đây là đạo hàm rất thanh lịch, đo lường sai số tuyến tính giữa xác suất dự đoán và nhãn thực tế."*

### Câu 3: Làm thế nào em xử lý lan truyền ngược qua hàm kích hoạt ReLU?
- **Trả lời:** *"Dạ thưa thầy/cô, hàm ReLU định nghĩa là $H = \max(0, Z)$. Đạo hàm của nó là $\frac{\partial H}{\partial Z} = 1$ khi $Z > 0$ và bằng $0$ khi $Z \le 0$. Trong NumPy, em thực hiện phép nhân Hadamard giữa gradient từ lớp trên dội về ($dH$) với một mặt nạ boolean: `dZ = dH * (Z > 0)`. Điều này cho phép ngắt dòng gradient đi qua những nơ-ron không được kích hoạt."*

### Câu 4: Kỹ thuật chống tràn số (Numerical Stability) trong code của em là gì?
- **Trả lời:** *"Dạ, có 2 chỗ dễ bị tràn số:*  
*1. Trong hàm Sigmoid $\frac{1}{1 + e^{-z}}$, nếu $z$ âm lớn thì $e^{-z}$ sẽ vượt quá giới hạn float64 (khoảng 709). Em dùng `np.clip(z, -500.0, 500.0)`.*  
*2. Trong hàm BCE có $\log(\hat{y})$, nếu $\hat{y} = 0$ thì $\log(0) = -\infty$. Em kẹp $\hat{y}$ trong khoảng `[1e-15, 1 - 1e-15]` bằng hàm `np.clip` trước khi lấy log."*

### Câu 5: Tại sao Accuracy của mô hình trên tập tiểu đường đạt 83.1% nhưng em lại nói mô hình chưa tốt nếu để ngưỡng 0.5?
- **Trả lời:** *"Dạ thưa thầy/cô, tập dữ liệu có sự mất cân bằng lớp nghiêm trọng: lớp âm (không tiểu đường) chiếm tới 82.7%. Nếu một mô hình ngây thơ luôn đoán tất cả là 0 thì tự nhiên đã có accuracy là 82.7%. Ở ngưỡng 0.5, mô hình chỉ bắt được 15.5% ca bệnh (Recall rất thấp). Trong chẩn đoán y tế, việc để lọt người có bệnh là tối kỵ, do đó accuracy ở đây bị sai lệch và bắt buộc phải dùng ROC-AUC, PR-AUC hoặc điều chỉnh ngưỡng để tối ưu Recall."*

### Câu 6: Dữ liệu được tiền xử lý và chuẩn hóa ra sao để đảm bảo không bị rò rỉ thông tin (Data Leakage)?
- **Trả lời:** *"Dạ, em chia dữ liệu theo tỉ lệ 70% Train, 15% Validation, 15% Test bằng kỹ thuật phân tầng (Stratified Split). Mọi giá trị thống kê để chuẩn hoá như Mean, Standard Deviation (đối với số) hoặc từ điển TF-IDF (đối với văn bản) đều **chỉ được tính trên tập Train**. Sau đó, em dùng đúng bộ thông số đó để transform sang tập Test. Tuyệt đối không tính Mean/Std trên toàn bộ tập dữ liệu trước khi chia."*

### Câu 7: Tại sao mô hình M3 Slim (ít tham số nhất) lại được xem là mô hình có tính ứng dụng cao nhất?
- **Trả lời:** *"Dạ thưa thầy/cô, mô hình M3 chỉ có 117 tham số (so với 849 tham số của M2), nhưng ROC-AUC đạt 0.8038 (chỉ kém M2 đúng 0.0005). Thời gian suy luận của M3 nhanh hơn 50% và tiêu tốn cực ít bộ nhớ RAM. Theo nguyên lý dao cạo Ockham (Occam's Razor) và bài toán triển khai thực tế (Production Deployment), M3 là sự đánh đổi hoàn hảo nhất giữa hiệu năng và chi phí tài nguyên."*

### Câu 8: Hiện tượng gì xảy ra khi học với Learning Rate quá nhỏ (0.001) so với quá lớn (0.5)?
- **Trả lời:** *"Dạ, với $\eta = 0.001$, sau 500 epochs trọng số chỉ mới dịch chuyển một đoạn rất ngắn trong không gian tham số, hàm mất mát vẫn dừng ở mức 0.4431 và mạng chưa đủ dốc để phân tách lớp (Recall = 0). Ngược lại, $\eta = 0.5$ hội tụ rất nhanh về đáy tối ưu. Sở dĩ $\eta = 0.5$ không bị nổ gradient vì dữ liệu đầu vào đã được chuẩn hoá Z-score đưa về mean=0, std=1, giúp mặt cong mất mát có tính điều hòa tốt."*

### Câu 9: Tại sao trên dữ liệu bảng (Tabular Data), Deep Learning lại không thắng được AdaBoost hay Random Forest?
- **Trả lời:** *"Dạ, dữ liệu bảng gồm các cột độc lập với nhau (tuổi tác, BMI, huyết áp) chứ không có mối tương quan không gian (spatial) như pixel ảnh hay tính chuỗi (sequential) như từ ngữ. Các cây quyết định tạo ra các siêu phẳng phân chia trực giao rất khớp với dạng dữ liệu này. Mạng nơ-ron cần nhiều dữ liệu hơn và có xu hướng bị trơn hoá quá mức (over-smoothing) trên các biên rời rạc của dữ liệu bảng."*

### Câu 10: Ngược lại, tại sao trên bài toán E-commerce, Deep Learning lại vượt lên dẫn đầu?
- **Trả lời:** *"Dạ thưa thầy/cô, dữ liệu E-commerce là các đoạn hội thoại hỗ trợ khách hàng bằng văn bản. Mô hình Machine Learning cổ điển dùng TF-IDF biểu diễn từ dưới dạng túi từ (Bag-of-words) rời rạc, bỏ qua hoàn toàn thứ tự và ngữ cảnh. Trong khi đó, mạng Deep Learning của em sử dụng **Lớp nhúng (Embedding Layer)**, cho phép nén từ vựng vào một không gian vector liên tục (dense vector). Mô hình tự học được việc các từ đồng nghĩa hoặc cụm từ thường đi kèm nhau sẽ có vị trí gần nhau, từ đó nắm bắt ngữ nghĩa tốt hơn và đạt F1-macro cao nhất (0.7918)."*

### Câu 11: Em sử dụng thuật toán tối ưu hóa (Optimizer) nào trong mô hình viết tay?
- **Trả lời:** *"Dạ, trong Notebook 04 viết tay thuần bằng NumPy, em sử dụng **Batch Gradient Descent (BGD)** kết hợp cập nhật theo toàn bộ tập train (Full-batch) hoặc Mini-batch. Việc tính đạo hàm trên toàn bộ ma trận $X$ giúp gradient mượt và hội tụ ổn định. Ở các bài toán sau (House Price & E-commerce), em sử dụng kết hợp bộ tối ưu Adam để tăng tốc độ hội tụ qua các thung lũng dốc."*

### Câu 12: Tại sao mô hình M4 Bottleneck (thắt nút cổ chai 4 nơ-ron) lại có kết quả kém nhất?
- **Trả lời:** *"Dạ, cấu trúc của M4 đi từ 16 nơ-ron co cụm xuống 4 nơ-ron rồi lại nở ra 8 nơ-ron. Việc ép 8 chiều đặc trưng ban đầu đi qua một nút thắt quá hẹp (4 nơ-ron) đã tạo ra sự mất mát thông tin (information loss), tương tự như việc nén dữ liệu quá mức mà bộ giải nén không khôi phục đủ thông tin để phân loại."*

### Câu 13: K-Nearest Neighbors (KNN) hoạt động ra sao và nhược điểm lớn nhất của nó trong dự án này là gì?
- **Trả lời:** *"Dạ, KNN là thuật toán học lười (Lazy learning), không có quá trình huấn luyện mà lưu toàn bộ dữ liệu mẫu. Khi có điểm mới, nó đo khoảng cách (Euclidean hoặc Cosine) tới $k$ điểm gần nhất. Nhược điểm lớn nhất của nó là **thời gian suy luận cực kỳ chậm** khi dữ liệu lên tới hàng trăm nghìn dòng, đồng thời bị suy giảm hiệu quả khi số chiều tăng (Curse of Dimensionality)."*

### Câu 14: AdaBoost khác gì so với Random Forest?
- **Trả lời:** *"Dạ, Random Forest là kỹ thuật **Bagging** (huấn luyện song song nhiều cây độc lập trên các mẫu con rồi bầu chọn số đông). Còn AdaBoost là kỹ thuật **Boosting** (huấn luyện tuần tự). Cây sau sẽ tập trung học vào các mẫu mà cây trước đó đoán sai bằng cách gán trọng số lớn hơn cho những mẫu này. Do đó AdaBoost có khả năng giảm Bias rất mạnh và thường đạt độ chính xác nhỉnh hơn trên tập test."*

### Câu 15: Nếu có thêm thời gian để nâng cấp hệ thống, em sẽ cải tiến điều gì?
- **Trả lời:** *"Dạ thưa thầy/cô, có 3 điểm em sẽ ưu tiên cải tiến:*  
*1. Về dữ liệu: Sử dụng đủ toàn bộ 21 thuộc tính ban đầu của bộ dữ liệu CDC BRFSS thay vì chọn trước 8 thuộc tính bằng tương quan tuyến tính, đồng thời tạo thêm các đặc trưng tương tác (interaction features).*  
*2. Về thuật toán: Thêm cơ chế Momentum hoặc thuật toán Adam viết tay vào code NumPy để đẩy nhanh tốc độ hội tụ.*  
*3. Về giải quyết mất cân bằng lớp: Áp dụng Focal Loss hoặc gán trọng số lớp (Class Weights) ngay trong hàm mất mát BCE để mạng nơ-ron tự động phạt nặng hơn khi đoán sai lớp thiểu số."*

---

# 7. HƯỚNG DẪN DEMO MÃ NGUỒN TRỰC TIẾP TẠI BÀN

Nếu giảng viên yêu cầu mở máy tính lên chạy thử ngay trước mặt, bạn hãy tự tin thực hiện các thao tác sau:

### Cách 1: Chạy mô hình Deep Learning viết tay (Thời gian: ~10 giây)
Mở PowerShell tại thư mục dự án và chạy:
```powershell
& "E:\smart system\ass4_env\python.exe" scripts/train_04_deep_learning.py --skip-lr-study
```
- **Hiện tượng trên màn hình:**
  - Chương trình sẽ nạp 253,680 dòng dữ liệu tiểu đường, làm sạch còn ~229,000 dòng.
  - Chuẩn hoá dữ liệu theo Z-score.
  - Chạy lần lượt 4 kiến trúc M1, M2, M3, M4 trong vòng vài giây.
  - In bảng so sánh chi tiết: Loss, Accuracy, Precision, Recall, F1, ROC-AUC và Ma trận nhầm lẫn (Confusion Matrix).
- **Điểm chỉ cho thầy cô xem:** Chỉ tay vào màn hình cho thấy thời gian chạy cực nhanh (chỉ 2-3s cho hơn 200,000 dòng) và giải thích: *"Dạ đây là kết quả huấn luyện thuần NumPy ma trận trên CPU máy em."*

### Cách 2: Chạy toàn bộ hệ thống bằng 1 lệnh duy nhất
```powershell
& "E:\smart system\ass4_env\python.exe" run_all.py
```
- Lệnh này sẽ chạy toàn bộ các bước 4, 5, 6, 7, 8 và xuất ra file JSON kết quả.

### Cách 3: Mở Báo cáo Master 97 trang
Chỉ cần mở file PDF đã biên soạn sẵn tại:
`E:\smart system\ass3\A3_02_NguyenDaiDung_103_Master.pdf`
- Cho thầy cô xem **Trang bìa**: Đúng tên Nguyễn Đại Dũng, B23DCVT103, Lớp E23VT01, Nhóm 02.
- Lật đến **Trang 18 (Phần Phụ lục)**: Chỉ cho thầy cô thấy toàn bộ cell code thực thi và kết quả `In/Out` của cả 5 notebooks và mã nguồn gói `ml/` dùng chung.

---

### LỜI CHÚC
*Bạn đã nắm trong tay một hệ thống hoàn chỉnh từ toán học lý thuyết, mã nguồn chuẩn hóa, thực nghiệm minh bạch cho tới báo cáo chuẩn học thuật 97 trang. Hãy giữ phong thái tự tin, trả lời rõ ràng và mạch lạc. Chúc bạn đạt điểm A+ trong buổi bảo vệ trước thầy cô!*
