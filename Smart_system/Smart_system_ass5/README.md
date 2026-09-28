# Assignment 5: Convolutional Neural Networks (CNN) Benchmark & Data Augmentation

Nghiên cứu, xây dựng và so sánh chuyên sâu mạng nơ-ron tích chập (CNN) trên hai nền tảng **Keras** và **PyTorch** qua 3 bộ dữ liệu:
1. **Fashion-MNIST** (70.000 ảnh sản phẩm thời trang Zalando 10 lớp, 28x28 grayscale)
2. **SVHN (Street View House Numbers)** (99.289 ảnh màu 32x32x3 số nhà ngoài đời thực từ Google Street View)
3. **CDC Diabetes Health Indicators** (Dữ liệu y tế dạng bảng lớn được tăng cường lên >513.000 dòng với mô hình 1D-CNN)

---

## 1. Cấu trúc thư mục độc lập phục vụ thuyết trình

```
Smart_system_ass5/
├── data/                                # NƠI LƯU TRỮ VÀ QUẢN LÝ DỮ LIỆU
│   ├── fashion_mnist/                   # Dataset ảnh 1: 70.000 ảnh thời trang 28x28
│   │   ├── fashion_mnist_data.npz       # Dữ liệu train/test đồng bộ hóa
│   │   ├── sample_fashion.png           # Ảnh mẫu 10 lớp thời trang phục vụ trình chiếu
│   │   └── info.json
│   ├── svhn/                            # Dataset ảnh 2: 99.289 ảnh màu Google Street View 32x32x3
│   │   ├── svhn_data.npz                # Dữ liệu train/test đồng bộ hóa
│   │   ├── sample_svhn.png              # Ảnh mẫu biển số nhà thực tế
│   │   └── info.json
│   └── diabetes/                        # Dataset bảng y tế lớn CDC Diabetes
│       ├── diabetes_raw.csv             # Dữ liệu gốc (253.680 dòng)
│       ├── diabetes_augmented.csv       # Dữ liệu sau tăng cường (513.703 dòng)
│       ├── distribution_comparison.png  # Biểu đồ phân phối lớp trước & sau tăng cường
│       └── diabetes_data.npz            # Dữ liệu 1D-CNN train/test
│
├── models_keras/                        # 3 THƯ MỤC RIÊNG CHO CÁC MÔ HÌNH KERAS
│   ├── 01_fashion_mnist/                # Model 3-layer, 5-layer, biểu đồ, confusion matrix
│   ├── 02_svhn/                         # Model 3-layer, 5-layer, biểu đồ, confusion matrix
│   └── 03_diabetes/                     # Model 1D-CNN 3-layer, 5-layer, biểu đồ, metrics
│
├── models_pytorch/                      # 3 THƯ MỤC RIÊNG CHO PYTORCH (GPU RTX 3060)
│   ├── 01_fashion_mnist/                # Trọng số .pth 3-layer, 5-layer, biểu đồ, metrics
│   ├── 02_svhn/                         # Trọng số .pth 3-layer, 5-layer, biểu đồ, metrics
│   └── 03_diabetes/                     # Trọng số .pth 1D-CNN 3-layer, 5-layer, metrics
│
├── results/                             # comparison_all_models.csv & các biểu đồ đối chuẩn tổng hợp
├── report/                              # FINAL_REPORT_ASS5.md (Báo cáo khoa học hoàn chỉnh tiếng Việt)
├── app.py                               # Ứng dụng Streamlit Web Demo tương tác thời gian thực
├── 01_cnn_theory_and_code.ipynb         # Yêu cầu 1: Khái niệm cơ bản & Code đối chiếu Keras vs PyTorch
├── 02_data_preparation_and_augmentation.ipynb # Yêu cầu 2: Data EDA & Tăng cường Diabetes
├── 03_keras_experiments.ipynb           # Yêu cầu 3: Toàn bộ code huấn luyện & đánh giá Keras
├── 04_pytorch_experiments.ipynb         # Yêu cầu 4: Toàn bộ code huấn luyện & đánh giá PyTorch GPU
└── 05_comparison_and_benchmarking.ipynb # Đánh giá đối chuẩn 12 mô hình
```

---

## 2. Bảng kết quả tổng hợp 12 mô hình thực nghiệm

| STT | Dataset | Framework | Kiến trúc | Tham số | Train Time (s) | Test Loss | Test Accuracy | Macro F1 | Folder Lưu Trữ |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Fashion-MNIST | Keras | 3-Layer | 50.186 | 161.92s | 0.2675 | **90.49%** | 0.9026 | `models_keras/01_fashion_mnist/` |
| **2** | Fashion-MNIST | Keras | 5-Layer | 469.098 | 1293.91s | 0.2104 | **92.32%** | 0.9230 | `models_keras/01_fashion_mnist/` |
| **3** | Fashion-MNIST | PyTorch (GPU) | 3-Layer | 50.186 | 19.91s | 0.2598 | **90.84%** | 0.9082 | `models_pytorch/01_fashion_mnist/` |
| **4** | Fashion-MNIST | PyTorch (GPU) | 5-Layer | 468.458 | 36.77s | 0.1947 | **92.90%** | 0.9285 | `models_pytorch/01_fashion_mnist/` |
| **5** | SVHN | Keras | 3-Layer | 60.362 | 345.61s | 0.5741 | **84.84%** | 0.8303 | `models_keras/02_svhn/` |
| **6** | SVHN | Keras | 5-Layer | 592.554 | 2548.41s | 0.2802 | **91.94%** | 0.9132 | `models_keras/02_svhn/` |
| **7** | SVHN | PyTorch (GPU) | 3-Layer | 60.362 | 28.40s | 0.5282 | **86.42%** | 0.8505 | `models_pytorch/02_svhn/` |
| **8** | SVHN | PyTorch (GPU) | 5-Layer | 591.914 | 66.19s | 0.2534 | **92.86%** | 0.9218 | `models_pytorch/02_svhn/` |
| **9** | Diabetes (1D) | Keras | 3-Layer | 7.299 | 81.04s | 0.6467 | **69.25%** | 0.6584 | `models_keras/03_diabetes/` |
| **10**| Diabetes (1D) | Keras | 5-Layer | 43.555 | 213.09s | 0.6516 | **68.62%** | 0.6638 | `models_keras/03_diabetes/` |
| **11**| Diabetes (1D) | PyTorch (GPU) | 3-Layer | 7.299 | 46.16s | 0.6238 | **70.00%** | 0.6664 | `models_pytorch/03_diabetes/` |
| **12**| Diabetes (1D) | PyTorch (GPU) | 5-Layer | 43.043 | 59.99s | 0.5822 | **71.35%** | 0.6652 | `models_pytorch/03_diabetes/` |

---

## 3. Khởi chạy Ứng dụng Demo (Streamlit) & Jupyter Server

* **Khởi chạy ứng dụng Web Streamlit:**
  ```powershell
  streamlit run Smart_system_ass5/app.py
  ```
* **Mở Jupyter Notebook trên trình duyệt:**
  Truy cập địa chỉ: `http://127.0.0.1:8888/tree`
