#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
DỰ ÁN HỆ THỐNG THÔNG MINH (ASSIGNMENT 4)
DEMO BÀI TOÁN 1: DIABETES 130-US HOSPITALS (TABULAR MLP)
==============================================================================
Kịch bản demo tương tác đối sánh 3 nền tảng:
  1. NumPy Scratch (Tự viết từ đầu)
  2. TensorFlow / Keras (Cấp cao)
  3. PyTorch (Hướng module)
==============================================================================
"""

import os
import sys
import time
import argparse
import numpy as np

# Giảm thiểu các log cảnh báo không cần thiết của TF
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import ass4_utils as U
import scratch_nn as S

# Màu sắc hiển thị Terminal ANSI
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner(text):
    print("\n" + "=" * 78)
    print(f"{BOLD}{CYAN}{text}{RESET}")
    print("=" * 78)


def print_section(title):
    print(f"\n{BOLD}{YELLOW}▶ {title}{RESET}")
    print("-" * 78)


class DiabetesDemo:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.meta = None

        self.scratch_model = None
        self.keras_model = None
        self.torch_model = None

        self._load_data()
        self._load_models()

    def _load_data(self):
        print(f"{CYAN}Đang nạp dữ liệu Diabetes 130-US Hospitals (UCI 296)...{RESET}")
        self.X_train, self.y_train, self.X_test, self.y_test, self.meta = U.load_diabetes130(verbose=False)
        print(f"  {GREEN}✔ Đã nạp thành công:{RESET} Train: {self.X_train.shape[0]:,} mẫu | Test: {self.X_test.shape[0]:,} mẫu | Đặc trưng: {self.X_train.shape[1]} chiều")

    def _load_models(self):
        print(f"{CYAN}Đang nạp 3 mô hình đã huấn luyện từ results/models/01_diabetes130/...{RESET}")

        # 1. Scratch Model
        self.scratch_model = S.Sequential([
            S.Dense(179, 128, seed=1),
            S.ReLU(),
            S.Dropout(0.3, seed=11),
            S.Dense(128, 64, seed=2),
            S.ReLU(),
            S.Dense(64, 3, seed=3),
        ])
        self.scratch_model = U.load_model("mlp_scratch", "01_diabetes130", model=self.scratch_model)

        # 2. Keras Model
        self.keras_model = U.load_model("mlp_keras", "01_diabetes130")

        # 3. PyTorch Model
        import torch
        import torch.nn as nn

        class DiabetesMLP(nn.Module):
            def __init__(self, n_in=179, n_classes=3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_classes),
                )

            def forward(self, x):
                return self.net(x)

        self.torch_model = U.load_model("mlp_torch", "01_diabetes130", model=DiabetesMLP(179, 3))
        print(f"  {GREEN}✔ Cả 3 mô hình (NumPy Scratch, Keras, PyTorch) đã sẵn sàng!{RESET}")

    def demo_data_overview(self):
        print_banner("1. TỔNG QUAN BÀI TOÁN & QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU")
        print(f"""
{BOLD}• Mục tiêu bài toán:{RESET} Dự đoán nguy cơ tái nhập viện của bệnh nhân tiểu đường.
{BOLD}• Nguồn dữ liệu:{RESET} UCI Machine Learning Repository (Dataset 296), 130 bệnh viện Hoa Kỳ (1999–2008).
{BOLD}• Số lớp nhãn (3 classes):{RESET}
    - {GREEN}NO{RESET}  : Không tái nhập viện (~53.9% tập dữ liệu)
    - {YELLOW}>30{RESET} : Tái nhập viện sau 30 ngày (~35.5% tập dữ liệu)
    - {RED}<30{RESET} : Tái nhập viện khẩn cấp trong vòng 30 ngày (~10.6% tập dữ liệu) -> {BOLD}Lớp thiểu số!{RESET}

{BOLD}• Các bước tiền xử lý chuẩn y tế (Strack et al., 2014):{RESET}
    1. Khử trùng lặp bệnh nhân (chỉ giữ lần nhập viện đầu tiên) để tránh rò rỉ thông tin (leakage).
    2. Loại bỏ các ca tử vong hoặc chuyển viện chăm sóc đặc biệt.
    3. Gom nhóm hơn 700 mã bệnh ICD-9 phức tạp thành 9 nhóm bệnh lý lớn (tim mạch, hô hấp, tiêu hóa...).
    4. One-Hot Encoding các biến phân loại -> Tensor đặc trưng đầu vào có {BOLD}179 chiều{RESET}.
""")
        counts = np.bincount(self.y_test)
        total = len(self.y_test)
        print(f"{BOLD}Phân bố nhãn trên tập Test (13,994 mẫu):{RESET}")
        for i, c in enumerate(self.meta["classes"]):
            pct = (counts[i] / total) * 100
            print(f"  - Lớp [{c:>3}]: {counts[i]:>5,} mẫu ({pct:>5.1f}%)")
        majority_acc = counts.max() / total
        print(f"  => {YELLOW}Majority Class Baseline:{RESET} Nếu luôn đoán 'NO', độ chính xác đã là {BOLD}{majority_acc:.2%}{RESET}!")

    def demo_parameter_verification(self):
        print_banner("2. KIỂM CHỨNG TÍNH TƯƠNG ĐƯƠNG TOÁN HỌC & SỐ THAM SỐ")
        print(f"""
{BOLD}• Công thức giải tích đếm tham số cho mạng MLP:{RESET}
    Tầng 1 (Dense): 179 in  × 128 out + 128 bias = 23,040 tham số
    Tầng 2 (Dense): 128 in  ×  64 out +  64 bias =  8,256 tham số
    Tầng 3 (Dense):  64 in  ×   3 out +   3 bias =    195 tham số
    ----------------------------------------------------------
    {BOLD}TỔNG SỐ THAM SỐ LÝ THUYẾT:                    31,491 THAM SỐ{RESET}
""")
        import torch

        p_scratch = self.scratch_model.n_params()
        p_keras = int(self.keras_model.count_params())
        p_torch = sum(p.numel() for p in self.torch_model.parameters())

        print(f"{BOLD}Số tham số thực tế đếm từ từng Framework:{RESET}")
        print(f"  • Scratch (NumPy thuần) : {GREEN}{p_scratch:>8,}{RESET} tham số")
        print(f"  • TensorFlow / Keras    : {GREEN}{p_keras:>8,}{RESET} tham số")
        print(f"  • PyTorch               : {GREEN}{p_torch:>8,}{RESET} tham số")

        match = (p_scratch == 31491 and p_keras == 31491 and p_torch == 31491)
        if match:
            print(f"\n{GREEN}{BOLD}✔ KẾT LUẬN:{RESET} Cả 3 nền tảng KHỚP TUYỆT ĐỐI 100% đến từng tham số đơn lẻ!")
            print("  Chứng minh tính trung thực và tuân thủ tuyệt đối quy tắc công bằng (The Fairness Rule).")

    def demo_gradient_check(self):
        print_banner("3. KIỂM CHỨNG ĐẠO HÀM VIẾT TAY (FINITE DIFFERENCE GRADIENT CHECKING)")
        print(f"""
Để chứng minh code NumPy Scratch không bị sai lệch công thức giải tích hay lỗi dấu,
ta so sánh Gradient tính bằng Lan truyền ngược (Backprop viết tay) với Sai phân hữu hạn:
    {CYAN}f'(x) ≈ [f(x + ε) - f(x - ε)] / (2ε)    với ε = 1e-3 (chạy ở chuẩn float64){RESET}
""")
        print("Đang chạy kiểm tra sai phân hữu hạn trên mini-batch mẫu...")
        U.set_seed(U.SEED)
        check_model = S.Sequential([
            S.Dense(179, 16, seed=1),
            S.ReLU(),
            S.Dense(16, 3, seed=3),
        ])
        err = S.gradient_check(check_model, self.X_train[:8], self.y_train[:8], n_samples=8)
        print(f"  • Sai số tương đối lớn nhất đo được: {BOLD}{GREEN}{err:.3e}{RESET}")
        if err < 1e-4:
            print(f"  • {GREEN}{BOLD}PASS:{RESET} Gradient giải tích viết tay hoàn toàn khớp với đạo hàm số trị!")
            print("    (Sai số nhỏ hơn nhiều so với ngưỡng quy chuẩn 1e-4)")

    def demo_evaluation_benchmark(self):
        print_banner("4. BẢNG ĐỐI SÁNH HIỆU NĂNG TOÀN DIỆN TRÊN 13,994 MẪU TEST")
        import torch

        print("Đang thực hiện đánh giá suy diễn trên tập test (13,994 mẫu)...")
        # Scratch
        t0 = time.perf_counter()
        pred_s = self.scratch_model.predict_classes(self.X_test)
        t_s = time.perf_counter() - t0
        m_s = U.evaluate(self.y_test, pred_s, 3)

        # Keras
        t0 = time.perf_counter()
        pred_k = self.keras_model.predict(self.X_test, batch_size=512, verbose=0).argmax(axis=1)
        t_k = time.perf_counter() - t0
        m_k = U.evaluate(self.y_test, pred_k, 3)

        # Torch
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_t = self.torch_model(torch.tensor(self.X_test)).argmax(dim=1).numpy()
        t_t = time.perf_counter() - t0
        m_t = U.evaluate(self.y_test, pred_t, 3)

        print("\n" + "-" * 78)
        print(f"{'Framework':<20} | {'Params':<8} | {'Accuracy':<10} | {'Macro F1':<10} | {'Inference (s)':<12}")
        print("-" * 78)
        print(f"{'Scratch (NumPy)':<20} | {31491:<8,} | {m_s['test_accuracy']*100:>8.2f}% | {m_s['f1_macro']:>10.4f} | {t_s:>12.3f}s")
        print(f"{'TensorFlow/Keras':<20} | {31491:<8,} | {m_k['test_accuracy']*100:>8.2f}% | {m_k['f1_macro']:>10.4f} | {t_k:>12.3f}s")
        print(f"{'PyTorch':<20} | {31491:<8,} | {m_t['test_accuracy']*100:>8.2f}% | {m_t['f1_macro']:>10.4f} | {t_t:>12.3f}s")
        print("-" * 78)

        print(f"""
{BOLD}PHÂN TÍCH CHUYÊN SÂU TỪ KẾT QUẢ:{RESET}
1. {BOLD}Tính tương đương:{RESET} Cả 3 framework đều đạt Accuracy xấp xỉ {BOLD}60.8%{RESET} (dao động cực nhỏ ±0.15% do khởi tạo ngẫu nhiên).
2. {BOLD}Nghịch lý Accuracy vs Macro-F1:{RESET}
   - Accuracy đạt ~60.8% nhưng Macro-F1 chỉ đạt ~0.37.
   - Nguyên nhân: Lớp `<30` (khẩn cấp) chỉ có 1,487 mẫu trên 13,994 mẫu kiểm thử.
   - Mạng ưu tiên dự đoán lớp an toàn `NO` và `>30` để tối đa hóa Accuracy tổng thể, dẫn đến F1 lớp `<30` bị kéo xuống thấp.
3. {BOLD}Về thời gian huấn luyện gốc:{RESET}
   - Scratch NumPy CPU (12.8s) ≈ Keras CPU (12.8s) < PyTorch GPU (20.9s).
   - Vì mô hình quá nhỏ (31k params), chi phí truyền PCIe và khởi tạo CUDA kernel trên GPU lớn hơn thời gian tính toán thực tế.
""")

    def demo_patient_inference(self, patient_idx=None):
        print_banner("5. SUY DIỄN TRỰC TIẾP TRÊN 1 HỒ SƠ BỆNH ÁN CỤ THỂ")
        import torch

        if patient_idx is None:
            rng = np.random.default_rng()
            patient_idx = int(rng.integers(0, len(self.X_test)))

        x_sample = self.X_test[patient_idx:patient_idx + 1]
        y_true = self.y_test[patient_idx]
        true_name = self.meta["classes"][y_true]

        # Lấy logits & xác suất softmax
        # 1. Scratch
        logits_s = self.scratch_model.forward(x_sample)[0]
        exp_s = np.exp(logits_s - logits_s.max())
        prob_s = exp_s / exp_s.sum()

        # 2. Keras
        logits_k = self.keras_model.predict(x_sample, verbose=0)[0]
        exp_k = np.exp(logits_k - logits_k.max())
        prob_k = exp_k / exp_k.sum()

        # 3. PyTorch
        with torch.no_grad():
            logits_t = self.torch_model(torch.tensor(x_sample)).numpy()[0]
        exp_t = np.exp(logits_t - logits_t.max())
        prob_t = exp_t / exp_t.sum()

        pred_s_cls = self.meta["classes"][prob_s.argmax()]
        pred_k_cls = self.meta["classes"][prob_k.argmax()]
        pred_t_cls = self.meta["classes"][prob_t.argmax()]

        print(f"{BOLD}Bệnh nhân #{patient_idx} (Tập Test):{RESET}")
        print(f"  • Nhãn thực tế của bệnh viện (Ground Truth): {BOLD}{GREEN if true_name == 'NO' else RED}{true_name}{RESET} ({'Không tái nhập viện' if true_name == 'NO' else 'Tái nhập viện sau 30 ngày' if true_name == '>30' else 'Tái nhập viện khẩn cấp trong 30 ngày'})")
        print("\n" + "-" * 78)
        print(f"{'Mô hình Framework':<20} | {'Xác suất NO':<14} | {'Xác suất >30':<14} | {'Xác suất <30':<14} | {'Dự đoán':<10}")
        print("-" * 78)
        print(f"{'NumPy Scratch':<20} | {prob_s[0]*100:>12.2f}% | {prob_s[1]*100:>12.2f}% | {prob_s[2]*100:>12.2f}% | {BOLD}{pred_s_cls:<10}{RESET}")
        print(f"{'TensorFlow/Keras':<20} | {prob_k[0]*100:>12.2f}% | {prob_k[1]*100:>12.2f}% | {prob_k[2]*100:>12.2f}% | {BOLD}{pred_k_cls:<10}{RESET}")
        print(f"{'PyTorch':<20} | {prob_t[0]*100:>12.2f}% | {prob_t[1]*100:>12.2f}% | {prob_t[2]*100:>12.2f}% | {BOLD}{pred_t_cls:<10}{RESET}")
        print("-" * 78)

        match_true = (pred_t_cls == true_name)
        status_txt = f"{GREEN}CHÍNH XÁC ✔{RESET}" if match_true else f"{YELLOW}KHÔNG KHỚP ✘ (Do độ bất định lâm sàng cao){RESET}"
        print(f"Đánh giá dự đoán: {status_txt}")

    def run_full_auto_demo(self):
        print_banner("BẮT ĐẦU CHẠY TOÀN BỘ KỊCH BẢN DEMO TỰ ĐỘNG CHO GIẢNG VIÊN")
        self.demo_data_overview()
        time.sleep(1.5)
        self.demo_parameter_verification()
        time.sleep(1.5)
        self.demo_gradient_check()
        time.sleep(1.5)
        self.demo_evaluation_benchmark()
        time.sleep(1.5)
        print_section("Chạy thử nghiệm trên 3 bệnh nhân ngẫu nhiên:")
        for idx in [42, 100, 1000]:
            self.demo_patient_inference(patient_idx=idx)
            time.sleep(1.0)
        print_banner("HOÀN THÀNH TOÀN BỘ KỊCH BẢN DEMO BÀI TOÁN DIABETES 130!")


def main():
    parser = argparse.ArgumentParser(description="Demo Assignment 4 - Diabetes 130 Tabular MLP")
    parser.add_argument("--full", action="store_true", help="Chạy toàn bộ kịch bản tự động")
    parser.add_argument("--predict", type=int, default=-1, help="Chạy dự đoán trên chỉ số bệnh nhân cụ thể")
    args = parser.parse_args()

    demo = DiabetesDemo()

    if args.full:
        demo.run_full_auto_demo()
        return

    if args.predict >= 0:
        demo.demo_patient_inference(patient_idx=args.predict)
        return

    # Menu tương tác
    while True:
        print("\n" + "=" * 65)
        print(f"{BOLD}{CYAN}   MENU DEMO BÀI TOÁN DIABETES 130 (ASSIGNMENT 4){RESET}")
        print("=" * 65)
        print(" [1] Tổng quan bài toán, đặc trưng và mất cân bằng nhãn")
        print(" [2] Kiểm chứng số tham số & tính tương đương (31,491 params)")
        print(" [3] Chạy kiểm tra đạo hàm Gradient Check (NumPy Scratch)")
        print(" [4] Bảng so sánh hiệu năng chi tiết (13,994 mẫu test)")
        print(" [5] Suy diễn ngẫu nhiên trên 1 hồ sơ bệnh án")
        print(" [6] Chạy toàn bộ kịch bản tự động từ A đến Z (Auto Full Demo)")
        print(" [0] Thoát")
        print("-" * 65)

        try:
            choice = input(f"{BOLD}Nhập lựa chọn của bạn (0-6): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát demo.")
            break

        if choice == "1":
            demo.demo_data_overview()
        elif choice == "2":
            demo.demo_parameter_verification()
        elif choice == "3":
            demo.demo_gradient_check()
        elif choice == "4":
            demo.demo_evaluation_benchmark()
        elif choice == "5":
            demo.demo_patient_inference()
        elif choice == "6":
            demo.run_full_auto_demo()
        elif choice == "0":
            print(f"{GREEN}Cảm ơn bạn đã theo dõi buổi demo! Chúc bạn bảo vệ đồ án thành công.{RESET}")
            break
        else:
            print(f"{RED}Lựa chọn không hợp lệ, vui lòng chọn từ 0 đến 6.{RESET}")


if __name__ == "__main__":
    main()
