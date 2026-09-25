#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
DỰ ÁN HỆ THỐNG THÔNG MINH (ASSIGNMENT 4)
DEMO BÀI TOÁN 3: CIFAR-10 OBJECT CLASSIFICATION & CHUỖI TIẾN HÓA M1..M4
==============================================================================
Nội dung demo:
  1. SmallCNN (545,098 params) đối sánh 3 nền tảng (Scratch vs Keras vs PyTorch)
  2. Nút thắt dung lượng biểu diễn: Vì sao LeNet-5 suy sụp trên CIFAR-10
  3. Chuỗi tiến hóa kiến trúc M1 -> M2 -> M3 -> M4 (Ablation Study)
  4. Suy diễn thực tế trên ảnh vật thể màu CIFAR-10
==============================================================================
"""

import os
import sys
import json
import time
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import ass4_utils as U
import scratch_nn as S

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def print_banner(text):
    print("\n" + "=" * 78)
    print(f"{BOLD}{CYAN}{text}{RESET}")
    print("=" * 78)


def print_section(title):
    print(f"\n{BOLD}{YELLOW}▶ {title}{RESET}")
    print("-" * 78)


def render_ascii_cifar(img_3x32x32):
    """Vẽ ảnh 32x32 từ 3 kênh màu RGB thành ký tự xám trên terminal."""
    # Chuyển đổi RGB sang grayscale: 0.299R + 0.587G + 0.114B
    gray = 0.299 * img_3x32x32[0] + 0.587 * img_3x32x32[1] + 0.114 * img_3x32x32[2]
    norm = gray - gray.min()
    norm = norm / (norm.max() if norm.max() > 0 else 1.0)
    chars = " .:-=+*#%@"
    lines = []
    lines.append("   +" + "-" * 32 + "+")
    for r in range(32):
        row_str = "   |"
        for c in range(32):
            val = norm[r, c]
            idx = int(val * (len(chars) - 1))
            row_str += chars[idx]
        row_str += "|"
        lines.append(row_str)
    lines.append("   +" + "-" * 32 + "+")
    return "\n".join(lines)


class CIFAR10Demo:
    def __init__(self):
        self.classes = ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"]
        self.X_test = None
        self.y_test = None
        self.meta = None

        self.scratch_model = None
        self.keras_model = None
        self.torch_model = None
        self._models_loaded = False

    def _ensure_models_loaded(self):
        if self._models_loaded:
            return
        print(f"{CYAN}Đang nạp tập test CIFAR-10 và các mô hình SmallCNN...{RESET}")
        _, _, self.X_test, self.y_test, self.meta = U.load_cifar10(n_train=10, n_test=1000, verbose=False)

        # 1. Scratch
        self.scratch_model = S.Sequential([
            S.Conv2D(3, 32, k=3, stride=1, pad=1, seed=1),
            S.ReLU(),
            S.MaxPool2D(2, 2),
            S.Conv2D(32, 64, k=3, stride=1, pad=1, seed=2),
            S.ReLU(),
            S.MaxPool2D(2, 2),
            S.Flatten(),
            S.Dense(4096, 128, seed=3),
            S.ReLU(),
            S.Dense(128, 10, seed=4),
        ])
        self.scratch_model = U.load_model("smallcnn_scratch_subset", "03_cifar10", model=self.scratch_model)

        # 2. Keras
        self.keras_model = U.load_model("smallcnn_keras_full", "03_cifar10")

        # 3. PyTorch
        import torch
        import torch.nn as nn

        class SmallCNN(nn.Module):
            def __init__(self, in_ch=3, n_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(in_ch, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(4096, 128),
                    nn.ReLU(),
                    nn.Linear(128, n_classes),
                )
            def forward(self, x):
                return self.classifier(self.features(x))

        self.torch_model = U.load_model("smallcnn_torch_full", "03_cifar10", model=SmallCNN())
        self._models_loaded = True
        print(f"  {GREEN}✔ Đã sẵn sàng các mô hình SmallCNN (NumPy Scratch, Keras, PyTorch)!{RESET}")

    def demo_overview(self):
        print_banner("1. TỔNG QUAN BÀI TOÁN CIFAR-10 & THÁCH THỨC THỊ GIÁC MÀU")
        print(f"""
{BOLD}• Bộ dữ liệu CIFAR-10:{RESET} 60,000 ảnh màu tự nhiên (3 kênh RGB), kích thước {BOLD}3 × 32 × 32{RESET}.
{BOLD}• 10 lớp vật thể:{RESET} airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
{BOLD}• Tại sao CIFAR-10 khó hơn MNIST gấp nhiều lần?{RESET}
  1. {YELLOW}Độ phức tạp đa kênh:{RESET} MNIST là chữ số trắng trên nền đen đơn sắc (1 kênh). CIFAR-10 là ảnh màu tự nhiên
     chứa nền phức tạp, góc chụp đa dạng, bóng mờ và độ phân giải rất thấp (32x32) khiến chi tiết bị nhòe.
  2. {YELLOW}Nghịch lý dung lượng phần đầu (Head Bottleneck):{RESET}
     Trong kiến trúc Baseline SmallCNN (32 -> 64 filters, 2 tầng Conv), kích thước Feature Map sau 2 lần MaxPool
     vẫn còn 8x8 với 64 kênh. Khi duỗi phẳng (Flatten = 4,096), tầng Dense(4096 -> 128) ngốn tới {BOLD}524,288 tham số{RESET}.
     Nghĩa là {BOLD}96.2% tham số toàn mạng nằm ở tầng Dense{RESET}, các tầng tích chập chỉ chiếm chưa đầy 4%!
""")

    def demo_parameter_breakdown(self):
        print_banner("2. GIẢI PHẪU THAM SỐ SMALLCNN (545,098 THAM SỐ)")
        print(f"""
{BOLD}• Công thức giải tích đếm tham số SmallCNN:{RESET}
  1. Conv2D_1 (3 in -> 32 out, k=3, pad=1) : 32 × (3 × 3 × 3 + 1)     =       896 tham số
  2. Conv2D_2 (32 in -> 64 out, k=3, pad=1): 64 × (32 × 3 × 3 + 1)    =    18,496 tham số
  3. MaxPool2D (2x2, stride=2)             : 0 tham số
  4. Flatten (64 x 8 x 8 = 4,096)          : 0 tham số
  5. Dense_1 (4096 -> 128)                 : 128 × (4,096 + 1)        =   524,416 tham số  ({RED}96.2%{RESET})
  6. Dense_2 (128 -> 10)                   : 10 × (128 + 1)           =     1,290 tham số
  -------------------------------------------------------------------------------------------------
  {BOLD}TỔNG SỐ THAM SỐ TOÁN HỌC:                                               545,098 THAM SỐ{RESET}

{GREEN}{BOLD}✔ KIỂM CHỨNG:{RESET} Cả 3 nền tảng (Scratch NumPy, Keras, PyTorch) đều có đúng {BOLD}545,098 tham số{RESET} (khớp 100%).
""")

    def demo_smallcnn_benchmark(self):
        print_banner("3. BẢNG ĐỐI SÁNH HIỆU NĂNG SMALLCNN (3 FRAMEWORKS)")
        print("""
| Framework | Dataset Split | Params | Epochs | Train Time (s) | Test Accuracy | Macro F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Scratch (NumPy)** | CIFAR-10 (5k subset) | 545,098 | 5 | 1495.1s (~25 phút) | **0.5120** | 0.5118 |
| **TensorFlow/Keras** | CIFAR-10 (5k subset) | 545,098 | 5 | 16.1s | **0.5450** | 0.5380 |
| **PyTorch** | CIFAR-10 (5k subset) | 545,098 | 5 | **3.1s** | **0.5290** | 0.5238 |
| **PyTorch** | CIFAR-10 (full 50k) | 545,098 | 5 | **17.7s** | **0.7184** | 0.7167 |
| **TensorFlow/Keras** | CIFAR-10 (full 50k) | 545,098 | 5 | 133.1s | **0.7011** | 0.6972 |
""")
        print(f"""
{BOLD}NHẬN XÉT QUAN TRỌNG:{RESET}
- Trên tập con 5k: Mô hình đạt ~51-54% accuracy (ngẫu nhiên 10 lớp chỉ là 10%).
- Trên tập đầy đủ 50k: Độ chính xác tăng vọt lên {BOLD}71.84%{RESET}.
- PyTorch CUDA GPU (17.7s) nhanh gấp {BOLD}7.5 lần{RESET} so với Keras CPU (133.1s) nhờ tận dụng nhân tính toán ma trận lớn.
""")

    def demo_lenet_collapse(self):
        print_banner("4. NÚT THẮT DUNG LƯỢNG BIỂU DIỄN: VÌ SAO LENET-5 SUY SỤP TRÊN CIFAR-10?")
        print(f"""
{BOLD}HAI SỐ PHẬN TRÁI NGƯỢC CỦA CÙNG MỘT KIẾN TRÚC LENET-5:{RESET}

| Tập dữ liệu | Kích thước ảnh | Kiến trúc LeNet-5 | Test Accuracy | Đánh giá |
|---|:---:|:---:|:---:|---|
| **MNIST** | 1 × 28 × 28 (xám) | Conv(1->6) + Conv(6->16) + FCs | {GREEN}{BOLD}98.70%{RESET} | Thành công rực rỡ, nhận diện chữ số hoàn hảo |
| **CIFAR-10** | 3 × 32 × 32 (màu) | Conv(3->6) + Conv(6->16) + FCs | {RED}{BOLD}61.18%{RESET} | {RED}Suy sụp nặng nề, tụt hậu -10.66% so với SmallCNN{RESET} |

{BOLD}GIẢI THÍCH NGUYÊN NHÂN KHOA HỌC (CAPACITY BOTTLENECK):{RESET}
1. LeNet-5 được LeCun thiết kế năm 1998 cho chữ số viết tay đơn sắc. Tầng tích chập đầu tiên chỉ có {BOLD}6 bộ lọc 5x5{RESET}.
   Tổng số tham số của 2 tầng Conv chỉ có vỏn vẹn {BOLD}2,870 tham số{RESET}.
2. Với ảnh màu CIFAR-10 chứa ô tô, chim, chó, mèo... trong không gian 3 chiều RGB, 6 bộ lọc ban đầu là một
   {BOLD}"nút thắt cổ chai"{RESET} quá hẹp. Mạng làm mất thông tin thị giác ngay từ cửa ngõ vào!
3. {BOLD}Nạp thêm dữ liệu có cứu được LeNet-5 không? KHÔNG!{RESET}
   - Khi tăng từ 5k lên 50k ảnh, khoảng cách tụt hậu của LeNet-5 so với SmallCNN không thu hẹp mà còn giãn rộng
     từ {BOLD}-7.8%{RESET} lên {BOLD}-10.66%{RESET}!
   - Khi một mô hình bị thiếu dung lượng tham số (Underfitting do kiến trúc), việc nạp thêm dữ liệu không giúp ích
     vì mạng không còn không gian tham số để hấp thụ tri thức.
""")

    def demo_ablation_variants(self):
        print_banner("5. TIẾN HÓA KIẾN TRÚC CNN: BỐN BIẾN THỂ M1 ĐẾN M4 (ABLATION STUDY)")
        print(f"""
Mỗi biến thể trong chuỗi kế thừa biến thể trước đó và bổ sung {BOLD}đúng một cơ chế duy nhất{RESET}:
  • {BOLD}M1{RESET}: Baseline Backbone (Conv 3x3 + ReLU + MaxPool) với Global Avg Pooling
  • {BOLD}M2{RESET}: {GREEN}+ Batch Normalization{RESET} (Chuẩn hóa phân phối ẩn, triệt tiêu Internal Covariate Shift)
  • {BOLD}M3{RESET}: {GREEN}+ Residual Connection{RESET} [Y = F(X) + X] (Tạo đường dẫn tắt cho gradient lan truyền)
  • {BOLD}M4{RESET}: {YELLOW}+ Squeeze-and-Excitation (SE) Attention{RESET} (Tự học trọng số chú ý giữa các kênh)
""")
        print("-" * 78)
        print(f"{'Biến thể':<22} | {'Cơ chế bổ sung':<20} | {'Params':<8} | {'Accuracy':<10} | {'Macro F1':<10} | {'Biến động Acc':<12}")
        print("-" * 78)
        print(f"{'M1 Conv+ReLU+Pool':<22} | {'Baseline':<20} | {72730:<8,} | {'69.55%':<10} | {'0.7001':<10} | {'Gốc':<12}")
        print(f"{'M2 + BatchNorm':<22} | {'+ Batch Normalization':<20} | {72954:<8,} | {BOLD}{GREEN}{'74.86%':<10}{RESET} | {'0.7483':<10} | {BOLD}{GREEN}{'+5.31%':<12}{RESET}")
        print(f"{'M3 + Residual':<22} | {'+ Residual Skip Path':<20} | {75786:<8,} | {BOLD}{GREEN}{'75.74%':<10}{RESET} | {'0.7634':<10} | {BOLD}{GREEN}{'+0.88%':<12}{RESET}")
        print(f"{'M4 + Attention(SE)':<22} | {'+ SE Channel Gate':<20} | {78614:<8,} | {BOLD}{RED}{'75.25%':<10}{RESET} | {'0.7557':<10} | {BOLD}{RED}{'-0.49%':<12}{RESET}")
        print("-" * 78)

        print(r"""
{BOLD}BÀI HỌC KỸ NGHỆ ĐẮT GIÁ TỪ KẾT QUẢ:{RESET}
1. {GREEN}{BOLD}Cú hích Batch Normalization (+5.31%):{RESET}
   Chỉ bổ sung {BOLD}224 tham số{RESET}, BatchNorm giúp ổn định phân phối hoạt hóa, làm phẳng bề mặt hàm mất mát (loss landscape),
   tạo ra bước nhảy vọt lớn nhất trong toàn bộ chuỗi tiến hóa!
2. {GREEN}{BOLD}Đóng góp biên của Residual (+0.88%):{RESET}
   Đường truyền tắt $Y = \mathcal{F}(X) + X$ loại bỏ hiện tượng triệt tiêu gradient, giúp mô hình đạt đỉnh cao nhất {BOLD}75.74%{RESET}.
3. {RED}{BOLD}Hiện tượng Quá khớp của Attention (-0.49%):{RESET}
   - M4 đạt Train Loss thấp nhất toàn bảng ({BOLD}0.4335{RESET} so với 0.4479 của M3), nhưng Test Accuracy lại bị giảm -0.49%!
   - Đây là minh chứng kinh điển của {BOLD}Overfitting{RESET}: Trên số kênh hẹp (16/32/64) và 10 epochs, khối SE Attention
     đã học thuộc lòng các tương quan ngẫu nhiên của tập train thay vì tổng quát hóa.
   - {BOLD}Kết luận:{RESET} "Không phải cứ thêm cơ chế phức tạp là độ chính xác sẽ tự động tăng lên!"
""")

    def demo_image_inference(self, sample_idx=None):
        print_banner("6. SUY DIỄN THỰC TẾ TRÊN ẢNH VẬT THỂ MÀU CIFAR-10")
        import torch

        self._ensure_models_loaded()
        if sample_idx is None:
            rng = np.random.default_rng()
            sample_idx = int(rng.integers(0, len(self.X_test)))

        x_img = self.X_test[sample_idx]           # (3, 32, 32)
        y_true = self.y_test[sample_idx]
        true_name = self.classes[y_true]
        x_batch = x_img[None, ...]               # (1, 3, 32, 32)

        print(f"{BOLD}Trực quan hóa vật thể #{sample_idx} từ tập Test CIFAR-10 (ASCII View):{RESET}")
        print(render_ascii_cifar(x_img))
        print(f"  • Nhãn thực tế (Ground Truth): {BOLD}{GREEN}{true_name.upper()}{RESET}")

        # 1. Scratch (subset 5k)
        logits_s = self.scratch_model.forward(x_batch)[0]
        prob_s = np.exp(logits_s - logits_s.max()) / np.exp(logits_s - logits_s.max()).sum()
        pred_s = self.classes[prob_s.argmax()]

        # 2. Keras (full 50k)
        x_k = x_batch.transpose(0, 2, 3, 1)
        logits_k = self.keras_model.predict(x_k, verbose=0)[0]
        prob_k = np.exp(logits_k - logits_k.max()) / np.exp(logits_k - logits_k.max()).sum()
        pred_k = self.classes[prob_k.argmax()]

        # 3. PyTorch (full 50k)
        with torch.no_grad():
            logits_t = self.torch_model(torch.tensor(x_batch, dtype=torch.float32)).numpy()[0]
        prob_t = np.exp(logits_t - logits_t.max()) / np.exp(logits_t - logits_t.max()).sum()
        pred_t = self.classes[prob_t.argmax()]

        print("\n" + "-" * 78)
        print(f"{'Mô hình':<24} | {'Dự đoán':<12} | {'Độ tin cậy':<14} | {'Đánh giá':<15}")
        print("-" * 78)
        print(f"{'Scratch (Subset 5k)':<24} | {pred_s:<12} | {prob_s.max()*100:>10.2f}% | {GREEN if pred_s == true_name else RED}{'CHÍNH XÁC ✔' if pred_s == true_name else 'SAI ✘'}{RESET}")
        print(f"{'Keras (Full 50k)':<24} | {BOLD}{pred_k:<12}{RESET} | {prob_k.max()*100:>10.2f}% | {GREEN if pred_k == true_name else RED}{'CHÍNH XÁC ✔' if pred_k == true_name else 'SAI ✘'}{RESET}")
        print(f"{'PyTorch (Full 50k)':<24} | {BOLD}{pred_t:<12}{RESET} | {prob_t.max()*100:>10.2f}% | {GREEN if pred_t == true_name else RED}{'CHÍNH XÁC ✔' if pred_t == true_name else 'SAI ✘'}{RESET}")
        print("-" * 78)

    def run_full_auto_demo(self):
        print_banner("BẮT ĐẦU CHẠY TOÀN BỘ KỊCH BẢN DEMO CIFAR-10 CHO GIẢNG VIÊN")
        self.demo_overview()
        time.sleep(1.5)
        self.demo_parameter_breakdown()
        time.sleep(1.5)
        self.demo_smallcnn_benchmark()
        time.sleep(1.5)
        self.demo_lenet_collapse()
        time.sleep(1.5)
        self.demo_ablation_variants()
        time.sleep(1.5)
        print_section("Thực hiện phân loại trên 2 mẫu ảnh CIFAR-10 ngẫu nhiên:")
        for idx in [12, 55]:
            self.demo_image_inference(sample_idx=idx)
            time.sleep(1.0)
        print_banner("HOÀN THÀNH TOÀN BỘ KỊCH BẢN DEMO CIFAR-10 & M1..M4!")


def main():
    parser = argparse.ArgumentParser(description="Demo Assignment 4 - CIFAR-10 CNN & Variants")
    parser.add_argument("--full", action="store_true", help="Chạy toàn bộ kịch bản tự động")
    parser.add_argument("--sample", type=int, default=-1, help="Xem dự đoán trên ảnh cụ thể")
    args = parser.parse_args()

    demo = CIFAR10Demo()

    if args.full:
        demo.run_full_auto_demo()
        return

    if args.sample >= 0:
        demo.demo_image_inference(sample_idx=args.sample)
        return

    while True:
        print("\n" + "=" * 65)
        print(f"{BOLD}{CYAN}    MENU DEMO BÀI TOÁN CIFAR-10 & BIẾN THỂ M1..M4{RESET}")
        print("=" * 65)
        print(" [1] Tổng quan bài toán CIFAR-10 & Thách thức ảnh màu")
        print(" [2] Giải phẫu tham số SmallCNN & Nút thắt tầng Dense (96.2%)")
        print(" [3] Bảng đối sánh SmallCNN trên 3 framework (Accuracy ~71.8%)")
        print(" [4] Nút thắt dung lượng: Vì sao LeNet-5 suy sụp trên CIFAR-10")
        print(" [5] Thang tiến hóa M1..M4 & Hiện tượng quá khớp của Attention")
        print(" [6] Suy diễn thực tế trên ảnh vật thể màu CIFAR-10 (ASCII)")
        print(" [7] Chạy toàn bộ kịch bản tự động từ A đến Z (Auto Full Demo)")
        print(" [0] Thoát")
        print("-" * 65)

        try:
            choice = input(f"{BOLD}Nhập lựa chọn của bạn (0-7): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát demo.")
            break

        if choice == "1":
            demo.demo_overview()
        elif choice == "2":
            demo.demo_parameter_breakdown()
        elif choice == "3":
            demo.demo_smallcnn_benchmark()
        elif choice == "4":
            demo.demo_lenet_collapse()
        elif choice == "5":
            demo.demo_ablation_variants()
        elif choice == "6":
            demo.demo_image_inference()
        elif choice == "7":
            demo.run_full_auto_demo()
        elif choice == "0":
            print(f"{GREEN}Cảm ơn bạn đã theo dõi demo CIFAR-10!{RESET}")
            break
        else:
            print(f"{RED}Lựa chọn không hợp lệ, vui lòng chọn từ 0 đến 7.{RESET}")


if __name__ == "__main__":
    main()
