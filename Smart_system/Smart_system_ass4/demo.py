#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
DỰ ÁN HỆ THỐNG THÔNG MINH (ASSIGNMENT 4)
TRÌNH ĐIỀU KHIỂN DEMO TỔNG HỢP CẢ 3 BÀI TOÁN (MASTER DEMO LAUNCHER)
==============================================================================
"""

import os
import sys
import subprocess

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    while True:
        print("\n" + "=" * 70)
        print(f"{BOLD}{CYAN}   HỆ THỐNG THÔNG MINH — TRUNG TÂM DEMO BẢO VỆ ĐỒ ÁN (A4){RESET}")
        print("=" * 70)
        print(f" {BOLD}[1]{RESET} Bài toán 1: {BOLD}Diabetes 130-US Hospitals{RESET} (Dữ liệu bảng - MLP 179 chiều)")
        print(f" {BOLD}[2]{RESET} Bài toán 2: {BOLD}MNIST Handwritten Digits{RESET} (Ảnh xám 1x28x28 - CNN 20k params)")
        print(f" {BOLD}[3]{RESET} Bài toán 3: {BOLD}CIFAR-10 & Thang tiến hóa M1..M4{RESET} (Ảnh màu 3x32x32)")
        print(f" {BOLD}[4]{RESET} Xem tóm tắt đối sánh toàn diện & Cheat Sheet phản biện")
        print(f" {BOLD}[0]{RESET} Thoát")
        print("-" * 70)

        try:
            choice = input(f"{BOLD}Chọn bài toán bạn muốn demo (0-4): {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nĐã thoát.")
            break

        if choice == "1":
            subprocess.run([sys.executable, os.path.join(HERE, "demo_diabetes.py")])
        elif choice == "2":
            subprocess.run([sys.executable, os.path.join(HERE, "demo_mnist.py")])
        elif choice == "3":
            subprocess.run([sys.executable, os.path.join(HERE, "demo_cifar10.py")])
        elif choice == "4":
            guide_path = os.path.join(HERE, "HUONG_DAN_THUYET_TRINH_VA_BAO_VE.md")
            print(f"\n{GREEN}Tài liệu hướng dẫn phản biện đầy đủ tại:{RESET} {guide_path}")
            print(f"{CYAN}Bản Master PDF 172 trang tại:{RESET} {os.path.join(HERE, 'pdf', 'FINAL_REPORT_A4.pdf')}")
            input(f"\n{BOLD}Nhấn Enter để quay lại menu chính...{RESET}")
        elif choice == "0":
            print(f"{GREEN}Chúc bạn có buổi demo và bảo vệ đồ án thành công rực rỡ!{RESET}\n")
            break
        else:
            print(f"{RED}Lựa chọn không hợp lệ, vui lòng chọn từ 0 đến 4.{RESET}")


if __name__ == "__main__":
    main()
