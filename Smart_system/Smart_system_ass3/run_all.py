"""Script điều khiển chạy toàn bộ hoặc từng phần thí nghiệm của Assignment 03.

Cách dùng:
    python run_all.py --all                 # Chạy toàn bộ 5 phần
    python run_all.py --step 4              # Chạy riêng Deep Learning tiểu đường (NumPy thuần)
    python run_all.py --step 5              # Chạy riêng K-Nearest Neighbors (3 category, 4 task)
    python run_all.py --step 6              # Chạy riêng AdaBoost (3 category, 4 task)
    python run_all.py --step 7              # Chạy riêng Deep Learning hồi quy giá nhà (NumPy thuần)
    python run_all.py --step 8              # Chạy riêng Deep Learning nhúng từ E-commerce (NumPy thuần)
    python run_all.py --step 4 --skip-lr-study # Chạy step 4 bỏ qua quét GD toàn tập (rất nhanh)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = {
    4: ("scripts/train_04_deep_learning.py", "Deep Learning từ đầu với NumPy (Tiểu đường)"),
    5: ("scripts/train_05_knn.py", "K-Nearest Neighbors (3 category, 4 task)"),
    6: ("scripts/train_06_adaboost.py", "AdaBoost (3 category, 4 task)"),
    7: ("scripts/train_07_deep_learning_house.py", "Deep Learning hồi quy giá nhà (NumPy thuần)"),
    8: ("scripts/train_08_deep_learning_ecommerce.py", "Deep Learning nhúng từ E-commerce (NumPy thuần)"),
}


def run_step(step_num: int, extra_args: list[str] = None):
    if step_num not in STEPS:
        print(f"Bước không hợp lệ: {step_num}. Chọn từ {list(STEPS.keys())}")
        return False

    script_rel, title = STEPS[step_num]
    script_path = ROOT / script_rel
    cmd = [PYTHON, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print("\n" + "=" * 75)
    print(f"BẮT ĐẦU BƯỚC {step_num}: {title}")
    print(f"Lệnh: {' '.join(cmd)}")
    print("=" * 75 + "\n")

    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"\n[LỖI] Bước {step_num} thất bại với mã lỗi {res.returncode}")
        return False
    print(f"\n[THÀNH CÔNG] Hoàn thành bước {step_num}: {title}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm Assignment 03")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ từ bước 4 đến bước 8")
    parser.add_argument("--step", type=int, choices=list(STEPS.keys()), help="Chọn bước cần chạy (4, 5, 6, 7, hoặc 8)")
    parser.add_argument("--skip-lr-study", action="store_true", help="Áp dụng cho bước 4: bỏ qua bài test GD toàn tập 1000 epoch để chạy nhanh hơn")

    args = parser.parse_args()

    if not args.all and args.step is None:
        parser.print_help()
        print("\nGợi ý: Thử chạy bước 4 nhanh bằng lệnh:")
        print("    python run_all.py --step 4 --skip-lr-study")
        return

    if args.all:
        for step_num in sorted(STEPS.keys()):
            extra = ["--skip-lr-study"] if step_num == 4 and args.skip_lr_study else []
            ok = run_step(step_num, extra)
            if not ok:
                print(f"Dừng quá trình chạy tại bước {step_num}")
                break
    else:
        extra = ["--skip-lr-study"] if args.step == 4 and args.skip_lr_study else []
        run_step(args.step, extra)


if __name__ == "__main__":
    main()
