# -*- coding: utf-8 -*-
"""
Export all PlantUML code blocks from report into individual files
Directory: D:\school_project\PTTK\code uml\
"""

import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

from generate_full_report import (
    PUML_UC_OVERALL, PUML_UC_MOD1, PUML_UC_MOD2,
    PUML_ENTITY_SYSTEM, PUML_CLASS_MOD1, PUML_CLASS_MOD2,
    PUML_STATE_MOD1, PUML_STATE_MOD2, PUML_COMM_MOD1, PUML_COMM_MOD2
)

output_dir = os.path.join("D:\\school_project\\PTTK", "code uml")
os.makedirs(output_dir, exist_ok=True)

diagrams = [
    (
        "01_usecase_tong_quan.puml",
        PUML_UC_OVERALL,
        "Biểu đồ Use Case tổng quan toàn hệ thống (Assignment 1)"
    ),
    (
        "02_usecase_chi_tiet_module1.puml",
        PUML_UC_MOD1,
        "Biểu đồ Use Case chi tiết Module 1: Customer searches for items (Assignment 1)"
    ),
    (
        "03_usecase_chi_tiet_module2.puml",
        PUML_UC_MOD2,
        "Biểu đồ Use Case chi tiết Module 2: Warehouse staff approves orders + export (Assignment 1)"
    ),
    (
        "04_so_do_lop_thuc_the_he_thong.puml",
        PUML_ENTITY_SYSTEM,
        "Sơ đồ lớp thực thể pha phân tích toàn hệ thống (Assignment 2 / Mục 1 trên Google Form)"
    ),
    (
        "05_so_do_lop_module1.puml",
        PUML_CLASS_MOD1,
        "Sơ đồ lớp phân tích tĩnh Module 1: Customer searches for items (Assignment 3 / Mục 2 trên Google Form)"
    ),
    (
        "06_so_do_lop_module2.puml",
        PUML_CLASS_MOD2,
        "Sơ đồ lớp phân tích tĩnh Module 2: Warehouse staff approves orders + export (Assignment 3 / Mục 3 trên Google Form)"
    ),
    (
        "07_bieu_do_trang_thai_module1.puml",
        PUML_STATE_MOD1,
        "Biểu đồ chuyển trạng thái Module 1: Customer searches for items (Assignment 3)"
    ),
    (
        "08_bieu_do_trang_thai_module2.puml",
        PUML_STATE_MOD2,
        "Biểu đồ chuyển trạng thái Module 2: Warehouse staff approves orders + export (Assignment 3)"
    ),
    (
        "09_communication_diagram_module1.puml",
        PUML_COMM_MOD1,
        "Biểu đồ giao tiếp Module 1: Customer searches for items (Assignment 4 / Mục 4 trên Google Form)"
    ),
    (
        "10_communication_diagram_module2.puml",
        PUML_COMM_MOD2,
        "Biểu đồ giao tiếp Module 2: Warehouse staff approves orders + export (Assignment 4 / Mục 5 trên Google Form)"
    )
]

for filename, content, desc in diagrams:
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Exported: {filename} ({desc})")

# Write README.md inside 'code uml'
readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write("# DANH MỤC CÁC FILE MÃ NGUỒN PLANTUML\n\n")
    f.write("Thư mục này chứa toàn bộ 10 file mã nguồn PlantUML trích xuất từ báo cáo `09D23DCVT103`.\n\n")
    f.write("### ⭐ BẢNG ĐỐI CHIẾU 5 MỤC NỘP BÀI TRÊN FORM TRỰC TUYẾN:\n\n")
    f.write("| Mục trên Form | File PlantUML tương ứng | Mô tả nội dung |\n")
    f.write("| :--- | :--- | :--- |\n")
    f.write("| **1. Sơ đồ lớp thực thể** | `04_so_do_lop_thuc_the_he_thong.puml` | Toàn bộ các lớp thực thể hệ thống và quan hệ đối tượng |\n")
    f.write("| **2. Sơ đồ lớp module 1** | `05_so_do_lop_module1.puml` | Lớp biên và lớp thực thể cho chức năng tìm kiếm mặt hàng |\n")
    f.write("| **3. Sơ đồ lớp module 2** | `06_so_do_lop_module2.puml` | Lớp biên và lớp thực thể cho chức năng duyệt và xuất đơn hàng |\n")
    f.write("| **4. Communication diagram module 1** | `09_communication_diagram_module1.puml` | Biểu đồ giao tiếp đánh số thông điệp tuần tự Module 1 |\n")
    f.write("| **5. Communication diagram module 2** | `10_communication_diagram_module2.puml` | Biểu đồ giao tiếp đánh số thông điệp tuần tự Module 2 |\n\n")
    f.write("### 📌 CÁC FILE BIỂU ĐỒ BỔ TRỢ KHÁC:\n\n")
    f.write("- `01_usecase_tong_quan.puml`: Biểu đồ Use Case toàn hệ thống (Assignment 1)\n")
    f.write("- `02_usecase_chi_tiet_module1.puml`: Biểu đồ Use Case chi tiết Module 1 (Assignment 1)\n")
    f.write("- `03_usecase_chi_tiet_module2.puml`: Biểu đồ Use Case chi tiết Module 2 (Assignment 1)\n")
    f.write("- `07_bieu_do_trang_thai_module1.puml`: Biểu đồ trạng thái Module 1 (Assignment 3)\n")
    f.write("- `08_bieu_do_trang_thai_module2.puml`: Biểu đồ trạng thái Module 2 (Assignment 3)\n")

print("\nREADME.md generated successfully inside 'code uml'!")
