"""Assemble all report sections and full executed notebooks into a single master PDF.

Produces a publication-grade, book-length master report matching the 165-page standard
of A4_02_HungNguyenBa_120.pdf.
"""

from __future__ import annotations

import os
import pypdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(HERE, "pdf")

SECTIONS = [
    # (filename, bookmark_title, parent_bookmark)
    ("FINAL_REPORT_A4_BODY.pdf", "Phần I: Báo cáo Khoa học & Phân tích Chuyên sâu (Chương 1 - 10)", None),
    ("00_inventory.pdf", "Phụ lục A: Notebook 00 — Kiểm kê Dữ liệu & Đảm bảo Tính Toàn vẹn", "Phần II: Toàn văn Thực nghiệm Notebooks"),
    ("01_diabetes130.pdf", "Phụ lục B: Notebook 01 — Dữ liệu bảng: Diabetes 130-US Hospitals", "Phần II: Toàn văn Thực nghiệm Notebooks"),
    ("02_mnist.pdf", "Phụ lục C: Notebook 02 — Dữ liệu ảnh: MNIST Phân loại chữ số viết tay", "Phần II: Toàn văn Thực nghiệm Notebooks"),
    ("03_cifar10.pdf", "Phụ lục D: Notebook 03 — Dữ liệu ảnh: CIFAR-10 Phân loại ảnh màu tự nhiên", "Phần II: Toàn văn Thực nghiệm Notebooks"),
    ("04_compare.pdf", "Phụ lục E: Notebook 04 — So sánh tổng hợp & Chuỗi tiến hóa kiến trúc CNN (M1..M4)", "Phần II: Toàn văn Thực nghiệm Notebooks"),
    ("06_mnist_lenet.pdf", "Phụ lục F.1: Notebook 06 — Hiện đại hóa LeNet-5 trên MNIST", "Phần II: Toàn văn Thực nghiệm Notebooks"),
    ("07_cifar10_lenet.pdf", "Phụ lục F.2: Notebook 07 — Hiện đại hóa LeNet-5 trên CIFAR-10", "Phần II: Toàn văn Thực nghiệm Notebooks"),
    ("APPENDIX_G_SOURCE.pdf", "Phụ lục G: Toàn văn Mã nguồn Thư viện cốt lõi (ass4_utils, scratch_nn, variants)", "Phần III: Toàn văn Mã nguồn & Bảng Thuật ngữ"),
    ("APPENDIX_H_GLOSSARY.pdf", "Phụ lục H: Bảng Tra cứu Thuật ngữ Chuyên ngành Deep Learning & CNN Việt – Anh", "Phần III: Toàn văn Mã nguồn & Bảng Thuật ngữ"),
]


def assemble():
    writer = pypdf.PdfWriter()
    current_page = 0
    parents = {}

    print("Assembling master PDF report...")
    for filename, title, parent_group in SECTIONS:
        path = os.path.join(PDF_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing component PDF: {path}")

        reader = pypdf.PdfReader(path)
        n_pages = len(reader.pages)
        start_page = current_page

        # Handle outline hierarchy
        parent_outline = None
        if parent_group:
            if parent_group not in parents:
                # Create parent bookmark at the start page of its first child
                parent_outline = writer.add_outline_item(parent_group, start_page)
                parents[parent_group] = parent_outline
            else:
                parent_outline = parents[parent_group]

        writer.add_outline_item(title, start_page, parent=parent_outline)

        # Append all pages from this section
        for page in reader.pages:
            writer.add_page(page)
            current_page += 1

        safe_title = title.encode("ascii", "backslashreplace").decode()
        print(f" - [{filename:<24}] : {n_pages:>3} pages (Page {start_page + 1:>3} to {current_page:>3}) -> {safe_title[:60]}")

    out_full = os.path.join(PDF_DIR, "FINAL_REPORT_A4_FULL.pdf")
    out_main = os.path.join(PDF_DIR, "FINAL_REPORT_A4.pdf")
    out_dung = os.path.join(PDF_DIR, "A4_02_NguyenDaiDung_103-1.pdf")

    with open(out_full, "wb") as f:
        writer.write(f)
    print(f"Saved master file: {out_full} ({os.path.getsize(out_full) / (1024*1024):.2f} MB)")

    try:
        with open(out_main, "wb") as f:
            writer.write(f)
        print(f"Saved primary file: {out_main} ({os.path.getsize(out_main) / (1024*1024):.2f} MB)")
    except PermissionError:
        print(f"[NOTE] '{out_main}' is currently open in user's PDF viewer, saved to '{out_full}' instead.")

    try:
        with open(out_dung, "wb") as f:
            writer.write(f)
        print(f"Saved student report: {out_dung} ({os.path.getsize(out_dung) / (1024*1024):.2f} MB)")
    except PermissionError:
        print(f"[NOTE] '{out_dung}' is currently open in user's PDF viewer.")

    print("=" * 70)
    print(f"MASTER REPORT ASSEMBLED SUCCESSFULLY!")
    print(f"Total Pages: {current_page} pages")
    print(f"Output 1: {out_main} ({os.path.getsize(out_main) / (1024*1024):.2f} MB)")
    print(f"Output 2: {out_full} ({os.path.getsize(out_full) / (1024*1024):.2f} MB)")
    print(f"Output 3: {out_dung} ({os.path.getsize(out_dung) / (1024*1024):.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    assemble()
