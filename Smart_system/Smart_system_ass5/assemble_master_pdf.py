"""
Master PDF Report Assembler for Assignment 5.
Matches the structure, styling, bookmark hierarchy, and publication standards of Assignment 4
(A4_02_NguyenDaiDung_103-1.pdf).

Generates:
1. report/A5_02_NguyenDaiDung_103-1.pdf
2. pdf/A5_02_NguyenDaiDung_103-1.pdf
3. report/FINAL_REPORT_ASS5_FULL.pdf
"""

import os
import sys
import json
import subprocess
import pypdf
import markdown

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "report")
HTML_DIR = os.path.join(HERE, "_html")
PDF_DIR = os.path.join(HERE, "pdf")

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

NOTEBOOKS = [
    ("01_cnn_theory_and_code", "Phụ lục A: Notebook 01 — Cơ sở Lý thuyết CNN & Code đối chiếu Keras vs PyTorch"),
    ("02_data_preparation_and_augmentation", "Phụ lục B: Notebook 02 — Chuẩn bị Dữ liệu & Kỹ thuật Tăng cường Dữ liệu Bảng (Mixup 513k dòng)"),
    ("03_keras_experiments", "Phụ lục C: Notebook 03 — Thực nghiệm Huấn luyện & Đánh giá trên Keras"),
    ("04_pytorch_experiments", "Phụ lục D: Notebook 04 — Thực nghiệm Huấn luyện & Đánh giá trên PyTorch GPU RTX 3060"),
    ("05_comparison_and_benchmarking", "Phụ lục E: Notebook 05 — Đánh giá Đối chuẩn Tổng hợp & Trực quan hóa"),
]

CSS = """
@page { 
    size: A4; 
    margin: 16mm 14mm; 
}

body {
    font-family: "Segoe UI", "Tahoma", "DejaVu Sans", sans-serif;
    font-size: 10.5pt;
    line-height: 1.58;
    color: #1a1a1a;
    max-width: 100%;
}

h1, h2, h3, h4 {
    font-family: "Segoe UI Semibold", "Segoe UI", Tahoma, sans-serif;
    line-height: 1.25;
    break-after: avoid;
    page-break-after: avoid;
}

h1 {
    font-size: 18pt;
    color: #0b3d62;
    border-bottom: 2px solid #0b3d62;
    padding-bottom: 5px;
    margin-top: 26px;
    break-before: page;
    page-break-before: page;
}

h1:nth-of-type(1),
h1:nth-of-type(2) {
    break-before: auto;
    page-break-before: auto;
}

h2 { 
    font-size: 13.5pt; 
    color: #14538a; 
    margin-top: 20px; 
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 3px;
}

h3 { 
    font-size: 11.5pt; 
    color: #2d3748; 
    margin-top: 14px; 
}

code {
    font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 9pt;
    background: #f1f5f9;
    padding: 1px 5px;
    border-radius: 4px;
    color: #0f172a;
}

pre {
    font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 8.5pt;
    line-height: 1.35;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #0284c7;
    border-radius: 5px;
    padding: 10px 12px;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-wrap: anywhere;
    break-inside: avoid;
    page-break-inside: avoid;
}

pre code { 
    background: none; 
    padding: 0; 
    font-size: inherit; 
}

table {
    border-collapse: collapse;
    width: 100%;
    font-size: 9pt;
    margin: 14px 0;
    break-inside: avoid;
    page-break-inside: avoid;
}

th, td { 
    border: 1px solid #cbd5e1; 
    padding: 6px 9px; 
    text-align: left; 
}

th { 
    background: #e2e8f0; 
    font-weight: 600; 
    color: #0f172a;
}

tr:nth-child(even) td { 
    background: #f8fafc; 
}

blockquote {
    margin: 12px 0;
    padding: 8px 14px;
    border-left: 4px solid #eab308;
    background: #fefce8;
    break-inside: avoid;
    page-break-inside: avoid;
}

blockquote p { 
    margin: 4px 0; 
}

hr { 
    border: none; 
    border-top: 1px solid #cbd5e1; 
    margin: 20px 0; 
}

ul, ol { 
    margin: 8px 0; 
    padding-left: 22px; 
}

li { 
    margin: 3px 0; 
}

strong { 
    color: #0b3d62; 
}

/* MathJax styling */
mjx-container {
    display: inline-block;
    margin: 0 !important;
}
mjx-container[jax="SVG"][display="true"] {
    display: block !important;
    text-align: center;
    margin: 12px 0 !important;
}
"""

MATHJAX_HEADER = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  },
  svg: { fontCache: 'global' },
  startup: {
    pageReady: () => {
      return MathJax.startup.defaultPageReady().then(() => {
        document.body.classList.add('mathjax-ready');
      });
    }
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""

def render_markdown_body():
    """Renders Part I & Part III from FINAL_REPORT_ASS5.md to PDF."""
    md_source = os.path.join(REPORT_DIR, "FINAL_REPORT_ASS5.md")
    html_output = os.path.join(HTML_DIR, "FINAL_REPORT_ASS5_BODY.html")
    pdf_output = os.path.join(PDF_DIR, "FINAL_REPORT_ASS5_BODY.pdf")

    with open(md_source, "r", encoding="utf-8") as f:
        md_text = f.read()

    body_html = markdown.markdown(
        md_text, 
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )

    full_html = (
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
        "<title>Báo Cáo Tổng Kết Assignment 5</title><style>" + CSS + "</style>"
        + MATHJAX_HEADER +
        "</head><body>" + body_html + "</body></html>"
    )

    os.makedirs(HTML_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    with open(html_output, "w", encoding="utf-8") as f:
        f.write(full_html)

    file_url = "file:///" + html_output.replace("\\", "/")
    cmd = [
        EDGE_PATH, 
        "--headless=new", 
        "--disable-gpu", 
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_output}", 
        file_url
    ]
    print(f"Rendering Markdown Body to: {pdf_output}...")
    subprocess.run(cmd, capture_output=True)
    print(f"  Body PDF size: {os.path.getsize(pdf_output) / 1024:.1f} KB")
    return pdf_output

def render_notebooks():
    """Converts 5 executed notebooks to HTML and PDF."""
    os.makedirs(HTML_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    nb_pdfs = []

    for nb_name, title in NOTEBOOKS:
        pdf_path = os.path.join(PDF_DIR, f"{nb_name}.pdf")
        if os.path.exists(pdf_path):
            size_kb = os.path.getsize(pdf_path) / 1024
            print(f"  Using existing PDF: {nb_name}.pdf ({size_kb:.1f} KB)")
            nb_pdfs.append((pdf_path, title))
            continue

        nb_path = os.path.join(HERE, f"{nb_name}.ipynb")
        if not os.path.exists(nb_path):
            print(f"Warning: Notebook {nb_path} not found.")
            continue

        print(f"Converting {nb_name}.ipynb to HTML...")
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "html", "--embed-images", 
             "--output-dir", HTML_DIR, nb_path],
            capture_output=True, text=True
        )

        html_path = os.path.join(HTML_DIR, f"{nb_name}.html")
        pdf_path = os.path.join(PDF_DIR, f"{nb_name}.pdf")

        if os.path.exists(html_path):
            file_url = "file:///" + html_path.replace("\\", "/")
            cmd = [
                EDGE_PATH, 
                "--headless=new", 
                "--disable-gpu", 
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}", 
                file_url
            ]
            print(f"  Rendering {nb_name} to PDF...")
            subprocess.run(cmd, capture_output=True)
            if os.path.exists(pdf_path):
                size_kb = os.path.getsize(pdf_path) / 1024
                print(f"  -> {pdf_path} ({size_kb:.1f} KB)")
                nb_pdfs.append((pdf_path, title))

    return nb_pdfs

def assemble_master():
    body_pdf = render_markdown_body()
    nb_pdfs = render_notebooks()

    writer = pypdf.PdfWriter()
    current_page = 0
    parents = {}

    print("\nAssembling Master PDF Document...")

    # 1. Section I: Main Scientific Report (Chapters 1 - 10)
    reader_body = pypdf.PdfReader(body_pdf)
    n_body = len(reader_body.pages)
    start_body = current_page
    for p in reader_body.pages:
        writer.add_page(p)
        current_page += 1
    writer.add_outline_item(
        "Phần I: Báo cáo Khoa học & Phân tích Chuyên sâu (Chương 1 - 10)", 
        start_body
    )
    print(f" - Added Phần I: {n_body} pages (Page {start_body+1} to {current_page})")

    # 2. Section II: All Executed Notebooks
    parent_nb = writer.add_outline_item("Phần II: Toàn văn Thực nghiệm Notebooks", current_page)
    for pdf_path, nb_title in nb_pdfs:
        reader_nb = pypdf.PdfReader(pdf_path)
        n_pages = len(reader_nb.pages)
        start_p = current_page
        for p in reader_nb.pages:
            writer.add_page(p)
            current_page += 1
        writer.add_outline_item(nb_title, start_p, parent=parent_nb)
        print(f" - Added [{os.path.basename(pdf_path)}]: {n_pages} pages (Page {start_p+1} to {current_page}) -> {nb_title[:55]}...")

    # Output targets matching Assignment 4
    target_dung = os.path.join(REPORT_DIR, "A5_02_NguyenDaiDung_103-1.pdf")
    target_dung_pdf = os.path.join(PDF_DIR, "A5_02_NguyenDaiDung_103-1.pdf")
    target_full = os.path.join(REPORT_DIR, "FINAL_REPORT_ASS5_FULL.pdf")
    target_main = os.path.join(REPORT_DIR, "FINAL_REPORT_ASS5.pdf")

    for tgt in [target_dung, target_dung_pdf, target_full, target_main]:
        try:
            with open(tgt, "wb") as f:
                writer.write(f)
            mb = os.path.getsize(tgt) / (1024 * 1024)
            print(f"Saved: {tgt} ({mb:.2f} MB)")
        except PermissionError:
            print(f"[NOTE] '{tgt}' is locked by viewer.")

    print("=" * 70)
    print("MASTER REPORT ASSEMBLED SUCCESSFULLY!")
    print(f"Total Master Pages: {current_page} pages")
    print(f"Primary Target: {target_dung}")
    print("=" * 70)

if __name__ == "__main__":
    assemble_master()
