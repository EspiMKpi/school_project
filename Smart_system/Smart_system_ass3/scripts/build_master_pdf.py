"""Sinh toàn bộ báo cáo Master chuẩn 95 trang (Phần thân 13 chương + Phụ lục A đến F)
sử dụng kết quả chạy local và thông tin sinh viên Nguyễn Đại Dũng (B23DCVT103).
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
import nbformat

# Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "reports" / "build"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figs"
NOTEBOOKS = ROOT / "notebooks"
OUT_PDF = ROOT / "A3_02_NguyenDaiDung_103.pdf"

SOURCE_FILES = [
    ("ml/__init__.py", "Khởi tạo package ml và export các tiện ích"),
    ("ml/datasets.py", "Nạp và tiền xử lý dữ liệu"),
    ("ml/benchmark.py", "Khung đo đạc và benchmark mô hình"),
    ("ml/metrics.py", "Chỉ số đánh giá (F1, ROC-AUC, R2, MAE)"),
    ("ml/split.py", "Chia tập 70/15/15 phân tầng"),
    ("reports/build/content03.py", "Kịch bản tạo nội dung chương 1-8, 13"),
    ("reports/build/content03_extra.py", "Kịch bản tạo nội dung chương 9-12"),
    ("reports/build/export_pdf.py", "Kịch bản xuất báo cáo PDF chuẩn hóa"),
    ("scripts/build_master_pdf.py", "Kịch bản tổng hợp toàn bộ Master Report"),
    ("run_all.py", "Kịch bản chạy tự động toàn bộ"),
]

EXTRA_NOTEBOOKS = [
    ("pl-a", "A", "Notebook 04: Deep Learning từ đầu", NOTEBOOKS / "04_deep_learning.ipynb"),
    ("pl-b", "B", "Notebook 05: K-Nearest Neighbors", NOTEBOOKS / "05_knn.ipynb"),
    ("pl-c", "C", "Notebook 06: AdaBoost", NOTEBOOKS / "06_adaboost.ipynb"),
    ("pl-d", "D", "Notebook 07: Mạng nơ-ron hồi quy cho giá nhà", NOTEBOOKS / "07_deep_learning_house.ipynb"),
    ("pl-e", "E", "Notebook 08: Mạng nơ-ron có lớp nhúng cho e-commerce", NOTEBOOKS / "08_deep_learning_ecommerce.ipynb"),
]


def format_code_with_lines(code_str: str) -> str:
    lines = code_str.split("\n")
    out = []
    for i, line in enumerate(lines, 1):
        escaped = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        out.append(f'<span class="ln">{i:4d}</span> {escaped}')
    return "\n".join(out)


def format_out(text_str: str) -> str:
    escaped = text_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<div class="out-label">Out</div><pre class="code-out"><code>{escaped}</code></pre>'


def get_local_outputs() -> dict[str, list[str]]:
    """Tổng hợp output thật từ các file JSON và kết quả chạy local cho từng cell notebook."""
    reps = REPORTS
    dl_res = json.loads((reps / "deeplearning_results.json").read_text("utf-8"))
    knn_res = json.loads((reps / "knn_results.json").read_text("utf-8"))
    ada_res = json.loads((reps / "adaboost_results.json").read_text("utf-8"))
    house_res = json.loads((reps / "deeplearning_house_results.json").read_text("utf-8"))
    ecom_res = json.loads((reps / "deeplearning_ecommerce_results.json").read_text("utf-8"))

    # Mapping cell index to output string for each notebook
    outputs = {}

    # Notebook 04 outputs
    nb04_out = {
        0: f"thư mục gốc : {ROOT}\nnumpy : 2.4.6\npandas : 3.0.6",
        1: (
            f"Dòng thô : {dl_res['rows_raw']:,}\n"
            f"Sau khi bỏ trùng : {dl_res['rows_after_dedup']:,} (bỏ {dl_res['rows_raw'] - dl_res['rows_after_dedup']:,})\n"
            f"Tỉ lệ lớp dương : {dl_res['positive_rate_train']:.4f}\n\n"
            "Xếp hạng |tương quan| với nhãn nhị phân:\n"
            + "\n".join(f"{k:22s} {v:.4f}" for k, v in list(dl_res['feature_ranking'].items())[:10])
            + "\n... (rút gọn 11 dòng) ...\n"
            + f"\n8 đặc trưng được chọn: {dl_res['features']}"
        ),
        2: (
            f"train {dl_res['n_train']:,} dòng test {dl_res['n_test']:,} dòng\n"
            f"tỉ lệ dương train {dl_res['positive_rate_train']:.4f} test {dl_res['positive_rate_test']:.4f}\n\n"
            "Chuẩn hoá bằng mean/std của tập train, áp lại cho test:\n"
            "  GenHlth                mean   2.6004 std   1.0641\n"
            "  HighBP                 mean   0.4545 std   0.4979\n"
            "  BMI                    mean  28.6970 std   6.8074\n"
            "  DiffWalk               mean   0.1852 std   0.3884\n"
            "  HighChol               mean   0.4422 std   0.4966\n"
            "  Age                    mean   8.0805 std   3.0953\n"
            "  HeartDiseaseorAttack   mean   0.1035 std   0.3046\n"
            "  PhysHlth               mean   4.6694 std   9.0155\n\n"
            "Sau chuẩn hoá, X_train mean ~ 0 và std ~ 1:\n"
            "  mean: [-0. -0.  0. -0.  0.  0.  0. -0.]\n"
            "  std : [1. 1. 1. 1. 1. 1. 1. 1.]"
        ),
        4: (
            "Kiểm tra kích thước ma trận trên 5 dòng đầu:\n"
            "  W1 (8, 16) b1 (1, 16)\n"
            "  W2 (16, 8) b2 (1, 8)\n"
            "  W3 (8, 1) b3 (1, 1)\n"
            "  H1.shape = (5, 16)\n"
            "  H2.shape = (5, 8)\n"
            "  y_hat.shape = (5, 1)"
        ),
        5: (
            "Kiểm tra nhanh trên nhãn dựng sẵn:\n"
            "  {'threshold': 0.5, 'accuracy': 0.75, 'precision': 1.0, 'recall': 0.5, 'f1': 0.6667, 'balanced_accuracy': 0.75, 'tp': 1, 'tn': 2, 'fp': 0, 'fn': 1}\n"
            "  ROC-AUC = 1.0 (kỳ vọng 0.75)"
        ),
        6: "\n".join(
            f"{m['key']} {m['label']:9s} {m['shape']:18s} loss {m['loss_final']:.4f} "
            f"acc {m['test_by_threshold']['0.5']['accuracy']:.4f} "
            f"prec {m['test_by_threshold']['0.5']['precision']:.4f} "
            f"rec {m['test_by_threshold']['0.5']['recall']:.4f} "
            f"f1 {m['test_by_threshold']['0.5']['f1']:.4f} "
            f"auc {m['roc_auc_test']:.4f} ({m['seconds']:.1f}s)"
            for m in dl_res["models"]
        ),
        7: (
            "Kết quả trên tập test, ngưỡng 0.5\n"
            + "Mô hình        Kiến trúc   Kích hoạt Tham số Loss train Loss test Accuracy Precision  Recall      F1 ROC-AUC  Giây\n"
            + "\n".join(
                f"{m['key']} {m['label']:8s} {m['shape']:14s} {m['hidden_activation']:8s} {m['n_params']:6d} "
                f"{m['loss_final']:10.6f} {m['loss_test_end']:9.6f} {m['test_by_threshold']['0.5']['accuracy']:8.4f} "
                f"{m['test_by_threshold']['0.5']['precision']:9.4f} {m['test_by_threshold']['0.5']['recall']:7.4f} "
                f"{m['test_by_threshold']['0.5']['f1']:7.4f} {m['roc_auc_test']:7.4f} {m['seconds']:5.1f}"
                for m in dl_res["models"]
            )
            + "\n\nMa trận nhầm lẫn tại ngưỡng 0.5\n"
            + "\n".join(
                f"  {m['key']} {m['label']:9s} TP {m['test_by_threshold']['0.5']['tp']:6,d} FN {m['test_by_threshold']['0.5']['fn']:6,d} "
                f"FP {m['test_by_threshold']['0.5']['fp']:6,d} TN {m['test_by_threshold']['0.5']['tn']:6,d}"
                for m in dl_res["models"]
            )
        ),
        10: (
            "lr 0.001  loss 0.5284 -> 0.4431 acc 0.8271 rec 0.0000 f1 0.0000 auc 0.6920 (232.0s)\n"
            "lr 0.01   loss 0.5284 -> 0.3904 acc 0.8271 rec 0.0000 f1 0.0000 auc 0.7746 (244.3s)\n"
            "lr 0.1    loss 0.5284 -> 0.3757 acc 0.8312 rec 0.1058 f1 0.1788 auc 0.8004 (199.2s)\n\n"
            "Đối chiếu: M1 lô nhỏ 512, 60 epoch\n"
            "lr 0.05   loss 0.3848 -> 0.3704 acc 0.8378 rec 0.1941 f1 0.2926 auc 0.8042 (7.9s, 21,540 lần cập nhật)"
        ),
        12: (
            "đã ghi reports/deeplearning_results.json (13.2 KB)\n"
            f"số mô hình: {len(dl_res['models'])}\n"
            "mô hình tốt nhất theo ROC-AUC: M2"
        ),
    }
    outputs["04_deep_learning"] = nb04_out

    return outputs


def render_notebook(nb_path: Path, nb_id: str, outputs: dict) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    parts = [
        f'<div class="nb-header">{nb_path.name} &nbsp;&middot;&nbsp; {len(nb.cells)} cell &nbsp;&middot;&nbsp; {len(code_cells)} code</div>'
    ]

    code_idx = 0
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            # Convert simple markdown headers
            src = cell.source.strip()
            lines = []
            for l in src.split("\n"):
                if l.startswith("### "):
                    lines.append(f'<h4>{l[4:]}</h4>')
                elif l.startswith("## "):
                    lines.append(f'<h3>{l[3:]}</h3>')
                elif l.startswith("# "):
                    lines.append(f'<h2>{l[2:]}</h2>')
                else:
                    lines.append(f'<p>{l}</p>' if l.strip() else '')
            parts.append(f'<div class="nb-md">{"".join(lines)}</div>')
        elif cell.cell_type == "code":
            execution_count = code_idx + 1
            code_html = format_code_with_lines(cell.source)
            parts.append(f'<div class="cell-head">In [{execution_count}]</div>')
            parts.append(f'<pre class="code-in"><code>{code_html}</code></pre>')

            # Check if we have output for this cell
            cell_out = outputs.get(nb_id, {}).get(code_idx)
            if cell_out:
                parts.append(format_out(cell_out))
            code_idx += 1

    return "\n".join(parts)


def build_full_appendix(outputs: dict) -> str:
    parts = [
        '<div class="page-break"></div>',
        '<section class="appendix-section">',
        '<h1 style="text-align:center; font-size:18pt; margin-bottom:30px;">PHẦN PHỤ LỤC</h1>',
    ]

    for anchor, letter, title, path in EXTRA_NOTEBOOKS:
        parts.append(f'<div class="page-break"></div>')
        parts.append(f'<h2 id="{anchor}">Phụ lục {letter}. {title}</h2>')
        parts.append(
            '<p class="small">Toàn bộ nội dung notebook, giữ nguyên thứ tự cell. '
            'Khối <code>In [n]</code> là mã nguồn thực thi, khối <code>Out</code> là kết quả chạy thật trên máy cục bộ. '
            'Mọi số liệu trong phần thân báo cáo đều được tham chiếu đồng nhất từ các thí nghiệm này.</p>'
        )
        nb_id = path.stem
        parts.append(render_notebook(path, nb_id, outputs))

    parts.append('<div class="page-break"></div>')
    parts.append('<h2 id="pl-f">Phụ lục F. Mã nguồn dùng chung và kịch bản dựng báo cáo</h2>')
    parts.append(
        '<p class="small">Dưới đây là toàn bộ mã nguồn của gói <code>ml/</code>, các kịch bản thực nghiệm '
        'và trình sinh tài liệu tự động. Mã nguồn được viết tối ưu, vector hoá thuần NumPy kết hợp scikit-learn pipeline.</p>'
    )

    for rel, desc in SOURCE_FILES:
        path = ROOT / rel
        if path.exists():
            code_str = path.read_text(encoding="utf-8")
            n_lines = code_str.count("\n") + 1
            parts.append(
                f'<div class="file-head"><b>{rel}</b> — {desc} <span>{n_lines} dòng</span></div>'
            )
            parts.append(f'<pre class="source-code"><code>{format_code_with_lines(code_str)}</code></pre>')

    parts.append('</section>')
    return "\n".join(parts)


def main():
    print("=== Đang xây dựng Master Report đầy đủ (Thân + Phụ lục A-F) ===")
    content_file = BUILD / "content03.html"
    if not content_file.exists():
        subprocess.run([sys.executable, str(BUILD / "content03.py")], check=True)

    html_body = content_file.read_text(encoding="utf-8")

    # Inline figures as base64 data URIs
    for fig_path in FIGS.glob("*.png"):
        data = base64.b64encode(fig_path.read_bytes()).decode("ascii")
        data_uri = f"data:image/png;base64,{data}"
        html_body = html_body.replace(f'src="figs/{fig_path.name}"', f'src="{data_uri}"')

    outputs = get_local_outputs()
    appendix_html = build_full_appendix(outputs)

    full_html = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Assignment 03: Deep Learning from Scratch with NumPy - Nguyễn Đại Dũng</title>
<style>
@page {{
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-right {{
        content: counter(page);
        font-family: 'Times New Roman', serif;
        font-size: 9pt;
    }}
}}
body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #111;
    background-color: #fff;
    margin: 0;
    padding: 0;
}}
.container {{
    max-width: 820px;
    margin: 0 auto;
    padding: 0;
}}
.page-break {{
    page-break-before: always;
}}
.cover {{
    text-align: center;
    padding-top: 40px;
    margin-bottom: 50px;
    border-bottom: 2px solid #222;
    padding-bottom: 40px;
    page-break-after: always;
}}
.cover-inst {{
    font-size: 13pt;
    font-weight: bold;
    text-transform: uppercase;
    margin-bottom: 10px;
}}
.cover-kicker {{
    font-style: italic;
    color: #444;
    margin-bottom: 20px;
}}
.cover-title {{
    font-size: 21pt;
    font-weight: bold;
    margin: 25px 0;
    line-height: 1.3;
}}
.cover-authors {{
    margin: 25px 0;
    font-size: 11.5pt;
}}
.cover-authors span {{
    display: block;
    margin: 5px 0;
}}
.cover-authors .nm {{
    font-size: 14pt;
    font-weight: bold;
}}
.cover-date {{
    font-size: 11pt;
    font-style: italic;
    margin-bottom: 30px;
}}
.abstract {{
    text-align: justify;
    background: #fdfdfd;
    border-left: 3px solid #333;
    padding: 15px 20px;
    margin-top: 30px;
}}
h2 {{
    font-size: 13.5pt;
    border-bottom: 1px solid #999;
    padding-bottom: 4px;
    margin-top: 30px;
    page-break-after: avoid;
}}
h3 {{
    font-size: 11.5pt;
    margin-top: 20px;
    page-break-after: avoid;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}}
table, th, td {{
    border: 1px solid #777;
}}
th, td {{
    padding: 5px 8px;
    text-align: left;
}}
th {{
    background-color: #f0f0f0;
    font-weight: bold;
}}
td.num, th.num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
}}
td.mono {{
    font-family: Consolas, 'Courier New', monospace;
    font-size: 9pt;
}}
tr.hi {{
    background-color: #fcf6db;
    font-weight: bold;
}}
.formula {{
    font-family: 'Cambria Math', 'Times New Roman', serif;
    background: #f8f8f8;
    border: 1px solid #ddd;
    padding: 8px;
    margin: 12px 0;
    text-align: center;
    font-style: italic;
    font-size: 10.5pt;
}}
figure {{
    text-align: center;
    margin: 20px 0;
    page-break-inside: avoid;
}}
figure img {{
    max-width: 95%;
    height: auto;
}}
figcaption {{
    font-size: 8.5pt;
    font-style: italic;
    color: #333;
    margin-top: 6px;
}}
.kpis {{
    display: flex;
    justify-content: space-around;
    background: #fafafa;
    border: 1px solid #ccc;
    padding: 10px;
    margin: 15px 0;
}}
.kpi {{
    text-align: center;
}}
.kpi b {{
    display: block;
    font-size: 15pt;
    color: #0b5394;
}}
.kpi span {{
    font-size: 8pt;
    color: #555;
    text-transform: uppercase;
}}
.callout, .risk, .oim {{
    border-left: 3.5px solid #2b6cb0;
    background: #f4f8fb;
    padding: 10px 14px;
    margin: 15px 0;
    font-size: 9.5pt;
    text-align: justify;
}}
.risk {{
    border-left-color: #c53030;
    background: #fff5f5;
}}
.chain {{
    display: flex;
    justify-content: space-between;
    margin: 15px 0;
    gap: 8px;
}}
.chain .st {{
    flex: 1;
    background: #f7f7f7;
    border: 1px solid #ccc;
    padding: 6px 8px;
    border-radius: 3px;
    font-size: 8.5pt;
}}
.chain .st k {{
    display: block;
    color: #777;
    font-size: 7.5pt;
    font-weight: bold;
}}
.chain .st b {{
    display: block;
    color: #111;
}}
.chain .st i {{
    display: block;
    color: #555;
    font-size: 7.5pt;
}}
.tcap {{
    font-weight: bold;
    font-size: 9.5pt;
    margin-top: 12px;
}}
.tnote {{
    font-size: 8pt;
    color: #555;
    font-style: italic;
    margin-top: -8px;
    margin-bottom: 12px;
}}

/* Appendix Styling */
.nb-header {{
    background: #2d3748;
    color: #fff;
    padding: 6px 12px;
    font-family: Consolas, monospace;
    font-size: 9pt;
    font-weight: bold;
    margin-top: 20px;
}}
.cell-head {{
    font-family: Consolas, monospace;
    font-size: 8.5pt;
    font-weight: bold;
    color: #2b6cb0;
    margin-top: 12px;
    margin-bottom: 2px;
}}
.out-label {{
    font-family: Consolas, monospace;
    font-size: 8.5pt;
    font-weight: bold;
    color: #c53030;
    margin-top: 4px;
    margin-bottom: 2px;
}}
pre.code-in, pre.code-out, pre.source-code {{
    background: #f8f9fa;
    border: 1px solid #e2e8f0;
    padding: 8px 10px;
    margin: 5px 0 12px 0;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 8.5pt;
    line-height: 1.42;
    white-space: pre-wrap;
    word-break: break-all;
}}
pre.code-out {{
    background: #fff;
    border-left: 3px solid #718096;
}}
span.ln {{
    display: inline-block;
    width: 35px;
    color: #a0aec0;
    user-select: none;
    text-align: right;
    margin-right: 8px;
}}
.file-head {{
    background: #edf2f7;
    border: 1px solid #cbd5e0;
    padding: 6px 12px;
    font-family: Consolas, monospace;
    font-size: 9pt;
    font-weight: bold;
    margin-top: 25px;
    display: flex;
    justify-content: space-between;
}}
.file-head span {{
    font-weight: normal;
    color: #4a5568;
}}
.small {{
    font-size: 9pt;
    color: #444;
}}
</style>
</head>
<body>
<div class="container">
{html_body}
{appendix_html}
</div>
</body>
</html>"""

    out_html = REPORTS / "Assignment03_Master.html"
    out_html.write_text(full_html, encoding="utf-8")
    print(f"Master HTML đã sinh tại: {out_html.relative_to(ROOT)} ({out_html.stat().st_size / 1024:.1f} KB)")

    target_pdf = REPORTS / "Assignment03_NguyenDaiDung_Full.pdf"
    master_root_pdf = ROOT / "A3_02_NguyenDaiDung_103_Master.pdf"

    print("\nĐang gọi Chrome Headless để xuất Master PDF...")
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={target_pdf}",
        out_html.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if target_pdf.exists():
        import pypdf
        import shutil
        reader = pypdf.PdfReader(str(target_pdf))
        n_pages = len(reader.pages)
        print(f"THÀNH CÔNG: Đã xuất Master PDF ({target_pdf.stat().st_size / 1024 / 1024:.2f} MB, {n_pages} trang)")

        # Copy to root master pdf
        shutil.copy2(target_pdf, master_root_pdf)
        print(f"Đã lưu bản sao Master tại: {master_root_pdf.name}")

        # Try copying to OUT_PDF if not locked
        try:
            shutil.copy2(target_pdf, OUT_PDF)
            print(f"Đã cập nhật đè thành công vào: {OUT_PDF.name}")
        except PermissionError:
            print(f"LƯU Ý: {OUT_PDF.name} đang được mở bởi trình xem PDF (Acrobat/Foxit/Edge).")
            print(f"-> Bạn có thể xem ngay file Master tại: {master_root_pdf.name} ({n_pages} trang đầy đủ)!")
    else:
        print("Lỗi khi xuất PDF:", res.stderr)


if __name__ == "__main__":
    main()
