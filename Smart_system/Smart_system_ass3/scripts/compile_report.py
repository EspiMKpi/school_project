"""Tập hợp content03.html thành báo cáo hoàn chỉnh report03.html kèm CSS giao diện học thuật."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

# Handle Windows console encoding
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "reports" / "build"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figs"


def main():
    content_file = BUILD / "content03.html"
    if not content_file.exists():
        print("Chưa có content03.html, đang chạy content03.py...")
        import subprocess
        import sys
        subprocess.run([sys.executable, str(BUILD / "content03.py")], check=True)

    html_body = content_file.read_text(encoding="utf-8")

    # Inline figures as base64 data URIs so report03.html is 100% self-contained
    for fig_path in FIGS.glob("*.png"):
        data = base64.b64encode(fig_path.read_bytes()).decode("ascii")
        data_uri = f"data:image/png;base64,{data}"
        html_body = html_body.replace(f'src="figs/{fig_path.name}"', f'src="{data_uri}"')

    full_html = f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Assignment 03: Deep Learning from Scratch with NumPy</title>
<style>
body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #111;
    background-color: #f8f9fa;
    margin: 0;
    padding: 20px;
}}
.container {{
    max-width: 900px;
    margin: 0 auto;
    background: #fff;
    padding: 50px 70px;
    box-shadow: 0 0 15px rgba(0,0,0,0.1);
    border-radius: 4px;
}}
.cover {{
    text-align: center;
    margin-bottom: 40px;
    border-bottom: 2px solid #222;
    padding-bottom: 30px;
}}
.cover-inst {{
    font-size: 13pt;
    font-weight: bold;
    text-transform: uppercase;
    margin-bottom: 10px;
}}
.cover-kicker {{
    font-style: italic;
    color: #555;
    margin-bottom: 15px;
}}
.cover-title {{
    font-size: 20pt;
    font-weight: bold;
    margin: 20px 0;
    line-height: 1.3;
}}
.cover-authors {{
    margin: 20px 0;
    font-size: 11pt;
}}
.cover-authors span {{
    display: block;
    margin: 4px 0;
}}
.cover-authors .nm {{
    font-size: 13pt;
    font-weight: bold;
}}
.cover-date {{
    font-size: 11pt;
    font-style: italic;
    margin-bottom: 25px;
}}
.abstract {{
    text-align: justify;
    background: #fdfdfd;
    border-left: 3px solid #333;
    padding: 15px 20px;
    margin-top: 20px;
}}
h2 {{
    font-size: 14pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 5px;
    margin-top: 35px;
}}
h3 {{
    font-size: 12pt;
    margin-top: 25px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 10pt;
}}
table, th, td {{
    border: 1px solid #999;
}}
th, td {{
    padding: 6px 10px;
    text-align: left;
}}
th {{
    background-color: #f2f2f2;
    font-weight: bold;
}}
td.num, th.num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
}}
td.mono {{
    font-family: Consolas, monospace;
    font-size: 9.5pt;
}}
tr.hi {{
    background-color: #fff9e6;
    font-weight: bold;
}}
.formula {{
    font-family: 'Cambria Math', 'Times New Roman', serif;
    background: #f7f7f7;
    border: 1px solid #e0e0e0;
    padding: 10px;
    margin: 15px 0;
    text-align: center;
    font-style: italic;
    font-size: 11pt;
}}
figure {{
    text-align: center;
    margin: 25px 0;
}}
figure img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
    border-radius: 3px;
}}
figcaption {{
    font-size: 9pt;
    font-style: italic;
    color: #444;
    margin-top: 8px;
}}
.kpis {{
    display: flex;
    justify-content: space-around;
    background: #fafafa;
    border: 1px solid #ddd;
    padding: 15px;
    margin: 20px 0;
    border-radius: 4px;
}}
.kpi {{
    text-align: center;
}}
.kpi b {{
    display: block;
    font-size: 16pt;
    color: #0b5394;
}}
.kpi span {{
    font-size: 8.5pt;
    color: #555;
    text-transform: uppercase;
}}
.callout, .risk, .oim {{
    border-left: 4px solid #3d85c6;
    background: #f0f7fc;
    padding: 12px 18px;
    margin: 20px 0;
    font-size: 10pt;
}}
.risk {{
    border-left-color: #cc0000;
    background: #fdf2f2;
}}
.callout .lab, .risk .lab {{
    font-weight: bold;
    display: block;
    margin-bottom: 5px;
    text-transform: uppercase;
    font-size: 8.5pt;
    letter-spacing: 0.5px;
}}
.chain {{
    display: flex;
    justify-content: space-between;
    margin: 20px 0;
    gap: 10px;
}}
.chain .st {{
    flex: 1;
    background: #f5f5f5;
    border: 1px solid #ccc;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 9pt;
}}
.chain .st k {{
    display: block;
    color: #888;
    font-size: 8pt;
    font-weight: bold;
}}
.chain .st b {{
    display: block;
    color: #111;
    margin: 2px 0;
}}
.chain .st i {{
    display: block;
    color: #555;
    font-size: 8pt;
}}
.tcap {{
    font-weight: bold;
    font-size: 10pt;
    margin-top: 15px;
}}
.tnote {{
    font-size: 8.5pt;
    color: #555;
    font-style: italic;
    margin-top: -10px;
    margin-bottom: 15px;
}}
</style>
</head>
<body>
<div class="container">
{html_body}
</div>
</body>
</html>"""

    out_file = REPORTS / "Assignment03_BaoCao.html"
    out_file.write_text(full_html, encoding="utf-8")
    print(f"Báo cáo hoàn chỉnh đã sinh tại: {out_file.relative_to(ROOT)} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
