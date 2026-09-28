"""
Script to render FINAL_REPORT_ASS5.md to HTML and export to professional PDF via Microsoft Edge.
Matches the styling and PDF generation of Assignment 4.
"""

import os
import subprocess
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "report")
HTML_DIR = os.path.join(HERE, "_html")
PDF_DIR = os.path.join(HERE, "pdf")

MD_SOURCE = os.path.join(REPORT_DIR, "FINAL_REPORT_ASS5.md")
HTML_OUTPUT = os.path.join(HTML_DIR, "FINAL_REPORT_ASS5.html")
PDF_OUTPUT = os.path.join(REPORT_DIR, "FINAL_REPORT_ASS5.pdf")

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

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

def generate_html():
    os.makedirs(HTML_DIR, exist_ok=True)
    with open(MD_SOURCE, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text, 
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"]
    )

    full_html = (
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
        f"<title>Báo Cáo Tổng Kết Assignment 5</title><style>{CSS}</style>"
        f"{MATHJAX_HEADER}"
        f"</head>"
        f"<body>{html_body}</body></html>"
    )

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated HTML: {HTML_OUTPUT}")
    return HTML_OUTPUT

def generate_pdf(html_path):
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    
    file_url = "file:///" + html_path.replace("\\", "/")
    
    # We will export to report/FINAL_REPORT_ASS5.pdf and pdf/FINAL_REPORT_ASS5.pdf
    targets = [PDF_OUTPUT, os.path.join(PDF_DIR, "FINAL_REPORT_ASS5.pdf")]
    
    if os.path.exists(EDGE_PATH):
        for target in targets:
            cmd = [
                EDGE_PATH, 
                "--headless=new", 
                "--disable-gpu", 
                "--no-pdf-header-footer",
                f"--print-to-pdf={target}", 
                file_url
            ]
            print(f"Rendering PDF via Microsoft Edge to: {target}...")
            res = subprocess.run(cmd, capture_output=True)
            if os.path.exists(target):
                size_mb = os.path.getsize(target) / (1024 * 1024)
                print(f"  SUCCESS -> {target} ({size_mb:.2f} MB)")
            else:
                print(f"  Error rendering {target}: {res.stderr.decode(errors='ignore')}")
    else:
        print("Microsoft Edge not found at default path.")

if __name__ == "__main__":
    h_path = generate_html()
    generate_pdf(h_path)
