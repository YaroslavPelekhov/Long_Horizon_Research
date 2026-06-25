"""Compile a markdown file to a styled PDF via xhtml2pdf. Usage: python build_pdf.py IN.md OUT.pdf"""
import sys
import markdown
from xhtml2pdf import pisa

src = sys.argv[1] if len(sys.argv) > 1 else "PAPER_DRAFT_v0.md"
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".pdf"

body = markdown.markdown(open(src, encoding="utf-8").read(),
                         extensions=["tables", "fenced_code", "sane_lists"])

CSS = """
@page { size: A4; margin: 1.8cm 1.9cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5px; line-height: 1.42; color: #16181d; }
h1 { font-size: 19px; color: #0f1115; margin: 0 0 4px; }
h2 { font-size: 14px; color: #1a1d22; margin: 16px 0 5px; border-bottom: 1px solid #c9d0d8; padding-bottom: 2px; }
h3 { font-size: 11.5px; color: #2a2f36; margin: 11px 0 3px; }
p { margin: 5px 0; }
em { color: #555c66; }
strong { color: #0b3d2e; }
code { font-family: "Courier New", monospace; font-size: 9.5px; background: #f0f2f4; padding: 1px 3px; }
pre { background: #f4f6f8; border: 1px solid #d8dee6; padding: 7px 9px; font-family: "Courier New", monospace;
      font-size: 9px; line-height: 1.3; white-space: pre-wrap; }
pre code { background: transparent; padding: 0; font-size: 9px; }
table { border-collapse: collapse; width: 100%; margin: 7px 0; font-size: 9.5px; }
th, td { border: 1px solid #c2cad4; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eef1f4; font-weight: bold; }
hr { border: none; border-top: 1px solid #d0d6dd; margin: 12px 0; }
ul, ol { margin: 5px 0 5px 0; padding-left: 18px; }
li { margin: 2px 0; }
"""
html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
with open(out, "wb") as f:
    res = pisa.CreatePDF(html, dest=f, encoding="utf-8")
print(f"{'OK' if not res.err else 'ERRORS'} → {out}")
