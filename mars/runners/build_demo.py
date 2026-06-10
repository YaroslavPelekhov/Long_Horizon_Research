"""Embed the generated trace JSON into the HTML to make a self-contained demo.

Reorders showcases so the most striking ones (weak+MDL solves while strong fails)
appear first. Output: demo/compression_demo_built.html (open directly, offline).
"""
from __future__ import annotations
import json
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
demo = _PROJ / "demo"
data = json.loads((demo / "compression_demo_data.json").read_text())

def rank(s):
    mdl = s["weak_mdl"]["solved"]; strong = s["strong_raw"]["solved"]
    if mdl and not strong: return 0      # the money shot: weak+arch beats strong
    if mdl and strong: return 1
    return 2                              # honest boundary (not solved) last
data["showcase"].sort(key=rank)

tmpl = (demo / "compression_demo.html").read_text()
built = tmpl.replace("/*__DEMO_DATA__*/ null", json.dumps(data))
out = demo / "compression_demo_built.html"
out.write_text(built, encoding="utf-8")
print(f"wrote {out}  ({len(built)//1024} KB, {len(data['showcase'])} showcases)")
order = [(s.get('letter',s['kind']), s['weak_mdl']['solved'], s['strong_raw']['solved']) for s in data['showcase']]
print("order (letter, mdl_solved, strong_solved):", order)
