"""Bangun dashboard.html (satu file mandiri) dari index.html + assets/cctvmakota-logo.svg.

Jalankan setiap kali index.html atau logo berubah:
    python build_single_file.py
"""
from pathlib import Path
import base64

folder = Path(__file__).parent
index = (folder / "index.html").read_text(encoding="utf8")
ref = "assets/cctvmakota-logo.svg"
logo = "data:image/svg+xml;base64," + base64.b64encode((folder / ref).read_bytes()).decode("ascii")

count = index.count(ref)
assert count, f"{ref} tidak dipakai di index.html"
(folder / "dashboard.html").write_text(index.replace(ref, logo), encoding="utf8")   # img + favicon

print(f"logo SVG di-embed di {count} tempat")
print(f"output: {(folder / 'dashboard.html').stat().st_size:,} bytes")
