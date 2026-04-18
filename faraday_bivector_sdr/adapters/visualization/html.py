from __future__ import annotations
import json

def _html_page(title: str, body_html: str, scripts: str = "") -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }}
#canvas {{ border: 1px solid #ccc; image-rendering: pixelated; }}
legend {{ font-size: 12px; color: #555; }}
</style></head>
<body>{body_html}
<script>
{scripts}
</script>
</body></html>"""

def write_waterfall_html(freqs_hz, psd_db_stack, path: str) -> None:
    data = {"freqs_hz": freqs_hz.tolist(), "psd_db": [row.tolist() for row in psd_db_stack]}
    canvas_html = '<h3>Waterfall</h3><canvas id="canvas" width="800" height="400"></canvas><legend>Frequency vs time (rows)</legend>'
    script = f"""
const data = {json.dumps(data)};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const rows = data.psd_db.length;
const cols = data.psd_db[0].length;
const img = ctx.createImageData(cols, rows);
function colorMap(val) {{
  let t = (val + 120.0) / 60.0; t = Math.max(0.0, Math.min(1.0, t)); const g = Math.floor(t*255); return [g, g, g];
}}
for (let y=0; y<rows; ++y) {{
  for (let x=0; x<cols; ++x) {{
    const v = data.psd_db[y][x];
    const [r,g,b] = colorMap(v);
    const idx = 4*(y*cols + x);
    img.data[idx+0]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
  }}
}}
ctx.putImageData(img, 0, 0);
"""
    html = _html_page("Waterfall", canvas_html, script)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

def write_rangedoppler_html(range_m, doppler_hz, power_db, path: str) -> None:
    data = {"range_m": range_m.tolist(), "doppler_hz": doppler_hz.tolist(), "power_db": [row.tolist() for row in power_db]}
    canvas_html = '<h3>Range-Doppler</h3><canvas id="canvas" width="800" height="400"></canvas><legend>Range (x) vs Doppler (y)</legend>'
    script = f"""
const data = {json.dumps(data)};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const rows = data.power_db.length;
const cols = data.power_db[0].length;
const img = ctx.createImageData(cols, rows);
function colorMap(val) {{
  let t = (val + 100.0) / 50.0; t = Math.max(0.0, Math.min(1.0, t)); const r = Math.floor(255*t); const b = Math.floor(255*(1.0-t)); const g = 20; return [r, g, b];
}}
for (let y=0; y<rows; ++y) {{
  for (let x=0; x<cols; ++x) {{
    const v = data.power_db[y][x];
    const [r,g,b] = colorMap(v);
    const idx = 4*(y*cols + x);
    img.data[idx+0]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
  }}
}}
ctx.putImageData(img, 0, 0);
"""
    html = _html_page("Range-Doppler", canvas_html, script)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
