from __future__ import annotations
import json

def write_polmap_html(range_m, doppler_hz, maps_db: dict, path: str) -> None:
    metrics = list(maps_db.keys())
    data = {
        "range_m": range_m.tolist(),
        "doppler_hz": doppler_hz.tolist(),
        "maps": {k: [row.tolist() for row in v] for k, v in maps_db.items()},
        "metrics": metrics
    }
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Polarimetric Map</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }}
#c {{ border:1px solid #ccc; image-rendering: pixelated; }}
select {{ margin-left: 8px; }}
</style></head><body>
<h3>Polarimetric Map <select id="metric"></select></h3>
<canvas id="c" width="800" height="400"></canvas>
<script>
const data = {json.dumps(data)};
const sel = document.getElementById('metric');
data.metrics.forEach(m => {{ const o = document.createElement('option'); o.value=m; o.textContent=m; sel.appendChild(o); }});
const canvas = document.getElementById('c'); const ctx = canvas.getContext('2d');
function colorMap(val) {{
  let t = (val + 100.0)/50.0; t = Math.max(0.0, Math.min(1.0, t));
  const r = Math.floor(255*t), b = Math.floor(255*(1.0-t)), g = 20;
  return [r,g,b];
}}
function draw(metric) {{
  const M = data.maps[metric]; const rows = M.length; const cols = M[0].length;
  const img = ctx.createImageData(cols, rows);
  for (let y=0;y<rows;++y) {{
    for (let x=0;x<cols;++x) {{
      const v = M[y][x]; const [r,g,b] = colorMap(v);
      const idx = 4*(y*cols + x); img.data[idx]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
    }}
  }}
  ctx.putImageData(img, 0, 0);
}}
sel.onchange = () => draw(sel.value);
draw(sel.value || data.metrics[0]);
</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
