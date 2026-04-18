from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .control import GLOBAL_CONTROL
from .params import GLOBAL_PARAMS
from .taps import PolMapTap, RDTap, SpectrumTap

# In-memory bookmarks and presets (process lifetime)
_BOOKMARKS: Dict[str, float] = {}          # name -> freq Hz
_PRESETS: Dict[str, Dict[str, Any]] = {}   # name -> {gain_db, cf_hz, bw_hz, etc.}

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>faraday-bivector-sdr Web UI</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
.grid { display: grid; grid-template-columns: 360px 1fr; gap: 12px; }
.panel { border: 1px solid #ddd; padding: 12px; border-radius: 6px; }
h3 { margin: 0 0 8px 0; }
label { display: block; margin: 6px 0 4px 0; font-size: 12px; color: #555; }
.row { display:flex; align-items:center; gap: 8px; flex-wrap: wrap; }
.value { display:inline-block; width:90px; text-align:right; }
input[type=range] { width: 280px; }
canvas { border: 1px solid #ccc; image-rendering: pixelated; }
small { color:#777; }
button { margin-right: 8px; }
select { min-width: 240px; }
hr { border: none; border-top: 1px solid #eee; margin: 12px 0; }
.kv { font-size: 12px; color: #555; margin-top: 6px; }
</style>
</head>
<body>
<div class="grid">
  <div class="panel" id="controls">
    <h3>Controls</h3>
    <div id="params"></div>

    <hr />

    <div>
      <label>Pipeline Tap</label>
      <select id="tapSelect"></select>

      <div class="row" style="margin-top:8px;">
        <button id="btnUseSpec">Use Spectrum</button>
        <button id="btnUseRD">Use Range-Doppler</button>
        <button id="btnUsePolMap">Use PolMap</button>
      </div>

      <div id="polmapControls" style="margin-top:10px; display:none;">
        <label>PolMap Metric</label>
        <select id="polMetric"></select>
      </div>

      <div class="kv" id="statusLine"></div>
      <div class="kv" id="dataLine"></div>
    </div>

    <hr />

    <div>
      <h3>Device Control</h3>
      <div id="devices"></div>
      <small>
        Note: changing center frequency / bandwidth while streaming may disrupt some devices.
      </small>
    </div>

    <hr />

    <div>
      <h3>Bookmarks</h3>
      <div id="bkmks"></div>
      <div class="row" style="margin-top:6px;">
        <input id="bmName" placeholder="name" />
        <input id="bmFreq" placeholder="freq Hz" />
        <button id="bmAdd">Add</button>
      </div>
    </div>

    <hr />

    <div>
      <h3>Presets</h3>
      <div id="presets"></div>
      <div class="row" style="margin-top:6px;">
        <input id="psName" placeholder="name" />
        <input id="psData" placeholder='{"gain_db":20}' />
        <button id="psAdd">Add</button>
      </div>
    </div>
  </div>

  <div class="panel">
    <h3>Live View</h3>
    <div class="row" style="margin-bottom:8px;">
      <small>Spectrum/Waterfall update continuously when a spectrum tap exists. Heatmap shows RD/PolMap based on selected tap.</small>
    </div>

    <h3 style="margin-top:6px;">Spectrum</h3>
    <canvas id="spec" width="900" height="220"></canvas>

    <h3 style="margin-top:12px;">Waterfall (Spectrum)</h3>
    <canvas id="wf" width="900" height="220"></canvas>

    <h3 style="margin-top:12px;">Heatmap (RD / PolMap)</h3>
    <canvas id="heat" width="900" height="420"></canvas>
  </div>
</div>

<script>
const config = __CONFIG__;

const specCanvas = document.getElementById('spec'); const specCtx = specCanvas.getContext('2d');
const wfCanvas = document.getElementById('wf'); const wfCtx = wfCanvas.getContext('2d');
const heatCanvas = document.getElementById('heat'); const heatCtx = heatCanvas.getContext('2d');

const polControls = document.getElementById('polmapControls');
const polMetricSel = document.getElementById('polMetric');
const statusLine = document.getElementById('statusLine');
const dataLine = document.getElementById('dataLine');

let currentTap = (config.taps && config.taps[0] && config.taps[0].id) || "";
let currentTapType = (config.taps && config.taps[0] && config.taps[0].type) || "spectrum";

let lastPolMetrics = [];
let selectedPolMetric = "";

function setStatus(text) { statusLine.textContent = text || ''; }
function setData(text) { dataLine.textContent = text || ''; }

function mkSlider(label, min, max, step, init, cb) {
  const wrap = document.createElement('div');
  const lab = document.createElement('label'); lab.textContent = label;
  const row = document.createElement('div'); row.className = 'row';
  const inp = document.createElement('input'); inp.type='range'; inp.min=min; inp.max=max; inp.step=step; inp.value=init;
  const val = document.createElement('span'); val.className='value'; val.textContent=init;
  inp.oninput = () => { val.textContent = inp.value; cb(parseFloat(inp.value)); };
  row.appendChild(inp); row.appendChild(val); wrap.appendChild(lab); wrap.appendChild(row);
  return wrap;
}

async function api(path) { const r = await fetch(path); return await r.json(); }
async function post(path, payload) {
  const r = await fetch(path, {method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  return await r.json();
}

function drawSpectrum(psd) {
  if (!psd || psd.length===0) return;
  specCtx.clearRect(0,0,specCanvas.width,specCanvas.height);
  const N = psd.length; const W = specCanvas.width; const H = specCanvas.height;
  const minDb = -120, maxDb = 0;
  specCtx.beginPath();
  for (let i=0; i<N; ++i) {
    const x = Math.floor(i * W / N);
    const t = (psd[i]-minDb)/(maxDb-minDb);
    const y = H - Math.floor(Math.max(0,Math.min(1,t)) * H);
    if (i===0) specCtx.moveTo(x,y); else specCtx.lineTo(x,y);
  }
  specCtx.strokeStyle = '#0b7'; specCtx.lineWidth = 1; specCtx.stroke();
}

function appendWaterfallRow(psd) {
  if (!psd || psd.length===0) return;
  const rows = wfCanvas.height; const cols = wfCanvas.width;

  // Pull current pixels
  const img = wfCtx.getImageData(0,0,cols,rows);

  // Scroll up by 1 row (manual copy; more reliable than putImageData with negative y)
  for (let y=0; y<rows-1; ++y) {
    for (let x=0; x<cols; ++x) {
      const dst = 4*(y*cols + x);
      const src = 4*((y+1)*cols + x);
      img.data[dst+0] = img.data[src+0];
      img.data[dst+1] = img.data[src+1];
      img.data[dst+2] = img.data[src+2];
      img.data[dst+3] = 255;
    }
  }

  // Draw new row at bottom
  const y = rows-1;
  for (let x=0; x<cols; ++x) {
    const i = Math.floor(x * psd.length / cols);
    const v = psd[i];

    // red-blue ramp
    let t = (v + 120.0) / 60.0;
    t = Math.max(0.0, Math.min(1.0, t));
    const r = Math.floor(255*t);
    const b = Math.floor(255*(1.0-t));
    const g = 30;

    const idx = 4*(y*cols + x);
    img.data[idx+0]=r;
    img.data[idx+1]=g;
    img.data[idx+2]=b;
    img.data[idx+3]=255;
  }

  wfCtx.putImageData(img, 0, 0);
}

function drawHeatmap(matrix, minDb, maxDb) {
  if (!matrix || matrix.length === 0 || matrix[0].length === 0) return;
  const rows = matrix.length;
  const cols = matrix[0].length;

  const W = heatCanvas.width;
  const H = heatCanvas.height;

  const img = heatCtx.createImageData(W, H);

  for (let y=0; y<H; ++y) {
    const my = Math.floor(y * rows / H);
    for (let x=0; x<W; ++x) {
      const mx = Math.floor(x * cols / W);
      const v = matrix[my][mx];

      let t = (v - minDb) / (maxDb - minDb);
      t = Math.max(0.0, Math.min(1.0, t));

      const r = Math.floor(255*t);
      const b = Math.floor(255*(1.0-t));
      const g = 30;

      const idx = 4*(y*W + x);
      img.data[idx+0]=r;
      img.data[idx+1]=g;
      img.data[idx+2]=b;
      img.data[idx+3]=255;
    }
  }
  heatCtx.putImageData(img, 0, 0);
}

function ensurePolMetricOptions(metrics) {
  const m = metrics || [];
  const same = (m.length === lastPolMetrics.length) && m.every((x,i)=>x===lastPolMetrics[i]);
  if (same) return;

  lastPolMetrics = m.slice();
  polMetricSel.innerHTML = '';

  m.forEach(name => {
    const o = document.createElement('option');
    o.value = name;
    o.textContent = name;
    polMetricSel.appendChild(o);
  });

  selectedPolMetric = (m[0] || "");
  polMetricSel.value = selectedPolMetric;

  polMetricSel.onchange = () => {
    selectedPolMetric = polMetricSel.value;
  };
}

function firstTapId(typeName) {
  const taps = (config.taps || []);
  for (let i=0; i<taps.length; ++i) {
    if (taps[i].type === typeName) return taps[i].id;
  }
  return "";
}

const spectrumTapId = firstTapId("spectrum");

function setupControls() {
  const pdiv = document.getElementById('params');
  (config.params || []).forEach(p => {
    const s = mkSlider(p.label || p.name, p.min, p.max, p.step || 1, p.init || 0, (v) => {
      api(`/api/set_param?name=${encodeURIComponent(p.name)}&value=${encodeURIComponent(v)}`);
    });
    pdiv.appendChild(s);
  });

  const ddiv = document.getElementById('devices');
  (config.devices || []).forEach(d => {
    const wrap = document.createElement('div'); wrap.className='panel'; wrap.style.margin='8px 0';
    const title = document.createElement('div'); title.innerHTML = `<b>${d.id}</b> <small>adapter=${d.adapter||''}</small>`;
    wrap.appendChild(title);
    (d.controls || []).forEach(c => {
      const s = mkSlider(`${c.field} (ch ${c.channel||0})`, c.min, c.max, c.step || 1, c.init || 0, (v) => {
        const q = new URLSearchParams({dev:d.id, field:c.field, channel:String(c.channel||0), value:String(v)});
        api(`/api/set_device?${q}`).catch(console.error);
      });
      wrap.appendChild(s);
    });
    ddiv.appendChild(wrap);
  });

  const tapSel = document.getElementById('tapSelect');
  (config.taps || []).forEach(t => {
    const o = document.createElement('option');
    o.value = `${t.type}:${t.id}`;
    o.textContent = `${t.type}:${t.id}`;
    tapSel.appendChild(o);
  });

  tapSel.onchange = () => {
    const [typ,id] = tapSel.value.split(':');
    currentTapType = typ;
    currentTap = id;
    polControls.style.display = (currentTapType === 'polmap') ? 'block' : 'none';
    setStatus(`Selected tap: ${currentTapType}:${currentTap}`);
  };

  if (tapSel.value) {
    const [typ,id] = tapSel.value.split(':');
    currentTapType = typ;
    currentTap = id;
  }
  polControls.style.display = (currentTapType === 'polmap') ? 'block' : 'none';
  setStatus(`Selected tap: ${currentTapType}:${currentTap}`);

  document.getElementById('btnUseSpec').onclick = () => {
    const id = firstTapId("spectrum");
    if (id) { currentTapType="spectrum"; currentTap=id; }
    polControls.style.display = 'none';
  };
  document.getElementById('btnUseRD').onclick = () => {
    const id = firstTapId("rd");
    if (id) { currentTapType="rd"; currentTap=id; }
    polControls.style.display = 'none';
  };
  document.getElementById('btnUsePolMap').onclick = () => {
    const id = firstTapId("polmap");
    if (id) { currentTapType="polmap"; currentTap=id; }
    polControls.style.display = 'block';
  };

  function refreshBm(data) {
    const b = document.getElementById('bkmks'); b.innerHTML = '';
    Object.entries(data || {}).forEach(([name,f]) => {
      const row = document.createElement('div'); row.className='row';
      const span = document.createElement('span'); span.textContent = `${name}: ${f} Hz`;
      const btn = document.createElement('button'); btn.textContent='Tune'; btn.onclick = () => {
        const d = (config.devices && config.devices[0]) || null;
        if (d) {
          const q = new URLSearchParams({dev:d.id, field:'center_freq', channel:'0', value:String(f)});
          api(`/api/set_device?${q}`);
        }
      };
      row.appendChild(span); row.appendChild(btn); b.appendChild(row);
    });
  }
  api('/api/bookmarks').then(refreshBm);
  document.getElementById('bmAdd').onclick = async () => {
    const name = (document.getElementById('bmName').value || '').trim();
    const freq = parseFloat(document.getElementById('bmFreq').value || '0');
    await post('/api/bookmarks', {action:'add', name, freq});
    const d = await api('/api/bookmarks'); refreshBm(d);
  };

  function refreshPs(data) {
    const p = document.getElementById('presets'); p.innerHTML = '';
    Object.entries(data || {}).forEach(([name,obj]) => {
      const row = document.createElement('div'); row.className='row';
      const span = document.createElement('span'); span.textContent = `${name}`;
      const btn = document.createElement('button'); btn.textContent='Apply'; btn.onclick = () => {
        const d = (config.devices && config.devices[0]) || null;
        if (d) {
          if (obj.center_freq_hz!=null) { const q = new URLSearchParams({dev:d.id, field:'center_freq', channel:'0', value:String(obj.center_freq_hz)}); api(`/api/set_device?${q}`); }
          if (obj.bandwidth_hz!=null)   { const q = new URLSearchParams({dev:d.id, field:'bandwidth',   channel:'0', value:String(obj.bandwidth_hz)}); api(`/api/set_device?${q}`); }
          if (obj.gain_db!=null)        { const q = new URLSearchParams({dev:d.id, field:'gain',        channel:'0', value:String(obj.gain_db)}); api(`/api/set_device?${q}`); }
        }
      };
      row.appendChild(span); row.appendChild(btn); p.appendChild(row);
    });
  }
  api('/api/presets').then(refreshPs);
  document.getElementById('psAdd').onclick = async () => {
    const name = (document.getElementById('psName').value || '').trim();
    const raw = (document.getElementById('psData').value || '{}');
    let obj = {}; try { obj = JSON.parse(raw); } catch(e) {}
    await post('/api/presets', {action:'add', name, obj});
    const d = await api('/api/presets'); refreshPs(d);
  };
}

async function pump() {
  // Always try to update spectrum/waterfall if a spectrum tap exists.
  let specOk = false;
  try {
    if (spectrumTapId) {
      const s = await api(`/api/spectrum_latest?tap=${encodeURIComponent(spectrumTapId)}`);
      if (s && s.ok && s.psd_db) {
        specOk = true;
        drawSpectrum(s.psd_db);
        appendWaterfallRow(s.psd_db);
      }
    }
  } catch(e) {}

  // Heatmap depends on selected tap type
  try {
    if (currentTap && currentTapType === 'rd') {
      const d = await api(`/api/rd_latest?tap=${encodeURIComponent(currentTap)}`);
      if (d && d.ok && d.power_db) {
        drawHeatmap(d.power_db, -90.0, 0.0);
        setData(`RD ok=true  bins=${(d.power_db.length||0)}x${((d.power_db[0]||[]).length||0)}`);
      } else {
        setData("RD ok=false (no frames yet)");
      }
    } else if (currentTap && currentTapType === 'polmap') {
      const d = await api(`/api/polmap_latest?tap=${encodeURIComponent(currentTap)}`);
      if (d && d.ok && d.maps_db) {
        ensurePolMetricOptions(d.metrics || []);
        const metric = selectedPolMetric || (d.metrics && d.metrics[0]) || '';
        const mat = d.maps_db[metric];
        if (mat) {
          drawHeatmap(mat, -40.0, 40.0);
          setData(`PolMap ok=true metric=${metric}`);
        } else {
          setData("PolMap ok=true but missing metric matrix");
        }
      } else {
        setData("PolMap ok=false (no frames yet)");
      }
    } else {
      setData(specOk ? "Spectrum ok=true" : "Spectrum ok=false (no frames yet)");
    }
  } catch(e) {}

  setTimeout(pump, 300);
}

function init() {
  setupControls();
  pump();
}
init();
</script>
</body>
</html>
"""

def _json(obj: Any) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _enumerate_devices() -> Any:
    try:
        import SoapySDR  # type: ignore
    except Exception:
        return {"soapy": [], "note": "SoapySDR Python not available; source env/soapy_site.sh"}
    try:
        lst = SoapySDR.Device.enumerate()  # type: ignore
    except Exception as e:
        return {"soapy": [], "error": str(e)}
    out = []
    for item in lst:
        try:
            out.append({k: str(v) for k, v in item.items()})
        except Exception:
            out.append(str(item))
    return {"soapy": out}


class Handler(BaseHTTPRequestHandler):
    server_version = "FBSDrWeb/2.3"

    def do_GET(self) -> None:
        url = urlparse(self.path)

        if url.path in ("/", "/index.html"):
            cfg = getattr(self.server, "config", {"params": [], "devices": [], "taps": []})  # type: ignore
            html = INDEX_HTML.replace("__CONFIG__", json.dumps(cfg))
            self._ok("text/html; charset=utf-8", html.encode("utf-8"))
            return

        if url.path == "/api/state":
            cfg = getattr(self.server, "config", {"params": [], "devices": [], "taps": []})  # type: ignore
            self._ok_json({"config": cfg})
            return

        if url.path == "/api/devices":
            self._ok_json(_enumerate_devices())
            return

        if url.path == "/api/set_param":
            qs = parse_qs(url.query)
            name = qs.get("name", [""])[0]
            value = float(qs.get("value", ["0"])[0])
            GLOBAL_PARAMS.set(name, value)
            self._ok_json({"ok": True, "param": name, "value": value})
            return

        if url.path == "/api/set_device":
            qs = parse_qs(url.query)
            dev = qs.get("dev", [""])[0]
            field = qs.get("field", [""])[0]
            channel = int(qs.get("channel", ["0"])[0])
            value = float(qs.get("value", ["0"])[0])
            ctl = GLOBAL_CONTROL.get(dev)
            ok = False
            if ctl:
                if field in ("gain", "gain_db"):
                    ok = ctl.set_rx_gain(channel, value)
                elif field in ("center_freq", "center_freq_hz", "cf"):
                    ok = ctl.set_rx_center_freq(channel, value)
                elif field in ("bandwidth", "bandwidth_hz", "bw"):
                    ok = ctl.set_rx_bandwidth(channel, value)
            self._ok_json({"ok": bool(ok), "dev": dev, "field": field, "channel": channel, "value": value})
            return

        if url.path == "/api/spectrum_latest":
            qs = parse_qs(url.query)
            tap_id = qs.get("tap", [""])[0]
            fr = SpectrumTap.get_latest(tap_id)
            if fr is None:
                self._ok_json({"ok": False})
                return
            self._ok_json(
                {
                    "ok": True,
                    "freqs_hz": fr.freqs_hz.tolist(),
                    "psd_db": fr.psd_db.tolist(),
                    "ts": fr.timestamp_ns or 0,
                }
            )
            return

        if url.path == "/api/rd_latest":
            qs = parse_qs(url.query)
            tap_id = qs.get("tap", [""])[0]
            fr = RDTap.get_latest(tap_id)
            if fr is None:
                self._ok_json({"ok": False})
                return
            self._ok_json(
                {
                    "ok": True,
                    "range_m": fr.range_m.tolist(),
                    "doppler_hz": fr.doppler_hz.tolist(),
                    "power_db": fr.power_db.tolist(),
                    "ts": fr.timestamp_ns or 0,
                }
            )
            return

        if url.path == "/api/polmap_latest":
            qs = parse_qs(url.query)
            tap_id = qs.get("tap", [""])[0]
            fr = PolMapTap.get_latest(tap_id)
            if fr is None:
                self._ok_json({"ok": False})
                return
            maps = {k: v.tolist() for k, v in fr.maps_db.items()}
            self._ok_json(
                {
                    "ok": True,
                    "metrics": fr.metric_names,
                    "range_m": fr.range_m.tolist(),
                    "doppler_hz": fr.doppler_hz.tolist(),
                    "maps_db": maps,
                    "ts": fr.timestamp_ns or 0,
                }
            )
            return

        if url.path == "/api/bookmarks":
            self._ok_json(_BOOKMARKS)
            return

        if url.path == "/api/presets":
            self._ok_json(_PRESETS)
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {}

        if url.path == "/api/bookmarks":
            act = payload.get("action")
            if act == "add":
                name = str(payload.get("name") or "").strip()
                freq = float(payload.get("freq") or 0.0)
                if name:
                    _BOOKMARKS[name] = freq
                    self._ok_json({"ok": True})
                    return
            self._ok_json({"ok": False})
            return

        if url.path == "/api/presets":
            act = payload.get("action")
            if act == "add":
                name = str(payload.get("name") or "").strip()
                obj = payload.get("obj") or {}
                if name:
                    _PRESETS[name] = obj
                    self._ok_json({"ok": True})
                    return
            self._ok_json({"ok": False})
            return

        self.send_error(404, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _ok(self, ctype: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _ok_json(self, obj: Any) -> None:
        self._ok("application/json; charset=utf-8", _json(obj))


def start_server(port: int, config: dict) -> threading.Thread:
    srv = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    setattr(srv, "config", config)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return t
