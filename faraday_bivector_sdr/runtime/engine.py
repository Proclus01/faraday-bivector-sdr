from __future__ import annotations
import asyncio
import numpy as np
from typing import AsyncIterator, Optional
from ..domain.types import SpectralFrame, FaradayProjection, RangeDopplerFrame, PolMapFrame
from ..adapters.storage.npz import NPZRecorder
from ..adapters.visualization.html import write_waterfall_html, write_rangedoppler_html
from ..adapters.visualization.html_polmap import write_polmap_html

async def consume_spectrum_console(stream: AsyncIterator[SpectralFrame], frames: int = 8, topk: int = 3) -> None:
    count = 0
    async for spec in stream:
        idx = np.argsort(spec.psd_db)[-topk:][::-1]
        peaks = [(float(spec.freqs_hz[i]), float(spec.psd_db[i])) for i in idx]
        print(f"[Spectrum] frame={count} center={spec.meta.mode.center_freq_hz:.3f} Hz peaks={peaks}")
        count += 1
        if count >= frames: break
        await asyncio.sleep(0)

async def record_npz_projection(proj: FaradayProjection, path: str, max_frames: Optional[int] = None) -> None:
    rec = NPZRecorder()
    await rec.record(proj, path=path, max_frames=max_frames)
    print(f"[Recorder] Saved projection to {path}")

async def record_npz_spectrum(stream: AsyncIterator[SpectralFrame], path: str, max_frames: Optional[int] = None) -> None:
    freqs = None; psd_db_list = []; ts_list = []; count = 0
    async for spec in stream:
        if freqs is None: freqs = spec.freqs_hz.astype(np.float64, copy=True)
        psd_db_list.append(spec.psd_db.astype(np.float64, copy=True))
        ts_list.append(0 if spec.timestamp_ns is None else int(spec.timestamp_ns))
        count += 1
        if max_frames is not None and count >= max_frames: break
        await asyncio.sleep(0)
    if freqs is None:
        np.savez_compressed(path, freqs=np.zeros(0), psd=np.zeros((0,0)), timestamps=np.zeros(0, dtype=np.int64))
    else:
        psd_arr = np.stack(psd_db_list, axis=0); ts_arr = np.array(ts_list, dtype=np.int64)
        np.savez_compressed(path, freqs=freqs, psd=psd_arr, timestamps=ts_arr)
    print(f"[Recorder] Saved spectrum to {path}")

async def consume_spectrum_collect_html(stream: AsyncIterator[SpectralFrame], path: str, frames: int = 16) -> None:
    rows = []; freqs = None; count = 0
    async for spec in stream:
        if freqs is None: freqs = spec.freqs_hz.astype(np.float64, copy=True)
        rows.append(spec.psd_db.astype(np.float64, copy=True))
        count += 1
        if count >= frames: break
        await asyncio.sleep(0)
    if freqs is None:
        print("[Waterfall] No data to visualize."); return
    mat = np.stack(rows, axis=0)
    write_waterfall_html(freqs, mat, path)
    print(f"[Waterfall] Wrote HTML to {path}")

async def consume_range_doppler_html(stream: AsyncIterator[RangeDopplerFrame], path: str, frames: int = 1) -> None:
    count = 0
    async for rd in stream:
        write_rangedoppler_html(rd.range_m, rd.doppler_hz, rd.power_db.real.astype(float), path)
        print(f"[Range-Doppler] Wrote HTML to {path}")
        count += 1
        if count >= frames: break
        await asyncio.sleep(0)

async def consume_polmap_html(stream: AsyncIterator[PolMapFrame], path: str, frames: int = 1) -> None:
    count = 0
    async for pm in stream:
        write_polmap_html(pm.range_m, pm.doppler_hz, pm.maps_db, path)
        print(f"[PolMap] Wrote HTML to {path}")
        count += 1
        if count >= frames: break
        await asyncio.sleep(0)
