from __future__ import annotations
import asyncio
from typing import List, AsyncIterator, Tuple
import numpy as np
from ..types import FaradayProjection, BufferFrame, ProjectionMeta, Mode

C0 = 299792458.0

def _unit_vector_from_azel(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    v = np.array([x, y, z], dtype=np.float64)
    n = np.linalg.norm(v) + 1e-18
    return v / n

async def _zip_frames(streams: List[AsyncIterator[BufferFrame]]) -> AsyncIterator[List[BufferFrame]]:
    iters = [s.__aiter__() for s in streams]
    while True:
        frames = []
        for it in iters:
            try:
                f = await it.__anext__()
            except StopAsyncIteration:
                return
            frames.append(f)
        yield frames

def align_time_phase(projs: List[FaradayProjection]) -> List[FaradayProjection]:
    """
    Best-effort constant per-channel phase alignment relative to channel 0 (consumes first frame internally for measurement).
    """
    corrected: List[FaradayProjection] = []
    for i, p in enumerate(projs):
        if i == 0:
            corrected.append(p)
        else:
            async def gen(idx=i) -> AsyncIterator[BufferFrame]:
                it_ref = projs[0].stream.__aiter__()
                it_cur = projs[idx].stream.__aiter__()
                try:
                    fr_ref = await it_ref.__anext__()
                    fr_cur = await it_cur.__anext__()
                except StopAsyncIteration:
                    return
                L = min(fr_ref.samples.shape[0], fr_cur.samples.shape[0])
                cross = fr_cur.samples[:L].astype(np.complex64, copy=False) * np.conj(fr_ref.samples[:L].astype(np.complex64, copy=False))
                phi = float(np.angle(np.mean(cross + 1e-20)))
                corr = np.exp(-1j * phi).astype(np.complex64)
                yield BufferFrame(samples=(fr_cur.samples.astype(np.complex64, copy=False) * corr), timestamp_ns=fr_cur.timestamp_ns)
                async for fr in it_cur:
                    yield BufferFrame(samples=(fr.samples.astype(np.complex64, copy=False) * corr), timestamp_ns=fr.timestamp_ns)
            corrected.append(FaradayProjection(meta=p.meta, stream=gen()))
    return corrected

def beamform_narrowband(projs: List[FaradayProjection], element_positions_m: List[Tuple[float,float,float]],
                        az_deg: float, el_deg: float, normalize: bool = True) -> FaradayProjection:
    assert len(projs) == len(element_positions_m) and len(projs) >= 1
    fc = float(projs[0].meta.mode.center_freq_hz)
    k_hat = _unit_vector_from_azel(az_deg, el_deg)
    positions = np.array(element_positions_m, dtype=np.float64)
    taus = (positions @ k_hat) / C0
    weights = np.exp(-1j * (2.0 * np.pi * fc) * taus).astype(np.complex64)
    if normalize and weights.size > 0:
        weights = weights / np.sqrt(weights.size)

    async def gen() -> AsyncIterator[BufferFrame]:
        async for frames in _zip_frames([p.stream for p in projs]):
            L = min(f.samples.shape[0] for f in frames)
            acc = np.zeros(L, dtype=np.complex64)
            for i, fr in enumerate(frames):
                acc += weights[i] * fr.samples[:L].astype(np.complex64, copy=False)
            ts = None
            for fr in frames:
                if fr.timestamp_ns is not None:
                    ts = fr.timestamp_ns; break
            yield BufferFrame(samples=acc, timestamp_ns=ts)

    ref_meta = projs[0].meta
    new_mode = Mode(center_freq_hz=ref_meta.mode.center_freq_hz, sample_rate_hz=ref_meta.mode.sample_rate_hz,
                    bandwidth_hz=ref_meta.mode.bandwidth_hz,
                    lo_chain=tuple(list(ref_meta.mode.lo_chain) + [("beamform_nb", (az_deg, el_deg))]))
    new_meta = ProjectionMeta(mode=new_mode, polarization=ref_meta.polarization, pattern=ref_meta.pattern, frame=ref_meta.frame,
                              gain_db=ref_meta.gain_db, noise_figure_db=ref_meta.noise_figure_db,
                              tags=dict(ref_meta.tags, beamformer="narrowband"))
    return FaradayProjection(meta=new_meta, stream=gen())
