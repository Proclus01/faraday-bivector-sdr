from __future__ import annotations
import numpy as np
from typing import AsyncIterator
from ..types import FaradayProjection, BufferFrame, Mode, ProjectionMeta
from ...runtime.params import GLOBAL_PARAMS

def dyn_frequency_shift(proj: FaradayProjection, param: str, initial_phase: float = 0.0) -> FaradayProjection:
    """
    Real-time frequency shift by delta = GLOBAL_PARAMS.get(param) [Hz].
    """
    mode = proj.meta.mode
    sr = float(mode.sample_rate_hz)

    async def gen() -> AsyncIterator[BufferFrame]:
        phase = float(initial_phase)
        last_delta = None
        w = 0.0
        async for frame in proj.stream:
            delta_hz = GLOBAL_PARAMS.get(param, 0.0)
            if last_delta is None or delta_hz != last_delta:
                w = 2.0 * np.pi * (delta_hz / sr)
                last_delta = delta_hz
            x = frame.samples.astype(np.complex64, copy=False)
            n = np.arange(x.shape[0], dtype=np.float64)
            phi = w * n + phase
            lo = np.exp(1j * phi).astype(np.complex64)
            y = (x * lo).astype(np.complex64, copy=False)
            phase = float(phi[-1] + w)
            yield BufferFrame(samples=y, timestamp_ns=frame.timestamp_ns)

    new_mode = Mode(
        center_freq_hz=mode.center_freq_hz,
        sample_rate_hz=mode.sample_rate_hz,
        bandwidth_hz=mode.bandwidth_hz,
        lo_chain=tuple(list(mode.lo_chain) + [("dyn_shift", param)]),
    )
    new_meta = ProjectionMeta(
        mode=new_mode,
        polarization=proj.meta.polarization,
        pattern=proj.meta.pattern,
        frame=proj.meta.frame,
        gain_db=proj.meta.gain_db,
        noise_figure_db=proj.meta.noise_figure_db,
        tags=dict(proj.meta.tags),
    )
    return FaradayProjection(meta=new_meta, stream=gen())

def dyn_gain(proj: FaradayProjection, param: str, unit: str = "linear") -> FaradayProjection:
    """
    Real-time gain scaling by GLOBAL_PARAMS.get(param). If unit='db', uses 10^(g/20).
    """
    async def gen() -> AsyncIterator[BufferFrame]:
        last_val = None
        scale = 1.0
        async for frame in proj.stream:
            g = GLOBAL_PARAMS.get(param, 1.0)
            if unit == "db":
                if last_val is None or g != last_val:
                    scale = float(10.0**(g/20.0))
                    last_val = g
            else:
                scale = float(g)
            yield BufferFrame(samples=(frame.samples.astype(np.complex64, copy=False) * scale),
                              timestamp_ns=frame.timestamp_ns)
    return FaradayProjection(meta=proj.meta, stream=gen())
