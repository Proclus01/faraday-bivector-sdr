from __future__ import annotations
import numpy as np
from typing import AsyncIterator
from ..types import FaradayProjection, BufferFrame, Mode, ProjectionMeta

def frequency_shift(proj: FaradayProjection, delta_hz: float, initial_phase: float = 0.0) -> FaradayProjection:
    mode = proj.meta.mode
    sr = float(mode.sample_rate_hz)
    w = 2.0 * np.pi * (delta_hz / sr)

    async def gen() -> AsyncIterator[BufferFrame]:
        phase = float(initial_phase)
        async for frame in proj.stream:
            x = frame.samples
            n = np.arange(x.shape[0], dtype=np.float64)
            phi = w * n + phase
            lo = np.exp(1j * phi).astype(np.complex64)
            y = (x * lo).astype(np.complex64, copy=False)
            phase = float(phi[-1] + w)
            yield BufferFrame(samples=y, timestamp_ns=frame.timestamp_ns)

    new_mode = Mode(
        center_freq_hz=mode.center_freq_hz + delta_hz,
        sample_rate_hz=mode.sample_rate_hz,
        bandwidth_hz=mode.bandwidth_hz,
        lo_chain=tuple(list(mode.lo_chain) + [("shift", delta_hz)]),
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
