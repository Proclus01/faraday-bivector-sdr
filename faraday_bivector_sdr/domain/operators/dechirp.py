from __future__ import annotations
import numpy as np
from typing import AsyncIterator
from ..types import FaradayProjection, BufferFrame, Mode, ProjectionMeta

def dechirp_local(proj: FaradayProjection, sweep_bw_hz: float, sweep_time_s: float, initial_phase: float = 0.0) -> FaradayProjection:
    """
    Multiply by conj of local FMCW chirp reference (baseband). Assumes chirp repeats every sweep_time_s.
    """
    sr = float(proj.meta.mode.sample_rate_hz)
    Ns = max(1, int(round(sweep_time_s * sr)))
    k = float(sweep_bw_hz) / float(sweep_time_s)  # Hz/slope

    # precompute one period
    n = np.arange(Ns, dtype=np.float64)
    phase = 2.0 * np.pi * (0.5 * (k / sr) * n**2) + initial_phase  # integral of freq over samples
    ref = np.exp(-1j * phase).astype(np.complex64)  # conjugate
    async def gen() -> AsyncIterator[BufferFrame]:
        offset = 0
        async for frame in proj.stream:
            x = frame.samples.astype(np.complex64, copy=False)
            y = np.empty_like(x)
            i = 0
            while i < x.shape[0]:
                take = min(Ns - offset, x.shape[0] - i)
                y[i:i+take] = x[i:i+take] * ref[offset:offset+take]
                i += take
                offset = (offset + take) % Ns
            yield BufferFrame(samples=y, timestamp_ns=frame.timestamp_ns)

    new_mode = Mode(center_freq_hz=proj.meta.mode.center_freq_hz, sample_rate_hz=sr,
                    bandwidth_hz=proj.meta.mode.bandwidth_hz,
                    lo_chain=tuple(list(proj.meta.mode.lo_chain) + [("dechirp_local", (sweep_bw_hz, sweep_time_s))]))
    new_meta = ProjectionMeta(mode=new_mode, polarization=proj.meta.polarization, pattern=proj.meta.pattern, frame=proj.meta.frame,
                              gain_db=proj.meta.gain_db, noise_figure_db=proj.meta.noise_figure_db, tags=dict(proj.meta.tags))
    return FaradayProjection(meta=new_meta, stream=gen())
