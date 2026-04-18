from __future__ import annotations
from typing import AsyncIterator, Optional, Tuple
import numpy as np
from ..types import FaradayProjection, BufferFrame, RangeDopplerFrame

C0 = 299792458.0

def range_doppler(proj: FaradayProjection, n_fast: int, n_slow: int, window_fast: str = "hann", window_slow: str = "hann",
                  mode: str = "pulsed") -> AsyncIterator[RangeDopplerFrame]:
    sr = float(proj.meta.mode.sample_rate_hz)
    wf = np.hanning(n_fast).astype(np.float64) if window_fast == "hann" else np.ones(n_fast, dtype=np.float64)
    ws = np.hanning(n_slow).astype(np.float64) if window_slow == "hann" else np.ones(n_slow, dtype=np.float64)
    pri_s = float(n_fast) / sr
    rng_axis = np.arange(n_fast, dtype=np.float64) * (C0 / (2.0 * sr))
    async def gen() -> AsyncIterator[RangeDopplerFrame]:
        slow_idx = 0
        stack = np.zeros((n_slow, n_fast), dtype=np.complex64)
        last_ts = None
        async for frame in proj.stream:
            x = frame.samples.astype(np.complex64, copy=False)
            if x.shape[0] < n_fast: continue
            stack[slow_idx, :] = x[:n_fast] * wf.astype(np.complex64, copy=False)
            slow_idx += 1
            last_ts = frame.timestamp_ns
            if slow_idx >= n_slow:
                slow_idx = 0
                R = np.fft.fft(stack, n=n_fast, axis=1)
                R = (R * ws[:, None].astype(np.complex64, copy=False))
                RD = np.fft.fftshift(np.fft.fft(R, n=n_slow, axis=0), axes=0)
                power = 20.0 * np.log10(np.abs(RD) + 1e-20)
                doppler_axis = np.fft.fftshift(np.fft.fftfreq(n_slow, d=pri_s)).astype(np.float64)
                yield RangeDopplerFrame(range_m=rng_axis.copy(), doppler_hz=doppler_axis, power_db=power, timestamp_ns=last_ts, meta=proj.meta)
    return gen()
