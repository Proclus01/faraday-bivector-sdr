from __future__ import annotations
from typing import AsyncIterator
import numpy as np
from ..types import FaradayProjection, SpectralFrame

def estimate_spectrum(proj: FaradayProjection, fft_size: int = 4096, avg: int = 8, window: str = "hann") -> AsyncIterator[SpectralFrame]:
    sr = float(proj.meta.mode.sample_rate_hz)
    win = np.hanning(fft_size).astype(np.float64) if window == "hann" else np.ones(fft_size, dtype=np.float64)
    win_power = np.sum(win**2)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0/sr)).astype(np.float64)
    acc = np.zeros(fft_size, dtype=np.float64); acc_count = 0
    buf = np.zeros(fft_size, dtype=np.complex64); cursor = 0; last_ts = None
    async def gen() -> AsyncIterator[SpectralFrame]:
        nonlocal acc, acc_count, buf, cursor, last_ts
        async for frame in proj.stream:
            x = frame.samples.astype(np.complex64, copy=False); i = 0
            while i < x.shape[0]:
                take = min(fft_size - cursor, x.shape[0] - i)
                buf[cursor:cursor+take] = x[i:i+take]; cursor += take; i += take
                if cursor == fft_size:
                    last_ts = frame.timestamp_ns; cursor = 0
                    X = np.fft.fft(buf * win.astype(np.complex64, copy=False), n=fft_size)
                    psd = (np.abs(np.fft.fftshift(X))**2).astype(np.float64) / (win_power + 1e-12)
                    acc += psd; acc_count += 1
                    if acc_count >= avg:
                        psd_avg = acc / max(1, acc_count)
                        psd_db = 10.0 * np.log10(psd_avg + 1e-20)
                        yield SpectralFrame(freqs_hz=freqs.copy(), psd_db=psd_db, timestamp_ns=last_ts, meta=proj.meta)
                        acc[:] = 0.0; acc_count = 0
    return gen()
