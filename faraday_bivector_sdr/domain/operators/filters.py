from __future__ import annotations
from typing import AsyncIterator
import numpy as np
from ..types import FaradayProjection, BufferFrame, Mode, ProjectionMeta
from ..accel.accelerate import convolve_real_ir_complex_sig

def _design_lowpass(fc_hz: float, sr_hz: float, taps: int, window: str = "hann") -> np.ndarray:
    fc = float(fc_hz) / float(sr_hz)
    if fc <= 0.0:
        h = np.zeros(taps, dtype=np.float64); h[(taps-1)//2] = 1.0; return h
    n = np.arange(taps) - (taps - 1) / 2.0
    h = 2.0 * fc * np.sinc(2.0 * fc * n)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(taps) / (taps - 1))) if window == "hann" else np.ones(taps)
    h *= w; h /= np.sum(h)
    return h.astype(np.float64)

def _design_bandpass(f1_hz: float, f2_hz: float, sr_hz: float, taps: int, window: str = "hann") -> np.ndarray:
    f1 = max(0.0, min(abs(f1_hz), sr_hz * 0.5)); f2 = max(0.0, min(abs(f2_hz), sr_hz * 0.5))
    if f2 < f1: f1, f2 = f2, f1
    h2 = _design_lowpass(f2, sr_hz, taps, window); h1 = _design_lowpass(f1, sr_hz, taps, window)
    h = h2 - h1; s = np.sum(np.abs(h))
    if s > 0: h /= s / (taps * 0.5)
    return h.astype(np.float64)

def bandpass_fir(proj: FaradayProjection, low_hz: float, high_hz: float, taps: int = 257, window: str = "hann") -> FaradayProjection:
    sr = float(proj.meta.mode.sample_rate_hz)
    h = _design_bandpass(low_hz, high_hz, sr, taps, window).astype(np.float64); L = int(h.shape[0])
    assert L >= 2 and (L % 2 == 1), "Use odd number of taps."
    async def gen() -> AsyncIterator[BufferFrame]:
        state = np.zeros(L - 1, dtype=np.complex64)
        async for frame in proj.stream:
            x = frame.samples.astype(np.complex64, copy=False)
            xpad = np.concatenate([state, x], axis=0)
            y = convolve_real_ir_complex_sig(h, xpad)
            state = xpad[-(L - 1):]
            yield BufferFrame(samples=y, timestamp_ns=frame.timestamp_ns)
    new_bw = min(proj.meta.mode.bandwidth_hz, max(1.0, abs(high_hz - low_hz)))
    new_mode = Mode(proj.meta.mode.center_freq_hz, proj.meta.mode.sample_rate_hz, new_bw,
                    tuple(list(proj.meta.mode.lo_chain) + [("bandpass", (low_hz, high_hz))]))
    new_meta = ProjectionMeta(mode=new_mode, polarization=proj.meta.polarization, pattern=proj.meta.pattern, frame=proj.meta.frame,
                              gain_db=proj.meta.gain_db, noise_figure_db=proj.meta.noise_figure_db, tags=dict(proj.meta.tags))
    return FaradayProjection(meta=new_meta, stream=gen())

def decimate(proj: FaradayProjection, factor: int) -> FaradayProjection:
    if factor <= 1: return proj
    sr = float(proj.meta.mode.sample_rate_hz)
    async def gen() -> AsyncIterator[BufferFrame]:
        carry = np.empty(0, dtype=np.complex64)
        async for frame in proj.stream:
            x = frame.samples.astype(np.complex64, copy=False)
            if carry.size > 0: x = np.concatenate([carry, x], axis=0)
            n_out = (x.shape[0]) // factor
            y = x[:n_out * factor:factor]
            carry_len = x.shape[0] - n_out * factor
            carry = x[-carry_len:] if carry_len > 0 else np.empty(0, dtype=np.complex64)
            yield BufferFrame(samples=y, timestamp_ns=frame.timestamp_ns)
    new_mode = Mode(proj.meta.mode.center_freq_hz, proj.meta.mode.sample_rate_hz / factor,
                    min(proj.meta.mode.bandwidth_hz, sr / (2.0 * factor)),
                    tuple(list(proj.meta.mode.lo_chain) + [("decimate", factor)]))
    new_meta = ProjectionMeta(mode=new_mode, polarization=proj.meta.polarization, pattern=proj.meta.pattern, frame=proj.meta.frame,
                              gain_db=proj.meta.gain_db, noise_figure_db=proj.meta.noise_figure_db, tags=dict(proj.meta.tags))
    return FaradayProjection(meta=new_meta, stream=gen())

def fractional_delay(proj: FaradayProjection, delay_samples: float, taps: int = 129, window: str = "hann") -> FaradayProjection:
    """
    Apply fractional delay (can be negative) using windowed-sinc FIR. delay_samples can be any real number.
    """
    D = float(delay_samples)
    n = np.arange(taps) - (taps - 1) / 2.0
    # Ideal shifted impulse: sinc(n - D)
    h = np.sinc(n - D)
    if window == "hann":
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(taps) / (taps - 1)))
        h *= w
    h = h / np.sum(h)
    h = h.astype(np.float64)

    async def gen() -> AsyncIterator[BufferFrame]:
        state = np.zeros(taps - 1, dtype=np.complex64)
        async for frame in proj.stream:
            x = frame.samples.astype(np.complex64, copy=False)
            xpad = np.concatenate([state, x], axis=0)
            y = convolve_real_ir_complex_sig(h, xpad)
            state = xpad[-(taps - 1):]
            yield BufferFrame(samples=y, timestamp_ns=frame.timestamp_ns)

    new_mode = Mode(
        center_freq_hz=proj.meta.mode.center_freq_hz,
        sample_rate_hz=proj.meta.mode.sample_rate_hz,
        bandwidth_hz=proj.meta.mode.bandwidth_hz,
        lo_chain=tuple(list(proj.meta.mode.lo_chain) + [("frac_delay", D)]),
    )
    new_meta = ProjectionMeta(mode=new_mode, polarization=proj.meta.polarization, pattern=proj.meta.pattern, frame=proj.meta.frame,
                              gain_db=proj.meta.gain_db, noise_figure_db=proj.meta.noise_figure_db, tags=dict(proj.meta.tags))
    return FaradayProjection(meta=new_meta, stream=gen())
