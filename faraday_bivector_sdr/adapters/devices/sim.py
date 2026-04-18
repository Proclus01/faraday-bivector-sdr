from __future__ import annotations
import asyncio
import math
from dataclasses import dataclass
from typing import Dict, Any, AsyncIterator, List, Optional
import numpy as np
from ...ports.device import RadioDevicePort
from ...domain.types import (
    FaradayProjection,
    ProjectionMeta,
    PolarizationBasis,
    Pattern,
    FrameRef,
    Mode,
    BufferFrame,
)

@dataclass
class SimChannelConfig:
    center_freq_hz: float
    sample_rate_hz: float
    bandwidth_hz: float
    block_size: int = 4096
    tone_offsets_hz: Optional[List[float]] = None
    amplitudes: Optional[List[float]] = None
    initial_phases_rad: Optional[List[float]] = None
    integer_sample_delay: int = 0
    noise_std: float = 0.01
    seed: int = 1234
    max_frames: Optional[int] = 256

class SimDeviceAdapter(RadioDevicePort):
    def __init__(self, args: str = "") -> None:
        self.args = args
        self._opened = False

    def open_rx(self, chan_cfg: Dict[str, Any]) -> FaradayProjection:
        cfg = SimChannelConfig(**chan_cfg)
        self._opened = True
        meta = ProjectionMeta(
            mode=Mode(
                center_freq_hz=cfg.center_freq_hz,
                sample_rate_hz=cfg.sample_rate_hz,
                bandwidth_hz=cfg.bandwidth_hz,
                lo_chain=(cfg.center_freq_hz,),
            ),
            polarization=PolarizationBasis(name="UNKNOWN", matrix_to_xy=np.eye(2, dtype=np.complex64)),
            pattern=Pattern("omni"),
            frame=FrameRef("DEVICE", t0_ns=0),
            gain_db=0.0,
            noise_figure_db=None,
            tags={"adapter":"sim"},
        )
        async def gen() -> AsyncIterator[BufferFrame]:
            rng = np.random.default_rng(cfg.seed)
            sr = float(cfg.sample_rate_hz)
            M = int(cfg.block_size)
            tones = cfg.tone_offsets_hz or []
            amps = cfg.amplitudes or [1.0 for _ in tones]
            phases = cfg.initial_phases_rad or [0.0 for _ in tones]
            w = [2.0 * math.pi * (f / sr) for f in tones]
            ts_ns = 0
            delay_left = max(0, int(cfg.integer_sample_delay))
            frames_left = cfg.max_frames if (cfg.max_frames is not None) else -1
            while (frames_left != 0):
                n = np.arange(M, dtype=np.float64)
                x = np.zeros(M, dtype=np.complex64)
                for k, amp in enumerate(amps):
                    phi = w[k] * n + phases[k]
                    x += (amp * np.exp(1j * phi)).astype(np.complex64)
                    phases[k] = float(phi[-1] + w[k])
                if cfg.noise_std > 0.0:
                    noise = (rng.normal(0.0, cfg.noise_std, size=M) + 1j * rng.normal(0.0, cfg.noise_std, size=M)).astype(np.complex64)
                    x += noise
                if delay_left > 0:
                    if delay_left >= M:
                        out = np.zeros(M, dtype=np.complex64)
                        delay_left -= M
                    else:
                        out = np.concatenate([np.zeros(delay_left, dtype=np.complex64), x[:-delay_left]], axis=0)
                        delay_left = 0
                else:
                    out = x
                yield BufferFrame(samples=out, timestamp_ns=ts_ns)
                ts_ns += int(1e9 * (M / sr))
                if frames_left > 0:
                    frames_left -= 1
                await asyncio.sleep(0)
        return FaradayProjection(meta=meta, stream=gen())

    def close(self) -> None:
        self._opened = False
