from __future__ import annotations
import asyncio
from typing import Dict, Any, AsyncIterator
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

class NPZPlaybackAdapter(RadioDevicePort):
    def __init__(self, args: str = "") -> None:
        self.args = args

    def open_rx(self, chan_cfg: Dict[str, Any]) -> FaradayProjection:
        path = chan_cfg["path"]
        data = np.load(path, allow_pickle=False)
        samples = data["samples"].astype(np.complex64, copy=False)
        timestamps = data["timestamps"].astype(np.int64, copy=False)
        meta = ProjectionMeta(
            mode=Mode(center_freq_hz=float(data["center_freq_hz"]),
                      sample_rate_hz=float(data["sample_rate_hz"]),
                      bandwidth_hz=float(data["bandwidth_hz"]),
                      lo_chain=tuple(data["lo_chain"].tolist())),
            polarization=PolarizationBasis(name=str(data["pol_name"]), matrix_to_xy=np.eye(2, dtype=np.complex64)),
            pattern=Pattern(str(data["pattern"])),
            frame=FrameRef(str(data["frame_name"]), int(data["t0_ns"])),
            gain_db=float(data["gain_db"]),
            noise_figure_db=None,
            tags={"adapter":"file_npz","source_path":path},
        )
        block_size = int(chan_cfg.get("block_size", 8192))
        async def gen() -> AsyncIterator[BufferFrame]:
            i = 0; N = samples.shape[0]
            while i < N:
                take = min(block_size, N - i)
                out = samples[i:i+take]
                ts = int(timestamps[i]) if i < timestamps.shape[0] else None
                i += take
                yield BufferFrame(samples=out, timestamp_ns=ts)
                await asyncio.sleep(0)
        return FaradayProjection(meta=meta, stream=gen())

    def close(self) -> None:
        pass
