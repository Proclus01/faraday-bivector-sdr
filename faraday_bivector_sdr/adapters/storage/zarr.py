from __future__ import annotations
from typing import Optional
import numpy as np
from ...ports.storage import RecorderPort
from ...domain.types import FaradayProjection

class ZarrRecorder(RecorderPort):
    def __init__(self, chunks: int = 16384) -> None:
        self.chunks = chunks
    async def record(self, proj: FaradayProjection, path: str, max_frames: Optional[int] = None) -> None:
        try:
            import zarr  # type: ignore
        except Exception as e:
            raise RuntimeError("Zarr is not installed. Install with: pip install zarr") from e
        samples_list = []; ts_list = []; count = 0
        async for frame in proj.stream:
            samples_list.append(frame.samples.copy())
            ts_list.append(0 if frame.timestamp_ns is None else int(frame.timestamp_ns))
            count += 1
            if max_frames is not None and count >= max_frames: break
        samples = np.concatenate(samples_list).astype(np.complex64, copy=False) if samples_list else np.zeros(0, dtype=np.complex64)
        timestamps = np.array(ts_list, dtype=np.int64)
        g = zarr.open(path, mode="w")
        a = g.create_dataset("samples", data=samples, chunks=self.chunks, dtype="c8")
        t = g.create_dataset("timestamps", data=timestamps, chunks=self.chunks, dtype="i8")
        meta = proj.meta
        g.attrs["center_freq_hz"] = float(meta.mode.center_freq_hz)
        g.attrs["sample_rate_hz"] = float(meta.mode.sample_rate_hz)
        g.attrs["bandwidth_hz"] = float(meta.mode.bandwidth_hz)
        g.attrs["lo_chain"] = [str(x) for x in meta.mode.lo_chain]
        g.attrs["pol_name"] = meta.polarization.name
        g.attrs["pattern"] = meta.pattern.description if meta.pattern else "unknown"
        g.attrs["frame_name"] = meta.frame.name
        g.attrs["t0_ns"] = int(meta.frame.t0_ns)
        g.attrs["gain_db"] = float(meta.gain_db)
