from __future__ import annotations
from typing import Optional
import numpy as np
from ...ports.storage import RecorderPort
from ...domain.types import FaradayProjection

class ZarrStreamRecorder(RecorderPort):
    """
    Appendable streaming Zarr writer. Requires 'zarr'.
    """
    def __init__(self, chunk: int = 16384) -> None:
        self.chunk = chunk

    async def record(self, proj: FaradayProjection, path: str, max_frames: Optional[int] = None) -> None:
        try:
            import zarr  # type: ignore
        except Exception as e:
            raise RuntimeError("Zarr is not installed. pip install zarr") from e
        g = zarr.open(path, mode="w")
        g.attrs["center_freq_hz"] = float(proj.meta.mode.center_freq_hz)
        g.attrs["sample_rate_hz"] = float(proj.meta.mode.sample_rate_hz)
        g.attrs["bandwidth_hz"] = float(proj.meta.mode.bandwidth_hz)
        g.attrs["lo_chain"] = [str(x) for x in proj.meta.mode.lo_chain]
        g.attrs["pol_name"] = proj.meta.polarization.name
        g.attrs["pattern"] = proj.meta.pattern.description if proj.meta.pattern else "unknown"
        g.attrs["frame_name"] = proj.meta.frame.name
        g.attrs["t0_ns"] = int(proj.meta.frame.t0_ns)
        g.attrs["gain_db"] = float(proj.meta.gain_db)
        samples = g.create_dataset("samples", shape=(0,), chunks=(self.chunk,), dtype="c8", maxshape=(None,))
        stamps = g.create_dataset("timestamps", shape=(0,), chunks=(self.chunk,), dtype="i8", maxshape=(None,))
        count = 0
        async for fr in proj.stream:
            # append
            N = fr.samples.shape[0]
            samples.resize(samples.shape[0] + N)
            samples[-N:] = fr.samples
            stamps.resize(stamps.shape[0] + 1)
            stamps[-1] = 0 if fr.timestamp_ns is None else int(fr.timestamp_ns)
            count += 1
            if max_frames is not None and count >= max_frames:
                break
