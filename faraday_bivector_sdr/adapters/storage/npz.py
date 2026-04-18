from __future__ import annotations
import numpy as np
from typing import Optional
from ...ports.storage import RecorderPort
from ...domain.types import FaradayProjection

class NPZRecorder(RecorderPort):
    async def record(self, proj: FaradayProjection, path: str, max_frames: Optional[int] = None) -> None:
        samples_list = []; ts_list = []; count = 0
        async for frame in proj.stream:
            samples_list.append(frame.samples.copy())
            ts_list.append(0 if frame.timestamp_ns is None else int(frame.timestamp_ns))
            count += 1
            if max_frames is not None and count >= max_frames: break
        if not samples_list:
            samples = np.zeros(0, dtype=np.complex64); timestamps = np.zeros(0, dtype=np.int64)
        else:
            samples = np.concatenate(samples_list).astype(np.complex64, copy=False)
            timestamps = np.array(ts_list, dtype=np.int64)
        meta = proj.meta
        np.savez_compressed(
            path,
            samples=samples,
            timestamps=timestamps,
            center_freq_hz=np.array(meta.mode.center_freq_hz),
            sample_rate_hz=np.array(meta.mode.sample_rate_hz),
            bandwidth_hz=np.array(meta.mode.bandwidth_hz),
            lo_chain=np.array(meta.mode.lo_chain, dtype=object),
            pol_name=np.array(meta.polarization.name),
            pattern=np.array(meta.pattern.description if meta.pattern else "unknown"),
            frame_name=np.array(meta.frame.name),
            t0_ns=np.array(meta.frame.t0_ns),
            gain_db=np.array(meta.gain_db),
        )
