from __future__ import annotations
import asyncio
from typing import List, AsyncIterator, Tuple, Optional
import numpy as np
from ..types import BufferFrame

async def zip_align_by_timestamp(streams: List[AsyncIterator[BufferFrame]], sample_rate_hz: float,
                                 tolerance_samples: int = 4) -> AsyncIterator[Tuple[BufferFrame, ...]]:
    """
    Align frames across multiple streams by timestamp within tolerance.
    If timestamps are None, fall back to naive zipping by arrival order.
    Emits tuples of frames (one per stream) when alignment found; drops unmatched.
    """
    N = len(streams)
    iters = [s.__aiter__() for s in streams]
    buffers: List[Optional[BufferFrame]] = [None] * N
    queues: List[asyncio.Queue] = [asyncio.Queue(maxsize=8) for _ in range(N)]

    async def pump(i: int):
        try:
            async for fr in streams[i]:
                await queues[i].put(fr)
            await queues[i].put(None)
        except Exception:
            await queues[i].put(None)

    for i in range(N):
        asyncio.create_task(pump(i))

    tol_ns = int((tolerance_samples / max(1.0, sample_rate_hz)) * 1e9)

    alive = [True] * N
    while all(alive):
        # get one from each if empty
        for i in range(N):
            if buffers[i] is None:
                fr = await queues[i].get()
                if fr is None:
                    alive[i] = False
                    break
                buffers[i] = fr
        if not all(alive):
            break

        ts = [b.timestamp_ns for b in buffers]  # type: ignore
        if any(t is None for t in ts):
            # fallback zip: emit as-is
            yield tuple(buffers)  # type: ignore
            buffers = [None] * N
            continue

        tmin = min(ts)  # type: ignore
        tmax = max(ts)  # type: ignore
        if int(tmax) - int(tmin) <= tol_ns:
            # aligned
            yield tuple(buffers)  # type: ignore
            buffers = [None] * N
        else:
            # advance the earliest
            idx = int(np.argmin(np.array(ts, dtype=np.int64)))
            buffers[idx] = None
            # loop continues
