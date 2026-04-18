from __future__ import annotations
import asyncio
from typing import AsyncIterator, List, Tuple, Optional
import numpy as np
from ..types import RangeDopplerFrame

async def zip_align_rd_by_timestamp(streams: List[AsyncIterator[RangeDopplerFrame]], pri_s: float,
                                    tolerance_pri: int = 1) -> AsyncIterator[Tuple[RangeDopplerFrame, ...]]:
    """
    Align Range-Doppler frames by timestamps (nearest within tolerance_pri * PRI).
    If timestamps missing, zip by arrival order.
    """
    N = len(streams)
    queues = [asyncio.Queue(maxsize=8) for _ in range(N)]

    async def pump(i: int):
        try:
            async for fr in streams[i]:
                await queues[i].put(fr)
            await queues[i].put(None)
        except Exception:
            await queues[i].put(None)

    for i in range(N):
        asyncio.create_task(pump(i))

    tol_ns = int(pri_s * 1e9 * max(0, int(tolerance_pri)))
    alive = [True] * N
    buffers: List[Optional[RangeDopplerFrame]] = [None]*N
    while all(alive):
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
            yield tuple(buffers)  # type: ignore
            buffers = [None]*N
            continue
        tmin = min(ts)  # type: ignore
        tmax = max(ts)  # type: ignore
        if int(tmax) - int(tmin) <= tol_ns:
            yield tuple(buffers)  # type: ignore
            buffers = [None]*N
        else:
            idx = int(np.argmin(np.array(ts, dtype=np.int64)))
            buffers[idx] = None
