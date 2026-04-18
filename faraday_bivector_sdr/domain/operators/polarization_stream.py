from __future__ import annotations
import asyncio
from typing import List, Tuple, AsyncIterator
import numpy as np
from ..types import FaradayProjection, BufferFrame, ProjectionMeta, Mode
from .timestamp_align import zip_align_by_timestamp
from .polarization_static import xy_to_rl

async def _tee_stream(stream: AsyncIterator[BufferFrame], copies: int) -> List[AsyncIterator[BufferFrame]]:
    queues = [asyncio.Queue(maxsize=8) for _ in range(copies)]
    async def fanout():
        async for fr in stream:
            for q in queues:
                await q.put(fr)
        for q in queues:
            await q.put(None)
    asyncio.create_task(fanout())
    async def gen(q):
        while True:
            item = await q.get()
            if item is None: break
            yield item
    return [gen(q) for q in queues]

def pol_xy_to_rl(projs: List[FaradayProjection]) -> List[FaradayProjection]:
    """
    Combine two linear-pol projections [X,Y] into circular [R,L].
    Streams are timestamp-aligned (best effort) then transformed sample-wise.
    """
    assert len(projs) == 2, "Provide [X, Y] projections"
    X, Y = projs

    async def gen() -> AsyncIterator[Tuple[BufferFrame, BufferFrame]]:
        async for fx, fy in zip_align_by_timestamp([X.stream, Y.stream],
                                                   sample_rate_hz=float(X.meta.mode.sample_rate_hz),
                                                   tolerance_samples=4):
            yield fx, fy

    async def gen_R() -> AsyncIterator[BufferFrame]:
        async for fx, fy in gen():
            L = min(fx.samples.shape[0], fy.samples.shape[0])
            r, l = xy_to_rl(fx.samples[:L], fy.samples[:L])
            yield BufferFrame(samples=r, timestamp_ns=fx.timestamp_ns)

    async def gen_L() -> AsyncIterator[BufferFrame]:
        async for fx, fy in gen():
            L = min(fx.samples.shape[0], fy.samples.shape[0])
            r, l = xy_to_rl(fx.samples[:L], fy.samples[:L])
            yield BufferFrame(samples=l, timestamp_ns=fy.timestamp_ns)

    base = X.meta
    new_mode = Mode(center_freq_hz=base.mode.center_freq_hz, sample_rate_hz=base.mode.sample_rate_hz,
                    bandwidth_hz=base.mode.bandwidth_hz,
                    lo_chain=tuple(list(base.mode.lo_chain) + [("pol_xy_to_rl", None)]))
    meta_R = ProjectionMeta(mode=new_mode, polarization=base.polarization, pattern=base.pattern,
                            frame=base.frame, gain_db=base.gain_db, noise_figure_db=base.noise_figure_db, tags=dict(base.tags))
    meta_L = meta_R
    return [FaradayProjection(meta=meta_R, stream=gen_R()),
            FaradayProjection(meta=meta_L, stream=gen_L())]

def select(projs: List[FaradayProjection], index: int) -> FaradayProjection:
    return projs[index]
