from __future__ import annotations
import asyncio
from typing import List, AsyncIterator
from ..types import FaradayProjection, BufferFrame, ProjectionMeta

def tee(proj: FaradayProjection, copies: int = 2) -> List[FaradayProjection]:
    """
    Fan out a projection into N identical streams using async queues.
    """
    queues = [asyncio.Queue(maxsize=8) for _ in range(copies)]

    async def fanout():
        async for fr in proj.stream:
            for q in queues:
                await q.put(fr)
        for q in queues:
            await q.put(None)

    asyncio.create_task(fanout())

    async def gen(q) -> AsyncIterator[BufferFrame]:
        while True:
            item = await q.get()
            if item is None:
                break
            yield item

    return [FaradayProjection(meta=proj.meta, stream=gen(q)) for q in queues]
