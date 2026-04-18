from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator, List

async def tee_async(stream: AsyncIterator[Any], copies: int = 2) -> List[AsyncIterator[Any]]:
    """
    Fan out an async iterator to N subscribers without consuming frames twice.
    """
    queues = [asyncio.Queue(maxsize=8) for _ in range(copies)]

    async def fanout():
        try:
            async for item in stream:
                for q in queues:
                    try:
                        await q.put(item)
                    except (asyncio.CancelledError, RuntimeError):
                        # RuntimeError may happen on shutdown when loop is closing
                        return
        except asyncio.CancelledError:
            pass
        finally:
            for q in queues:
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

    asyncio.create_task(fanout())

    async def gen(q: asyncio.Queue):
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                yield item
        except asyncio.CancelledError:
            return

    return [gen(q) for q in queues]
