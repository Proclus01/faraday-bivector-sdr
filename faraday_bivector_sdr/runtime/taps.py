from __future__ import annotations
import asyncio
from threading import Lock
from typing import Dict, Optional, Any, AsyncIterator

class _TapLatest:
    def __init__(self) -> None:
        self._lock = Lock()
        self._data: Dict[str, Any] = {}

    def set_latest(self, tap_id: str, frame: Any) -> None:
        with self._lock:
            self._data[tap_id] = frame

    def get_latest(self, tap_id: str) -> Optional[Any]:
        with self._lock:
            return self._data.get(tap_id)

SpectrumTap = _TapLatest()
RDTap = _TapLatest()
PolMapTap = _TapLatest()

async def feed_spectrum_tap(stream: AsyncIterator[Any], tap_id: str) -> None:
    try:
        async for frame in stream:
            SpectrumTap.set_latest(tap_id, frame)
    except asyncio.CancelledError:
        return

async def feed_rd_tap(stream: AsyncIterator[Any], tap_id: str) -> None:
    try:
        async for frame in stream:
            RDTap.set_latest(tap_id, frame)
    except asyncio.CancelledError:
        return

async def feed_polmap_tap(stream: AsyncIterator[Any], tap_id: str) -> None:
    try:
        async for frame in stream:
            PolMapTap.set_latest(tap_id, frame)
    except asyncio.CancelledError:
        return
