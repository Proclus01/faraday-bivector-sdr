from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from ..domain.types import FaradayProjection

class RecorderPort(ABC):
    @abstractmethod
    async def record(self, proj: FaradayProjection, path: str, max_frames: Optional[int] = None) -> None: ...

class PlaybackPort(ABC):
    @abstractmethod
    def open(self, path: str) -> FaradayProjection: ...
