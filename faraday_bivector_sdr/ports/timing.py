from __future__ import annotations
from abc import ABC, abstractmethod

class TimingSyncPort(ABC):
    @abstractmethod
    def set_clock_source(self, source: str) -> None: ...
    @abstractmethod
    def set_time_now(self, time_ns: int) -> None: ...
    @abstractmethod
    def get_time_now(self) -> int: ...
