from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..domain.types import FaradayProjection

class RadioDevicePort(ABC):
    @abstractmethod
    def open_rx(self, chan_cfg: Dict[str, Any]) -> FaradayProjection:
        raise NotImplementedError
    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

class TxHandle(ABC):
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...
