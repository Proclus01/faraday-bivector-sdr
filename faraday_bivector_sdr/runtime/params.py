from __future__ import annotations
import threading

class ParameterStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vals = {}

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._vals[name] = float(value)

    def get(self, name: str, default: float = 0.0) -> float:
        with self._lock:
            return float(self._vals.get(name, default))

GLOBAL_PARAMS = ParameterStore()
