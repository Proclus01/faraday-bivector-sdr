from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .types import FaradayProjection

class FieldOperator(Protocol):
    def __call__(self, proj: FaradayProjection) -> FaradayProjection: ...

@dataclass(frozen=True)
class AntennaFunctional:
    name: str
    description: str = "generic"
