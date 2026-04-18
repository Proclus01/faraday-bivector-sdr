from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class DeviceDef:
    id: str
    adapter: str
    args: str = ""
    rx_channels: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ProjectionDef:
    id: str
    device: str
    channel: int
    antenna: Optional[str] = None

@dataclass
class PipelineDef:
    id: str
    input: Optional[str] = None
    inputs: Optional[List[str]] = None
    ops: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class OutputDef:
    type: str
    params: Dict[str, Any]

@dataclass
class Config:
    devices: List[DeviceDef]
    projections: List[ProjectionDef]
    pipelines: List[PipelineDef]
    outputs: List[OutputDef] = field(default_factory=list)
