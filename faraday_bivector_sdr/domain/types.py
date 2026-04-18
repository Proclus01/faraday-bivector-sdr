from __future__ import annotations
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any, Optional, Tuple, List
import numpy as np

ComplexBuffer = np.ndarray

@dataclass(frozen=True)
class PolarizationBasis:
    name: str
    matrix_to_xy: np.ndarray

@dataclass(frozen=True)
class Mode:
    center_freq_hz: float
    sample_rate_hz: float
    bandwidth_hz: float
    lo_chain: Tuple[Any, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class Pattern:
    description: str = "omni"

@dataclass(frozen=True)
class FrameRef:
    name: str = "DEVICE"
    t0_ns: int = 0

@dataclass
class ProjectionMeta:
    mode: Mode
    polarization: PolarizationBasis
    pattern: Optional[Pattern]
    frame: FrameRef
    gain_db: float = 0.0
    noise_figure_db: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BufferFrame:
    samples: ComplexBuffer
    timestamp_ns: Optional[int] = None

@dataclass
class SpectralFrame:
    freqs_hz: np.ndarray
    psd_db: np.ndarray
    timestamp_ns: Optional[int]
    meta: ProjectionMeta

@dataclass
class RangeDopplerFrame:
    range_m: np.ndarray
    doppler_hz: np.ndarray
    power_db: np.ndarray
    timestamp_ns: Optional[int]
    meta: ProjectionMeta

@dataclass
class PolMapFrame:
    metric_names: List[str]
    maps_db: Dict[str, np.ndarray]  # name -> 2D array (doppler x range)
    range_m: np.ndarray
    doppler_hz: np.ndarray
    timestamp_ns: Optional[int]
    meta: ProjectionMeta

@dataclass
class FaradayProjection:
    meta: ProjectionMeta
    stream: AsyncIterator[BufferFrame]

    def copy_with(self, meta: Optional[ProjectionMeta] = None, stream: Optional[AsyncIterator[BufferFrame]] = None) -> "FaradayProjection":
        return FaradayProjection(meta=meta or self.meta, stream=stream or self.stream)
