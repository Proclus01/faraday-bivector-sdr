from __future__ import annotations

from typing import AsyncIterator, Dict, List, Optional

import numpy as np

from ..types import PolMapFrame, RangeDopplerFrame
from .rd_align import zip_align_rd_by_timestamp


def polmap_from_rd(
    streams: List[AsyncIterator[RangeDopplerFrame]],
    metrics: Optional[List[str]] = None,
    pri_s: float = 0.001,
    tolerance_pri: int = 1,
) -> AsyncIterator[PolMapFrame]:
    """
    Combine four RD streams [HH, HV, VH, VV] into polarimetric maps (non-coherent proxies).

    This is intended to be robust for both:
    - fully parallel polarimetric capture (all four streams advance together), and
    - time-division multiplexing (TDM) where HH/HV and VH/VV arrive in alternating bursts.

    tolerance_pri controls timestamp alignment tolerance in units of PRI.
    Larger values are useful for TDM where the streams can be offset by an entire burst.
    """
    if metrics is None:
        metrics = ["span", "copol_ratio", "xpol_ratio"]

    async def gen() -> AsyncIterator[PolMapFrame]:
        async for HH, HV, VH, VV in zip_align_rd_by_timestamp(
            streams,
            pri_s=float(pri_s),
            tolerance_pri=int(tolerance_pri),
        ):
            # Use power_db -> linear power
            P_HH = 10.0 ** (HH.power_db / 10.0)
            P_HV = 10.0 ** (HV.power_db / 10.0)
            P_VH = 10.0 ** (VH.power_db / 10.0)
            P_VV = 10.0 ** (VV.power_db / 10.0)

            out: Dict[str, np.ndarray] = {}

            if "span" in metrics:
                out["span"] = 10.0 * np.log10(P_HH + P_HV + P_VH + P_VV + 1e-30)

            if "copol_ratio" in metrics:
                num = P_HH + P_VV
                den = (P_HV + P_VH) + 1e-30
                out["copol_ratio"] = 10.0 * np.log10(num / den + 1e-30)

            if "xpol_ratio" in metrics:
                num = P_HV + P_VH
                den = (P_HH + P_VV) + 1e-30
                out["xpol_ratio"] = 10.0 * np.log10(num / den + 1e-30)

            meta = HH.meta  # reuse
            yield PolMapFrame(
                metric_names=list(out.keys()),
                maps_db=out,
                range_m=HH.range_m,
                doppler_hz=HH.doppler_hz,
                timestamp_ns=HH.timestamp_ns,
                meta=meta,
            )

    return gen()
