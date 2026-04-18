import asyncio
import numpy as np
from faraday_bivector_sdr.domain.types import RangeDopplerFrame, ProjectionMeta, Mode, PolarizationBasis, Pattern, FrameRef
from faraday_bivector_sdr.domain.operators.polmap import polmap_from_rd

async def gen_rd(power_db, rng, dop, ts):
    yield RangeDopplerFrame(range_m=rng, doppler_hz=dop, power_db=power_db, timestamp_ns=ts, meta=ProjectionMeta(
        mode=Mode(center_freq_hz=0.0, sample_rate_hz=1e6, bandwidth_hz=1e6, lo_chain=tuple()),
        polarization=PolarizationBasis("UNK", np.eye(2, dtype=np.complex64)),
        pattern=Pattern("omni"), frame=FrameRef("DEVICE", 0), gain_db=0.0, noise_figure_db=None, tags={}
    ))

def test_polmap_basic_metrics():
    rng = np.linspace(0, 1000, 64)
    dop = np.linspace(-100, 100, 32)
    P_HH = np.zeros((32,64))-40.0
    P_HV = np.zeros((32,64))-60.0
    P_VH = np.zeros((32,64))-60.0
    P_VV = np.zeros((32,64))-40.0
    streams = [
        gen_rd(P_HH, rng, dop, 0),
        gen_rd(P_HV, rng, dop, 0),
        gen_rd(P_VH, rng, dop, 0),
        gen_rd(P_VV, rng, dop, 0),
    ]
    async def run():
        gen = polmap_from_rd(streams, metrics=["span","copol_ratio","xpol_ratio"], pri_s=0.001)
        fr = await gen.__anext__()
        return fr
    pm = asyncio.run(run())
    assert "span" in pm.metric_names and "copol_ratio" in pm.metric_names and "xpol_ratio" in pm.metric_names
    # copol_ratio should be high (co-pol stronger)
    assert float(np.mean(pm.maps_db["copol_ratio"])) > 0.0
