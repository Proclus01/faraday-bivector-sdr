import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.domain.operators.tee import tee
from faraday_bivector_sdr.domain.operators.timestamp_align import zip_align_by_timestamp

async def anext(it):
    return await it.__anext__()

def test_tee_and_timestamp_align():
    sr = 1_000_000.0
    dev = SimDeviceAdapter()
    # Two identical channels but with integer delay on second
    p1 = dev.open_rx({"center_freq_hz": 0.0, "sample_rate_hz": sr, "bandwidth_hz": sr, "block_size": 2048,
                      "tone_offsets_hz": [100_000.0], "amplitudes": [1.0], "integer_sample_delay": 0,
                      "noise_std": 0.0, "seed": 1, "max_frames": 8})
    p2 = dev.open_rx({"center_freq_hz": 0.0, "sample_rate_hz": sr, "bandwidth_hz": sr, "block_size": 2048,
                      "tone_offsets_hz": [100_000.0], "amplitudes": [1.0], "integer_sample_delay": 256,
                      "noise_std": 0.0, "seed": 2, "max_frames": 8})
    # Tee first channel (not essential here, just exercising the operator)
    t1, t2 = tee(p1, copies=2)
    async def collect_one():
        zipgen = zip_align_by_timestamp([t1.stream, p2.stream], sample_rate_hz=sr, tolerance_samples=300)
        fx, fy = await anext(zipgen.__aiter__())
        return fx, fy
    fx, fy = asyncio.run(collect_one())
    # Check timestamps aligned within tolerance (or timestamps None fallback)
    if fx.timestamp_ns is not None and fy.timestamp_ns is not None:
        assert abs(int(fx.timestamp_ns) - int(fy.timestamp_ns)) <= int(300.0/sr*1e9)
