import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.domain.operators.spectral import estimate_spectrum
from faraday_bivector_sdr.runtime.stream import tee_async
from faraday_bivector_sdr.runtime.taps import feed_spectrum_tap, SpectrumTap

async def run_once():
    dev = SimDeviceAdapter()
    proj = dev.open_rx({
        "center_freq_hz": 0.0, "sample_rate_hz": 1_000_000.0, "bandwidth_hz": 1_000_000.0,
        "block_size": 4096, "tone_offsets_hz": [100_000.0], "amplitudes": [1.0],
        "noise_std": 0.0, "seed": 123, "max_frames": 8
    })
    spec = estimate_spectrum(proj, fft_size=1024, avg=2)
    a, b = await tee_async(spec, copies=2)
    # Consume b into tap
    task = asyncio.create_task(feed_spectrum_tap(b, "t1"))
    # Pull one frame from a
    async for fr in a:
        # give time to tap
        await asyncio.sleep(0.05)
        latest = await SpectrumTap.get_latest("t1")
        assert latest is not None
        return
    await task

def test_tap_smoke():
    asyncio.run(run_once())
