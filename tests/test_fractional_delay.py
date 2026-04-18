import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.domain.operators.filters import fractional_delay

async def get_one_frame(stream):
    async for fr in stream:
        return fr
    return None

def test_fractional_delay_phase_slope():
    sr = 2_000_000.0
    f0 = 200_000.0
    dev = SimDeviceAdapter()
    p = dev.open_rx({"center_freq_hz": 0.0, "sample_rate_hz": sr, "bandwidth_hz": sr,
                     "block_size": 4096, "tone_offsets_hz": [f0], "amplitudes": [1.0],
                     "noise_std": 0.0, "seed": 3, "max_frames": 4})
    mu = 0.5  # half-sample delay
    d = fractional_delay(p, delay_samples=mu, taps=129)
    fr = asyncio.run(get_one_frame(d.stream))
    assert fr is not None
    # The fractional delay introduces a linear phase vs frequency; for single tone, approximate phase shift 2*pi*f0*mu/sr
    expected = 2*np.pi*f0*mu/sr
    # Estimate phase difference between delayed and original first block (use original from device again)
    fr0 = asyncio.run(get_one_frame(p.stream))
    L = min(fr.samples.shape[0], fr0.samples.shape[0])
    cross = fr.samples[:L] * np.conj(fr0.samples[:L])
    phi = np.angle(np.mean(cross + 1e-20))
    # Compare modulo 2pi
    diff = np.angle(np.exp(1j*(phi - expected)))
    assert abs(diff) < 0.2  # ~11 degrees tolerance
