import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.domain.operators.beamforming import align_time_phase

async def get_two_frames(stream):
    res = []
    async for fr in stream:
        res.append(fr)
        if len(res) >= 2:
            break
    return res

def test_phase_alignment_two_channels():
    sr = 1_000_000.0
    dev = SimDeviceAdapter()
    # Two channels: same tone, second has +60 deg phase
    p1 = dev.open_rx({
        "center_freq_hz": 0.0, "sample_rate_hz": sr, "bandwidth_hz": sr,
        "block_size": 2048, "tone_offsets_hz": [100_000.0], "amplitudes": [1.0],
        "initial_phases_rad": [0.0], "noise_std": 0.0, "seed": 1, "max_frames": 4
    })
    p2 = dev.open_rx({
        "center_freq_hz": 0.0, "sample_rate_hz": sr, "bandwidth_hz": sr,
        "block_size": 2048, "tone_offsets_hz": [100_000.0], "amplitudes": [1.0],
        "initial_phases_rad": [np.deg2rad(60.0)], "noise_std": 0.0, "seed": 2, "max_frames": 4
    })
    p1_aligned, p2_aligned = align_time_phase([p1, p2])
    # Grab one frame from each and check mean phase difference near 0
    async def fetch():
        it1 = p1_aligned.stream.__aiter__(); it2 = p2_aligned.stream.__aiter__()
        f1 = await it1.__anext__(); f2 = await it2.__anext__()
        return f1, f2
    f1, f2 = asyncio.run(fetch())
    # Estimate phase difference
    L = min(f1.samples.shape[0], f2.samples.shape[0])
    cross = f2.samples[:L] * np.conj(f1.samples[:L])
    phi = float(np.angle(np.mean(cross + 1e-20)))
    assert abs(phi) < 0.1  # within ~6 degrees
