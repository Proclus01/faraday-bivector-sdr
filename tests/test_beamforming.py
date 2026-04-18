import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.domain.operators.beamforming import beamform_narrowband, align_time_phase

async def get_one_frame(stream):
    async for fr in stream:
        return fr
    return None

def test_beamforming_narrowband_power_gain():
    fc = 1_000_000_000.0
    sr = 2_000_000.0
    d = 0.1
    f_off = 100_000.0
    dev = SimDeviceAdapter()
    p1 = dev.open_rx({"center_freq_hz": fc, "sample_rate_hz": sr, "bandwidth_hz": sr, "block_size": 4096,
                      "tone_offsets_hz": [f_off], "amplitudes": [1.0], "initial_phases_rad": [0.0], "noise_std": 0.0, "seed": 3, "max_frames": 8})
    p2 = dev.open_rx({"center_freq_hz": fc, "sample_rate_hz": sr, "bandwidth_hz": sr, "block_size": 4096,
                      "tone_offsets_hz": [f_off], "amplitudes": [1.0], "initial_phases_rad": [np.deg2rad(60)], "noise_std": 0.0, "seed": 4, "max_frames": 8})
    p1a, p2a = align_time_phase([p1, p2])
    bf_good = beamform_narrowband([p1a, p2a], element_positions_m=[(0.0,0.0,0.0),(d,0.0,0.0)], az_deg=90.0, el_deg=0.0)
    bf_bad = beamform_narrowband([p1a, p2a], element_positions_m=[(0.0,0.0,0.0),(d,0.0,0.0)], az_deg=0.0, el_deg=0.0)
    f_good = asyncio.run(get_one_frame(bf_good.stream))
    f_bad = asyncio.run(get_one_frame(bf_bad.stream))
    Pg = float(np.mean(np.abs(f_good.samples)**2))
    Pb = float(np.mean(np.abs(f_bad.samples)**2))
    assert Pg > Pb
