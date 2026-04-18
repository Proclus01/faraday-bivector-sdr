import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.domain.operators.dynamic import dyn_frequency_shift, dyn_gain
from faraday_bivector_sdr.domain.operators.spectral import estimate_spectrum
from faraday_bivector_sdr.runtime.params import GLOBAL_PARAMS

async def get_one_spectrum_frame(stream):
    async for frame in stream:
        return frame
    return None

def test_dyn_frequency_shift_moves_peak():
    dev = SimDeviceAdapter()
    proj = dev.open_rx({
        "center_freq_hz": 0.0,
        "sample_rate_hz": 1_000_000.0,
        "bandwidth_hz": 1_000_000.0,
        "block_size": 4096,
        "tone_offsets_hz": [100_000.0],
        "amplitudes": [1.0],
        "noise_std": 0.0,
        "seed": 10,
        "max_frames": 16,
    })
    GLOBAL_PARAMS.set("off", -50_000.0)
    shifted = dyn_frequency_shift(proj, param="off")
    spec = estimate_spectrum(shifted, fft_size=2048, avg=2)
    fr = asyncio.run(get_one_spectrum_frame(spec))
    idx = int(np.argmax(fr.psd_db))
    f_peak = float(fr.freqs_hz[idx])
    bw = float(proj.meta.mode.sample_rate_hz)/2048.0
    assert abs(f_peak - 50_000.0) <= (1.5*bw)

def test_dyn_gain_scales_power():
    dev = SimDeviceAdapter()
    proj = dev.open_rx({
        "center_freq_hz": 0.0,
        "sample_rate_hz": 1_000_000.0,
        "bandwidth_hz": 1_000_000.0,
        "block_size": 2048,
        "tone_offsets_hz": [100_000.0],
        "amplitudes": [1.0],
        "noise_std": 0.0,
        "seed": 11,
        "max_frames": 4,
    })
    GLOBAL_PARAMS.set("gdb", 20.0)  # +20 dB ~ x10 amplitude
    g = dyn_gain(proj, param="gdb", unit="db")
    async def get_blocks():
        async for fr in g.stream:
            return fr
    fr = asyncio.run(get_blocks())
    pwr = float(np.mean(np.abs(fr.samples)**2))
    assert pwr > 5.0  # expect increased power
