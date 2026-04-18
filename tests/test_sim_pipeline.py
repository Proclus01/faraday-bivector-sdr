import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.domain.operators.filters import bandpass_fir
from faraday_bivector_sdr.domain.operators.spectral import estimate_spectrum

async def get_one_spectrum_frame(stream):
    async for frame in stream:
        return frame
    return None

def test_spectrum_peak_detection():
    dev = SimDeviceAdapter()
    proj = dev.open_rx({
        "center_freq_hz": 0.0,
        "sample_rate_hz": 1_000_000.0,
        "bandwidth_hz": 1_000_000.0,
        "block_size": 4096,
        "tone_offsets_hz": [100_000.0],
        "amplitudes": [1.0],
        "noise_std": 0.01,
        "seed": 1,
        "max_frames": 32,
    })
    proj_bp = bandpass_fir(proj, low_hz=10_000.0, high_hz=300_000.0, taps=257)
    spec_stream = estimate_spectrum(proj_bp, fft_size=4096, avg=4)
    frame = asyncio.run(get_one_spectrum_frame(spec_stream))
    assert frame is not None
    idx = int(np.argmax(frame.psd_db))
    f_peak = float(frame.freqs_hz[idx])
    bin_width = float(proj.meta.mode.sample_rate_hz)/4096.0
    assert abs(f_peak - 100_000.0) <= (1.5 * bin_width)
