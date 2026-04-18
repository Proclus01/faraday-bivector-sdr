import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.domain.operators.frequency import frequency_shift
from faraday_bivector_sdr.domain.operators.spectral import estimate_spectrum

async def get_one_spectrum_frame(stream):
    async for frame in stream:
        return frame
    return None

def test_frequency_shift_moves_tone():
    dev = SimDeviceAdapter()
    proj = dev.open_rx({
        "center_freq_hz": 0.0,
        "sample_rate_hz": 2_000_000.0,
        "bandwidth_hz": 2_000_000.0,
        "block_size": 4096,
        "tone_offsets_hz": [200_000.0],
        "amplitudes": [1.0],
        "noise_std": 0.0,
        "seed": 2,
        "max_frames": 16,
    })
    shifted = frequency_shift(proj, delta_hz=-50_000.0)
    spec_stream = estimate_spectrum(shifted, fft_size=2048, avg=2)
    frame = asyncio.run(get_one_spectrum_frame(spec_stream))
    assert frame is not None
    idx = int(np.argmax(frame.psd_db))
    f_peak = float(frame.freqs_hz[idx])
    bin_width = float(proj.meta.mode.sample_rate_hz)/2048.0
    assert abs(f_peak - 150_000.0) <= (1.5 * bin_width)
