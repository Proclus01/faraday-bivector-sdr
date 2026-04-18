import os
import asyncio
import numpy as np
from faraday_bivector_sdr.adapters.devices.sim import SimDeviceAdapter
from faraday_bivector_sdr.adapters.devices.file_npz import NPZPlaybackAdapter
from faraday_bivector_sdr.adapters.storage.npz import NPZRecorder

async def capture(proj, path):
    rec = NPZRecorder()
    await rec.record(proj, path=path, max_frames=4)

async def get_frames(proj, n=2):
    out = []
    i = 0
    async for frame in proj.stream:
        out.append(frame.samples.copy())
        i += 1
        if i >= n:
            break
    return out

def test_npz_record_and_playback(tmp_path):
    dev = SimDeviceAdapter()
    proj = dev.open_rx({
        "center_freq_hz": 10e6,
        "sample_rate_hz": 1e6,
        "bandwidth_hz": 1e6,
        "block_size": 1024,
        "tone_offsets_hz": [50_000.0],
        "amplitudes": [1.0],
        "noise_std": 0.0,
        "seed": 11,
        "max_frames": 4,
    })
    path = os.path.join(tmp_path, "capture.npz")
    asyncio.run(capture(proj, path))
    # Playback
    pb = NPZPlaybackAdapter()
    proj2 = pb.open_rx({"path": path, "block_size": 1024})
    frames = asyncio.run(get_frames(proj2, n=2))
    assert len(frames) == 2
    # The tone should still be present; check mean power > 0
    for f in frames:
        power = float(np.mean(np.abs(f)**2))
        assert power > 0.1
