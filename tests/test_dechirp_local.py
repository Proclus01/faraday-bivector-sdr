import asyncio
import numpy as np
from faraday_bivector_sdr.domain.types import FaradayProjection, BufferFrame, ProjectionMeta, Mode, PolarizationBasis, Pattern, FrameRef
from faraday_bivector_sdr.domain.operators.dechirp import dechirp_local

async def gen_chirp(sr, sweep_bw, sweep_time, blocks=8, block_size=2048):
    Ns = int(round(sr * sweep_time))
    k = sweep_bw / sweep_time
    n = np.arange(Ns)
    phase = 2*np.pi*(0.5*(k/sr)*n**2)
    chirp = np.exp(1j*phase).astype(np.complex64)
    # Repeat chirp across blocks
    idx = 0
    ts = 0
    for b in range(blocks):
        buf = np.empty(block_size, dtype=np.complex64)
        for i in range(block_size):
            buf[i] = chirp[idx]
            idx = (idx+1) % Ns
        yield BufferFrame(samples=buf, timestamp_ns=ts)
        ts += int(1e9 * (block_size / sr))

def test_dechirp_local_reduces_chirp_to_tone():
    sr = 1_000_000.0
    sweep_bw = 200_000.0
    sweep_time = 0.01
    meta = ProjectionMeta(
        mode=Mode(center_freq_hz=0.0, sample_rate_hz=sr, bandwidth_hz=sr, lo_chain=tuple()),
        polarization=PolarizationBasis("UNK", np.eye(2, dtype=np.complex64)),
        pattern=Pattern("omni"),
        frame=FrameRef("DEVICE", 0),
        gain_db=0.0,
        noise_figure_db=None,
        tags={}
    )
    proj = FaradayProjection(meta=meta, stream=gen_chirp(sr, sweep_bw, sweep_time))
    de = dechirp_local(proj, sweep_bw_hz=sweep_bw, sweep_time_s=sweep_time)
    # After dechirp, signal should be near-DC. Compute spectrum and check a strong low-bin.
    async def get_block():
        async for fr in de.stream:
            return fr
    fr = asyncio.run(get_block())
    X = np.fft.fft(fr.samples)
    mag = np.abs(X)
    peak_bin = int(np.argmax(mag))
    assert peak_bin < 10  # near DC
