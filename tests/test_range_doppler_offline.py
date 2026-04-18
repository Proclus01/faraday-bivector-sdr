import numpy as np
from faraday_bivector_sdr.domain.operators.radar import range_doppler, C0, compute_range_doppler_from_stack

def test_offline_rd_peak_location():
    n_slow = 32; n_fast = 256
    r0 = 30; d0 = 5
    sr = 1e6
    pri_s = 1e-3
    rng = np.random.default_rng(0)
    stack = (rng.normal(size=(n_slow, n_fast)) + 1j*rng.normal(size=(n_slow, n_fast))).astype(np.complex64) * 0.0
    for m in range(n_slow):
        for n in range(n_fast):
            stack[m, n] += np.exp(1j * (2*np.pi * r0 * n / n_fast + 2*np.pi * d0 * m / n_slow))
    rng_axis, doppler_axis, P = compute_range_doppler_from_stack(stack, sr, pri_s)
    m_idx = int(np.argmax(P)) // P.shape[1]
    n_idx = int(np.argmax(P)) % P.shape[1]
    assert abs(m_idx - d0) <= 1
    assert abs(n_idx - r0) <= 1
