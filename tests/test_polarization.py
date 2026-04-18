import numpy as np
from faraday_bivector_sdr.domain.operators.polarization import xy_to_rl, rl_to_xy, M_XY_TO_RL, M_RL_TO_XY

def test_xy_rl_roundtrip():
    # Random complex XY signals
    rng = np.random.default_rng(0)
    x = (rng.normal(size=1024) + 1j*rng.normal(size=1024)).astype(np.complex64)
    y = (rng.normal(size=1024) + 1j*rng.normal(size=1024)).astype(np.complex64)

    r, l = xy_to_rl(x, y)
    x2, y2 = rl_to_xy(r, l)
    # Round-trip within tolerance
    assert np.allclose(x, x2, atol=1e-5)
    assert np.allclose(y, y2, atol=1e-5)

def test_pure_rhcp_from_xy():
    # RHCP = (X - jY)/sqrt(2)
    n = 512
    t = np.arange(n)
    # Create a unit-amplitude RHCP tone; in XY, set Y = jX so R = sqrt(2)X, L = 0
    x = np.ones(n, dtype=np.complex64)
    y = (1j * x).astype(np.complex64)
    r, l = xy_to_rl(x, y)
    # R has power, L near zero
    assert np.all(np.abs(l) < 1e-5)
    assert np.allclose(r, np.sqrt(2.0) * x, atol=1e-5)
