from __future__ import annotations
import numpy as np

try:
    import numba  # type: ignore
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

def convolve_real_ir_complex_sig(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Convolve complex x with real h, return complex result. May be jitted if numba available.
    """
    if NUMBA_AVAILABLE:
        return _nb_convolve_real_ir_complex_sig(h, x)
    # fallback
    return np.convolve(x, h, mode="valid").astype(np.complex64)

if NUMBA_AVAILABLE:
    @numba.njit(cache=True, fastmath=True)
    def _nb_convolve_real_ir_complex_sig(h: np.ndarray, x: np.ndarray) -> np.ndarray:
        L = h.shape[0]
        N = x.shape[0] - L + 1
        out = np.empty(N, dtype=np.complex64)
        for n in range(N):
            acc_r = 0.0
            acc_i = 0.0
            for k in range(L):
                xr = x[n + L - 1 - k].real
                xi = x[n + L - 1 - k].imag
                hk = h[k]
                acc_r += xr * hk
                acc_i += xi * hk
            out[n] = np.complex64(acc_r + 1j * acc_i)
        return out
