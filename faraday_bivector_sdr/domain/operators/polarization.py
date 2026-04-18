from __future__ import annotations
import numpy as np
from typing import Tuple

SQ2_INV = 1.0 / np.sqrt(2.0)
M_XY_TO_RL = np.array([[1.0, -1j],[1.0, 1j]], dtype=np.complex64) * SQ2_INV
M_RL_TO_XY = np.linalg.inv(M_XY_TO_RL).astype(np.complex64)

def xy_to_rl(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.vstack([x.astype(np.complex64, copy=False), y.astype(np.complex64, copy=False)])
    rl = M_XY_TO_RL @ v
    return rl[0], rl[1]

def rl_to_xy(r: np.ndarray, l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.vstack([r.astype(np.complex64, copy=False), l.astype(np.complex64, copy=False)])
    xy = M_RL_TO_XY @ v
    return xy[0], xy[1]

def estimate_jones_from_cal(x_meas: complex, y_meas: complex, x_true: complex = 1.0+0j, y_true: complex = 0.0+0j) -> np.ndarray:
    J = np.zeros((2, 2), dtype=np.complex64)
    J[:, 0] = np.array([x_meas, y_meas], dtype=np.complex64)
    J[:, 1] = np.array([0.0+0j, 1.0+0j], dtype=np.complex64)
    return J

def apply_jones(x: np.ndarray, y: np.ndarray, J: np.ndarray, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.linalg.inv(J) if inverse else J
    v = np.vstack([x.astype(np.complex64, copy=False), y.astype(np.complex64, copy=False)])
    out = mat @ v
    return out[0], out[1]
