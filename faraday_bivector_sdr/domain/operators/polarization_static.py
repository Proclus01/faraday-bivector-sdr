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
