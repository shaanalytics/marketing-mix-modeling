"""Feature transformations commonly used in Marketing Mix Modeling."""

from __future__ import annotations
import numpy as np
import pandas as pd

def geometric_adstock(values: pd.Series | np.ndarray, decay: float, *, normalize: bool = False) -> np.ndarray:
    """Apply geometric carryover to a one-dimensional media series in time order."""
    if not 0 <= decay < 1:
        raise ValueError("decay must be in the interval [0, 1).")
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not np.isfinite(x).all():
        raise ValueError("values must contain only finite numbers.")
    if (x < 0).any():
        raise ValueError("values must be non-negative.")
    result = np.empty_like(x, dtype=float)
    carryover = 0.0
    for i, value in enumerate(x):
        carryover = value + decay * carryover
        result[i] = carryover
    if normalize and len(result):
        result *= 1.0 - decay
    return result

def hill_saturation(values: pd.Series | np.ndarray, half_saturation: float, slope: float = 1.0) -> np.ndarray:
    """Apply a Hill saturation curve with outputs bounded between zero and one."""
    if half_saturation <= 0:
        raise ValueError("half_saturation must be greater than zero.")
    if slope <= 0:
        raise ValueError("slope must be greater than zero.")
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not np.isfinite(x).all():
        raise ValueError("values must contain only finite numbers.")
    if (x < 0).any():
        raise ValueError("values must be non-negative.")
    x_power = np.power(x, slope)
    half_power = half_saturation ** slope
    return x_power / (x_power + half_power)
