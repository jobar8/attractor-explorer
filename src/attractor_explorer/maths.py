"""Functions to calculate trajectories of attractors."""

import numpy as np
import pandas as pd
from numba import jit
from numpy.typing import NDArray


@jit
def trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Given an attractor fn with up to six parameters a-e, compute n trajectory points
    (starting from x0,y0). Numba-optimized to run at machine-code speeds.
    """
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    for i in np.arange(n - 1):
        x[i + 1], y[i + 1] = fn(x[i], y[i], a, b, c, d, e, f)
    return x, y


def trajectory(fn, x0, y0, a, b=None, c=None, d=None, e=None, f=None, n=1000000) -> pd.DataFrame:
    """
    Given an attractor fn with up to six parameters a-e, compute n trajectory points
    (starting from x0,y0) and return as a Pandas dataframe with columns x,y.
    """
    xs, ys = trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n)
    return pd.DataFrame({'x': xs, 'y': ys})
