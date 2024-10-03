"""Functions to calculate trajectories of attractors."""

from multiprocessing import Pool

import numpy as np
import pandas as pd
from numba import jit
from numpy.typing import NDArray

RNG = np.random.default_rng(12)


@jit(nopython=True)
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


# @jit(nopython=True)
def calculate_coords(fn, origin, xlim, ylim, n_points):
    """Calculate trajectory of an attractor and limit the output to given 2D boundaries."""
    x_coords = []
    y_coords = []

    xn, yn = origin
    for _ in np.arange(n_points):
        x_next, y_next = fn(xn, yn)
        if (x_next >= xlim[0]) and (x_next <= xlim[1]) and (y_next >= ylim[0]) and (y_next <= ylim[1]):
            x_coords.append(x_next)
            y_coords.append(y_next)
        xn, yn = x_next, y_next
    return x_coords, y_coords


# # @jit(nopython=True)
# def limited_trajectory(
#     func,
#     origin: tuple[float, float] = (0, 0),
#     xlim: tuple[float, float] = (-2, 2),
#     ylim: tuple[float, float] = (-2, 2),
#     n_points: int = 1000000,
# ) -> pd.DataFrame:
#     """Calculate trajectory of an attractor and limit the output to given 2D boundaries."""
#     x, y = calculate_coords(func, origin, xlim, ylim, n_points)
#     return pd.DataFrame({'x': x, 'y': y})


def unpack_arguments(x0, y0, a, b=None, c=None, d=None, e=None, f=None):
    """Return full list of arguments required by trajectory function."""
    return x0, y0, a, b, c, d, e, f


# @jit(nopython=True)
def limited_trajectory(
    func,
    args,
    origin: tuple[float, float] = (0, 0),
    xlim: tuple[float, float] = (-2, 2),
    ylim: tuple[float, float] = (-2, 2),
    n_points: int = 1000000,
) -> pd.DataFrame:
    """Calculate trajectory of an attractor and limit the output to given 2D boundaries."""
    # args[0] = origin[0]
    # args[1] = origin[1]
    # x, y = trajectory_coords(func, *args, n=n_points)
    x0, y0, a, b, c, d, e, f = unpack_arguments(*args)
    x, y = trajectory_coords(func, origin[0], origin[1], a, b, c, d, e, f, n_points)
    cropped_x = x[(x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])]
    cropped_y = y[(x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])]
    return pd.DataFrame({'x': cropped_x, 'y': cropped_y})


# def compute_multiple(
#     func,
#     xlim: tuple[float, float],
#     ylim: tuple[float, float],
#     n_points: int,
#     n_origins: int = 24,
#     nprocs: int = 8,
# ) -> list[pd.DataFrame]:
#     """Create image of the attractor's trajectory limited to a given region."""
#     xmin, xmax = xlim
#     ymin, ymax = ylim
#     origins = RNG.uniform((xmin, ymin), (xmax, ymax), size=(n_origins, 2))
#     args = [(func, origin, xlim, ylim, n_points) for origin in origins]
#     # all_dfs = [limited_trajectory(*arg) for arg in args]
#     with Pool(nprocs) as p:
#         all_dfs = p.starmap(limited_trajectory, args)
#     return all_dfs


# def compute_multiple(
#     func,
#     xlim: tuple[float, float],
#     ylim: tuple[float, float],
#     n_points: int,
#     n_origins: int = 24,
#     nprocs: int = 8,
# ) -> list[pd.DataFrame]:
#     """Create image of the attractor's trajectory limited to a given region."""
#     xmin, xmax = xlim
#     ymin, ymax = ylim
#     origins = RNG.uniform((xmin, ymin), (xmax, ymax), size=(n_origins, 2))
#     args = [(func, origin, xlim, ylim, n_points) for origin in origins]
#     # all_dfs = [limited_trajectory(*arg) for arg in args]
#     with Pool(nprocs) as p:
#         all_dfs = p.starmap(limited_trajectory, args)
#     return all_dfs


def compute_multiple(
    func,
    arguments,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    n_points: int,
    n_origins: int = 24,
    nprocs: int = 8,
) -> list[pd.DataFrame]:
    """Create image of the attractor's trajectory limited to a given region."""
    xmin, xmax = xlim
    ymin, ymax = ylim
    origins = RNG.uniform((xmin, ymin), (xmax, ymax), size=(n_origins, 2))
    all_args = [(func, arguments, origin, xlim, ylim, n_points) for origin in origins]
    all_dfs = [limited_trajectory(*args) for args in all_args]
    # with Pool(nprocs) as p:
    #     all_dfs = p.starmap(limited_trajectory, args)
    return all_dfs
