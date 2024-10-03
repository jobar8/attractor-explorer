"""Test functions for the maths.py module."""

import pandas as pd
import pytest
from numba import jit
from numpy import sin

from attractor_explorer import maths


@jit(nopython=True)
def fn(x, y, a, b, c, d, *o):  # noqa: ARG001
    return sin(b * y) + c * sin(b * x), sin(a * x) + d * sin(a * y)


def test_trajectory_coords():
    """Test for the trajectory_coords() function."""
    x0 = 0.5
    y0 = 0.5
    a = 1
    b = 2
    c = 1
    d = -0.5
    e = 2
    f = 2
    n = 5
    x, y = maths.trajectory_coords(fn, x0, y0, a, b, c, d, e, f, n)
    assert x == pytest.approx([0.5, 1.68294197, 0.2388541, 1.44372879, -0.03872752])
    assert y == pytest.approx([0.5, 0.23971277, 0.87500646, -0.14718441, 1.06526456])


def test_trajectory():
    """Test for the trajectory() function."""
    x0 = 0.5
    y0 = 0.5
    a = 1
    b = 2
    c = 1
    d = -0.5
    e = 2
    f = 2
    n = 5
    points = maths.trajectory(fn, x0, y0, a, b, c, d, e, f, n)
    assert isinstance(points, pd.DataFrame)
