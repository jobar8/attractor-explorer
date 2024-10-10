"""Tests for the shared.py module."""

import datashader as ds
import numpy as np
import pandas as pd

from attractor_explorer.shared import render_attractor


def test_render():
    trajectory = pd.DataFrame(np.random.default_rng().random((30, 2)), columns=list('xy'))
    image = list(render_attractor(trajectory, plot_type='points', cmap=None, size=400))
    assert isinstance(image[0], ds.tf.Image)
