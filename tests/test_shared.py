"""Tests for the shared.py module."""

from unittest.mock import MagicMock, patch

import datashader as ds
import numpy as np
import pandas as pd

from attractor_explorer.shared import render_attractor, save_image


def test_render():
    trajectory = pd.DataFrame(np.random.default_rng().random((30, 2)), columns=list('xy'))
    image = list(render_attractor(trajectory, plot_type='points', cmap=None, size=400))
    assert isinstance(image[0], ds.tf.Image)


@patch('attractor_explorer.shared.ds.tf.set_background')
def test_save_image(mock_set_background):
    # Mocking the image object
    mock_img = MagicMock()
    mock_output_image = MagicMock()
    mock_set_background.return_value = mock_output_image
    mock_output_image.to_pil.return_value.save = MagicMock()

    output_path = 'test_output.png'
    color = 'blue'

    save_image(mock_img, output_path, color)
    mock_set_background.assert_called_once_with(mock_img, color=color)
    mock_output_image.to_pil.return_value.save.assert_called_once_with(output_path, format='png')
