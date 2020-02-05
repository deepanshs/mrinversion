import csdmpy as cp
import numpy as np

from mrinversion.util import supersampled_coordinates

# import pytest


def test_supersampling():
    xy_grid = [
        cp.Dimension(
            type="linear", count=20, coordinates_offset="0 Hz", increment="0.5 kHz"
        ),
        cp.Dimension(
            type="linear", count=20, coordinates_offset="0 Hz", increment="0.5 kHz"
        ),
    ]

    y = np.arange(20) * 0.5
    for i in range(10):
        oversample = i + 1
        y_oversampled = supersampled_coordinates(xy_grid[1], supersampling=oversample)
        y_reduced = y_oversampled.reshape(20, oversample).sum(axis=-1) / oversample
        assert np.allclose(y_reduced.value, y)
