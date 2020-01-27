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

    # xy_grid = [
    #     cp.Dimension(type="linear", count=20, increment="0.5 kHz"),
    #     cp.Dimension(
    #         type="linear", count=10, coordinates_offset="10 kHz", increment="0.5 kHz"
    #     ),
    # ]

    # assert xy_grid[0].coordinates_offset.value == 0
    # assert xy_grid[1].coordinates_offset.value == 10
    # assert str(xy_grid[1].coordinates_offset.unit) == "kHz"

    # xy_grid[0].coordinates_offset = "5 Hz"
    # assert xy_grid[0].coordinates_offset.value == 5

    # error = "Expecting an instance of type"
    # with pytest.raises(TypeError, match=".*{0}.*".format(error)):
    #     xy_grid[0].coordinates_offset = {5, "Hz"}

    # xy_grid = 10
