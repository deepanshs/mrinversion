# -*- coding: utf-8 -*-
import csdmpy as cp
import numpy as np

from mrinversion.util import supersampled_coordinates

# import pytest


def test_supersampling():
    dim = cp.LinearDimension(count=20, coordinates_offset="0 Hz", increment="0.5 kHz")

    y = np.arange(20) * 0.5
    for i in range(10):
        oversample = i + 1
        y_oversampled = supersampled_coordinates(dim, supersampling=oversample)
        y_reduced = y_oversampled.reshape(-1, oversample).mean(axis=-1)
        assert np.allclose(y_reduced.value, y)
