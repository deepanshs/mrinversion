# -*- coding: utf-8 -*-
import csdmpy as cp
import numpy as np

from mrinversion.kernel.utils import _supersampled_coordinates
from mrinversion.kernel.utils import _x_y_to_zeta_eta
from mrinversion.kernel.utils import _x_y_to_zeta_eta_distribution
from mrinversion.kernel.utils import x_y_to_zeta_eta
from mrinversion.kernel.utils import zeta_eta_to_x_y


def test_x_y_to_zeta_eta():

    x = np.random.rand(16) * 3000
    y = np.random.rand(16) * 3000
    x[-1] = y[-1] = 56.0
    x[-2] = y[-2] = 0.0
    factor_ = 4 / np.pi
    zeta_ = []
    eta_ = []
    for x_, y_ in zip(x, y):
        z = np.sqrt(x_ ** 2 + y_ ** 2)
        if x_ < y_:
            eta_.append(factor_ * np.arctan(x_ / y_))
            zeta_.append(z)
        elif x_ > y_:
            eta_.append(factor_ * np.arctan(y_ / x_))
            zeta_.append(-z)
        else:
            zeta_.append(z)
            eta_.append(1.0)

        z_temp, e_temp = x_y_to_zeta_eta(x_, y_)
        assert zeta_[-1] == z_temp
        assert eta_[-1] == e_temp

    zeta, eta = _x_y_to_zeta_eta(x, y)
    assert np.allclose(zeta, np.asarray(zeta_))
    assert np.allclose(eta, np.asarray(eta_))


def test_zeta_eta_to_x_y():
    sq25 = np.sqrt(2) * 25
    x, y = zeta_eta_to_x_y([10, sq25, -10, -sq25], [0, 1, 0, 1])
    assert np.allclose(x, [0, 25, 10, 25])
    assert np.allclose(y, [10, 25, 0, 25])


def test_x_y_to_zeta_eta_distribution():
    inverse_dimension = [
        cp.Dimension(type="linear", count=4, increment="3 kHz"),
        cp.Dimension(type="linear", count=4, increment="3 kHz"),
    ]

    x = np.arange(4) * 3000
    y = np.arange(4) * 3000
    factor_ = 4 / np.pi
    zeta_ = []
    eta_ = []
    for y_ in y:
        for x_ in x:
            z = np.sqrt(x_ ** 2 + y_ ** 2)
            if x_ < y_:
                eta_.append(factor_ * np.arctan(x_ / y_))
                zeta_.append(z)
            elif x_ > y_:
                eta_.append(factor_ * np.arctan(y_ / x_))
                zeta_.append(-z)
            else:
                zeta_.append(z)
                eta_.append(1.0)

    zeta, eta = _x_y_to_zeta_eta_distribution(inverse_dimension, supersampling=1)

    assert np.allclose(zeta, np.asarray(zeta_))
    assert np.allclose(eta, np.asarray(eta_))


def test_supersampling():
    dim = cp.LinearDimension(count=20, coordinates_offset="0 Hz", increment="0.5 kHz")

    y = np.arange(20) * 0.5
    for i in range(10):
        oversample = i + 1
        y_oversampled = _supersampled_coordinates(dim, supersampling=oversample)
        y_reduced = y_oversampled.reshape(-1, oversample).mean(axis=-1)
        assert np.allclose(y_reduced.value, y)
