# -*- coding: utf-8 -*-
import numpy as np

from mrinversion.kernel.lineshape import _x_y_to_zeta_eta
from mrinversion.kernel.lineshape import zeta_eta_to_x_y


def test_x_y_to_zeta_eta():

    x = np.random.rand(16) * 3000
    y = np.random.rand(16) * 3000
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

    zeta, eta = _x_y_to_zeta_eta(x, y)
    assert np.allclose(zeta, np.asarray(zeta_))
    assert np.allclose(eta, np.asarray(eta_))


def test_x_y_from_zeta_eta():
    x, y = zeta_eta_to_x_y([10, np.sqrt(2) * 25], [0, 1])
    assert np.allclose(x, [0, 25])
    assert np.allclose(y, [10, 25])
