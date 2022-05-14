# -*- coding: utf-8 -*-
import csdmpy as cp
import numpy as np

from mrinversion.kernel import T1
from mrinversion.kernel import T2

kernel_dimension = cp.Dimension(type="linear", count=5, increment="20 ms")

count = 10
coords = 10 ** ((np.arange(count) / (count - 1)) * (2 - (-3)) - 3)
inverse_kernel_dimension = cp.as_dimension(array=coords, unit="s")
print(inverse_kernel_dimension)


def test_T2_kernel():
    T2_obj = T2(
        kernel_dimension=kernel_dimension,
        inverse_dimension=dict(
            count=count,
            minimum="1e-3 s",
            maximum="1e2 s",
            scale="log",
            label="log (T2 / s)",
        ),
    )
    K = T2_obj.kernel(supersampling=1)

    x = kernel_dimension.coordinates
    x_inverse = inverse_kernel_dimension.coordinates
    amp = np.exp(np.tensordot(-x, (1 / x_inverse), 0))
    # amp /= amp.max()
    assert np.allclose(K, amp)


def test_T1_kernel():
    T1_obj = T1(
        kernel_dimension=kernel_dimension,
        inverse_dimension=dict(
            count=count,
            minimum="1e-3 s",
            maximum="1e2 s",
            scale="log",
            label="log (T2 / s)",
        ),
    )
    K = T1_obj.kernel(supersampling=1)

    x = kernel_dimension.coordinates
    x_inverse = inverse_kernel_dimension.coordinates
    amp = 1 - np.exp(np.tensordot(-x, (1 / x_inverse), 0))
    # amp /= amp.max()
    assert np.allclose(K, amp)
