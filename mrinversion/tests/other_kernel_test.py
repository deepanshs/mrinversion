# -*- coding: utf-8 -*-
import csdmpy as cp
import numpy as np

from mrinversion.kernel import T1
from mrinversion.kernel import T2

kernel_dimension = cp.Dimension(type="linear", count=96, increment="20 ms")

inverse_kernel_dimension = cp.Dimension(
    type="monotonic", coordinates=["1ms", "10ms", "100ms", "1s", "2s"]
)


def test_T2_kernel():
    T2_obj = T2(
        kernel_dimension=kernel_dimension,
        inverse_kernel_dimension=inverse_kernel_dimension,
    )
    K = T2_obj.kernel(supersampling=1)

    x = kernel_dimension.coordinates
    x_inverse = inverse_kernel_dimension.coordinates
    amp = np.exp(np.tensordot(-x, (1 / x_inverse), 0))
    amp /= amp[:, 0].sum()
    assert np.allclose(K, amp)


def test_T1_kernel():
    T1_obj = T1(
        kernel_dimension=kernel_dimension,
        inverse_kernel_dimension=inverse_kernel_dimension,
    )
    K = T1_obj.kernel(supersampling=1)

    x = kernel_dimension.coordinates
    x_inverse = inverse_kernel_dimension.coordinates
    amp = 1 - np.exp(np.tensordot(-x, (1 / x_inverse), 0))
    amp /= amp[:, 0].sum()
    assert np.allclose(K, amp)
