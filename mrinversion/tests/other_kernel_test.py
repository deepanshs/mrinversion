import csdmpy as cp
import numpy as np

from mrinversion.kernel import T1
from mrinversion.kernel import T2

direct_dimension = cp.Dimension(type="linear", count=96, increment="20 ms")

inverse_dimension = cp.Dimension(
    type="monotonic", coordinates=["1ms", "10ms", "100ms", "1s", "2s"]
)


def test_T2_kernel():
    T2_obj = T2(direct_dimension=direct_dimension, inverse_dimension=inverse_dimension)
    K = T2_obj.kernel(supersampling=1)

    x = direct_dimension.coordinates
    x_inverse = inverse_dimension.coordinates
    amp = np.exp(np.tensordot(-x, (1 / x_inverse), 0))
    amp /= amp[:, 0].sum()
    assert np.allclose(K, amp)


def test_T1_kernel():
    T1_obj = T1(direct_dimension=direct_dimension, inverse_dimension=inverse_dimension)
    K = T1_obj.kernel(supersampling=1)

    x = direct_dimension.coordinates
    x_inverse = inverse_dimension.coordinates
    amp = 1 - np.exp(np.tensordot(-x, (1 / x_inverse), 0))
    amp /= amp[:, 0].sum()
    assert np.allclose(K, amp)
