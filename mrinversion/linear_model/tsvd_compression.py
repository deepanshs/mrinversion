# -*- coding: utf-8 -*-
import csdmpy as cp

from mrinversion.linear_model.linear_inversion import reduced_subspace_kernel_and_data
from mrinversion.linear_model.linear_inversion import TSVD


class TSVDCompression:
    """SVD compression.

    Args:
        K: The kernel.
        s: The data.
        r: The number of singular values used in data compression.

    Attributes
    ----------

    truncation_index: int
        The number of singular values retained.

    compressed_K: ndarray
        The compressed kernel.

    compressed_s: ndarray of CSDM object
        The compressed data.
    """

    def __init__(self, K, s, r=None):

        U, S, VT, r_ = TSVD(K)
        if r is None:
            r = r_
        self.truncation_index = r

        if isinstance(s, cp.CSDM):
            signal = s.dependent_variables[0].components[0].T
        else:
            signal = s
        (
            self.compressed_K,
            compressed_signal,
            _,  # projectedSignal,
            __,  # guess_solution,
        ) = reduced_subspace_kernel_and_data(U[:, :r], S[:r], VT[:r, :], signal)
        factor = signal.size / compressed_signal.size
        print(f"compression factor = {factor}")

        if isinstance(s, cp.CSDM):
            self.compressed_s = cp.as_csdm(compressed_signal.T.copy())
            if len(s.dimensions) > 1:
                self.compressed_s.dimensions[1] = s.dimensions[1]
        else:
            self.compressed_s = compressed_signal
