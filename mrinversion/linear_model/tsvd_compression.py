# -*- coding: utf-8 -*-
import csdmpy as cp
import numpy as np

from mrinversion.linear_model.linear_inversion import find_optimum_singular_value
from mrinversion.linear_model.linear_inversion import reduced_subspace_kernel_and_data


class TSVDCompression:
    """SVD compression.

        Args:
            K: The kernel.
            s: The data.
            r: An integer defining the number of singular values used in
                compression.

        Attributes:
            truncation_index: The number of singular values retained,
            compressed_K: The compressed kernel.
            compressed_s: Tje compressed data.
        """

    def __init__(self, K, s, r=None):
        self.U, self.S, self.VT = np.linalg.svd(K, full_matrices=False)
        if r is None:
            r = find_optimum_singular_value(self.S)

        self.K = K
        self.s = s
        self.truncation_index = r
        self.compress(r)

    def compress(self, r):
        """Compress the kernel and data up to first r singular values using SVD

        Args:
            r: An integer defining the number of singular values used in
                compression.
        """
        if isinstance(self.s, cp.CSDM):
            signal = self.s.dependent_variables[0].components[0].T
        else:
            signal = self.s
        r = self.truncation_index
        (
            self.compressed_K,
            compressed_signal,
            _,  # projectedSignal,
            __,  # guess_solution,
        ) = reduced_subspace_kernel_and_data(
            self.U[:, :r], self.S[:r], self.VT[:r, :], signal
        )
        factor = signal.size / compressed_signal.size
        print(f"compression factor = {factor}")

        if isinstance(self.s, cp.CSDM):
            self.compressed_s = cp.as_csdm(compressed_signal.T)
            if len(self.s.dimensions) > 1:
                self.compressed_s.dimensions[1] = self.s.dimensions[1]
        else:
            self.compressed_s = compressed_signal
