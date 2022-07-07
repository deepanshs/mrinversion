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

        signal = s.y[0].components[0].T if isinstance(s, cp.CSDM) else s
        (
            self.compressed_K,
            compressed_signal,
            projectedSignal,
            __,  # guess_solution,
        ) = reduced_subspace_kernel_and_data(U[:, :r], S[:r], VT[:r, :], signal)

        factor = signal.size / compressed_signal.size
        print(f"compression factor = {factor}")

        self.filtered_s = projectedSignal
        self.compressed_s = compressed_signal

        if isinstance(s, cp.CSDM):
            self.compressed_s = cp.as_csdm(self.compressed_s.T.copy())
            self.filtered_s = cp.as_csdm(self.filtered_s.T.copy())
            if len(s.x) > 1:
                self.compressed_s.x[1] = s.x[1]
                self.filtered_s.x[1] = s.x[1]
