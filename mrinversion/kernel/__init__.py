# from copy import deepcopy
import numpy as np

from mrinversion.kernel.lineshape import DAS
from mrinversion.kernel.lineshape import MAF
from mrinversion.kernel.lineshape import NuclearShieldingTensor
from mrinversion.kernel.lineshape import SpinningSidebands
from mrinversion.linear_model.linear_inversion import reduced_subspace_kernel_and_data
from mrinversion.linear_model.linear_inversion import TSVD


# def TSVD(self, index=None):
#     K, s, r = reduce_problem(self.kernel, self.signal, tsvd_index)
#     return K, s, r


# class NMRRelaxation:
#     def __new__(self, method, dimension, inverse_dimension, supersampling=1,
#                   **kwargs):
#         if method is "T2":
#             return relaxation_kernel.T2(
#                 x, nx=2, rangex=[0, 1], oversample=1, log_scale=True
#             )


def reduce_problem(K, signal, tsvd_index=None):
    if tsvd_index is None:
        U, S, VT, r = TSVD(K)
    else:
        U, S, VT = np.linalg.svd(K, full_matrices=False)
        r = tsvd_index

    (
        k_tilde,
        s_tilde,
        _,  # projectedSignal,
        __,  # guess_solution,
    ) = reduced_subspace_kernel_and_data(U[:, :r], S[:r], VT[:r, :], signal)
    return k_tilde, s_tilde, r
