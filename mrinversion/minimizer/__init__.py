import numpy as np

from mrinversion.minimizer import fista
from mrinversion.minimizer.linear_inversion import reduced_subspace_kernel_and_data
from mrinversion.minimizer.linear_inversion import TSVD


class Minimizer:
    def __init__(self, kernel=None):
        self.kernel = kernel
        self.hyperparameter = 1e-5
        self.truncation_index = None
        self.fista_config = {
            "hyperparameter": 1e-5,
            "maxiter": 5000,
            "nonnegative": True,
            "tol": 1e-9,
            "npros": 1,
        }
        self.function = None
        self.iter = 0
        self.cpu_time = None

    def minimize(self, signal):
        if self.truncation_index is None:
            U, S, VT, r = TSVD(self.kernel)
            self.truncation_index = r
        else:
            U, S, VT, r = np.linalg.svd(self.kernel, full_matrices=False)

        k_tilde, s_tilde, projectedSignal = reduced_subspace_kernel_and_data(
            U[:, :r], S[:r], VT[:r, :], signal
        )
        # print("Truncation points", r)

        if s_tilde.ndim == 1:
            s_tilde = s_tilde[:, np.newaxis]

        # number_of_crosssections = s_tilde.shape[1]
        s_max = S[0] ** 2

        zf, self.function, self.iter, self.cpu_time = fista.fista(
            matrix=k_tilde, s=s_tilde, Linv=(1 / s_max), **self.fista_config
        )

        self.function.shape = (s_tilde.shape[1], VT.shape[1])
        zf = U[:, :r] @ zf.reshape(s_tilde.T.shape).T
        zf = np.squeeze(zf)

        return zf.T
