import csdmpy as cp
import numpy as np

from mrinversion.kernel import relaxation
from mrinversion.linear_model import LassoFistaCV
from mrinversion.linear_model import TSVDCompression


def test_fista():
    domain = "https://ssnmr.org/resources/mrinversion"
    filename = f"{domain}/test1_signal.csdf"
    signal = cp.load(filename)
    sigma = 0.0008

    datafile = f"{domain}/test1_t2.csdf"
    true_dist = cp.load(datafile)

    kernel_dimension = signal.dimensions[0]
    relaxT2 = relaxation.T2(
        kernel_dimension=kernel_dimension,
        inverse_dimension=dict(
            count=64,
            minimum="1e-2 s",
            maximum="1e3 s",
            scale="log",
            label="log (T2 / s)",
        ),
    )
    inverse_dimension = relaxT2.inverse_dimension
    K = relaxT2.kernel(supersampling=1)

    new_system = TSVDCompression(K, signal)
    compressed_K = new_system.compressed_K
    compressed_s = new_system.compressed_s

    assert new_system.truncation_index == 29

    lambdas = 10 ** (-5 + 4 * (np.arange(16) / 15))

    f_lasso_cv = LassoFistaCV(
        lambdas=lambdas,  # A numpy array of lambda values.
        folds=5,  # The number of folds in n-folds cross-validation.
        sigma=sigma,  # noise standard deviation
        inverse_dimension=inverse_dimension,  # previously defined inverse dimensions.
    )

    f_lasso_cv.fit(K=compressed_K, s=compressed_s)

    sol = f_lasso_cv.f
    assert np.argmax(sol.y[0].components[0]) == np.argmax(true_dist.y[0].components[0])

    residuals = f_lasso_cv.residuals(K=K, s=signal)
    std = residuals.std()
    np.testing.assert_almost_equal(std.value, sigma, decimal=2)
