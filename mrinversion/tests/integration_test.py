import csdmpy as cp
import numpy as np
from csdmpy import statistics as stats

from mrinversion.kernel import relaxation
from mrinversion.kernel.nmr import ShieldingPALineshape
from mrinversion.linear_model import LassoFistaCV
from mrinversion.linear_model import SmoothLasso
from mrinversion.linear_model import TSVDCompression


def test_01():
    domain = "https://ssnmr.org/resources/mrinversion"
    filename = f"{domain}/8lnwmg0dr7y6egk40c2orpkmmugh9j7c.csdf"
    data_object = cp.load(filename)
    data_object = data_object.real
    _ = [item.to("ppm", "nmr_frequency_ratio") for item in data_object.dimensions]

    data_object = data_object.T
    data_object_truncated = data_object[:, 155:180]

    anisotropic_dimension = data_object_truncated.dimensions[0]
    inverse_dimensions = [
        cp.LinearDimension(count=25, increment="400 Hz", label="x"),
        cp.LinearDimension(count=25, increment="400 Hz", label="y"),
    ]

    lineshape = ShieldingPALineshape(
        anisotropic_dimension=anisotropic_dimension,
        inverse_dimension=inverse_dimensions,
        channel="29Si",
        magnetic_flux_density="9.4 T",
        rotor_angle="87.14°",
        rotor_frequency="14 kHz",
        number_of_sidebands=4,
    )
    K = lineshape.kernel(supersampling=2)

    new_system = TSVDCompression(K, data_object_truncated)
    compressed_K = new_system.compressed_K
    compressed_s = new_system.compressed_s

    assert new_system.truncation_index == 87

    s_lasso = SmoothLasso(
        alpha=2.07e-7, lambda1=7.85e-6, inverse_dimension=inverse_dimensions
    )
    s_lasso.fit(K=compressed_K, s=compressed_s)
    f_sol = s_lasso.f

    residuals = s_lasso.residuals(K=K, s=data_object_truncated)

    # assert np.allclose(residuals.mean().value, 0.00048751)
    np.testing.assert_almost_equal(residuals.std().value, 0.00336372, decimal=2)

    f_sol /= f_sol.max()
    [item.to("ppm", "nmr_frequency_ratio") for item in f_sol.dimensions]

    Q4_region = f_sol[0:8, 0:8, 3:18]
    Q4_region.description = "Q4 region"

    Q3_region = f_sol[0:8, 11:22, 8:20]
    Q3_region.description = "Q3 region"

    # Analysis
    int_Q4 = stats.integral(Q4_region)  # volume of the Q4 distribution
    mean_Q4 = stats.mean(Q4_region)  # mean of the Q4 distribution
    std_Q4 = stats.std(Q4_region)  # standard deviation of the Q4 distribution

    int_Q3 = stats.integral(Q3_region)  # volume of the Q3 distribution
    mean_Q3 = stats.mean(Q3_region)  # mean of the Q3 distribution
    std_Q3 = stats.std(Q3_region)  # standard deviation of the Q3 distribution

    np.testing.assert_almost_equal(
        (100 * int_Q4 / (int_Q4 + int_Q3)).value, 60.45388973909665, decimal=1
    )

    np.testing.assert_almost_equal(
        np.asarray([mean_Q4[0].value, mean_Q4[1].value, mean_Q4[2].value]),
        np.asarray([8.604842824865958, 9.05845796147297, -103.6976331077773]),
        decimal=0,
    )

    np.testing.assert_almost_equal(
        np.asarray([mean_Q3[0].value, mean_Q3[1].value, mean_Q3[2].value]),
        np.asarray([10.35036818411856, 79.02481579085152, -90.58326773441284]),
        decimal=0,
    )

    np.testing.assert_almost_equal(
        np.asarray([std_Q4[0].value, std_Q4[1].value, std_Q4[2].value]),
        np.asarray([4.525457744683861, 4.686253809896416, 5.369228151035292]),
        decimal=0,
    )

    np.testing.assert_almost_equal(
        np.asarray([std_Q3[0].value, std_Q3[1].value, std_Q3[2].value]),
        np.asarray([6.138761032132587, 7.837190479891721, 4.210912435356488]),
        decimal=0,
    )


def test_inversion():
    domain = "https://www.ssnmr.org/sites/default/files/mrsimulator"
    filename = f"{domain}/MAS_SE_PIETA_5%25Li2O_FT.csdf"
    data_object = cp.load(filename)

    # Inversion only requires the real part of the complex dataset.
    data_object = data_object.real
    sigma = 1110.521  # data standard deviation

    # Convert the MAS dimension from Hz to ppm.
    data_object.dimensions[0].to("ppm", "nmr_frequency_ratio")

    data_object = data_object.T
    data_object_truncated = data_object[:, 1220:-1220]

    data_object_truncated.dimensions[0].to("s")  # set coordinates to 's'
    kernel_dimension = data_object_truncated.dimensions[0]

    relaxT2 = relaxation.T2(
        kernel_dimension=kernel_dimension,
        inverse_dimension=dict(
            count=32,
            minimum="1e-3 s",
            maximum="1e4 s",
            scale="log",
            label="log (T2 / s)",
        ),
    )
    inverse_dimension = relaxT2.inverse_dimension
    K = relaxT2.kernel(supersampling=20)

    new_system = TSVDCompression(K, data_object_truncated)
    compressed_K = new_system.compressed_K
    compressed_s = new_system.compressed_s

    assert new_system.truncation_index == 18

    # setup the pre-defined range of alpha and lambda values
    lambdas = 10 ** (-4 + 5 * (np.arange(32) / 31))

    # setup the smooth lasso cross-validation class
    s_lasso = LassoFistaCV(
        lambdas=lambdas,  # A numpy array of lambda values.
        sigma=sigma,  # data standard deviation
        folds=5,  # The number of folds in n-folds cross-validation.
        inverse_dimension=inverse_dimension,  # previously defined inverse dimensions.
    )

    # run the fit method on the compressed kernel and compressed data.
    s_lasso.fit(K=compressed_K, s=compressed_s)
    np.testing.assert_almost_equal(s_lasso.hyperparameters["lambda"], 0.116, decimal=1)

    s_lasso.cv_plot()

    residuals = s_lasso.residuals(K=K, s=data_object_truncated)
    np.testing.assert_almost_equal(residuals.std().value, 1526.48, decimal=0)
