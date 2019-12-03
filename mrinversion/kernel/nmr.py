import numpy as np
from mrsimulator import Dimension
from mrsimulator.tests.tests import _one_d_simulator

# from .linearInversion import reduced_subspace_kernel_and_data
# from .linearInversion import TSVD
# from .minimizer import fista


def cal_aniso_eta_for_x_y_distribution(n_x, x_range, n_y, y_range, oversampling):
    """Return a list of zeta-eta coordinates from a list of x-y coordinates."""

    print(n_x, x_range, n_y, y_range, oversampling)
    n_x_ = n_x * oversampling
    x = (np.arange(n_x_) / n_x_) * (x_range[1] - x_range[0])
    n_y_ = n_y * oversampling
    y = (np.arange(n_y_) / n_y_) * (y_range[1] - y_range[0])

    x -= 0.5 * (x[1] - x[0]) * oversampling
    y -= 0.5 * (y[1] - y[0]) * oversampling

    x_, y_ = np.meshgrid(np.abs(x), np.abs(y))

    zeta = np.sqrt((x_ + x_range[0]) ** 2 + (y_ + y_range[0]) ** 2)
    eta = np.ones(zeta.shape)
    index = np.where(x_ > y_)
    zeta[index] = -zeta[index]
    eta[index] = (4.0 / np.pi) * np.arctan(y_[index] / x_[index])

    index = np.where(x_ < y_)
    eta[index] = (4.0 / np.pi) * np.arctan(x_[index] / y_[index])

    return zeta.ravel(), eta.ravel()


def kernel(dimension, points, extent, oversampling, **kwargs):
    """
    Generate a kernel of NMR anisotropic lineshapes.

    :ivar dimension: The Spectroscopic dimension from the mrsimulator package.
    :ivar points: A tuple or list of two intergers. The first integer defines
                  the number od points along the x-dimension and the second
                  integer is the number of points along the y-dimension.
    :ivar extent: A list of lists. Each sub-list contains two floats of the
                  form (min, max) where min and max are the minimum and the
                  maximun bounds along the respective dimension. The first list
                  contains the bounds along the x-dimension while the second
                  list contains the bounds of y-dimension.
    :ivar oversampling: An integer. The x-y grid is partially averaged by first
                        over-sampling the grid by a factor of `oversampling`.
    """
    if not isinstance(dimension, [dict, Dimension]):
        raise ValueError(
            "Attribute dimension must be an instance of `Dimension` or `dict` class."
        )
    if isinstance(dimension, dict):
        dimension = Dimension.parse_dict_with_units(dimension)

    n_x, n_y = points
    x_range, y_range = extent

    spectral_width = dimension.spectral_width
    increment = spectral_width / dimension.number_of_points
    rotor_frequency = dimension.rotor_frequency
    reference_offset = dimension.reference_offset - spectral_width / 2.0

    aniso, eta = cal_aniso_eta_for_x_y_distribution(
        n_x, x_range, n_y, y_range, oversampling
    )
    iso = np.zeros(n_x * n_y * oversampling ** 2, dtype=np.float64)

    if dimension.spin == 0.5:
        if rotor_frequency != 0:
            iso /= rotor_frequency
            aniso /= rotor_frequency
            number_of_sidebands = dimension.number_of_points
            increment = 1
            if dimension.number_of_points % 2 == 0:
                reference_offset += -number_of_sidebands / 2.0
            else:
                reference_offset += -(number_of_sidebands - 1) / 2

        else:
            number_of_sidebands = 1

        freq, amp = _one_d_simulator(
            number_of_points=dimension.number_of_points,
            reference_offset=reference_offset,
            increment=increment,
            isotropic_chemical_shift=iso,
            shielding_anisotropy=aniso,
            shielding_asymmetry=eta,
            number_of_sidebands=number_of_sidebands,
            rotor_angle=dimension.rotor_angle * 180.0 / np.pi,
            sample_rotation_frequency_in_Hz=rotor_frequency,
            **kwargs,
        )

    print(kwargs.keys())
    if "remove_second_order_quad_iso" in kwargs.keys():
        remove_second_order_quad_iso = kwargs["remove_second_order_quad_iso"]
    else:
        remove_second_order_quad_iso = 0

    print("remove_second_order_quad_iso", remove_second_order_quad_iso)

    if "larmor_frequency" in kwargs.keys():
        larmor_frequency = kwargs["larmor_frequency"]
    else:
        larmor_frequency = dimension.larmor_frequency

    print("larmor_frequency", larmor_frequency)

    if dimension.spin >= 1:
        freq, amp = _one_d_simulator(
            number_of_points=dimension.number_of_points,
            reference_offset=reference_offset,
            larmor_frequency=larmor_frequency,
            spin_quantum_number=dimension.spin,
            increment=increment,
            isotropic_chemical_shift=iso,
            quadrupolar_coupling_constant=aniso,
            quadrupolar_asymmetry=eta,
            number_of_sidebands=1,
            remove_second_order_quad_iso=remove_second_order_quad_iso,
            rotor_angle=dimension.rotor_angle * 180.0 / np.pi,
            sample_rotation_frequency=rotor_frequency,
        )

    K = amp.reshape(n_x, oversampling, n_y, oversampling, dimension.number_of_points)
    K = K.sum(axis=(1, 3))
    K /= K[0, 0].sum(axis=-1)

    if x_range[0] == 0:
        K[0, :, :] /= 2.0
    if y_range[0] == 0:
        K[:, 0, :] /= 2.0
    K = K.reshape(n_x * n_y, dimension.number_of_points).T

    x = (np.arange(n_x) / n_x) * (x_range[1] - x_range[0]) + x_range[0]
    y = (np.arange(n_y) / n_y) * (y_range[1] - y_range[0]) + y_range[0]
    dimensions = [x, y, freq]

    return K, dimensions
