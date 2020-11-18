# -*- coding: utf-8 -*-
from copy import deepcopy

import csdmpy as cp
import numpy as np
from mrsimulator.methods import BlochDecaySpectrum

from .utils import _x_y_to_zeta_eta_distribution

__dimension_list__ = (cp.Dimension, cp.LinearDimension, cp.MonotonicDimension)

__dimension_name__ = ("Dimension", "LinearDimension", "MonotonicDimension")


class BaseModel:
    """Base kernel class."""

    def __init__(self, kernel_dimension, inverse_kernel_dimension, n_dir, n_inv):
        kernel = self.__class__.__name__
        message = (
            f"Exactly {n_inv} inverse dimension(s) is/are required for the "
            f"{kernel} kernel."
        )
        if isinstance(inverse_kernel_dimension, list):
            if len(inverse_kernel_dimension) != n_inv:
                raise ValueError(message)
            if n_inv == 1:
                inverse_kernel_dimension = inverse_kernel_dimension[0]
        else:
            if n_inv != 1:
                raise ValueError(message)

        inverse_kernel_dimension = _check_csdm_dimension(
            inverse_kernel_dimension, "inverse_dimension"
        )

        message = (
            f"Exactly {n_dir} direct dimension(s) is/are required for the "
            f"{kernel} kernel."
        )
        if isinstance(kernel_dimension, list):
            if len(kernel_dimension) != n_dir:
                raise ValueError(message)
            if n_dir == 1:
                kernel_dimension = kernel_dimension[0]
        else:
            if n_dir != 1:
                raise ValueError(message)

        kernel_dimension = _check_csdm_dimension(kernel_dimension, "kernel_dimension")

        self.kernel_dimension = kernel_dimension
        self.inverse_kernel_dimension = inverse_kernel_dimension

    def _averaged_kernel(self, amp, supersampling):
        """Return the kernel by averaging over the supersampled grid cells."""
        shape = ()
        inverse_kernel_dimension = self.inverse_kernel_dimension
        if not isinstance(self.inverse_kernel_dimension, list):
            inverse_kernel_dimension = [self.inverse_kernel_dimension]

        for item in inverse_kernel_dimension[::-1]:
            shape += (item.count, supersampling)
        shape += (self.kernel_dimension.count,)

        K = amp.reshape(shape)

        inv_len = len(inverse_kernel_dimension)
        axes = tuple([2 * i + 1 for i in range(inv_len)])
        K = K.mean(axis=axes)

        section = [*[0 for i in range(inv_len)], slice(None, None, None)]
        K /= K[tuple(section)].sum()

        section = [slice(None, None, None) for _ in range(inv_len + 1)]
        for i, item in enumerate(inverse_kernel_dimension):
            if item.coordinates[0].value == 0:
                section_ = deepcopy(section)
                section_[i] = 0
                K[tuple(section_)] /= 2.0

        inv_size = np.asarray([item.count for item in inverse_kernel_dimension]).prod()
        K = K.reshape(inv_size, self.kernel_dimension.count).T

        return K


class LineShape(BaseModel):
    """Base line-shape kernel generation class."""

    def __init__(
        self,
        kernel_dimension,
        inverse_kernel_dimension,
        channel,
        magnetic_flux_density="9.4 T",
        rotor_angle="54.735 deg",
        rotor_frequency=None,
        number_of_sidebands=None,
    ):
        super().__init__(kernel_dimension, inverse_kernel_dimension, 1, 2)

        kernel = self.__class__.__name__
        dim_types = ["frequency", "dimensionless"]
        _check_dimension_type(self.kernel_dimension, "anisotropic", dim_types, kernel)
        _check_dimension_type(
            self.inverse_kernel_dimension, "inverse", dim_types, kernel
        )

        dim = self.kernel_dimension

        temp_method = BlochDecaySpectrum.parse_dict_with_units(
            {"channels": [channel], "magnetic_flux_density": magnetic_flux_density}
        )
        # larmor frequency from method.
        B0 = temp_method.spectral_dimensions[0].events[0].magnetic_flux_density  # in T
        gamma = temp_method.channels[0].gyromagnetic_ratio  # in MHz/T
        self.larmor_frequency = -gamma * B0  # in MHz

        spectral_width = dim.increment * dim.count
        reference_offset = dim.coordinates_offset
        if dim.complex_fft is False:
            reference_offset = dim.coordinates_offset + spectral_width / 2.0

        if dim.increment.unit.physical_type == "dimensionless":
            lf = abs(self.larmor_frequency)
            val = spectral_width.to("ppm")
            spectral_width = f"{val.value * lf} Hz"
            val = reference_offset.to("ppm")
            reference_offset = f"{val.value * lf} Hz"

        spectral_dimensions = [
            dict(
                count=dim.count,
                reference_offset=str(reference_offset),
                spectral_width=str(spectral_width),
            )
        ]

        if rotor_frequency is None:
            if dim.increment.unit.physical_type == "dimensionless":
                rotor_frequency = f"{dim.increment.to('ppm').value * lf} Hz"
            else:
                rotor_frequency = str(dim.increment)

        self.method_args = {
            "channels": [channel],
            "magnetic_flux_density": magnetic_flux_density,
            "rotor_angle": rotor_angle,
            "rotor_frequency": rotor_frequency,
            "spectral_dimensions": spectral_dimensions,
        }

        self.number_of_sidebands = number_of_sidebands
        if number_of_sidebands is None:
            self.number_of_sidebands = dim.count

    def _get_zeta_eta(self, supersampling):
        """Return zeta and eta coordinates over x-y grid"""

        zeta, eta = _x_y_to_zeta_eta_distribution(
            self.inverse_kernel_dimension, supersampling
        )
        return zeta, eta


def _check_csdm_dimension(dimensions, dimension_id):
    if not isinstance(dimensions, (list, *__dimension_list__)):
        raise ValueError(
            f"The value of the `{dimension_id}` attribute must be one of "
            f"`{__dimension_name__}` objects."
        )

    # copy the list
    # dimensions = deepcopy(dimensions)
    if isinstance(dimensions, __dimension_list__):
        return dimensions

    for i, item in enumerate(dimensions):
        if not isinstance(item, __dimension_list__):
            raise ValueError(
                f"The element at index {i} of the `{dimension_id}` list must be an "
                f"instance of one of the {__dimension_name__} classes."
            )
        # dimensions[i] = item
    return dimensions


def _check_dimension_type(dimensions, direction, dimension_quantity, kernel_type):
    # if isinstance(dimension_quantity, tuple):
    #     dimension_quantity = list(dimension_quantity)
    # if isinstance(dimension_quantity, str):
    #     dimension_quantity = [dimension_quantity]

    if not isinstance(dimensions, list):
        if dimensions.quantity_name not in dimension_quantity:
            raise ValueError(
                f"A {direction} dimension with quantity name `{dimension_quantity}` "
                f"is required for the `{kernel_type}` kernel, instead got "
                f"`{dimensions.quantity_name}` as the quantity name for the dimension."
            )
        return
    for i, item in enumerate(dimensions):
        if item.quantity_name not in dimension_quantity:
            raise ValueError(
                f"A {direction} dimension with quantity name `{dimension_quantity}` "
                f"is required for the `{kernel_type}` kernel, instead got "
                f"`{item.quantity_name}` as the quantity name for the dimension at "
                f"index {i}."
            )
