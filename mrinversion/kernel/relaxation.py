# -*- coding: utf-8 -*-
import numpy as np

from .utils import _supersampled_coordinates
from mrinversion.kernel.base import BaseModel


class T2(BaseModel):
    r"""
    A class for simulating the kernel of T2 decaying functions,

    .. math::
            y = \exp(-x/x_\text{inv}).

    Args:
        kernel_dimension: A Dimension object, or an equivalent dictionary object. This
            dimension must represent the T2 decay dimension.
        inverse_kernel_dimension: A list of two Dimension objects, or equivalent
            dictionary objects representing the `x`-`y` coordinate grid.
    """

    def __init__(self, kernel_dimension, inverse_kernel_dimension):
        super().__init__(kernel_dimension, inverse_kernel_dimension, 1, 1)

    def kernel(self, supersampling=1):
        """
        Return the kernel of T2 decaying functions.

        Args:
            supersampling: An integer. Each cell is supersampled by the factor
                    `supersampling`.
        Returns:
            A numpy array.
        """
        x = self.kernel_dimension.coordinates
        x_inverse = _supersampled_coordinates(
            self.inverse_kernel_dimension, supersampling=supersampling
        )
        amp = np.exp(np.tensordot(-(1 / x_inverse), x, 0))
        return self._averaged_kernel(amp, supersampling)


class T1(BaseModel):
    r"""
    A class for simulating the kernel of T1 recovery functions,

    .. math::
            y = 1 - \exp(-x/x_\text{inv}).

    Args:
        kernel_dimension: A Dimension object, or an equivalent dictionary object.
        This dimension must represent the T2 decay dimension.
        inverse_kernel_dimension: A list of two Dimension objects, or equivalent
                dictionary objects representing the `x`-`y` coordinate grid.
    """

    def __init__(self, kernel_dimension, inverse_kernel_dimension):
        super().__init__(kernel_dimension, inverse_kernel_dimension, 1, 1)

    def kernel(self, supersampling=1):
        x = self.kernel_dimension.coordinates
        x_inverse = _supersampled_coordinates(
            self.inverse_kernel_dimension, supersampling=supersampling
        )
        amp = 1 - np.exp(np.tensordot(-(1 / x_inverse), x, 0))
        return self._averaged_kernel(amp, supersampling)
