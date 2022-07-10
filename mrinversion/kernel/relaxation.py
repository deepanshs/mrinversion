import csdmpy as cp
import numpy as np
from csdmpy.units import string_to_quantity

from .utils import _supersampled_coordinates
from mrinversion.kernel.base import BaseModel


class BaseRelaxation(BaseModel):
    def __init__(self, kernel_dimension, inverse_dimension):
        minimum = inverse_dimension["minimum"]
        maximum = inverse_dimension["maximum"]
        count = inverse_dimension.get("count", 32)
        scale = inverse_dimension.get("scale", "linear")
        label = inverse_dimension.get("label", None)

        unit = kernel_dimension.coordinates[0].unit
        x_min = string_to_quantity(minimum).to(unit).value
        x_max = string_to_quantity(maximum).to(unit).value
        check_log = scale == "log"

        if check_log:
            x_min, x_max = np.log10(x_min), np.log10(x_max)

        coords = (np.arange(count) / (count - 1)) * (x_max - x_min) + x_min

        if check_log:
            coords = 10 ** (coords)

        lbl_ = f"log({self.__class__.__name__} / {unit})" if check_log else None
        label = label if label is not None else lbl_
        inverse_dimension = cp.as_dimension(array=coords, unit=str(unit), label=label)

        meta = {
            "log": check_log,
            "label": f"log({self.__class__.__name__} / {unit})" if check_log else None,
        }
        inverse_dimension.application = {"com.github.deepanshs.mrinversion": meta}
        super().__init__(kernel_dimension, inverse_dimension, 1, 1)


class T2(BaseRelaxation):
    r"""A class for simulating the kernel of T2 decaying functions,

    .. math::
            y = \exp(-x/x_\text{inv}).

    Args:
        kernel_dimension: A Dimension object, or an equivalent dictionary object. This
            dimension must represent the T2 decay dimension.
        inverse_dimension: A list of two Dimension objects, or equivalent
            dictionary objects representing the `x`-`y` coordinate grid.
    """

    def __init__(self, kernel_dimension, inverse_dimension):
        super().__init__(kernel_dimension, inverse_dimension)

    def kernel(self, supersampling=1):
        """Return the kernel of T2 decaying functions.

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
        return self._averaged_kernel(amp, supersampling, xy_grid=False)


class T1(BaseRelaxation):
    r"""A class for simulating the kernel of T1 recovery functions,

    .. math::
            y = 1 - \exp(-x/x_\text{inv}).

    Args:
        kernel_dimension: A Dimension object, or an equivalent dictionary object.
        This dimension must represent the T2 decay dimension.
        inverse_dimension: A list of  Dimension objects, or equivalent
                dictionary objects representing the `x`-`y` coordinate grid.
    """

    def __init__(self, kernel_dimension, inverse_dimension):
        super().__init__(kernel_dimension, inverse_dimension)

    def kernel(self, supersampling=1):
        x = self.kernel_dimension.coordinates
        x_inverse = _supersampled_coordinates(
            self.inverse_kernel_dimension, supersampling=supersampling
        )
        amp = 1 - np.exp(np.tensordot(-(1 / x_inverse), x, 0))
        return self._averaged_kernel(amp, supersampling, xy_grid=False)
