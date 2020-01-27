from copy import deepcopy

import csdmpy as cp
import numpy as np


def supersampled_coordinates(dimension, supersampling=1):
    r"""The coordinates along the dimension.

        Args:
            supersampling: An integer used in supersampling the coordinates along the
                dimension, If :math:`n` is the count, :math:`\Delta_x` is the
                increment, :math:`x_0` is the coordinates offset along the dimension,
                and :math:`m` is the supersampling, a total of :math:`mn` coordinates
                are sampled using

                .. math::
                    x = [0 .. (nm-1)] \Delta_x + \x_0 - \frac{1}{2} \Delta_x (m-1)

                where :math:`\Delta_x' = \frac{\Delta_x}{m}`.

        Returns:
            An `Quantity` array of coordinates.
        """
    increment = dimension.increment / supersampling
    array = np.arange(dimension.count * supersampling) * increment
    array += dimension.coordinates_offset
    # shift the coordinates by half a bin for proper averaging
    array -= 0.5 * increment * (supersampling - 1)
    return array


def _check_csdm_dimension(dimensions, dimension_id):
    dim_type = cp.Dimension.__name__
    if not isinstance(dimensions, (list, cp.Dimension)):
        raise ValueError(
            f"The value of the `{dimension_id}` attribute must be a `{dim_type}` "
            f"object, or a list of the `{dim_type}` objects, or equivalent "
            "dictionary objects."
        )

    # copy the list
    dimensions = deepcopy(dimensions)
    if isinstance(dimensions, cp.Dimension):
        return dimensions

    for i, item in enumerate(dimensions):
        if isinstance(item, (dict, cp.Dimension)):
            if isinstance(item, dict):
                item = cp.Dimension(item)
        else:
            raise ValueError(
                f"The element at index {i} of the `{dimension_id}` list must be an "
                f"instance of the {dim_type} class, or an equivalent dictionary "
                "object."
            )
        dimensions[i] = item
    return dimensions


def _check_dimension_type(dimensions, direction, dimension_quantity, kernel_type):
    if not isinstance(dimensions, list):
        if dimensions.quantity_name != dimension_quantity:
            raise ValueError(
                f"A {direction} dimension with quantity name `{dimension_quantity}` "
                f"is required for the `{kernel_type}` kernel, instead got "
                f"`{dimensions.quantity_name}` as the quantity name for the dimension."
            )
        return
    for i, item in enumerate(dimensions):
        if item.quantity_name != dimension_quantity:
            raise ValueError(
                f"A {direction} dimension with quantity name `{dimension_quantity}` "
                f"is required for the `{kernel_type}` kernel, instead got "
                f"`{item.quantity_name}` as the quantity name for the dimension at "
                f"index {i}."
            )
