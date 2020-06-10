# -*- coding: utf-8 -*-
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
    array = dimension.coordinates
    if dimension.type == "linear":
        increment = dimension.increment / supersampling
        array = np.arange(dimension.count * supersampling) * increment
        array += dimension.coordinates_offset
        # shift the coordinates by half a bin for proper averaging
        array -= 0.5 * increment * (supersampling - 1)
    return array
