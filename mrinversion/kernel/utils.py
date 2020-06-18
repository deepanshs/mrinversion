# -*- coding: utf-8 -*-
import numpy as np


def x_y_to_zeta_eta(x, y):
    r"""Convert the coordinates :math:`(x,y)` to :math:`(\zeta, \eta)` using the
        following definition,

        .. math::
            \zeta = \sqrt(x^2 + y^2)
            \eta = (4/\pi) \tan^{-1} |x/y|,

        if :math:`|x| \le |y|`, otherwise,

        .. math::
            \zeta = -\sqrt(x^2 + y^2)
            \eta = (4/\pi) \tan^{-1} |y/x|.

        Args:
            x: floats or Quantity object. The coordinate x.
            y: floats or Quantity object. The coordinate y.

        Return:
            zeta: The coordinate :math:`zeta`.
            eta: The coordinate :math:`\eta`.
    """
    x_unit = y_unit = 1
    if x.__class__.__name__ == "Quantity":
        x_unit = x.unit
        x = x.value
    if y.__class__.__name__ == "Quantity":
        y_unit = y.unit
        y = y.value
    if x_unit != y_unit:
        raise ValueError(
            f"x and y must have same dimensionality; x ({x_unit}) != y ({y_unit})."
        )

    zeta = np.sqrt(x ** 2 + y ** 2)  # + offset
    eta = 1.0
    if x > y:
        zeta = -zeta
        eta = (4.0 / np.pi) * np.arctan(y / x)

    if x < y:
        eta = (4.0 / np.pi) * np.arctan(x / y)

    return zeta * x_unit, eta


def _x_y_to_zeta_eta(x, y):
    r"""Convert the coordinates :math:`(x,y)` to :math:`(\zeta, \eta)` using the
        following definition,

        .. math::
            \zeta = \sqrt(x^2 + y^2)
            \eta = (4/\pi) \tan^{-1} |x/y|,

        if :math:`|x| \le |y|`, otherwise,

        .. math::
            \zeta = -\sqrt(x^2 + y^2)
            \eta = (4/\pi) \tan^{-1} |y/x|.

        Args:
            x: ndarray or list of floats. The coordinate x.
            y: ndarray or list of floats. The coordinate y.

        Return:
            zeta: 1D-ndarray. The coordinate :math:`zeta`.
            eta: 1D-ndarray. The coordinate :math:`\eta`.
    """
    x = np.abs(x)
    y = np.abs(y)
    zeta = np.sqrt(x ** 2 + y ** 2)  # + offset
    eta = np.ones(zeta.shape)
    index = np.where(x > y)
    zeta[index] = -zeta[index]
    eta[index] = (4.0 / np.pi) * np.arctan(y[index] / x[index])

    index = np.where(x < y)
    eta[index] = (4.0 / np.pi) * np.arctan(x[index] / y[index])

    return zeta.ravel(), eta.ravel()


def zeta_eta_to_x_y(zeta, eta):
    r"""Convert the coordinates :math:`(\zeta,\eta)` to :math:`(x, y)` using the
        following definition,

        .. math::
            x = |\zeta| \sin\theta
            y = |\zeta| \cos\theta,

        if :math:`\zeta \ge 0`, otherwise,

        .. math::
            x = |\zeta| \cos\theta
            y = |\zeta| \sin\theta,

        where :math:`\theta = \pi\eta/4`.

        Args:
            x: ndarray or list of floats. The coordinate x.
            y: ndarray or list of floats. The coordinate y.

        Return:
            zeta: 1D-ndarray. The coordinate :math:`zeta`.
            eta: 1D-ndarray. The coordinate :math:`\eta`.
    """
    zeta = np.asarray(zeta)
    eta = np.asarray(eta)

    theta = np.pi * eta / 4.0
    x = np.zeros(zeta.size)
    y = np.zeros(zeta.size)

    index = np.where(zeta >= 0)
    x[index] = zeta[index] * np.sin(theta[index])
    y[index] = zeta[index] * np.cos(theta[index])

    index = np.where(zeta < 0)
    x[index] = -zeta[index] * np.cos(theta[index])
    y[index] = -zeta[index] * np.sin(theta[index])

    return x.ravel(), y.ravel()


def cal_zeta_eta_from_x_y_distribution(dimension, grid, supersampling):
    """Return a list of zeta-eta coordinates from a list of x-y coordinates."""
    # if grid.x.coordinates_offset != grid.y.coordinates_offset:
    #     raise ValueError("coordinates_offset for x and y grid must be identical")

    x_coordinates = supersampled_coordinates(grid[0], supersampling=supersampling)
    y_coordinates = supersampled_coordinates(grid[1], supersampling=supersampling)

    if x_coordinates.unit.physical_type == "frequency":
        x_coordinates = x_coordinates.to("Hz").value
        y_coordinates = y_coordinates.to("Hz").value
        # offset = grid.x.coordinates_offset.to("Hz").value

    elif x_coordinates.unit.physical_type == "dimensionless":
        x_coordinates = (x_coordinates * dimension.larmor_frequency).to("").value
        y_coordinates = (y_coordinates * dimension.larmor_frequency).to("").value
        # offset = grid.x.coordinates_offset.to("").value

    x_mesh, y_mesh = np.meshgrid(
        np.abs(x_coordinates), np.abs(y_coordinates), indexing="xy"
    )

    # y_offset = y_coordinates[0]
    # x_offset = x_coordinates[0]
    return _x_y_to_zeta_eta(x_mesh, y_mesh)


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
