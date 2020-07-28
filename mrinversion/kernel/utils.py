# -*- coding: utf-8 -*-
import numpy as np


def x_y_to_zeta_eta(x, y):
    r"""Convert the coordinates :math:`(x,y)` to :math:`(\zeta, \eta)` using the
        following definition,

        .. math::
            \left.\begin{array}{rl}
            \zeta &= \sqrt{x^2 + y^2}, \\
            \eta &= \frac{4}{\pi} \tan^{-1} \left| \frac{x}{y} \right|
            \end{array} {~~~~~~~~} \right\} {~~~~~~~~} |x| \le |y|.

        .. math::
            \left.\begin{array}{rl}
            \zeta &= -\sqrt{x^2 + y^2}, \\
            \eta &= \frac{4}{\pi} \tan^{-1} \left| \frac{y}{x} \right|
            \end{array} {~~~~~~~~} \right\} {~~~~~~~~} |x| > |y|.

        Args:
            x: floats or Quantity object. The coordinate x.
            y: floats or Quantity object. The coordinate y.

        Return:
            A list of two ndarrays. The first array is the :math:`\zeta`
            coordinates. The second array is the :math:`\eta` coordinates.
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
    """Same as def x_y_to_zeta_eta, but for ndarrays."""
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
            \left. \begin{array}{rl}
            x &= |\zeta| \sin\theta, \\
            y &= |\zeta| \cos\theta
            \end{array} {~~~~~~~~} \right\} {~~~~~~~~} \zeta \ge 0

        .. math::
            \left. \begin{array}{rl}
            x &= |\zeta| \cos\theta, \\
            y &= |\zeta| \sin\theta
            \end{array} {~~~~~~~~} \right\} {~~~~~~~~} \zeta < 0

        where :math:`\theta = \frac{\pi}{4}\eta`.

        Args:
            x: ndarray or list of floats. The coordinate x.
            y: ndarray or list of floats. The coordinate y.

        Return:
            A list of ndarrays. The first array holds the coordinate :math:`x`. The
            second array holds the coordinates :math:`y`.
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


def _x_y_to_zeta_eta_distribution(grid, supersampling):
    """Return a list of zeta-eta coordinates from a list of x-y coordinates."""
    # if grid.x.coordinates_offset != grid.y.coordinates_offset:
    #     raise ValueError("coordinates_offset for x and y grid must be identical")

    x_coordinates = _supersampled_coordinates(grid[0], supersampling=supersampling)
    y_coordinates = _supersampled_coordinates(grid[1], supersampling=supersampling)

    if x_coordinates.unit.physical_type == "frequency":
        x_coordinates = x_coordinates.to("Hz").value
        y_coordinates = y_coordinates.to("Hz").value
        # offset = grid.x.coordinates_offset.to("Hz").value

    elif x_coordinates.unit.physical_type == "dimensionless":
        x_coordinates = x_coordinates.to("ppm").value
        y_coordinates = y_coordinates.to("ppm").value
        # x_coordinates = (x_coordinates * dimension.larmor_frequency).to("").value
        # y_coordinates = (y_coordinates * dimension.larmor_frequency).to("").value
        # offset = grid.x.coordinates_offset.to("").value

    x_mesh, y_mesh = np.meshgrid(
        np.abs(x_coordinates), np.abs(y_coordinates), indexing="xy"
    )

    # y_offset = y_coordinates[0]
    # x_offset = x_coordinates[0]
    return _x_y_to_zeta_eta(x_mesh, y_mesh)


def _supersampled_coordinates(dimension, supersampling=1):
    r"""The coordinates along the dimension.

    Args:
        supersampling: An integer used in supersampling the coordinates along the
            dimension, If :math:`n` is the count, :math:`\Delta_x` is the increment,
            :math:`x_0` is the coordinates offset along the dimension, and :math:`m` is
            the supersampling factor, a total of :math:`mn` coordinates are sampled
            using

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
