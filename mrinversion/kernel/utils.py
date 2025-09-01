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
            f"x and y must have same dimensionality: x ({x_unit}) != y ({y_unit})."
        )

    zeta = np.sqrt(x**2 + y**2)  # + offset
    eta = 1.0
    if x > y:
        zeta = -zeta
        eta = (4.0 / np.pi) * np.arctan(y / x)

    if x < y:
        eta = (4.0 / np.pi) * np.arctan(x / y)

    return zeta * x_unit, eta


def _x_y_to_zeta_eta(x, y, eta_bound=1, calc_pos=False):
    """Same as def x_y_to_zeta_eta, but for ndarrays."""
    x = np.abs(x).ravel()
    y = np.abs(y).ravel()
    zeta = np.sqrt(x**2 + y**2)  # + offset
    abundances = np.ones(zeta.shape)
    eta = np.ones(zeta.shape)
    eta = eta.tolist()
    zeta = zeta.tolist()
    del_these = []
    for index, _ in enumerate(x):
        if x[index] > y[index]:
            if not calc_pos:
                zeta[index] = -zeta[index]
                this_eta = (4.0 / np.pi) * np.arctan(y[index] / x[index])
                if this_eta < eta_bound:
                    eta[index] = this_eta
                else:
                    abundances[index] = 0
            else:
                abundances[index] = 0

        elif x[index] < y[index]:
            this_eta = (4.0 / np.pi) * np.arctan(x[index] / y[index])
            if this_eta < eta_bound:
                eta[index] = this_eta
            else:
                abundances[index] = 0
        elif x[index] == y[index] and eta_bound < 1:
            abundances[index] = 0

    if eta_bound == 1 and not calc_pos:
        return np.asarray(zeta), np.asarray(eta)
    else:
        for idx in del_these[::-1]:
            del zeta[idx], eta[idx]
        return np.asarray(zeta), np.asarray(eta), abundances


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


def _x_y_to_zeta_eta_distribution(grid, supersampling, eta_bound=1, calc_pos=False):
    """Return a list of zeta-eta coordinates from a list of x-y coordinates."""
    x_coordinates = _supersampled_coordinates(grid[0], supersampling=supersampling)
    y_coordinates = _supersampled_coordinates(grid[1], supersampling=supersampling)

    if x_coordinates.unit.physical_type == "frequency":
        x_coordinates = x_coordinates.to("Hz").value
        y_coordinates = y_coordinates.to("Hz").value

    elif x_coordinates.unit.physical_type == "dimensionless":
        x_coordinates = x_coordinates.to("ppm").value
        y_coordinates = y_coordinates.to("ppm").value

    x_mesh, y_mesh = np.meshgrid(
        np.abs(x_coordinates), np.abs(y_coordinates), indexing="xy"
    )

    return _x_y_to_zeta_eta(x_mesh, y_mesh, eta_bound, calc_pos)


# def _x_y_to_cq_eta_distribution(grid, supersampling):
#     """Return a list of zeta-eta coordinates from a list of x-y coordinates."""
#     x_coordinates = _supersampled_coordinates(grid[0], supersampling=supersampling)
#     y_coordinates = _supersampled_coordinates(grid[1], supersampling=supersampling)

#     if x_coordinates.unit.physical_type == "frequency":
#         x_coordinates = x_coordinates.to("Hz").value
#         y_coordinates = y_coordinates.to("Hz").value

#     elif x_coordinates.unit.physical_type == "dimensionless":
#         x_coordinates = x_coordinates.to("ppm").value
#         y_coordinates = y_coordinates.to("ppm").value

#     x_mesh, y_mesh = np.meshgrid(
#         np.abs(x_coordinates), np.abs(y_coordinates), indexing="xy"
#     )

#     return _x_y_to_cq_eta(x_mesh, y_mesh)


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
    if dimension.type == "linear":
        increment = dimension.increment / supersampling
        array = np.arange(dimension.count * supersampling) * increment
        array += dimension.coordinates_offset
        # shift the coordinates by half a bin for proper averaging
        array -= 0.5 * increment * (supersampling - 1)

    if dimension.type == "monotonic":
        coordinates = dimension.coordinates
        unit = coordinates[0].unit
        size = coordinates.size

        diff = np.zeros(size)
        diff[1:] = (coordinates[1:] - coordinates[:-1]) / supersampling
        diff *= unit

        s2 = supersampling // 2
        eo = 0.5 if supersampling % 2 == 0 else 0

        array = np.zeros(size * supersampling) * unit
        for i in range(supersampling):
            array[i::supersampling] = coordinates + (i - s2 + eo) * diff

    return array


# def cq_eta_to_x_y(cq, eta):
#     r"""Convert the coordinates :math:`(C_q,\eta)` to :math:`(x, y)` using the
#         following definition,

#         .. math::
#             \begin{array}{rl}
#             x &= C_q \sin\theta, \\
#             y &= C_q \cos\theta
#             \end{array} {~~~~~~~~}

#         where :math:`\theta = \frac{\pi}{2}\eta`.

#         Args:
#             x: ndarray or list of floats. The coordinate x.
#             y: ndarray or list of floats. The coordinate y.

#         Return:
#             A list of ndarrays. The first array holds the coordinate :math:`x`. The
#             second array holds the coordinates :math:`y`.
#     """
#     cq = np.asarray(cq)
#     eta = np.asarray(eta)

#     theta = np.pi * eta / 2.0
#     x = np.zeros(cq.size)
#     y = np.zeros(cq.size)

#     index = np.arange(len(cq))
#     x[index] = cq[index] * np.cos(theta[index])
#     y[index] = cq[index] * np.sin(theta[index])

#     return x.ravel(), y.ravel()


# def _x_y_to_cq_eta(x, y):
#     """Same as def x_y_to_zeta_eta, but for ndarrays."""
#     x = np.abs(x)
#     y = np.abs(y)
#     cq = np.sqrt(x**2 + y**2)  # + offset
#     eta = np.ones(cq.shape)
#     index = np.arange(len(cq))
#     eta[index] = (2.0 / np.pi) * np.arctan(y[index] / x[index])

#     return cq.ravel(), eta.ravel()
