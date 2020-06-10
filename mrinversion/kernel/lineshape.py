# -*- coding: utf-8 -*-
import numpy as np
from astropy.units import Quantity
from mrsimulator import Simulator
from mrsimulator import SpinSystem
from mrsimulator.methods import BlochDecayCentralTransitionSpectrum
from mrsimulator.methods import BlochDecaySpectrum

from mrinversion.kernel.base import _check_dimension_type
from mrinversion.kernel.base import BaseModel
from mrinversion.util import supersampled_coordinates


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
    if isinstance(x, Quantity):
        x_unit = x.unit
        x = x.value
    if isinstance(y, Quantity):
        y_unit = y.unit
        y = y.value
    if x_unit != y_unit:
        raise ValueError("x and y have different quantity types.")

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
        dim_types = "frequency"
        _check_dimension_type(self.kernel_dimension, "anisotropic", dim_types, kernel)
        _check_dimension_type(
            self.inverse_kernel_dimension, "inverse", dim_types, kernel
        )

        dim = self.kernel_dimension

        spectral_width = dim.increment * dim.count
        reference_offset = dim.coordinates_offset
        if dim.complex_fft is False:
            reference_offset = dim.coordinates_offset + spectral_width / 2.0

        spectral_dimensions = [
            dict(
                count=dim.count,
                reference_offset=str(reference_offset),
                spectral_width=str(spectral_width),
            )
        ]

        if rotor_frequency is None:
            rotor_frequency = str(dim.increment)

        self.method_args = {
            "channels": [channel],
            "magnetic_flux_density": magnetic_flux_density,
            "rotor_angle": rotor_angle,
            "rotor_frequency": rotor_frequency,
            "spectral_dimensions": spectral_dimensions,
        }

        if number_of_sidebands is None:
            number_of_sidebands = dim.count

        self.simulator = Simulator()
        self.simulator.config.number_of_sidebands = number_of_sidebands
        self.simulator.config.decompose_spectrum = "spin_system"

    def _get_zeta_eta(self, supersampling):
        """Return zeta and eta coordinates over x-y grid"""

        zeta, eta = cal_zeta_eta_from_x_y_distribution(
            self.kernel_dimension, self.inverse_kernel_dimension, supersampling
        )
        return zeta, eta


class NuclearShieldingLineshape(LineShape):
    """
        A generalized class for simulating the pure anisotropic NMR nuclear shielding
        line-shape kernel.

        Args:
            anisotropic_dimension: A Dimension object, or an equivalent dictionary
                    object. This dimension must represent the pure anisotropic
                    dimension.
            inverse_dimension: A list of two Dimension objects, or equivalent
                    dictionary objects representing the `x`-`y` coordinate grid.
            channel: The channel is an isotope symbol of the nuclei given as the atomic
                number followed by the atomic symbol, for example, `1H`, `13C`, and
                `29Si`. This nucleus must correspond to the recorded frequency
                resonances.
            magnetic_flux_density: The magnetic flux density of the external static
                    magnetic field. The default value is 9.4 T.
            rotor_angle: The angle of the sample holder (rotor) relative to the
                    direction of the external magnetic field. The default value is
                    54.735 deg (magic angle).
            rotor_frequency: The effective sample spin rate. Depending on the NMR
                    sequence, this value may be less than the physical sample rotation
                    frequency. The default is 14 kHz.
            number_of_sidebands: The number of sidebands to simulate along the
                    anisotropic dimension. The default value is 1.
    """

    def __init__(
        self,
        anisotropic_dimension,
        inverse_dimension,
        channel,
        magnetic_flux_density="9.4 T",
        rotor_angle="54.735 deg",
        rotor_frequency="14 kHz",
        number_of_sidebands=1,
    ):
        super().__init__(
            anisotropic_dimension,
            inverse_dimension,
            channel,
            magnetic_flux_density,
            rotor_angle,
            rotor_frequency,
            number_of_sidebands,
        )

    def kernel(self, supersampling=1):
        """
        Return the NMR nuclear shielding anisotropic line-shape kernel.

        Args:
            supersampling: An integer. Each cell is supersampled by the factor
                    `supersampling` along every dimension.
        Returns:
            A numpy array containing the line-shape kernel.
        """

        method = BlochDecaySpectrum.parse_dict_with_units(self.method_args)
        isotope = self.method_args["channels"][0]
        zeta, eta = self._get_zeta_eta(supersampling)
        # convert zeta to ppm
        B0 = method.spectral_dimensions[0].events[0].magnetic_flux_density  # in T
        gamma = method.channels[0].gyromagnetic_ratio  # in MHz/T
        larmor_frequency = -gamma * B0  # in MHZ
        zeta /= larmor_frequency  # zeta in ppm
        spin_systems = [
            SpinSystem(
                sites=[dict(isotope=isotope, shielding_symmetric=dict(zeta=z, eta=e))]
            )
            for z, e in zip(zeta, eta)
        ]

        dim = method.spectral_dimensions[0]
        if dim.origin_offset == 0:
            dim.origin_offset = larmor_frequency * 1e6  # in Hz
        for dim_i in self.inverse_kernel_dimension:
            if dim_i.origin_offset.value == 0:
                dim_i.origin_offset = f"{larmor_frequency} MHz"

        self.simulator.spin_systems = spin_systems
        self.simulator.methods = [method]
        self.simulator.run(pack_as_csdm=False)

        amp = self.simulator.methods[0].simulation

        return self._averaged_kernel(amp, supersampling)


class MAF(NuclearShieldingLineshape):
    r"""
        A specialized class for simulating the pure anisotropic NMR nuclear shielding
        line-shape kernel resulting from the 2D MAF spectra.

        Args:
            anisotropic_dimension: A Dimension object, or an equivalent dictionary
                    object. This dimension must represent the pure anisotropic
                    dimension.
            inverse_dimension: A list of two Dimension objects, or equivalent
                    dictionary objects representing the `x`-`y` coordinate grid.
            channel: The isotope symbol of the nuclei given as the atomic number
                    followed by the atomic symbol, for example, `1H`, `13C`, and
                    `29Si`. This nucleus must correspond to the recorded frequency
                    resonances.
            magnetic_flux_density: The magnetic flux density of the external static
                    magnetic field. The default value is 9.4 T.

        **Assumptions:**
        The simulated line-shapes correspond to an infinite speed spectrum spinning at
        :math:`90^\circ`.
    """

    def __init__(
        self,
        anisotropic_dimension,
        inverse_dimension,
        channel,
        magnetic_flux_density="9.4 T",
    ):

        super().__init__(
            anisotropic_dimension,
            inverse_dimension,
            channel,
            magnetic_flux_density,
            "90 deg",
            "200 GHz",
            1,
        )


class SpinningSidebands(NuclearShieldingLineshape):
    r"""
        A specialized class for simulating the pure anisotropic spinning sideband
        amplitudes of the nuclear shielding resonances resulting from a 2D sideband
        separation spectra.

        Args:
            anisotropic_dimension: A Dimension object, or an equivalent dictionary
                    object. This dimension must represent the pure anisotropic
                    dimension.
            inverse_dimension: A list of two Dimension objects, or equivalent
                    dictionary objects representing the `x`-`y` coordinate grid.
            channel: The isotope symbol of the nuclei given as the atomic number
                    followed by the atomic symbol, for example, `1H`, `13C`, and
                    `29Si`. This nucleus must correspond to the recorded frequency
                    resonances.
            magnetic_flux_density: The magnetic flux density of the external static
                    magnetic field. The default value is 9.4 T.

        **Assumption:**
        The simulated line-shapes correspond to a finite speed spectrum spinning at the
        magic angle, :math:`54.735^\circ`, where the spin rate is the increment along
        the anisotropic dimension.
    """

    def __init__(
        self,
        anisotropic_dimension,
        inverse_dimension,
        channel,
        magnetic_flux_density="9.4 T",
    ):

        super().__init__(
            anisotropic_dimension,
            inverse_dimension,
            channel,
            magnetic_flux_density,
            "54.735 deg",
            None,
            None,
        )


class DAS(LineShape):
    def __init__(
        self,
        anisotropic_dimension,
        inverse_kernel_dimension,
        channel,
        magnetic_flux_density="9.4 T",
        rotor_angle="54.735 deg",
        rotor_frequency="600 Hz",
        number_of_sidebands=None,
    ):
        super().__init__(
            anisotropic_dimension,
            inverse_kernel_dimension,
            channel,
            magnetic_flux_density,
            rotor_angle,
            rotor_frequency,
            number_of_sidebands,
            # "DAS",
        )

    def kernel(self, supersampling):
        method = BlochDecayCentralTransitionSpectrum.parse_dict_with_units(
            self.method_args
        )
        isotope = self.method_args["channels"][0]
        Cq, eta = self._get_zeta_eta(supersampling)
        spin_systems = [
            SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))])
            for cq_, e in zip(Cq, eta)
        ]

        self.simulator.spin_systems = spin_systems
        self.simulator.methods = [method]
        self.simulator.run(pack_as_csdm=False)

        amp = self.simulator.methods[0].simulation

        return self._averaged_kernel(amp, supersampling)
