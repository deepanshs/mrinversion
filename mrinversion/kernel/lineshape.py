# -*- coding: utf-8 -*-
from mrsimulator import Simulator
from mrsimulator import SpinSystem
from mrsimulator.methods import BlochDecaySpectrum

from .utils import _x_y_to_zeta_eta_distribution
from mrinversion.kernel.base import _check_dimension_type
from mrinversion.kernel.base import BaseModel


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

        zeta, eta = _x_y_to_zeta_eta_distribution(
            self.inverse_kernel_dimension, supersampling
        )
        return zeta, eta


class ShieldingPALineshape(LineShape):
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

        x_csdm = self.inverse_kernel_dimension[0]
        if x_csdm.coordinates.unit.physical_type == "frequency":

            # larmor frequency from method.
            B0 = method.spectral_dimensions[0].events[0].magnetic_flux_density  # in T
            gamma = method.channels[0].gyromagnetic_ratio  # in MHz/T
            larmor_frequency = -gamma * B0  # in MHz

            # convert zeta to ppm if given in frequency units.
            zeta /= larmor_frequency  # zeta in ppm

            for dim_i in self.inverse_kernel_dimension:
                if dim_i.origin_offset.value == 0:
                    dim_i.origin_offset = f"{abs(larmor_frequency)} MHz"

        spin_systems = [
            SpinSystem(
                sites=[dict(isotope=isotope, shielding_symmetric=dict(zeta=z, eta=e))]
            )
            for z, e in zip(zeta, eta)
        ]

        dim = method.spectral_dimensions[0]
        if dim.origin_offset == 0:
            dim.origin_offset = larmor_frequency * 1e6  # in Hz

        self.simulator.spin_systems = spin_systems
        self.simulator.methods = [method]
        self.simulator.run(pack_as_csdm=False)

        amp = self.simulator.methods[0].simulation

        return self._averaged_kernel(amp, supersampling)


class MAF(ShieldingPALineshape):
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


class SpinningSidebands(ShieldingPALineshape):
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


# class DAS(LineShape):
#     def __init__(
#         self,
#         anisotropic_dimension,
#         inverse_kernel_dimension,
#         channel,
#         magnetic_flux_density="9.4 T",
#         rotor_angle="54.735 deg",
#         rotor_frequency="600 Hz",
#         number_of_sidebands=None,
#     ):
#         super().__init__(
#             anisotropic_dimension,
#             inverse_kernel_dimension,
#             channel,
#             magnetic_flux_density,
#             rotor_angle,
#             rotor_frequency,
#             number_of_sidebands,
#             # "DAS",
#         )

#     def kernel(self, supersampling):
#         method = BlochDecayCentralTransitionSpectrum.parse_dict_with_units(
#             self.method_args
#         )
#         isotope = self.method_args["channels"][0]
#         Cq, eta = self._get_zeta_eta(supersampling)
#         spin_systems = [
#             SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))])
#             for cq_, e in zip(Cq, eta)
#         ]

#         self.simulator.spin_systems = spin_systems
#         self.simulator.methods = [method]
#         self.simulator.run(pack_as_csdm=False)

#         amp = self.simulator.methods[0].simulation

#         return self._averaged_kernel(amp, supersampling)
