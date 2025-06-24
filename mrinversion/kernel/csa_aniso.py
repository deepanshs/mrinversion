from copy import deepcopy

import csdmpy as cp
import numpy as np
from mrsimulator.utils import get_spectral_dimensions

from mrsimulator import Simulator
from mrsimulator import SpinSystem
from mrsimulator.method import Method, SpectralEvent
from mrsimulator.method.lib import BlochDecaySpectrum

from mrinversion.kernel.base import LineShape


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
        args_ = deepcopy(self.method_args)
        method = BlochDecaySpectrum.parse_dict_with_units(args_)
        isotope = args_["channels"][0]
        zeta, eta = self._get_zeta_eta(supersampling)

        x_csdm = self.inverse_kernel_dimension[0]
        if x_csdm.coordinates.unit.physical_type == "frequency":
            # convert zeta to ppm if given in frequency units.
            zeta /= self.larmor_frequency  # zeta in ppm

            for dim_i in self.inverse_kernel_dimension:
                if dim_i.origin_offset.value == 0:
                    dim_i.origin_offset = f"{abs(self.larmor_frequency)} MHz"

        spin_systems = [
            SpinSystem(
                sites=[dict(isotope=isotope, shielding_symmetric=dict(zeta=z, eta=e))]
            )
            for z, e in zip(zeta, eta)
        ]

        sim = Simulator()
        sim.config.number_of_sidebands = self.number_of_sidebands
        sim.config.decompose_spectrum = "spin_system"

        sim.spin_systems = spin_systems
        sim.methods = [method]
        sim.run(pack_as_csdm=False)

        amp = sim.methods[0].simulation.real
        return self._averaged_kernel(amp, supersampling)


class MAF(ShieldingPALineshape):
    r"""A specialized class for simulating the pure anisotropic NMR nuclear shielding
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
    r"""A specialized class for simulating the pure anisotropic spinning sideband
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

    def kernel(self, supersampling, eta_bound = 1):
        # update method for DAS spectra events
        das_event = dict(
            transition_queries=[{"ch1": {"P": [-1], "D": [0]}}],
            freq_contrib=["Quad2_4"],
        )
        self.method_args["spectral_dimensions"][0]["events"] = [das_event]

        method = Method.parse_dict_with_units(self.method_args)
        isotope = self.method_args["channels"][0]
        
        if eta_bound == 1:
            Cq, eta = self._get_zeta_eta(supersampling, eta_bound)
        else: 
            Cq, eta, abundances = self._get_zeta_eta(supersampling, eta_bound)
       
        if eta_bound == 1:
            spin_systems = [
                SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))])
                for cq_, e in zip(Cq, eta)
            ]
        else:
            spin_systems = [
                SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))], abundance=abun)
                for cq_, e,abun in zip(Cq, eta,abundances)
            ]
        sim = Simulator()
        sim.config.number_of_sidebands = self.number_of_sidebands
        sim.config.decompose_spectrum = "spin_system"

        sim.spin_systems = spin_systems
        sim.methods = [method]
        sim.run(pack_as_csdm=False)

        amp = sim.methods[0].simulation.real
        
        return self._averaged_kernel(amp, supersampling)


class MQMAS(LineShape):
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

    def kernel(self, supersampling, eta_bound = 1):
        # update method for DAS spectra events
        mqmas_events = [
            dict(
                fraction= -9/50,
                transition_queries=[{"ch1": {"P": [-3], "D": [0]}}]
            ),
            dict(
                fraction= 27/50,
                transition_queries=[{"ch1": {"P": [-1], "D": [0]}}]
            ),
        ]
        
        self.method_args["spectral_dimensions"][0]["events"] = mqmas_events

        method = Method.parse_dict_with_units(self.method_args)
        isotope = self.method_args["channels"][0]
        
        if eta_bound == 1:
            Cq, eta = self._get_zeta_eta(supersampling, eta_bound)
        else: 
            Cq, eta, abundances = self._get_zeta_eta(supersampling, eta_bound)
       
        if eta_bound == 1:
            spin_systems = [
                SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))])
                for cq_, e in zip(Cq, eta)
            ]
        else:
            spin_systems = [
                SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))], abundance=abun)
                for cq_, e,abun in zip(Cq, eta,abundances)
            ]
        sim = Simulator()
        sim.config.number_of_sidebands = self.number_of_sidebands
        sim.config.decompose_spectrum = "spin_system"

        sim.spin_systems = spin_systems
        sim.methods = [method]
        sim.run(pack_as_csdm=False)

        amp = sim.methods[0].simulation.real
        print(f'amp shape: {amp.shape}')
        return self._averaged_kernel(amp, supersampling)
    

class SL_MQMASnodist(LineShape):
    def __init__(
        self,
        anisotropic_dimension,
        inverse_kernel_dimension,
        channel,
        exp_dict,
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
            #
            #  "DAS",
        )
        self.exp_dict = exp_dict
        self.anisotropic_dimension = anisotropic_dimension

    def kernel(self, supersampling, eta_bound = 1):
        import sys
        sys.path.insert(0, '/home/lexicon2810/github-repos-WSL/mrsmqmas')
        # import src.processing as smproc
        import src.simulation as smsim
        # import src.fitting as smfit
        
        isotope = self.method_args["channels"][0]
        if eta_bound == 1:
            Cq, eta = self._get_zeta_eta(supersampling, eta_bound)
        else: 
            Cq, eta, abundances = self._get_zeta_eta(supersampling, eta_bound)

        # print(f'Cq: {Cq}')
        # print(f'eta: {eta}')
        if eta_bound == 1:
            spin_systems = [
                SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))])
                for cq_, e in zip(Cq, eta)
            ]
        else:
            spin_systems = [
                SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))], abundance=abun)
                for cq_, e,abun in zip(Cq, eta,abundances)
            ]
        # print(self.anisotropic_dimension)
        
        obj = cp.CSDM(dimensions=[self.anisotropic_dimension])
        spec_dim = get_spectral_dimensions(obj)
        # print(obj.x[0])
        # print(self.anisotropic_dimension)
        # print(spec_dim)

        amp = np.asarray([smsim.simulate_onesite_lineshape(
            self.exp_dict, 
            mysys, 
            spec_dim[0], 
            input_type='c0_c4', 
            contribs='c0_c4', 
            return_array=True,
            distorted=False) for mysys in spin_systems])
        # sim = Simulator()
        # sim.config.number_of_sidebands = self.number_of_sidebands
        # sim.config.decompose_spectrum = "spin_system"

        # sim.spin_systems = spin_systems
        # sim.methods = [method]
        # sim.run(pack_as_csdm=False)
        # obj_pre = cp.CSDM(
        #     dimensions=[
        #         aniso
        #     ]
        # )
        # amp = sim.methods[0].simulation.real
        # print(f'amp shape: {amp.shape}')
        return self._averaged_kernel(amp, supersampling)
        

class SL_MQMAS(LineShape):
    def __init__(
        self,
        anisotropic_dimension,
        inverse_kernel_dimension,
        channel,
        exp_dict,
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
            #
            #  "DAS",
        )
        self.exp_dict = exp_dict
        self.anisotropic_dimension = anisotropic_dimension

    def kernel(self, supersampling, eta_bound = 1):
        import sys
        sys.path.insert(0, '/home/lexicon2810/github-repos-WSL/mrsmqmas')
        # import src.processing as smproc
        import src.simulation as smsim
        # import src.fitting as smfit
        
        isotope = self.method_args["channels"][0]
        if eta_bound == 1:
            Cq, eta = self._get_zeta_eta(supersampling, eta_bound)
        else: 
            Cq, eta, abundances = self._get_zeta_eta(supersampling, eta_bound)
        
        print(f'Cq: {Cq}')
        print(f'eta: {eta}')
        if eta_bound == 1:
            spin_systems = [
                SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))])
                for cq_, e in zip(Cq, eta)
            ]
        else:
            spin_systems = [
                SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=cq_, eta=e))], abundance=abun)
                for cq_, e,abun in zip(Cq, eta,abundances)
            ]
        # print(self.anisotropic_dimension)
        
        obj = cp.CSDM(dimensions=[self.anisotropic_dimension])
        spec_dim = get_spectral_dimensions(obj)

        amp = np.asarray([smsim.simulate_onesite_lineshape(
            self.exp_dict, 
            mysys, 
            spec_dim[0], 
            input_type='c0_c4', 
            contribs='c0_c4', 
            return_array=True,
            distorted=True) for mysys in spin_systems])
        # sim = Simulator()
        # sim.config.number_of_sidebands = self.number_of_sidebands
        # sim.config.decompose_spectrum = "spin_system"

        # sim.spin_systems = spin_systems
        # sim.methods = [method]
        # sim.run(pack_as_csdm=False)

        # amp = sim.methods[0].simulation.real
        print(f'amp shape: {amp.shape}')
        return self._averaged_kernel(amp, supersampling)
    