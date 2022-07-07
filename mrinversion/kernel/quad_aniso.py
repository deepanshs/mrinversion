# from copy import deepcopy
# import numpy as np
# from mrsimulator import Simulator
# from mrsimulator import SpinSystem
# from mrsimulator.method import Method
# from mrinversion.kernel.base import LineShape
# class MQMAS(LineShape):
#     """
#     A generalized class for simulating the pure anisotropic NMR nuclear shielding
#     line-shape kernel.
#     Args:
#         anisotropic_dimension: A Dimension object, or an equivalent dictionary
#                 object. This dimension must represent the pure anisotropic
#                 dimension.
#         inverse_dimension: A list of two Dimension objects, or equivalent
#                 dictionary objects representing the `x`-`y` coordinate grid.
#         channel: The channel is an isotope symbol of the nuclei given as the atomic
#             number followed by the atomic symbol, for example, `1H`, `13C`, and
#             `29Si`. This nucleus must correspond to the recorded frequency
#             resonances.
#         magnetic_flux_density: The magnetic flux density of the external static
#                 magnetic field. The default value is 9.4 T.
#     """
#     def __init__(
#         self,
#         anisotropic_dimension,
#         inverse_dimension,
#         channel,
#         magnetic_flux_density="9.4 T",
#     ):
#         super().__init__(
#             anisotropic_dimension,
#             inverse_dimension,
#             channel,
#             magnetic_flux_density,
#             rotor_angle=54.7356 * np.pi / 180,
#             rotor_frequency=1e9,
#             number_of_sidebands=1,
#         )
#     def kernel(self, supersampling=1):
#         """
#         Return the quadrupolar anisotropic line-shape kernel.
#         Args:
#             supersampling: An integer. Each cell is supersampled by the factor
#                     `supersampling` along every dimension.
#         Returns:
#             A numpy array containing the line-shape kernel.
#         """
#         self.inverse_kernel_dimension[0].application["half"] = True
#         args_ = deepcopy(self.method_args)
#         args_["spectral_dimensions"][0]["events"] = [
#             {"fraction": 27 / 17, "freq_contrib": ["Quad2_0"]},
#             {"fraction": 1, "freq_contrib": ["Quad2_4"]},
#         ]
#         method = Method.parse_dict_with_units(args_)
#         isotope = args_["channels"][0]
#         zeta, eta = self._get_zeta_eta(supersampling)
#         # new_size = zeta.size
#         # n1, n2 = [item.count for item in self.inverse_kernel_dimension]
#         # index = []
#         # for i in range(n2):
#         #     i_ = i * supersampling
#         #     for j in range(supersampling):
#         #         index = np.append(index, np.arange(n1 - i_) + (i_ + j) * n1 + i_)
#         # index = np.asarray(index, dtype=int)
#         # print(index)
#         # zeta = zeta[index]
#         # eta = eta[index]
#         # larmor frequency from method.
#         B0 = method.spectral_dimensions[0].events[0].magnetic_flux_density  # in T
#         gamma = method.channels[0].gyromagnetic_ratio  # in MHz/T
#         larmor_frequency = -gamma * B0  # in MHz
#         for dim_i in self.inverse_kernel_dimension:
#             if dim_i.origin_offset.value == 0:
#                 dim_i.origin_offset = f"{abs(larmor_frequency)} MHz"
#         spin_systems = [
#             SpinSystem(sites=[dict(isotope=isotope, quadrupolar=dict(Cq=z, eta=e))])
#             for z, e in zip(zeta, eta)
#         ]
#         dim = method.spectral_dimensions[0]
#         if dim.origin_offset == 0:
#             dim.origin_offset = larmor_frequency * 1e6  # in Hz
#         sim = Simulator()
#         sim.config.number_of_sidebands = self.number_of_sidebands
#         sim.config.decompose_spectrum = "spin_system"
#         sim.spin_systems = spin_systems
#         sim.methods = [method]
#         sim.run(pack_as_csdm=False)
#         amp = sim.methods[0].simulation.real
#         # amp2 = np.zeros((new_size, amp.shape[1]))
#         # amp2 = (amp.T + amp) / 2.0
#         # amp2[index] = amp
#         # print(amp2.shape, amp.shape)
#         kernel = self._averaged_kernel(amp, supersampling)
#         # print(kernel.shape)
#         n1, n2 = [item.count for item in self.inverse_kernel_dimension]
#         shape = kernel.shape[0]
#         kernel.shape = (shape, n1, n2)
#         for i in range(n1):
#             for j in range(n2):
#                 if i > j:
#                     kernel[:, i, j] = 0
#         kernel.shape = (shape, n1 * n2)
#         # if not half:
#         #     return kernel
#         index = np.where(kernel.sum(axis=0) != 0)[0]
#         self.inverse_kernel_dimension[0].application["index"] = index.tolist()
#         return kernel[:, index]
#         #
#         # return kernel.reshape(shape, n1 * n2)
