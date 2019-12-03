# coding: utf-8
# import os
# from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from csdmpy.units import string_to_quantity


__version__ = "0.0.9"


class Minimizer:
    # __slots__ = (
    #     ''
    # )
    def __init__(
        self,
        data_object=None,
        kernel=None,
        show_details=False,
        tolerance=1e-9,
        n_processors=4,
        nonnegative=True,
    ):

        self.list = [
            "nsa static",
            "T2 exponential decay",
            "T1 saturation recovery",
            "T1 inversion recovery",
        ]

        self.max_iteration = 15000
        self.iter = 1
        self.time = 0
        self.nonnegative = nonnegative
        self.oversampling = 1
        self.averageNumber = 10
        self.kernelType = kernel
        self.sigmaNoise = 1.0
        self.tolerance = tolerance
        self.npros = n_processors
        self.data_object = data_object
        self.show_details = show_details
        self.x = np.array([])
        self.inverse_vector = np.array([])
        # self.default_dimension = dimension

    def set_kernel_type(self, kernel):
        self.kernel_type = kernel

    def setup_kernel_parameters(
        self,
        kernel=None,
        n_points=None,
        minimum=None,
        maximum=None,
        scale="log",
        dimension=0,
        oversample=1,
        show_details=False,
    ):

        if kernel is not None:
            self.set_kernel_type(kernel)

        if show_details is not None:
            if isinstance(show_details, bool):
                self.show_details = show_details

        if self.kernelType == "T2 exponential decay":
            if n_points is not None:
                self.w3_nx = n_points
            if min is not None:
                self.w3x_min = string_to_quantity(minimum)
                self.unit = self.w3x_min.unit
            if max is not None:
                self.w3x_max = string_to_quantity(maximum).to(self.unit)
            if oversample is not None:
                self.w3_average = oversample
            self.w3_logOrLinear = scale
            self.w3_inversionDimension = dimension
            # svd_compress(self, self.data_object, show_plot=self.show_details)

    # def inverse(self, lambda_value, warm_start=True):
    #     s = self.maxSingularValue ** 2
    #     if warm_start:
    #         warm_f_k = np.asfortranarray(np.zeros((self.k_tilde.shape[1], 1)))
    #         zf, function, chi_square, iteri, cpu_time0, wall_time0 = fista(
    #             matrix=self.k_tilde,
    #             s=self.s_tilde.mean(axis=1),
    #             lambdaval=lambda_value,
    #             maxiter=self.max_iteration,
    #             f_k=warm_f_k,
    #             nonnegative=int(self.nonnegative),
    #             linv=(1 / s),
    #             tol=self.tolerance,
    #             npros=self.npros,
    #         )

    #         f_k_std = np.asfortranarray(
    #             np.tile(warm_f_k[:, 0], (self.s_tilde.shape[1], 1)).T
    #         )
    #     else:
    #         cpu_time0 = 0.0
    #         wall_time0 = 0.0
    #         f_k_std = np.asfortranarray(
    #             np.zeros((self.k_tilde.shape[1], self.s_tilde.shape[1]))
    #         )

    #     zf, function, chi_square, iteri, cpu_time, wall_time = fista(
    #         matrix=self.k_tilde,
    #         s=self.s_tilde,
    #         lambdaval=lambda_value,
    #         maxiter=self.max_iteration,
    #         f_k=f_k_std,
    #         nonnegative=int(self.nonnegative),
    #         linv=(1 / s),
    #         tol=self.tolerance,
    #         npros=self.npros,
    #     )

    #     print("Solving for T2 vector at lambda = {0}".format(lambda_value))
    #     print("cpu time = {0} s ".format(cpu_time + cpu_time0))
    #     print("wall time = {0} s".format(wall_time + wall_time0))
    #     print("number of iterations = {0}".format(iteri))

    #     fit = np.dot(self.U, zf)

    #     self.plot_data(f_k_std)

    #     self.inverse_vector = f_k_std
    #     self.fit = fit
    #     self.chi_square = chi_square
    #     self.function = function

    #     def print_report(self):
    #         if self.nonnegative == 1:
    #             cnst = 'True'
    #         else:
    #             cnst = 'False'
    # #         if len(data_object.dimensions)==1:

    #         self.reconstructedFit = np.dot(self.U, np.dot(np.diag(self.S), \
    #                                         np.dot(self.VT, self.f_k)))
    #         chi2 = ((self.data_object.signal.real-self.reconstructedFit)**2).sum()/
    #                           self.sigmaNoise**2
    #         std =  (np.squeeze(self.data_object.signal.real)  - \
    #                    np.squeeze(self.reconstructedFit)).std(ddof=1)

    #         print ('Non negative constrain           :  ', cnst)
    #         print ('chi-square                       :  ', chi2 )
    #         print ('standard deviation from residue  :  ', std )
    #         print ('standard deviation from input    :  ', self.sigmaNoise)
    #         print ('Iterations                       :  ', self.max_iteration)
    #         print ('Iterations performed             :  ', self.iter)
    #         print ('Lambda determined                :  ', self.lambd)
    #         print ('Execution time for inversion     :  ', self.time, ' s')

    def plot_data(self, f_k_std):
        _inversion_index = self.w3_inversionDimension
        _x = self.data_object.dimensions[_inversion_index - 1].value
        _y = self.x
        _unit = self.data_object.dimensions[_inversion_index].unit
        plt.contourf(_x, _y, f_k_std, origin="lower", cmap="viridis")
        plt.ylabel("log ($T_2$ / {0})".format(str(_unit)))
        plt.xlabel("dimension-{0}".format(str(_inversion_index - 1)))
        plt.show()
