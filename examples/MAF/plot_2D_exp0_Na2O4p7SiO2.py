#!/usr/bin/env python
# coding: utf-8
"""
2D MAF of Na2O 4.7SiO2 glass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
#%%
# The following example demonstrates the statistical learning of nuclear shielding
# tensor distribution from the 2D magic-angle flipping (MAF) spectrum. In this example,
# we use the 2D MAF spectrum of :math:`\text{Na}_2\text{O}\cdot4.7\text{SiO}_2` glass.
#
# Setup for figure.
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams["figure.figsize"] = 4, 3
rcParams["font.size"] = 9

#%%
# Import the dataset
# ------------------
#
# Load the example file as the CSDM file-object.
import csdmpy as cp

from mrinversion import examples

# the 2D MAF dataset in csdm format
data_object = cp.load(examples.exp1)

#%%
# The variable ``data_object`` is a `CSDM <https://csdmpy.readthedocs.io/en/latest/api/CSDM.html>`_
# object that holds the coordinates and the
# responses of the 2D MAF dataset. The plot of the MAF dataset is
cp.plot(data_object, cmap="gist_ncar_r")

#%%
# There are two dimensions in this dataset, where the dimension at index 0 is the
# isotropic chemical shift dimension, while the dimension at index 1 is the pure
# anisotropic dimension. The number of coordinates along each dimension is
print(data_object.shape)

#%%
# respectively.
#
# Notice, on this two-dimensional frequency grid, the signal
# only occupies a small fraction of the grid. It is best to truncate the dataset to
# the desired region before proceeding,
data_object_truncated = data_object[158:173, :]
cp.plot(data_object_truncated, cmap="gist_ncar_r")

#%%
# where we have truncated the dataset along the dimension at index 0, that is, the
# isotropic chemical shift dimension. The coordinates and responses from this truncated
# dataset are

# along the isotropic dimension
coordinates1 = data_object_truncated.dimensions[0].coordinates
# along the pure anisotropic dimension
coordinates2 = data_object_truncated.dimensions[1].coordinates
responses = data_object_truncated.dependent_variables[0].components[0]


#%%
# Set the anisotropic and inverse-dimension
# -----------------------------------------
#
# **The anisotropic-dimension**
#
# The anisotropic dimension of the 2D MAF dataset is the dimension at the index 1.

anisotropic_dimension = data_object_truncated.dimensions[1]

#%%
# **Inverse-dimension**
#
# The two inverse dimensions corresponds to the `x` and `y`-axis of the
# `x`-`y` grid.

inverse_dimension = [
    cp.LinearDimension(count=25, increment="370 Hz"),  # along the `x`-dimension.
    cp.LinearDimension(count=25, increment="370 Hz"),  # along the `y`-dimension.
]

#%%
# Generate the line-shape kernel
# ------------------------------

from mrinversion.kernel import NuclearShieldingTensor

method = NuclearShieldingTensor(
    anisotropic_dimension=anisotropic_dimension,
    inverse_dimension=inverse_dimension,
    isotope="29Si",
    magnetic_flux_density="9.4 T",
    rotor_angle="90 deg",
    rotor_frequency="14 kHz",
    number_of_sidebands=1,
)

#%%
# The above code generates an instance of the NuclearShieldingTensor class assigned
# to the variable ``method``.
# The two required arguments of this class are the `anisotropic_dimension` and
# `inverse_dimension`, as previously defined.
# The value of the remaining optional attributes such as the isotope, magnetic flux
# density, rotor angle, and rotor frequency is set to match the conditions under which
# the MAF spectrum was acquired. Note, for the MAF measurements, the rotor angle is
# usually :math:`90^\circ` for the anisotropic dimension. Once the
# NuclearShieldingTensor instance is created, use the kernel() method to generate
# the MAF lineshape kernel.
K = method.kernel(supersampling=1)
print(K.shape)

#%%
# The kernel ``K`` is a NumPy array of shape (128, 625), where the axis with 128 points
# corresponds to the anisotropic dimension, and the axis with 625 points are the features
# corresponding to the `x`-`y` coordinates.

#%%
# Data Compression
# ----------------

#%%
from mrinversion.linear_model import TSVDCompression

new_system = TSVDCompression(K, responses)
compressed_K = new_system.compressed_K
compressed_s = new_system.compressed_s

#%%
# Set up the inverse problem
# --------------------------
#
# Solve the smooth-lasso problem. You may skip this step and proceed to the
# statistical learning method. The statistical learning method is a
# computationally expensive process that requires you to provide a range of predefined
# hyperparameters.
# If you are unsure what range of hyperparameters to use, you can use this step for
# a quick look into the possible solution, given a guess value for the :math:`\alpha`
# and :math:`\lambda` hyperparameters, and decide on the hyperparameter range accordingly.

from mrinversion.linear_model import SmoothLasso

# guess alpha and lambda values.
s_lasso = SmoothLasso(alpha=1e-3, lambda1=5e-7, inverse_dimension=inverse_dimension)
s_lasso.fit(K=compressed_K, s=compressed_s)
f_sol = s_lasso.f
f_sol_sum_iso = f_sol.sum(axis=0)

#%%
# Here, ``f_sol`` is the solution corresponding to the hyperparameters :math:`\alpha=0.001`
# and :math:`\lambda=5\times 10^{-7}`. The plot of this solution follows

import numpy as np
import matplotlib.pyplot as plt
from mrinversion.plot import get_polar_grids

plt.figure(figsize=(3, 3))

# the plot of the tensor distribution solution.
inverse_dimension[0].to("ppm", "nmr_frequency_ratio")  # convert unit to ppm from Hz
inverse_dimension[1].to("ppm", "nmr_frequency_ratio")  # convert unit to ppm from Hz
x = inverse_dimension[0].coordinates  # the x coordinates
y = inverse_dimension[1].coordinates  # the y coordinates

levels = (np.arange(9) + 1) / 10
plt.contourf(x, y, f_sol_sum_iso / f_sol_sum_iso.max(), cmap="gist_ncar", levels=levels)
plt.xlim(0, 100)
plt.ylim(0, 100)
get_polar_grids(plt.gca())  # places a polar zeta-eta grid on the background.

plt.xlabel(inverse_dimension[0].axis_label)  # the x label
plt.ylabel(inverse_dimension[1].axis_label)  # the y label
plt.gca().set_aspect("equal")

# the plot of the true tensor distribution.
plt.tight_layout()
plt.show()

#%%
# The predicted spectrum from the solution may be evaluated using the `predict` method as

#%%
predicted_spectrum = s_lasso.predict(K)

fig, ax = plt.subplots(1, 2, figsize=(7, 3))
# the original spectrum
ax[0].imshow(responses.real, aspect="auto", origin="lower")
# the predicted spectrum
ax[1].imshow(predicted_spectrum, aspect="auto", origin="lower")
# plt.xlabel("frequency / Hz")
# plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()


# #%%
# # Statistical learning of the tensors
# # -----------------------------------
# #
# # Create a guess range of values for the :math:`\alpha` and :math:`\lambda`
# # hyperparameters.
# # The following code generates a range of :math:`\lambda` values sampled uniformly on a
# # log scale and ranging from :math:`10^{-5}` to :math:`10^{-7}`. The range for the
# # hyperparameter :math:`\alpha` is similarly sampled uniformly on a log scale, ranging
# # from :math:`10^{-2.5}` to :math:`10^{-4.5}`.

# lambdas = 10 ** (-5 - 2 * (np.arange(10) / 9))
# alphas = 10 ** (-2.5 - 2 * (np.arange(10) / 9))

# #%%
# from mrinversion.linear_model import SmoothLassoCV

# s_lasso_cv = SmoothLassoCV(alphas=alphas, lambdas=lambdas, sigma=0.005, folds=10)
# # solve the smooth lasso problem over the range of alpha and lambda values, using the
# # 10-folds cross-validation.
# s_lasso_cv.fit(compressed_K, compressed_s, f_shape=(25, 25))

# #%%
# # The optimized hyperparameters from the 10-folds cross-validation are
# s_lasso_cv.hyperparameter

# #%%
# # and the corresponding cross-validation metric (mean square error, MSE), follows

# plt.figure(figsize=(4, 3))
# plt.contour(
#     -np.log10(lambdas), -np.log10(alphas), np.log10(s_lasso_cv.cv_map), levels=25
# )
# plt.scatter(
#     -np.log10(s_lasso_cv.hyperparameter["lambda"]),
#     -np.log10(s_lasso_cv.hyperparameter["alpha"]),
#     marker="x",
#     color="k",
# )
# plt.xlabel(r"$-\log~\lambda$")
# plt.ylabel(r"$-\log~\alpha$")
# plt.tight_layout()
# plt.show()


# #%%
# # The optimum model selection from the 10-folds cross-validation is

# vector = s_lasso_cv.f

# #%%
# # and the corresponding plot of the model, along with the true tensor distribution
# # model is shown below.


# fig, ax = plt.subplots(1, 2, figsize=(6, 3))

# # plot of the selected model
# inverse_dimension[0].to("ppm", "nmr_frequency_ratio")
# inverse_dimension[1].to("ppm", "nmr_frequency_ratio")
# inv1 = inverse_dimension[0].coordinates
# inv2 = inverse_dimension[1].coordinates
# levels = (np.arange(9) + 1) / 10

# ax[0].contourf(inv1, inv2, vector / vector.max(), cmap="gist_ncar", levels=levels)

# ax[0].set_xlabel(inverse_dimension[0].axis_label)
# ax[0].set_ylabel(inverse_dimension[1].axis_label)
# ax[0].set_xlim(0, 100)
# ax[0].set_ylim(0, 100)
# get_polar_grids(ax[0])
# ax[0].set_aspect("equal")


# # the plot of the true tensor distribution.
# plot_true_distribution(ax[1])
# plt.tight_layout()
# plt.show()
