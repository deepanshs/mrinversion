#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bimodal distribution
^^^^^^^^^^^^^^^^^^^^
"""
# %%
# The following example demonstrates the statistical learning based nuclear shielding
# tensor distribution from the spinning sideband correlation measurement. In this
# example, we use a synthetic sideband amplitude spectrum from a bimodal tensor
# distribution.
#
# Import the dataset
# ------------------
#
# Importing the example file as the CSDM file-object.
import csdmpy as cp

# the 1D sideband amplitude cross-section data in csdm format
filename = "https://osu.box.com/shared/static/wjbhb6sif76mxfgndetew8mnrq6pw4pj.csdf"
data_object = cp.load(filename)

# the true tensor distribution for comparison
datafile = "https://osu.box.com/shared/static/xesah85nd2gtm9yefazmladi697khuwi.csdf"
true_data_object = cp.load(datafile)

# %%
# The variable ``data_object`` holds the coordinates and the responses of the 1D
# sideband cross-section, which are

# %%
coordinates = data_object.dimensions[0].coordinates
responses = data_object.dependent_variables[0].components[0]

# %%
# The corresponding sideband lineshape cross-section along with the 2D true tensor
# distribution of the synthetic dataset is shown below.

# %%
import numpy as np
import matplotlib.pyplot as plt
from mrinversion.plot import get_polar_grids

# the plot of the 1D sideband cross-section dataset.
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].plot(coordinates, responses)
ax[0].invert_xaxis()
ax[0].set_xlabel(data_object.dimensions[0].axis_label)

# the plot of the true tensor distribution.


def plot_true_distribution(ax):
    # convert the dimension coordinates of the true_data_object from Hz to pmm.
    true_data_object.dimensions[0].to("ppm", "nmr_frequency_ratio")
    true_data_object.dimensions[1].to("ppm", "nmr_frequency_ratio")

    # the coordinates along the x and y dimensions
    x_t = true_data_object.dimensions[0].coordinates
    y_t = true_data_object.dimensions[1].coordinates

    # the true tensor distribution
    true_tensor_dist = true_data_object.dependent_variables[0].components[0]

    # plot
    levels = (np.arange(9) + 1) / 10
    ax.contourf(
        x_t,
        y_t,
        true_tensor_dist / true_tensor_dist.max(),
        cmap="gist_ncar",
        levels=levels,
    )
    ax.set_xlabel(true_data_object.dimensions[0].axis_label)
    ax.set_ylabel(true_data_object.dimensions[1].axis_label)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    get_polar_grids(ax)
    ax.set_aspect("equal")


plot_true_distribution(ax[1])
plt.tight_layout()
plt.show()

# %%
# Set the direct and inverse-dimension
# ------------------------------------
#
# **The direct-dimension**
#
# The direct dimension is the pure anisotropic dimension, which in this case, is the
# only dimension.

anisotropic_dimension = cp.LinearDimension(
    count=32, increment="625 Hz", coordinates_offset="-10 kHz"
)

# %%
# **Indirect-dimension**
#
# The two inverse dimensions are corresponding to the `x` and `y`-axis of the
# `x`-`y` grid.

inverse_dimension = [
    cp.LinearDimension(count=25, increment="370 Hz"),  # along the `x`-dimension.
    cp.LinearDimension(count=25, increment="370 Hz"),  # along the `y`-dimension.
]

# %%
# Generate the line-shape kernel
# ------------------------------
#
# The following code creates a NuclearShieldingLineshape class object called ``method``.
# The two required arguments of this class are the direct and inverse dimensions, which
# are the key in generating the transformation kernel, transforming the data on the
# direct dimension to the data on the inverse-dimensions.
# The value of the remaining optional attributes such as the channel, magnetic flux
# density, rotor angle, and rotor frequency is set to match the conditions under which
# the sideband spectrum was acquired. Note, the rotor frequency is the effective
# anisotropic modulation frequency. This frequency may be less than the actual physical
# rotor frequency. The number of sidebands is usually the number of points along the
# sideband dimension. Once the NuclearShieldingLineshape instance is created, use the
# kernel() method to generate the sideband amplitude lineshape kernel.

# %%
from mrinversion.kernel import NuclearShieldingLineshape

method = NuclearShieldingLineshape(
    anisotropic_dimension=anisotropic_dimension,
    inverse_dimension=inverse_dimension,
    channel="29Si",
    magnetic_flux_density="9.4 T",
    rotor_angle="54.735 deg",
    rotor_frequency="625 Hz",
    number_of_sidebands=32,
)
K = method.kernel(supersampling=1)

# %%
# Data Compression
# ----------------

# %%
from mrinversion.linear_model import TSVDCompression

new_system = TSVDCompression(K, responses)
compressed_K = new_system.compressed_K
compressed_s = new_system.compressed_s

# %%
# Set up the inverse problem
# --------------------------
#
# Solve the smooth-lasso problem. You may skip this step and proceed to the
# statistical learning method. Usually, the statistical learning method is a
# time-consuming process that requires you to provide a range of predefined
# hyperparameters.
# If you are unsure what range of hyperparameters to use, you can use this step for
# a quick look into the possible solution, given a guess value for the :math:`\alpha`
# and :math:`\lambda` hyperparameters, and decide on the hyperparameter range accordingly.

from mrinversion.linear_model import SmoothLasso

# guess alpha and lambda values.
s_lasso = SmoothLasso(alpha=0.005, lambda1=5e-6, inverse_dimension=inverse_dimension)
s_lasso.fit(K=compressed_K, s=compressed_s)
f_sol = s_lasso.f

# %%
# Here, ``f_sol`` is the solution corresponding to hyperparameters :math:`\alpha=0.005`
# and :math:`\lambda=5\times 10^{-6}`. The plot of this solution follows

fig, ax = plt.subplots(1, 2, figsize=(6, 3))

# the plot of the tensor distribution solution.
inverse_dimension[0].to("ppm", "nmr_frequency_ratio")  # convert unit to ppm from Hz
inverse_dimension[1].to("ppm", "nmr_frequency_ratio")  # convert unit to ppm from Hz
x = inverse_dimension[0].coordinates  # the x coordinates
y = inverse_dimension[1].coordinates  # the y coordinates

levels = (np.arange(9) + 1) / 10
ax[0].contourf(x, y, f_sol / f_sol.max(), cmap="gist_ncar", levels=levels)
ax[0].set_xlim(0, 100)
ax[0].set_ylim(0, 100)
get_polar_grids(ax[0])  # places a polar zeta-eta grid on the background.

ax[0].set_xlabel(inverse_dimension[0].axis_label)  # the x label
ax[0].set_ylabel(inverse_dimension[1].axis_label)  # the y label
ax[0].set_aspect("equal")

# the plot of the true tensor distribution.
plot_true_distribution(ax[1])
plt.tight_layout()
plt.show()

# %%
# The predicted spectrum from the solution may be evaluated using the `predict` method as

# %%
predicted_spectrum = s_lasso.predict(K)

plt.figure(figsize=(4, 3))
plt.plot(coordinates, responses, color="black")  # the original spectrum
plt.plot(coordinates, predicted_spectrum, color="red")  # the predicted spectrum
plt.xlabel("frequency / Hz")
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

# %%
# Statistical learning of the tensors
# -----------------------------------
#
# Create a guess range of values for the :math:`\alpha` and :math:`\lambda`
# hyperparameters.
# The following code generates a range of :math:`\lambda` values sampled uniformly on a
# log scale and ranging from :math:`10^{-5}` to :math:`10^{-7}`. The range for the
# hyperparameter :math:`\alpha` is similarly sampled uniformly on a log scale, ranging
# from :math:`10^{-2.5}` to :math:`10^{-4.5}`.

lambdas = 10 ** (-5 - 2 * (np.arange(10) / 9))
alphas = 10 ** (-2.5 - 2 * (np.arange(10) / 9))

# %%
from mrinversion.linear_model import SmoothLassoCV

s_lasso_cv = SmoothLassoCV(
    alphas=alphas,
    lambdas=lambdas,
    inverse_dimension=inverse_dimension,
    sigma=0.005,
    folds=10,
)
# solve the smooth lasso problem over the range of alpha and lambda values, using the
# 10-folds cross-validation.
s_lasso_cv.fit(compressed_K, compressed_s)

# %%
# The optimized hyperparameters from the 10-folds cross-validation are
print(s_lasso_cv.hyperparameter)

# %%
# and the corresponding cross-validation metric (mean square error, MSE), follows

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

# %%
# The optimum model selection from the 10-folds cross-validation is

vector = s_lasso_cv.f

# %%
# and the corresponding plot of the model, along with the true tensor distribution
# model is shown below.

fig, ax = plt.subplots(1, 2, figsize=(6, 3))

# plot of the selected model
inverse_dimension[0].to("ppm", "nmr_frequency_ratio")
inverse_dimension[1].to("ppm", "nmr_frequency_ratio")
inv1 = inverse_dimension[0].coordinates
inv2 = inverse_dimension[1].coordinates
levels = (np.arange(9) + 1) / 10

ax[0].contourf(inv1, inv2, vector / vector.max(), cmap="gist_ncar", levels=levels)

ax[0].set_xlabel(inverse_dimension[0].axis_label)
ax[0].set_ylabel(inverse_dimension[1].axis_label)
ax[0].set_xlim(0, 100)
ax[0].set_ylim(0, 100)
get_polar_grids(ax[0])
ax[0].set_aspect("equal")

# the plot of the true tensor distribution.
plot_true_distribution(ax[1])
plt.tight_layout()
plt.show()
