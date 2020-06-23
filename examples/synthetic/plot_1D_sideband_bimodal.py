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
coordinates = data_object.dimensions[0].coordinates
responses = data_object.dependent_variables[0].components[0]

# %%
# The corresponding sideband lineshape cross-section along with the 2D true tensor
# distribution of the synthetic dataset is shown below.

# %%
import numpy as np
import matplotlib.pyplot as plt
from mrinversion.plot import get_polar_grids

# convert the dimension from `Hz` to `ppm`.
data_object.dimensions[0].to("ppm", "nmr_frequency_ratio")

# the plot of the 1D MAF cross-section dataset.
_, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={"projection": "csdm"})
ax[0].plot(data_object)
ax[0].invert_xaxis()


# a function for 2D plot.
def twoD_plot(ax, csdm_object, title=""):
    # convert the dimension coordinates of the csdm_object from Hz to pmm.
    csdm_object.dimensions[0].to("ppm", "nmr_frequency_ratio")
    csdm_object.dimensions[1].to("ppm", "nmr_frequency_ratio")

    levels = (np.arange(9) + 1) / 10
    ax.contourf(csdm_object, cmap="gist_ncar", levels=levels)
    ax.grid(None)
    ax.set_title(title)
    get_polar_grids(ax)
    ax.set_aspect("equal")


# the plot of the true tensor distribution.
twoD_plot(ax[1], true_data_object, title="True distribution")
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
    cp.LinearDimension(count=25, increment="370 Hz", label="x"),
    cp.LinearDimension(count=25, increment="370 Hz", label="y"),
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
#
from mrinversion.linear_model import TSVDCompression

new_system = TSVDCompression(K, data_object)
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
# and :math:`\lambda` hyperparameters, and decide on the hyperparameter range
# accordingly.
from mrinversion.linear_model import SmoothLasso

# guess alpha and lambda values.
s_lasso = SmoothLasso(alpha=0.001, lambda1=5e-6, inverse_dimension=inverse_dimension)
s_lasso.fit(K=compressed_K, s=compressed_s)
f_sol = s_lasso.f

# %%
# Here, ``f_sol`` is the solution corresponding to hyperparameters :math:`\alpha=0.005`
# and :math:`\lambda=5\times 10^{-6}`. The plot of this solution follows
_, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={"projection": "csdm"})

# the plot of the tensor distribution solution.
twoD_plot(ax[0], f_sol / f_sol.max(), title="Guess distribution")

# the plot of the true tensor distribution.
twoD_plot(ax[1], true_data_object, title="True distribution")
plt.tight_layout()
plt.show()

# %%
# The predicted spectrum from the solution may be evaluated using the `predict` method
# as
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
alphas = 10 ** (-4 - 2 * (np.arange(10) / 9))

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
plt.figure(figsize=(5, 3.5))
ax = plt.subplot(projection="csdm")
ax.contour(np.log10(s_lasso_cv.cv_map), levels=25)
ax.scatter(
    -np.log10(s_lasso_cv.hyperparameter["alpha"]),
    -np.log10(s_lasso_cv.hyperparameter["lambda"]),
    marker="x",
    color="k",
)
plt.tight_layout(pad=0.5)
plt.show()

# %%
# The optimum model selection from the 10-folds cross-validation is

vector = s_lasso_cv.f

# %%
# and the corresponding plot of the model, along with the true tensor distribution
# model is shown below.
_, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={"projection": "csdm"})

# the plot of the tensor distribution solution.
twoD_plot(ax[0], vector / vector.max(), title="Optimum distribution")

# the plot of the true tensor distribution.
twoD_plot(ax[1], true_data_object, title="True distribution")
plt.tight_layout()
plt.show()