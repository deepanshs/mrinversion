#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D MAF of MgO.SiO2 glass
^^^^^^^^^^^^^^^^^^^^^^^^
"""
# %%
# The following example illustrates an application of the statistical learning method
# applied to determine the distribution of the nuclear shielding tensor parameters from
# a 2D magic-angle flipping (MAF) spectrum. In this example,
# we use the 2D MAF spectrum [#f1]_ of :math:`2\text{MgO}\cdot\text{SiO}_2`
# glass.
#
# Before getting started
# ----------------------
#
# Import all relevant packages.
import csdmpy as cp
import matplotlib.pyplot as plt
from pylab import rcParams

from mrinversion.kernel import NuclearShieldingLineshape
from mrinversion.linear_model import SmoothLasso
from mrinversion.linear_model import TSVDCompression
from mrinversion.plot import plot_3d

# sphinx_gallery_thumbnail_number = 4

# %%
# **Setup for matplotlib figures**
rcParams["figure.figsize"] = 4.5, 3.5
rcParams["font.size"] = 9


# function for plotting 2D dataset
def plot2D(csdm_object, **kwargs):
    ax = plt.gca(projection="csdm")
    ax.imshow(csdm_object, cmap="gist_ncar_r", aspect="auto", **kwargs)
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# %%
# Dataset setup
# -------------
#
# Import the dataset
# ''''''''''''''''''
#
# Load the dataset. In this example, we import the dataset as the CSDM [#f2]_
# data-object.

# The 2D MAF dataset in csdm format
filename = "https://osu.box.com/shared/static/mai73gk6nv4uhwuwrm30rr8wczxv3uyt.csdf"
data_object = cp.load(filename)

# For inversion, we only interest ourselves with the real part of the complex dataset.
data_object = data_object.real
# We will also convert the coordinates of both dimensions from Hz to ppm.
_ = [item.to("ppm", "nmr_frequency_ratio") for item in data_object.dimensions]

# %%
# The variable ``data_object`` is a
# `CSDM <https://csdmpy.readthedocs.io/en/latest/api/CSDM.html>`_
# object that holds the 2D MAF dataset. The plot of the MAF dataset is
plot2D(data_object)

# %%
# There are two dimensions in this dataset. The dimension at index 0 is the
# isotropic chemical shift dimension, while the dimension at index 1 is the pure
# anisotropic dimension. The number of coordinates along the respective dimensions
# is
print(data_object.shape)

# %%
# Prepping the data for inversion
# '''''''''''''''''''''''''''''''
# **Step-1: Data Alignment**
#
# When using csdm objects with mrinversion, the dimension at index 0 must always be
# the dimension undergoing the linear inversion, which in this example is the
# pure anisotropic dimension. In the variable ``data_object``, the anisotropic dimension
# is already at index 0 and, therefore, no further action is required.
#
# **Step-2: Optimization**
#
# Also notice, that the MAF data only occupies a small fraction of the two-dimensional
# frequency grid. It is, therefore, best to truncate the dataset to the desired region
# before proceeding. Use the appropriate array indexing/slicing to select the signal
# region.
data_object_truncated = data_object[:, 37:74]
plot2D(data_object_truncated)

# %%
# Linear Inversion setup
# ----------------------
#
# Dimension setup
# '''''''''''''''
#
# **Anisotropic-dimension:**
# The dimension of the dataset which holds the pure anisotropic frequency
# contributions. In ``mrinversion``, this must always be the dimension at index 0 of
# the data object.
anisotropic_dimension = data_object_truncated.dimensions[0]

# %%
# **x-y dimensions:**
# The two inverse dimensions corresponding to the `x` and `y`-axis of the `x`-`y` grid.
inverse_dimensions = [
    cp.LinearDimension(count=30, increment="450 Hz", label="x"),  # the `x`-dimension.
    cp.LinearDimension(count=30, increment="450 Hz", label="y"),  # the `y`-dimension.
]

# %%
# Generate the line-shape kernel
# ''''''''''''''''''''''''''''''
#
# For MAF datasets, the line-shape kernel corresponds to the pure nuclear shielding
# anisotropy line-shapes. Use the :class:`~mrinversion.kernel.NuclearShieldingLineshape`
# class to generate a shielding line-shape kernel.
method = NuclearShieldingLineshape(
    anisotropic_dimension=anisotropic_dimension,
    inverse_dimension=inverse_dimensions,
    channel="29Si",
    magnetic_flux_density="9.4 T",
    rotor_angle="90 deg",
    rotor_frequency="12 kHz",
    number_of_sidebands=4,
)

# %%
# The above code generates an instance of the NuclearShieldingLineshape class, which we
# assigned to the variable ``method``.
# The two required arguments of this class are the `anisotropic_dimension` and
# `inverse_dimension`, as previously defined.
# The value of the remaining optional attributes such as the channel, magnetic flux
# density, rotor angle, and rotor frequency is set to match the conditions under which
# the MAF spectrum was acquired. Once the
# NuclearShieldingLineshape instance is created, use the kernel() method to generate
# the MAF lineshape kernel.
K = method.kernel(supersampling=1)
print(K.shape)

# %%
# The kernel ``K`` is a NumPy array of shape (128, 625), where the axis with 128 points
# corresponds to the anisotropic dimension, and the axis with 625 points are the
# features corresponding to the :math:`25\times 25` `x`-`y` coordinates.

# %%
# Data Compression
# ''''''''''''''''
#
new_system = TSVDCompression(K, data_object_truncated)
compressed_K = new_system.compressed_K
compressed_s = new_system.compressed_s

print(f"truncation_index = {new_system.truncation_index}")

# %%
# Solving inverse problem
# -----------------------
#
# Solve the smooth-lasso problem. Normally, one should use the statistical learning
# method to solve the problem over a range of α and λ values, and determine a nuclear
# shielding tensor distribution that best depicts the 2D MAF dataset.
# Given, the time constraints for building this documentation, we skip this step
# and evaluate the nuclear shielding tensor distribution at the pre-optimized α
# and λ values, where the optimum values are :math:`\alpha = 2.2\times 10^{-8}` and
# :math:`\lambda = 1.27\times 10^{-6}`.
# The following commented code was used in determining the optimum α and λ values.

# %%

# from mrinversion.linear_model import SmoothLassoCV
# import numpy as np

# lambdas = 10 ** (-4.8 - 1 * (np.arange(20) / 19))
# alphas = 10 ** (-2.5 - 2.5 * (np.arange(20) / 19))

# s_lasso = SmoothLassoCV(
#     alphas=alphas,
#     lambdas=lambdas,
#     sigma=0.015,
#     folds=10,
#     inverse_dimension=inverse_dimensions,
#     verbose=1
# )

# # run fit using the compressed kernel and compressed data.
# s_lasso.fit(compressed_K, compressed_s)

# # the optimum hyper-parameters, alpha and lambda, from the cross-validation.
# print(s_lasso.hyperparameter)
# # {'alpha': 3.359818286283781e-05, 'lambda': 5.324953129837531e-06}

# # the solution
# f_sol = s_lasso.f

# # the cross-validation error curve
# error_curve = s_lasso.cross_validation_curve

# %%
# If you use the above cross-validation, ``SmoothLassoCV`` method, you may skip the
# following section of code.

s_lasso = SmoothLasso(
    alpha=3.36e-5, lambda1=5.33e-6, inverse_dimension=inverse_dimensions
)
# run the fit method on the compressed kernel and compressed data.
s_lasso.fit(K=compressed_K, s=compressed_s)

# the solution
f_sol = s_lasso.f

# %%
# Here, ``f_sol`` is the solution corresponding to the optimized hyperparameters. To
# calculate the residuals between the data and predicted data(fit), use the `residuals`
# method, as follows,
residue = s_lasso.residuals(K, data_object_truncated.real)
plot2D(residue, vmax=data_object_truncated.max(), vmin=data_object_truncated.min())

# %%
# The standard deviation of the residuals is
residue.std()

# %%
# Saving the solution
# '''''''''''''''''''
#
# To serialize the solution to file, use the `save()` method of the CSDM object,
# for example,
f_sol.save("MgO.SiO2_inverse.csdf")  # save the solution
residue.save("MgO.SiO2_residue.csdf")  # save the residuals

# %%
# Data Visualization
# ------------------
#
# At this point, we have solved the inverse problem and obtained an optimum
# distribution of the nuclear shielding tensors from the 2D MAF dataset. You may use
# any data visualization and interpretation tool of choice for further analysis.
# In the following sections, we provide minimal visualization and analysis
# to complete the case study.
#
# **Visualizing the 3D solution**

# convert the coordinates of the solution, `f_sol`, from frequency units to ppm.
[item.to("ppm", "nmr_frequency_ratio") for item in f_sol.dimensions]

# The 3d plot of the solution
plt.figure(figsize=(5, 4.4))
ax = plt.gca(projection="3d")
plot_3d(
    ax,
    f_sol,
    theta_angle=15,
    angle=-132,
    x_lim=[0, 165],
    y_lim=[0, 165],
    z_lim=[-55, -115],
)
plt.tight_layout()
plt.show()


# %%
# References
# ----------
#
# .. [#f1] P. Zhang, C. Dunlap, P. Florian, P. J. Grandinetti, I. Farnan, J. F.
#       Stebbins, Silicon site distributions in an alkali silicate glass derived by
#       two-dimensional 29Si nuclear magnetic resonance, J. Non. Cryst. Solids 204
#       294–300 (1996). `doi:10.1016/S0022-3093(96)00601-1
#       <https://doi.org/doi:10.1016/S0022-3093(96)00601-1>`_.
#
# .. [#f2] Srivastava, D.J., Vosegaard, T., Massiot, D., Grandinetti, P.J. (2020) Core
#       Scientific Dataset Model: A lightweight and portable model and file format
#       for multi-dimensional scientific data.
#       `PLOS ONE 15(1): e0225953. <https://doi.org/10.1371/journal.pone.0225953>`_
