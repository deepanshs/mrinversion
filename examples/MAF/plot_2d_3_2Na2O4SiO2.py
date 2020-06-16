#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D MAF of 2Na2O 3SiO2 glass
^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
# sphinx_gallery_thumbnail_number = 5
# %%
# The following example illustrates an application of the statistical learning method
# applied to determine the distribution of the nuclear shielding tensor parameters from
# a 2D magic-angle flipping (MAF) spectrum. In this example,
# we use the 2D MAF spectrum [#f1]_ of :math:`2\text{Na}_2\text{O}\cdot3\text{SiO}_2`
# glass.
#
# Setup for matplotlib figure.
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams["figure.figsize"] = 4, 3
rcParams["font.size"] = 9

# %%
# Import the dataset
# ------------------
#
# Load the dataset. In this example, we import the dataset as the CSDM [#f2]_
# data-object.
import csdmpy as cp

# the 2D MAF dataset in csdm format
data_object = cp.load(
    "https://osu.box.com/shared/static/k405dsptwe1p43x8mfi1wc1geywrypzc.csdf"
)
# get the real part of the complex dataset
data_object = data_object.real

# %%
# The variable ``data_object`` is a
# `CSDM <https://csdmpy.readthedocs.io/en/latest/api/CSDM.html>`_
# object that holds the 2D MAF dataset. The plot of the MAF dataset is
cp.plot(data_object, cmap="gist_ncar_r", reverse_axis=[True, True])

# %%
# There are two dimensions in this dataset. The dimension at index 0 is the
# isotropic chemical shift dimension, while the dimension at index 1 is the pure
# anisotropic dimension. The number of coordinates along the respective dimensions
# is
print(data_object.shape)

# %%
# When using csdm objects with mrinversion, the dimension at index 0 must always be
# the dimension undergoing the linear inversion, which in this example is the
# pure anisotropic dimension. In the variable ``data_object``, the anisotropic dimension
# is already at index 0 and, therefore, no further action is required.
# Also notice, that the MAF data only occupies a small fraction of the two-dimensional
# frequency grid. It is, therefore, best to truncate the dataset to the desired region
# before proceeding. Use the appropriate array indexing/slicing to select the signal
# region.

data_object_truncated = data_object[:, 220:280]
cp.plot(data_object_truncated, cmap="gist_ncar_r", reverse_axis=[True, True])

# %%
# Set the anisotropic and inverse-dimension
# -----------------------------------------
#
# **The anisotropic-dimension**
#
# The anisotropic dimension of the 2D MAF dataset should always be the dimension at
# index 0.

anisotropic_dimension = data_object_truncated.dimensions[0]

# %%
# **Inverse-dimension**
#
# The two inverse dimensions correspond to the `x` and `y`-axis of the `x`-`y` grid.

inverse_dimensions = [
    # along the `x`-dimension.
    cp.LinearDimension(count=25, increment="500 Hz", label="x"),
    # along the `y`-dimension.
    cp.LinearDimension(count=25, increment="500 Hz", label="y"),
]

# %%
# Generate the line-shape kernel
# ------------------------------

from mrinversion.kernel import NuclearShieldingLineshape

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
# ----------------

# %%
from mrinversion.linear_model import TSVDCompression

new_system = TSVDCompression(K, data_object_truncated)
compressed_K = new_system.compressed_K
compressed_s = new_system.compressed_s

print(f"truncation_index = {new_system.truncation_index}")
# %%
# Set up the inverse problem
# --------------------------
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
# import numpy as np

# from mrinversion.linear_model import SmoothLassoCV

# lambdas = 10 ** (-4 - 3 * (np.arange(20) / 19))
# alphas = 10 ** (-4.5 - 5 * (np.arange(20) / 19))

# s_lasso = SmoothLassoCV(
#     alphas=alphas,
#     lambdas=lambdas,
#     sigma=0.003,
#     folds=10,
#     inverse_dimension=inverse_dimensions,
# )
# s_lasso.fit(compressed_K, compressed_s)

# print(s_lasso.hyperparameter)
# # 2.198392648862289e-08, 'lambda': 1.2742749857031348e-06}

# # the solution.
# f_sol = s_lasso.f

# # the cross-validation error curve
# error_curve = s_lasso.cross_validation_curve

# %%
# If you use the above cross-validation, ``SmoothLassoCV`` method, you may skip the
# following section of code.

from mrinversion.linear_model import SmoothLasso

# guess alpha and lambda values.
s_lasso = SmoothLasso(
    alpha=2.198e-8, lambda1=1.27e-6, inverse_dimension=inverse_dimensions
)
s_lasso.fit(K=compressed_K, s=compressed_s)

# # normalize the solution.
f_sol = s_lasso.f

# %%
# Here, ``f_sol`` is the solution corresponding to the optimized hyperparameters. To
# calculate the residuals between the data and predicted data(fit), use the `residuals`
# method, as follows,

residue = s_lasso.residuals(K, data_object_truncated.real)
cp.plot(
    residue,
    cmap="gist_ncar_r",
    vmax=data_object_truncated.real.max(),
    vmin=data_object_truncated.real.min(),
    reverse_axis=[True, True],
)

# %%
# The standard deviation of the residuals is
residue.std()

# %%
# **Serialize the solution**
#
# To serialize the solution to file, use the `save()` method of the CSDM object,
# for example,

f_sol.save("2Na2O.3SiO2_inverse.csdf")  # save the solution
residue.save("2Na2O.3SiO2_residue.csdf")  # save the residuals

# %%
# At this point, we have solved the inverse problem and obtained an optimum
# distribution of the nuclear shielding tensors from the 2D MAF dataset. You may use
# any data visualization and interpretation tool of choice for further analysis.
# In the following sections, we provide minimal visualization and analysis
# to complete the case study.
#
# Data Visualization
# ^^^^^^^^^^^^^^^^^^
#
from mrinversion.plot import plot_3d
from matplotlib import cm

# Normalize the solution so that the maximum amplitude is 1.
f_sol /= f_sol.max()

# convert the coordinates of the solution, `f_sol`, from frequency units to ppm.
[item.to("ppm", "nmr_frequency_ratio") for item in f_sol.dimensions]
plt.figure(figsize=(5, 4.4))
ax = plt.gca(projection="3d")
plot_3d(
    ax,
    f_sol,
    theta_angle=20,
    angle=-50,
    x_lim=[0, 150],
    y_lim=[0, 150],
    z_lim=[-60, -120],
)
plt.tight_layout()
plt.show()

# %%
# From the 3d plot, we observe two distinct volumes: one for the :math:`\text{Q}^4`
# sites and another for the :math:`\text{Q}^3` sites. To select the respective
# volumes, use the appropriate array indexing, as follows

Q4_region = f_sol[0:6, 0:6, 14:35]
Q4_region.description = "Q4 region"

Q3_region = f_sol[0:8, 7:, 20:39]
Q3_region.description = "Q3 region"

Q2_region = f_sol[:10, 6:18, 36:52]
Q2_region.description = "Q2 region"

# %%
# An approximate plot of the respective volumes is shown below.

max_2d = [
    f_sol.sum(axis=0).max().value,
    f_sol.sum(axis=1).max().value,
    f_sol.sum(axis=2).max().value,
]
max_1d = [
    f_sol.sum(axis=(1, 2)).max().value,
    f_sol.sum(axis=(0, 2)).max().value,
    f_sol.sum(axis=(0, 1)).max().value,
]

plt.figure(figsize=(5, 4.4))
ax = plt.gca(projection="3d")

# plot for Q4 region
plot_3d(
    ax,
    Q4_region,
    x_lim=[0, 150],
    y_lim=[0, 150],
    z_lim=[-60, -120],
    max_2d=max_2d,
    max_1d=max_1d,
    cmap=cm.Reds_r,
    box=False,
)
# plot for Q3 region
plot_3d(
    ax,
    Q3_region,
    x_lim=[0, 150],
    y_lim=[0, 150],
    z_lim=[-60, -120],
    max_2d=max_2d,
    max_1d=max_1d,
    cmap=cm.Blues_r,
    box=False,
)
# plot for Q2 region
plot_3d(
    ax,
    Q2_region,
    theta_angle=20,
    angle=-50,
    x_lim=[0, 150],
    y_lim=[0, 150],
    z_lim=[-60, -120],
    max_2d=max_2d,
    max_1d=max_1d,
    cmap=cm.Oranges_r,
    box=False,
)

ax.legend()
plt.tight_layout()
plt.show()

# %%
# References
# ^^^^^^^^^^
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
