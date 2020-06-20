#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D MAF of Na2O 4.7SiO2 glass
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
# %%
# The following example illustrates an application of the statistical learning method
# applied in determining the distribution of the nuclear shielding tensor parameters
# from a 2D magic-angle flipping (MAF) spectrum. In this example, we use the 2D MAF
# spectrum [#f1]_ of :math:`\text{Na}_2\text{O}\cdot4.7\text{SiO}_2` glass.
#
# Before getting started
# ----------------------
#
# Import all relevant packages.
import csdmpy as cp
import csdmpy.statistics as stats
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pylab import rcParams

from mrinversion.kernel import NuclearShieldingLineshape
from mrinversion.kernel.utils import x_y_to_zeta_eta
from mrinversion.linear_model import SmoothLasso
from mrinversion.linear_model import TSVDCompression
from mrinversion.plot import plot_3d

# sphinx_gallery_thumbnail_number = 5

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
filename = "https://osu.box.com/shared/static/8lnwmg0dr7y6egk40c2orpkmmugh9j7c.csdf"
data_object = cp.load(filename)

# For inversion, we only interest ourselves with the real part of the complex dataset.
data_object = data_object.real

# %%
# Here, the variable ``data_object`` is a
# `CSDM <https://csdmpy.readthedocs.io/en/latest/api/CSDM.html>`_
# object that holds the real part of the 2D MAF dataset. The plot of the 2D MAF dataset
# is
plot2D(data_object)

# %%
# There are two dimensions in this dataset. The dimension at index 0, the horizontal
# dimension in the figure, is the isotropic chemical shift dimension, while the
# dimension at index 1 is the pure anisotropic dimension. The number of coordinates
# along the respective dimensions is
print(data_object.shape)

# %%
# Prepping the data for inversion
# '''''''''''''''''''''''''''''''
# **Step-1: Data Alignment**
#
# When using the csdm object with mrinversion, the dimension at index 0 must always be
# the dimension undergoing the linear inversion, which in the examples, is the pure
# anisotropic dimension. In ``data_object``, however, the anisotropic dimension is at
# index 1. You may swap the anisotropic dimension to index 0 by using the transpose
# method.
# Also notice, that the MAF data only occupies a small fraction of the two-dimensional
# frequency grid. It is, therefore, best to truncate the dataset to the desired region
# before proceeding. Use the appropriate array indexing/slicing to select the signal
# region.
data_object_truncated = data_object.T[:, 155:180]
plot2D(data_object_truncated)

# %%
# In the above code, we first transpose the dataset and then truncate the isotropic
# dimension to isotropic chemical shifts between indexes ranging from 155 to 175.

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
    cp.LinearDimension(count=25, increment="400 Hz", label="x"),  # the `x`-dimension.
    cp.LinearDimension(count=25, increment="400 Hz", label="y"),  # the `y`-dimension.
]

# %%
# Generate the line-shape kernel
# ''''''''''''''''''''''''''''''
#
# For MAF datasets, the line-shape kernel corresponds to the pure nuclear shielding
# anisotropy line-shapes. Use the :class:`~mrinversion.kernel.NuclearShieldingLineshape`
# class to generate a shielding line-shape kernel.
lineshape = NuclearShieldingLineshape(
    anisotropic_dimension=anisotropic_dimension,
    inverse_dimension=inverse_dimensions,
    channel="29Si",
    magnetic_flux_density="9.4 T",
    rotor_angle="87.14°",
    rotor_frequency="14 kHz",
    number_of_sidebands=4,
)

# %%
# Here, ``lineshape`` is an instance of the
# :class:`~mrinversion.kernel.NuclearShieldingLineshape` class. The required arguments
# of this class are the `anisotropic_dimension`, `inverse_dimension`, and `channel`.
# We have already defined the first two arguments in the previous sub-section. The
# value of the `channel` argument is the nuclei observed in the MAF experiment. In this
# example, this value is '29Si'.
# The remaining attribute values, such as the `magnetic_flux_density`, `rotor_angle`,
# and `rotor_frequency`, are set to match the conditions under which the 2D MAF
# spectrum was acquired. Note for this particular MAF measurement the rotor angle was
# set to :math:`87.19^\circ` for the anisotropic dimension, not the usual
# :math:`90^\circ`. Once the NuclearShieldingLineshape instance is created, use the
# :meth:`~mrinversion.kernel.NuclearShieldingLineshape.kernel` method of the instance
# to generate the MAF lineshape kernel.
K = lineshape.kernel(supersampling=1)
print(K.shape)

# %%
# The kernel ``K`` is a NumPy array of shape (128, 625), where the axis with 128 points
# corresponds to the anisotropic dimension, and the axis with 625 points are the
# features corresponding to the :math:`25\times 25` `x`-`y` coordinates.

# %%
# Data Compression
# ''''''''''''''''
#
# Data compression is optional but recommended. It may reduce the size of the
# inverse problem and, thus, further computation time.
new_system = TSVDCompression(K, data_object_truncated)
compressed_K = new_system.compressed_K
compressed_s = new_system.compressed_s

print(f"truncation_index = {new_system.truncation_index}")
# %%
# Solving inverse problem
# -----------------------
#
# Solve the smooth-lasso problem. Ordinarily, one should use the statistical learning
# method to solve the inverse problem over a range of α and λ values and then determine
# the best nuclear shielding tensor parameter distribution for the given 2D MAF
# dataset.
# Given the time constraints for building this documentation, we skip this step
# and evaluate the distribution at pre-optimized α and λ values. The optimum values are
# :math:`\alpha = 2.07\times 10^{-7}` and :math:`\lambda = 7.85\times 10^{-6}`.
# The following commented code was used in determining the optimum α and λ values.

# %%

# from mrinversion.linear_model import SmoothLassoCV

# # set up the pre-defined range of alpha and lambda values
# lambdas = 10 ** (-4 - 3 * (np.arange(20) / 19))
# alphas = 10 ** (-4 - 3 * (np.arange(20) / 19))

# # set up the smooth lasso cross-validation class
# s_lasso = SmoothLassoCV(
#     alphas=alphas,        # A numpy array of alpha values.
#     lambdas=lambdas,      # A numpy array of lambda values.
#     sigma=0.003,          # The standard deviation of noise from the MAF data.
#     folds=10,             # The number of folds in n-folds cross-validation.
#     inverse_dimension=inverse_dimensions, # previously defined inverse dimensions.
#     verbose=1, # If non-zero, prints the progress as the computation proceeds.
#     max_iterations=20000, # maximum number of allowed iterations.
# )

# # run fit using the conpressed kernel and compressed data.
# s_lasso.fit(compressed_K, compressed_s)

# # the optimum hyper-parameters, alpha and lambda, from the cross-validation.
# print(s_lasso.hyperparameter)
# # {'alpha': 2.06913808111479e-07, 'lambda': 7.847599703514622e-06}

# # the solution
# f_sol = s_lasso.f

# # the cross-validation error curve
# error_curve = s_lasso.cross_validation_curve

# %%
# If you use the ``SmoothLassoCV`` method, skip the following section of code.

# set up the smooth lasso class
s_lasso = SmoothLasso(
    alpha=2.07e-7, lambda1=7.85e-6, inverse_dimension=inverse_dimensions
)
# run the fit method on the conpressed kernel and compressed data.
s_lasso.fit(K=compressed_K, s=compressed_s)

# the solution
f_sol = s_lasso.f

# %%
# Here, ``f_sol`` is the solution corresponding to the optimized hyperparameters. To
# calculate the residuals between the data and predicted data(fit), use the
# :meth:`~mrinversion.linear_model.SmoothLasso.residuals` method, as follows,
residue = s_lasso.residuals(K, data_object_truncated)
plot2D(residue, vmax=data_object_truncated.max(), vmin=data_object_truncated.min())

# %%
# The mean and standard deviation of the residuals are
residue.mean(), residue.std()

# %%
# Saving the solution
# '''''''''''''''''''
#
# To serialize the solution a file, use the `save()` method of the CSDM object,
# for example,
f_sol.save("Na2O.4.7SiO2_inverse.csdf")  # save the solution
residue.save("Na2O.4.7SiO2_residue.csdf")  # save the residuals

# %%
# Data Visualization
# ------------------
#
# At this point, we have solved the inverse problem and obtained an optimum
# distribution of the nuclear shielding tensor parameters from the 2D MAF dataset. You
# may use any data visualization and interpretation tool of choice for further
# analysis. In the following sections, we provide minimal visualization and analysis
# to complete the case study.
#
# **Visualizing the 3D solution**

# Convert the coordinates of the solution, `f_sol`, from frequency units to ppm.
[item.to("ppm", "nmr_frequency_ratio") for item in f_sol.dimensions]

# The 3d plot of the solution
plt.figure(figsize=(5, 4.4))
ax = plt.gca(projection="3d")
plot_3d(ax, f_sol, x_lim=[0, 140], y_lim=[0, 140], z_lim=[-50, -150])
plt.tight_layout()
plt.show()

# %%
# From the 3D plot, we observe two distinct volumes: one for the :math:`\text{Q}^4`
# sites and another for the :math:`\text{Q}^3` sites.
# Select the respective regions by using the appropriate array indexing,

Q4_region = f_sol[0:8, 0:8, 3:18]
Q4_region.description = "Q4 region"

Q3_region = f_sol[0:8, 11:22, 8:20]
Q3_region.description = "Q3 region"

# %%
# The plot of the respective volumes is shown below.

# Calculate the normalization factor the 2D contour and 1D projections from the
# original solution, `f_sol`. Use this normalization factor to scale the intensities
# from the sub-regions.
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
    x_lim=[0, 140],  # the x-limit
    y_lim=[0, 140],  # the y-limit
    z_lim=[-50, -150],  # the z-limit
    max_2d=max_2d,  # normalization factors for the 2D contours projections
    max_1d=max_1d,  # normalization factors for the 1D projections
    cmap=cm.Reds_r,  # colormap
    box=True,  # draw a box around the region
)
# plot for Q3 region
plot_3d(
    ax,
    Q3_region,
    x_lim=[0, 140],  # the x-limit
    y_lim=[0, 140],  # the y-limit
    z_lim=[-50, -150],  # the z-limit
    max_2d=max_2d,  # normalization factors for the 2D contours projections
    max_1d=max_1d,  # normalization factors for the 1D projections
    cmap=cm.Blues_r,  # colormap
    box=True,  # draw a box around the region
)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Because the Q4 and Q3 sites are fully resolved in this dataset, we can observe the
# contributions from these sites without having to build any model. For examples, the
# distribution of the isotropic chemical shifts for the Q4 and Q3 sites are

# Isotropic chemical shift projection of the MAF dataset.
data_iso = data_object_truncated.sum(axis=0)
data_iso /= data_iso.max()  # normalize

# Isotropic chemical shift projection of the tensor distribution dataset.
f_sol_iso = f_sol.sum(axis=(0, 1))
f_sol_iso_max = f_sol_iso.max()
f_sol_iso /= f_sol_iso_max  # normalize

# Isotropic chemical shift projection of the tensor distribution for the Q4 region.
Q4_region_iso = Q4_region.sum(axis=(0, 1))
Q4_region_iso /= f_sol_iso_max  # normalize

# Isotropic chemical shift projection of the tensor distribution for the Q3 region.
Q3_region_iso = Q3_region.sum(axis=(0, 1))
Q3_region_iso /= f_sol_iso_max  # normalize

plt.figure(figsize=(5.5, 3.5))
ax = plt.gca(projection="csdm")
ax.plot(f_sol_iso, "--k", label="tensor")
ax.plot(Q4_region_iso, "r", label="Q4")
ax.plot(Q3_region_iso, "b", label="Q3")
ax.plot(data_iso, "-k", label="MAF")
ax.set_title("Isotropic projection")
ax.invert_xaxis()
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Analysis
# --------
#
# For analysis, we use the
# `statistics <https://csdmpy.readthedocs.io/en/latest/api/statistics.html>`_
# module of the csdmpy package. In the following code, we perform the moment analysis
# of the 3d volumes for both the :math:`\text{Q}^4` and :math:`\text{Q}^3` sites
# up to the second moment.

int_Q4 = stats.integral(Q4_region)  # volume of the Q4 distribution
mean_Q4 = stats.mean(Q4_region)  # mean of the Q4 distribution
std_Q4 = stats.std(Q4_region)  # standard deviation of the Q4 distribution

int_Q3 = stats.integral(Q3_region)  # volume of the Q3 distribution
mean_Q3 = stats.mean(Q3_region)  # mean of the Q3 distribution
std_Q3 = stats.std(Q3_region)  # standard deviation of the Q3 distribution

print("Q4 statistics")
print(f"\tpopulation = {100 * int_Q4 / (int_Q4 + int_Q3)}%")
print("\tmean\n\t\tx:\t{0}\n\t\ty:\t{1}\n\t\tiso:\t{2}".format(*mean_Q4))
print("\tstandard deviation\n\t\tx:\t{0}\n\t\ty:\t{1}\n\t\tiso:\t{2}".format(*std_Q4))

print("Q3 statistics")
print(f"\tpopulation = {100 * int_Q3 / (int_Q4 + int_Q3)}%")
print("\tmean\n\t\tx:\t{0}\n\t\ty:\t{1}\n\t\tiso:\t{2}".format(*mean_Q3))
print("\tstandard deviation\n\t\tx:\t{0}\n\t\ty:\t{1}\n\t\tiso:\t{2}".format(*std_Q3))

# %%
# The statistics shown above are according to the respective dimensions, that is, the
# `x`, `y`, and the isotropic chemical shifts. To convert the `x` and `y` statistics
# to commonly used :math:`\zeta` and :math:`\eta` statistics, use the
# :func:`~mrinversion.kernel.utils.x_y_to_zeta_eta` function.
mean_ζη_Q3 = x_y_to_zeta_eta(*mean_Q3[0:2])

# error propagation for calculating the standard deviation
std_ζ = (std_Q3[0] * mean_Q3[0]) ** 2 + (std_Q3[1] * mean_Q3[1]) ** 2
std_ζ /= mean_Q3[0] ** 2 + mean_Q3[1] ** 2
std_ζ = np.sqrt(std_ζ)

std_η = (std_Q3[1] * mean_Q3[0]) ** 2 + (std_Q3[0] * mean_Q3[1]) ** 2
std_η /= (mean_Q3[0] ** 2 + mean_Q3[1] ** 2) ** 2
std_η = (4 / np.pi) * np.sqrt(std_η)

print("Q3 statistics")
print(f"\tpopulation = {100 * int_Q3 / (int_Q4 + int_Q3)}%")
print("\tmean\n\t\tζ:\t{0}\n\t\tη:\t{1}\n\t\tiso:\t{2}".format(*mean_ζη_Q3, mean_Q3[2]))
print(
    "\tstandard deviation\n\t\tζ:\t{0}\n\t\tη:\t{1}\n\t\tiso:\t{2}".format(
        std_ζ, std_η, std_Q3[2]
    )
)


# %%
# References
# ^^^^^^^^^^
#
# .. [#f1] Baltisberger, J. H., Florian, P., Keeler, E. G., Phyo, P. A., Sanders, K. J.,
#       Grandinetti, P. J.. Modifier cation effects on 29Si nuclear shielding
#       anisotropies in silicate glasses, J. Magn. Reson. 268 (2016) 95 – 106.
#       `doi:10.1016/j.jmr.2016.05.003 <https://doi.org/10.1016/j.jmr.2016.05.003>`_.
#
# .. [#f2] Srivastava, D.J., Vosegaard, T., Massiot, D., Grandinetti, P.J. (2020) Core
#       Scientific Dataset Model: A lightweight and portable model and file format
#       for multi-dimensional scientific data.
#       `PLOS ONE 15(1): e0225953. <https://doi.org/10.1371/journal.pone.0225953>`_
