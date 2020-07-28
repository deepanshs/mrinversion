#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2D MAT data of KMg0.5O 4.SiO2 glass
===================================
"""
# %%
# The following example illustrates an application of the statistical learning method
# applied in determining the distribution of the nuclear shielding tensor parameters
# from a 2D magic-angle turning (MAT) spectrum. In this example, we use the 2D MAT
# spectrum [#f1]_ of :math:`\text{KMg}_{0.5}\text{O}\cdot4\text{SiO}_2` glass.
#
# Setup for the matplotlib figure.
import csdmpy as cp
import csdmpy.statistics as stats
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pylab import rcParams

from mrinversion.kernel.nmr import ShieldingPALineshape
from mrinversion.kernel.utils import x_y_to_zeta_eta
from mrinversion.linear_model import SmoothLasso
from mrinversion.linear_model import TSVDCompression
from mrinversion.utils import plot_3d

# sphinx_gallery_thumbnail_number = 5

# %%
# Setup for the matplotlib figures.
rcParams["figure.figsize"] = 4, 3
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
# Load the dataset. Here, we import the dataset as the CSDM data-object.

# The 2D MAT dataset in csdm format
filename = "https://zenodo.org/record/3964531/files/KMg0_5-4SiO2-MAT.csdf"
data_object = cp.load(filename)

# For inversion, we only interest ourselves with the real part of the complex dataset.
data_object = data_object.real

# We will also convert the coordinates of both dimensions from Hz to ppm.
_ = [item.to("ppm", "nmr_frequency_ratio") for item in data_object.dimensions]

# %%
# Here, the variable ``data_object`` is a
# `CSDM <https://csdmpy.readthedocs.io/en/latest/api/CSDM.html>`_
# object that holds the real part of the 2D MAT dataset. The plot of the MAT dataset
# is
plot2D(data_object)

# %%
# There are two dimensions in this dataset. The dimension at index 0 is the pure
# anisotropic spinning sideband dimension, while the dimension at index 1 is the
# isotropic chemical shift dimension.
#
# Prepping the data for inversion
# '''''''''''''''''''''''''''''''
# **Step-1: Data Alignment**
#
# When using the csdm objects with the ``mrinversion`` package, the dimension at index
# 0 must be the dimension undergoing the linear inversion. In this example, we plan to
# invert the pure anisotropic shielding line-shape. In the ``data_object``, the
# anisotropic dimension is already at index 0 and, therefore, no further action is
# required.
#
# **Step-2: Optimization**
#
# Also notice, the signal from the 2D MAF dataset occupies a small fraction of the
# two-dimensional frequency grid. For optimum performance, truncate the dataset to the
# relevant region before proceeding. Use the appropriate array indexing/slicing to
# select the signal region.
data_object_truncated = data_object[:, 75:105]
plot2D(data_object_truncated)

# %%
# Linear Inversion setup
# ----------------------
#
# Dimension setup
# '''''''''''''''
#
# **Anisotropic-dimension:**
# The dimension of the dataset that holds the pure anisotropic frequency
# contributions. In ``mrinversion``, this must always be the dimension at index 0 of
# the data object.
anisotropic_dimension = data_object_truncated.dimensions[0]

# %%
# **x-y dimensions:**
# The two inverse dimensions corresponding to the `x` and `y`-axis of the `x`-`y` grid.
inverse_dimensions = [
    cp.LinearDimension(count=25, increment="370 Hz", label="x"),  # the `x`-dimension.
    cp.LinearDimension(count=25, increment="370 Hz", label="y"),  # the `y`-dimension.
]

# %%
# Generating the kernel
# '''''''''''''''''''''
#
# For MAF/PASS datasets, the kernel corresponds to the pure nuclear shielding anisotropy
# sideband spectra. Use the
# :class:`~mrinversion.kernel.nmr.ShieldingPALineshape` class to generate a
# shielding spinning sidebands kernel.
sidebands = ShieldingPALineshape(
    anisotropic_dimension=anisotropic_dimension,
    inverse_dimension=inverse_dimensions,
    channel="29Si",
    magnetic_flux_density="9.4 T",
    rotor_angle="54.735°",
    rotor_frequency="790 Hz",
    number_of_sidebands=anisotropic_dimension.count,
)

# %%
# Here, ``sidebands`` is an instance of the
# :class:`~mrinversion.kernel.nmr.ShieldingPALineshape` class. The required
# arguments of this class are the `anisotropic_dimension`, `inverse_dimension`, and
# `channel`. We have already defined the first two arguments in the previous
# sub-section. The value of the `channel` argument is the nucleus observed in the
# MAT/PASS experiment. In this example, this value is '29Si'.
# The remaining arguments, such as the `magnetic_flux_density`, `rotor_angle`,
# and `rotor_frequency`, are set to match the conditions under which the 2D MAT/PASS
# spectrum was acquired, which in this case corresponds to acquisition at
# the magic-angle and spinning at a rotor frequency of 790 Hz in a 9.4 T magnetic
# flux density.
#
# The value of the `rotor_frequency` argument is the effective anisotropic
# modulation frequency. In a MAT measurement, the anisotropic modulation frequency
# is the same as the physical rotor frequency. For other measurements like the extended
# chemical shift modulation sequences (XCS) [#f3]_, or its variants, the effective
# anisotropic modulation frequency is lower than the physical rotor frequency and
# should be set appropriately.
#
# The argument `number_of_sidebands` is the maximum number of computed
# sidebands in the kernel. For most two-dimensional isotropic v.s pure
# anisotropic spinning-sideband correlation measurements, the sampling along the
# sideband dimension is the rotor or the effective anisotropic modulation
# frequency. Therefore, the value of the `number_of_sidebands` argument is
# usually the number of points along the sideband dimension.
# In this example, this value is 32.
#
# Once the ShieldingPALineshape instance is created, use the kernel()
# method to generate the spinning sideband lineshape kernel.
K = sidebands.kernel(supersampling=1)
print(K.shape)

# %%
# The kernel ``K`` is a NumPy array of shape (32, 625), where the axes with 32 and
# 625 points are the spinning sidebands dimension and the features (x-y coordinates)
# corresponding to the :math:`25\times 25` `x`-`y` grid, respectively.

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
# Solving the inverse problem
# ---------------------------
#
# Smooth LASSO cross-validation
# '''''''''''''''''''''''''''''
#
# Solve the smooth-lasso problem. Ordinarily, one should use the statistical learning
# method to solve the inverse problem over a range of α and λ values and then determine
# the best nuclear shielding tensor parameter distribution for the given 2D MAT/PASS
# dataset. Considering the limited build time for the documentation, we skip this step
# and evaluate the distribution at pre-optimized α and λ values. The optimum values are
# :math:`\alpha = 3.79\times 10^{-6}` and :math:`\lambda = 1.83\times 10^{-6}`.
# The following commented code was used in determining the optimum α and λ values.

# %%

# from mrinversion.linear_model import SmoothLassoCV

# # setup the pre-defined range of alpha and lambda values
# lambdas = 10 ** (-4 - 3 * (np.arange(20) / 19))
# alphas = 10 ** (-4 - 3 * (np.arange(20) / 19))

# # setup the smooth lasso cross-validation class
# s_lasso = SmoothLassoCV(
#     alphas=alphas,  # A numpy array of alpha values.
#     lambdas=lambdas,  # A numpy array of lambda values.
#     sigma=0.00070,  # The standard deviation of noise from the MAF data.
#     folds=10,  # The number of folds in n-folds cross-validation.
#     inverse_dimension=inverse_dimensions,  # previously defined inverse dimensions.
#     verbose=1,  # If non-zero, prints the progress as the computation proceeds.
#     max_iterations=20000,  # The maximum number of allowed interations.
# )

# # run fit using the compressed kernel and compressed data.
# s_lasso.fit(compressed_K, compressed_s)

# # the optimum hyper-parameters, alpha and lambda, from the cross-validation.
# print(s_lasso.hyperparameters)
# # {'alpha': 3.7926901907322535e-06, 'lambda': 1.1288378916846883e-05}

# # the solution
# f_sol = s_lasso.f

# # the cross-validation error curve
# CV_metric = s_lasso.cross_validation_curve

# %%
# If you use the above ``SmoothLassoCV`` method, skip the following code-block.

# Setup the smooth lasso class
s_lasso = SmoothLasso(
    alpha=3.79e-6, lambda1=11.3e-6, inverse_dimension=inverse_dimensions
)
# run the fit method on the compressed kernel and compressed data.
s_lasso.fit(K=compressed_K, s=compressed_s)

# %%
# The optimum solution
# ''''''''''''''''''''
#
# The :attr:`~mrinversion.linear_model.SmoothLasso.f` attribute of the instance holds
# the solution,
f_sol = s_lasso.f  # f_sol is a CSDM object.

# %%
# where ``f_sol`` is the optimum solution.
#
# The fit residuals
# '''''''''''''''''
#
# To calculate the residuals between the data and predicted data(fit), use the
# :meth:`~mrinversion.linear_model.SmoothLasso.residuals` method, as follows,
residuals = s_lasso.residuals(K=K, s=data_object_truncated)
# residuals is a CSDM object.

# The plot of the residuals.
plot2D(residuals, vmax=data_object_truncated.max(), vmin=data_object_truncated.min())

# %%
# The standard deviation of the residuals is
residuals.std()

# %%
# Saving the solution
# '''''''''''''''''''
#
# To serialize the solution to a file, use the `save()` method of the CSDM object,
# for example,
f_sol.save("KMg_mixed_silicate_inverse.csdf")  # save the solution
residuals.save("KMg_mixed_silicate_residue.csdf")  # save the residuals

# %%
# Data Visualization
# ------------------
#
# At this point, we have solved the inverse problem and obtained an optimum
# distribution of the nuclear shielding tensor parameters from the 2D MAT dataset. You
# may use any data visualization and interpretation tool of choice for further
# analysis. In the following sections, we provide minimal visualization and analysis
# to complete the case study.
#
# Visualizing the 3D solution
# '''''''''''''''''''''''''''

# Normalize the solution
f_sol /= f_sol.max()

# Convert the coordinates of the solution, `f_sol`, from Hz to ppm.
[item.to("ppm", "nmr_frequency_ratio") for item in f_sol.dimensions]

# The 3D plot of the solution
plt.figure(figsize=(5, 4.4))
ax = plt.gca(projection="3d")
plot_3d(ax, f_sol, x_lim=[0, 120], y_lim=[0, 120], z_lim=[-50, -150])
plt.tight_layout()
plt.show()

# %%
# From the 3D plot, we observe two distinct regions: one for the :math:`\text{Q}^4`
# sites and another for the :math:`\text{Q}^3` sites.
# Select the respective regions by using the appropriate array indexing,

Q4_region = f_sol[0:8, 0:8, 5:25]
Q4_region.description = "Q4 region"

Q3_region = f_sol[0:8, 7:24, 7:25]
Q3_region.description = "Q3 region"

# %%
# The plot of the respective regions is shown below.

# Calculate the normalization factor for the 2D contours and 1D projections from the
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

# plot for the Q4 region
plot_3d(
    ax,
    Q4_region,
    x_lim=[0, 120],  # the x-limit
    y_lim=[0, 120],  # the y-limit
    z_lim=[-50, -150],  # the z-limit
    max_2d=max_2d,  # normalization factors for the 2D contours projections
    max_1d=max_1d,  # normalization factors for the 1D projections
    cmap=cm.Reds_r,  # colormap
    box=True,  # draw a box around the region
)
# plot for the Q3 region
plot_3d(
    ax,
    Q3_region,
    x_lim=[0, 120],  # the x-limit
    y_lim=[0, 120],  # the y-limit
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
# Visualizing the isotropic projections.
# ''''''''''''''''''''''''''''''''''''''
#
# Because the :math:`\text{Q}^4` and :math:`\text{Q}^3` regions are fully resolved
# after the inversion, evaluating the contributions from these regions is trivial.
# For examples, the distribution of the isotropic chemical shifts for these regions are

# Isotropic chemical shift projection of the 2D MAT dataset.
data_iso = data_object_truncated.sum(axis=0)
data_iso /= data_iso.max()  # normalize the projection

# Isotropic chemical shift projection of the tensor distribution dataset.
f_sol_iso = f_sol.sum(axis=(0, 1))

# Isotropic chemical shift projection of the tensor distribution for the Q4 region.
Q4_region_iso = Q4_region.sum(axis=(0, 1))

# Isotropic chemical shift projection of the tensor distribution for the Q3 region.
Q3_region_iso = Q3_region.sum(axis=(0, 1))

# Normalize the three projections.
f_sol_iso_max = f_sol_iso.max()
f_sol_iso /= f_sol_iso_max
Q4_region_iso /= f_sol_iso_max
Q3_region_iso /= f_sol_iso_max

# The plot of the different projections.
plt.figure(figsize=(5.5, 3.5))
ax = plt.gca(projection="csdm")
ax.plot(f_sol_iso, "--k", label="tensor")
ax.plot(Q4_region_iso, "r", label="Q4")
ax.plot(Q3_region_iso, "b", label="Q3")
ax.plot(data_iso, "-k", label="MAF")
ax.plot(data_iso - f_sol_iso - 0.1, "gray", label="residuals")
ax.set_title("Isotropic projection")
ax.invert_xaxis()
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Notice the shape of the isotropic chemical shift distribution for the
# :math:`\text{Q}^4` sites is skewed, which is expected.
#
#
# Analysis
# --------
#
# For the analysis, we use the
# `statistics <https://csdmpy.readthedocs.io/en/latest/api/statistics.html>`_
# module of the csdmpy package. Following is the moment analysis of the 3D volumes for
# both the :math:`\text{Q}^4` and :math:`\text{Q}^3` regions up to the second moment.

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
# ----------
#
# .. [#f1]  Walder, B. J., Dey, K. K., Kaseman, D. C., Baltisberger, J. H.,
#       Grandinetti, P. J. Sideband separation experiments in NMR with phase
#       incremented echo train acquisition, J. Chem. Phys. 138, 4803142, (2013).
#       `doi:10.1063/1.4803142. <https://doi.org/10.1063/1.4803142>`_
#
# .. [#f3] Gullion, T., Extended chemical-shift modulation, J. Mag. Res., **85**, 3,
#       (1989).
#       `10.1016/0022-2364(89)90253-9 <https://doi.org/10.1016/0022-2364(89)90253-9>`_
