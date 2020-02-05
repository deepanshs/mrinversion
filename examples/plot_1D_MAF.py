#!/usr/bin/env python
# coding: utf-8
"""
MAF Anisotropic line-shape
^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
#%%
# The following example demonstrate the statistical learning based nuclear shielding
# tensor distribution from the magic angle flipping spectrum. In this examples, we use
# a synthetic MAF lineshape.
#
# Import the dataset
# ------------------
#
# We import the example file as the CSDM file-object.
import csdmpy as cp

from mrinversion import examples

# the data in csdm format
data_object = cp.load(examples.MAF01)

# the true tensor distribution for comparison
true_data_object = cp.load(examples.true_distribution01)

#%%
# The variable ``data_object`` holds the coordinates and the responses of the MAF
# dataset, which are

#%%
coordinates = data_object.dimensions[0].coordinates
responses = data_object.dependent_variables[0].components[0]

#%%
# The corresponding MAF lineshape plot along with the 2D true tensor distribution of
# the synthetic dataset is shown below.

#%%
import numpy as np
import matplotlib.pyplot as plt
from mrinversion.plot import get_polar_grids

# the plot of the MAF dataset.
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].plot(coordinates, responses)
ax[0].invert_xaxis()
ax[0].set_xlabel(data_object.dimensions[0].axis_label)


# the plot of the true tensor distribution.
# convert the true_data_object's dimension coordinates to pmm from Hz.
true_data_object.dimensions[0].to("ppm", "nmr_frequency_ratio")
true_data_object.dimensions[1].to("ppm", "nmr_frequency_ratio")

# the coordinates along the x and y dimensions
x_t = true_data_object.dimensions[0].coordinates
y_t = true_data_object.dimensions[1].coordinates

# the true tensor distribution
true_tensor_dist = true_data_object.dependent_variables[0].components[0]

# plot
levels = (np.arange(9) + 1) / 10
ax[1].contourf(
    x_t, y_t, true_tensor_dist / true_tensor_dist.max(), cmap="gist_ncar", levels=levels
)
ax[1].set_xlabel(true_data_object.dimensions[0].axis_label)
ax[1].set_ylabel(true_data_object.dimensions[1].axis_label)
ax[1].set_xlim(0, 100)
ax[1].set_ylim(0, 100)
get_polar_grids(ax[1])
ax[1].set_aspect("equal")

plt.tight_layout(pad=0)
plt.show()

#%%
# Set the direct and inverse-dimension
# ------------------------------------
#
# **The direct-dimension**

#%%
direct_dimension = cp.LinearDimension(
    count=96, increment="208.33 Hz", coordinates_offset="-9999.84 Hz"
)

#%%
# **Indirect-dimension**

#%%
inverse_dimension = [
    cp.LinearDimension(count=25, increment="370 Hz"),
    cp.LinearDimension(count=25, increment="370 Hz"),
]

#%%
# Generate the line-shape kernel
# ------------------------------
#
# In the following code, we declare a NuclearShieldingTensor class object using the
# direct and the inverse dimensions. The value of the optional attributes of the
# NuclearShieldingTensor class, such as, the isotope, magnetic flux density, rotor
# angle, and rotor frequency are set to match the conditions under which the MAF
# spectrum was acquired. Notice, for MAF measurements, the rotor angle is usually
# :math:`90^\circ`. Once the NuclearShieldingTensor instance is created, in the
# following case, the variable ``method``, use the kernel method to generate the
# MAF lineshape kernel.

#%%
from mrinversion.kernel import NuclearShieldingTensor

method = NuclearShieldingTensor(
    direct_dimension=direct_dimension,
    inverse_dimension=inverse_dimension,
    isotope="29Si",
    magnetic_flux_density="9.4 T",
    rotor_angle="90 deg",
    rotor_frequency="14 kHz",
    number_of_sidebands=1,
)
K = method.kernel(supersampling=4)

#%%
# Data Compression
# ----------------

#%%
from mrinversion.linear_model import TSVDCompression

new_system = TSVDCompression(K, responses)
compressed_K = new_system.compressed_K
compressed_s = new_system.compressed_s
print(compressed_K.shape, compressed_s.shape)

#%%
# Set up the inverse problem
# --------------------------

#%%
from mrinversion.linear_model import SmoothLasso

s_lasso = SmoothLasso(alpha=0.005, lambda1=5e-6)
s_lasso.fit(K=compressed_K, s=compressed_s, f_shape=(25, 25))

#%%
# The solution is the value of the `f` attribute from the ``s_lasso`` instance,

#%%
f_sol = s_lasso.f

#%%
# **The plot of the solution**

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
# convert the true_data_object's dimension coordinates to pmm from Hz.
true_data_object.dimensions[0].to("ppm", "nmr_frequency_ratio")
true_data_object.dimensions[1].to("ppm", "nmr_frequency_ratio")

# the coordinates along the x and y dimensions
x_t = true_data_object.dimensions[0].coordinates
y_t = true_data_object.dimensions[1].coordinates

# the true tensor distribution
true_tensor_dist = true_data_object.dependent_variables[0].components[0]

# plot
levels = (np.arange(9) + 1) / 10
ax[1].contourf(
    x_t, y_t, true_tensor_dist / true_tensor_dist.max(), cmap="gist_ncar", levels=levels
)
ax[1].set_xlabel(true_data_object.dimensions[0].axis_label)
ax[1].set_ylabel(true_data_object.dimensions[1].axis_label)
ax[1].set_xlim(0, 100)
ax[1].set_ylim(0, 100)
get_polar_grids(ax[1])
ax[1].set_aspect("equal")

plt.tight_layout(pad=0.2)
plt.show()

#%%
# The predicted spectrum from the solution may be evaluated using the `predict` method as

#%%
predicted_spectrum = s_lasso.predict(K)

#%%
plt.figure(figsize=(4, 3))
plt.plot(coordinates, responses, color="black")  # the original spectrum
plt.plot(coordinates, predicted_spectrum, color="red")  # the predicted spectrum
plt.xlabel("frequency / Hz")
plt.tight_layout()
plt.show()


#%%
# Statistical learning of the tensors
# -----------------------------------

#%%
lambdas = 10 ** (-5 - 2 * (np.arange(5) / 4))
alphas = 10 ** (-2.5 - 2 * (np.arange(5) / 4))

#%%
from mrinversion.linear_model import SmoothLassoCV

s_lasso_cv = SmoothLassoCV(alphas=alphas, lambdas=lambdas, sigma=0.005)
s_lasso_cv.fit(compressed_K, compressed_s, f_shape=(25, 25))

#%%
# The optimized hyperparameters from the 10-folds cross-validation are

#%%
s_lasso_cv.hyperparameter

#%%
# and the corresponding cross-validation metric,

#%%
plt.figure(figsize=(4, 3))
plt.contour(
    -np.log10(lambdas), -np.log10(alphas), np.log10(s_lasso_cv.cv_map), levels=25
)
plt.scatter(
    -np.log10(s_lasso_cv.hyperparameter["lambda"]),
    -np.log10(s_lasso_cv.hyperparameter["alpha"]),
    marker="x",
    color="k",
)
plt.xlabel(r"$-\log~\lambda$")
plt.ylabel(r"$-\log~\alpha$")
plt.tight_layout()
plt.show()


#%%
# The optimum selected model from the 10-folds cross-validation is

#%%
vector = s_lasso_cv.f

#%%
# and the corresponding plots

#%%
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
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
# convert the true_data_object's dimension coordinates to pmm from Hz.
true_data_object.dimensions[0].to("ppm", "nmr_frequency_ratio")
true_data_object.dimensions[1].to("ppm", "nmr_frequency_ratio")

# the coordinates along the x and y dimensions
x_t = true_data_object.dimensions[0].coordinates
y_t = true_data_object.dimensions[1].coordinates

# the true tensor distribution
true_tensor_dist = true_data_object.dependent_variables[0].components[0]

# plot
levels = (np.arange(9) + 1) / 10
ax[1].contourf(
    x_t, y_t, true_tensor_dist / true_tensor_dist.max(), cmap="gist_ncar", levels=levels
)
ax[1].set_xlabel(true_data_object.dimensions[0].axis_label)
ax[1].set_ylabel(true_data_object.dimensions[1].axis_label)
ax[1].set_xlim(0, 100)
ax[1].set_ylim(0, 100)
get_polar_grids(ax[1])
ax[1].set_aspect("equal")
plt.show()
