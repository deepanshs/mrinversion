# -*- coding: utf-8 -*-
"""
2D T2-MAS data of Li2O2 glass
=============================
"""

# %%
# The following example is an application of the statistical learning method in
# determining the distribution of the T2 relaxation constants in glasses.
#
#
# Import all relevant packages.
import csdmpy as cp
import matplotlib.pyplot as plt
import numpy as np

from mrinversion.kernel import relaxation
from mrinversion.linear_model import LassoFistaCV, TSVDCompression

# sphinx_gallery_thumbnail_number = 4


def plot2D(csdm_object, **kwargs):
    plt.figure(figsize=(4.5, 3.5))
    csdm_object.plot(**kwargs)
    plt.tight_layout()
    plt.show()


# %%
# Dataset setup
# -------------
#
# Import the dataset
# ''''''''''''''''''
#
# Load the dataset as a CSDM data-object.

# The 2D MAF dataset in csdm format
filename = "T2MAS_5%Li2O.csdf"
data_object = cp.load(filename)

# Inversion only requires the real part of the complex dataset.
data_object = data_object.real
sigma = 1270.825  # data standard deviation

# Convert the MAS dimension from Hz to ppm.
data_object.dimensions[0].to("ppm", "nmr_frequency_ratio")
plot2D(data_object)

# %%
# There are two dimensions in this dataset. The dimension at index 0, the horizontal
# dimension in the figure, is the MAS dimension, while the dimension at
# index 1, the vertical dimension, is the signal decay from relaxation.
#
# Prepping the data for inversion
# '''''''''''''''''''''''''''''''
# **Step-1: Data Alignment**
#
# When using the csdm objects with the ``mrinversion`` package, the dimension at index
# 0 must be the dimension undergoing the linear inversion. In this example, we
# invert the signal decay from relaxation, that is, dimension-1. The first step is to
# swap the axes using a data transpose.
data_object = data_object.T

# %%
# **Step-2: Optimization**
#
# Notice, that the signal from the 2D T2-MAS dataset occupies a small fraction of the
# two-dimensional grid. Though you may choose to proceed with the inversion
# directly onto this dataset, it is not computationally optimum. For optimum
# performance, trim the dataset to the region of relevant signals. Use the appropriate
# array indexing/slicing to select the signal region.
data_object_truncated = data_object[:, 245:-245]
plot2D(data_object_truncated)

# %%
# Linear Inversion setup
# ----------------------
#
# Dimension setup
# '''''''''''''''
#
# In a generic linear-inverse problem, one needs to define two sets of dimensions---
# the dimensions undergoing a linear transformation, and the dimensions onto which
# the inversion method transforms the data. For T2 inversion, the two sets of
# dimensions are the signal decay time dimension (``kernel dimension``) and the
# reciprocal T2 (``inverse dimension``).
data_object_truncated.dimensions[0].to("s")  # set coordinates to 's'
kernel_dimension = data_object_truncated.dimensions[0]

# %%
# Generating the kernel
# '''''''''''''''''''''
#
# Use the :class:`~mrinversion.kernel.relaxation.T2` class to generate a T2 object
# and then use its ``kernel`` method to generate the T2 relaxation kernel..
relaxT2 = relaxation.T2(
    kernel_dimension=kernel_dimension,
    inverse_dimension=dict(
        count=64, minimum="1e-3 s", maximum="1e4 s", scale="log", label="log (T2 / s)"
    ),
)
inverse_dimension = relaxT2.inverse_dimension
K = relaxT2.kernel(supersampling=1)
print(K.shape)

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
# FISTA LASSO cross-validation
# '''''''''''''''''''''''''''''
#
# We solve the inverse Laplace problem using the statistical learning ``FISTALassoCV``
# method over a range of λ values and determine the best T2 parameter distribution for
# the given 2D T2-MAS dataset.

# setup the pre-defined range of alpha and lambda values
lambdas = 10 ** (-5 + 6 * (np.arange(32) / 31))

# setup the smooth lasso cross-validation class
s_lasso = LassoFistaCV(
    lambdas=lambdas,  # A numpy array of lambda values.
    sigma=sigma,  # data standard deviation
    folds=5,  # The number of folds in n-folds cross-validation.
    inverse_dimension=inverse_dimension,  # previously defined inverse dimensions.
    max_iterations=20000,  # maximum number of allowed iterations.
)

# run the fit method on the compressed kernel and compressed data.
s_lasso.fit(K=compressed_K, s=compressed_s)

# %%
# The optimum hyper-parameters
# ''''''''''''''''''''''''''''
#
# Use the :attr:`~mrinversion.linear_model.LassoFistaCV.hyperparameters` attribute of
# the instance for the optimum hyper-parameters, $\lambda$, determined from the
# cross-validation.
print(s_lasso.hyperparameters)

# %%
# The cross-validation curve
# ''''''''''''''''''''''''''
plt.figure(figsize=(4.5, 3.5))
s_lasso.cv_plot()
plt.tight_layout()
plt.show()

# %%
# The optimum solution
# ''''''''''''''''''''
f_sol = s_lasso.f
plot2D(f_sol, interpolation="none")

# %%
# The fit residuals
# '''''''''''''''''
#
# To calculate the residuals between the data and predicted data(fit), use the
# :meth:`~mrinversion.linear_model.LassoFistaCV.residuals` method, as follows,
residuals = s_lasso.residuals(K=K, s=data_object_truncated)
plot2D(residuals)

# %%
# The standard deviation of the residuals is
residuals.std()

# %%
# Saving the solution
# '''''''''''''''''''
#
# To serialize the solution (nuclear shielding tensor parameter distribution) to a
# file, use the `save()` method of the CSDM object, for example,
f_sol.save("T2_inverse.csdf")  # save the solution
residuals.save("T2_residue.csdf")  # save the residuals
