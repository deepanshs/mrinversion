#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inverse Laplace (ILT) T2 distribution
=====================================
"""
# %%
# The following example demonstrates the statistical learning based determination of
# the nuclear shielding tensor parameters from a one-dimensional cross-section of a
# spinning sideband correlation spectrum. In this example, we use a synthetic
# sideband amplitude spectrum from a unimodal tensor distribution.
#
# Before getting started
# ----------------------
#
# Import all relevant packages.
import csdmpy as cp
import matplotlib.pyplot as plt
import numpy as np

from mrinversion.kernel import relaxation
from mrinversion.linear_model import LassoFistaCV, TSVDCompression

# sphinx_gallery_thumbnail_number = 3

# %%
# Dataset setup
# -------------
#
# Generate a dataset
# ''''''''''''''''''
#
time = np.arange(128) * 0.3 + 0.01  # in s
T2_samples = [1, 10]  # in s
T2_weights = [1, 2]

signal = 0
for wt, t2 in zip(T2_weights, T2_samples):
    signal += wt * np.exp(-time / t2)

signal += np.random.normal(0, 0.01, size=signal.size)
signal = cp.as_csdm(signal)
signal.dimensions[0] = cp.as_dimension(array=time, unit="s")

plt.figure(figsize=(4.5, 3.5))
signal.plot()
plt.tight_layout()
plt.show()

# %%
# Linear Inversion setup
# ----------------------
#
# Generating the kernel
# '''''''''''''''''''''
#
kernel_dimension = signal.dimensions[0]

relaxT2 = relaxation.T2(
    kernel_dimension=kernel_dimension,
    inverse_dimension=dict(
        count=64, minimum="1e-2 s", maximum="1e3 s", scale="log", label="log (T2 / s)"
    ),
)
inverse_dimension = relaxT2.inverse_dimension
K = relaxT2.kernel(supersampling=1)
print(K.shape)

# %%
# Data Compression
# ''''''''''''''''
#
new_system = TSVDCompression(K, signal)
compressed_K = new_system.compressed_K
compressed_s = new_system.compressed_s

print(f"truncation_index = {new_system.truncation_index}")

# %%
# Statistical learning of the tensors
# -----------------------------------
#
# Fista LASSO cross-validation
# '''''''''''''''''''''''''''''
#
# Create a guess range of values for the :math:`\lambda`
# hyperparameters.
# The following code generates a range of :math:`\lambda` and :math:`\alpha` values
# that are uniformly sampled on the log scale.
lambdas = 10 ** (-6 + 6 * (np.arange(64) / 63))

# setup the smooth lasso cross-validation class
f_lasso_cv = LassoFistaCV(
    lambdas=lambdas,  # A numpy array of lambda values.
    folds=5,  # The number of folds in n-folds cross-validation.
    inverse_dimension=inverse_dimension,  # previously defined inverse dimensions.
    randomize=True,
    times=15,
)

# run the fit method on the compressed kernel and compressed data.
f_lasso_cv.fit(K=compressed_K, s=compressed_s)

# %%
# The optimum hyper-parameters
# ''''''''''''''''''''''''''''
#
# Use the :attr:`~mrinversion.linear_model.SmoothLassoCV.hyperparameters` attribute of
# the instance for the optimum hyper-parameters, :math:`\alpha` and :math:`\lambda`,
# determined from the cross-validation.
print(f_lasso_cv.hyperparameters)

# %%
# The cross-validation surface
# ''''''''''''''''''''''''''''
#
# Optionally, you may want to visualize the cross-validation error curve/surface. Use
# the :attr:`~mrinversion.linear_model.SmoothLassoCV.cross_validation_curve` attribute
# of the instance, as follows. The cross-validation metric is the mean square error
# (MSE).
plt.figure(figsize=(4.5, 3.5))
f_lasso_cv.cross_validation_curve.plot()
plt.tight_layout()
plt.show()


# %%
# The optimum solution
# ''''''''''''''''''''
#
# The :attr:`~mrinversion.linear_model.SmoothLassoCV.f` attribute of the instance holds
# the solution.
plt.figure(figsize=(4.5, 3.5))
f_lasso_cv.f.plot()
[plt.axvline(np.log10(sample), c="r", linestyle="--") for sample in T2_samples]
plt.tight_layout()
plt.show()


# %%
# Residuals
# '''''''''
residuals = f_lasso_cv.residuals(K=K, s=signal)

plt.figure(figsize=(4.5, 3.5))
residuals.plot()
plt.tight_layout()
plt.show()
