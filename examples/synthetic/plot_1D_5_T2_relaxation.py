#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inverse Laplace (ILT) T2 distribution
=====================================
"""
# %%
# The following example demonstrates the statistical learning based determination of
# the NMR T2 relaxation vis inverse Laplace transformation.
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
time = 2 ** (np.arange(80) * 0.13 - 3)  # in s
T2_samples = [1, 30]  # in s
T2_weights = [1, 2]

signal = 0
for wt, t2 in zip(T2_weights, T2_samples):
    signal += wt * np.exp(-time / t2)

sigma = 0.01
signal += np.random.normal(0, sigma, size=signal.size)
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

# %%
# Data Compression
# ''''''''''''''''
#
new_system = TSVDCompression(K, signal)
compressed_K = new_system.compressed_K
compressed_s = new_system.compressed_s

print(f"truncation_index = {new_system.truncation_index}")

# %%
# Fista LASSO cross-validation
# '''''''''''''''''''''''''''''
#
# Create a guess range of values for the :math:`\lambda` hyperparameters.
lambdas = 10 ** (-3 + 4 * (np.arange(16) / 15))

# setup the smooth lasso cross-validation class
f_lasso_cv = LassoFistaCV(
    lambdas=lambdas,  # A numpy array of lambda values.
    folds=5,  # The number of folds in n-folds cross-validation.
    sigma=sigma,  # noise standard deviation
    inverse_dimension=inverse_dimension,  # previously defined inverse dimensions.
)

# run the fit method on the compressed kernel and compressed data.
f_lasso_cv.fit(K=compressed_K, s=compressed_s)

# %%
# The optimum hyper-parameters
# ''''''''''''''''''''''''''''
print(f_lasso_cv.hyperparameters)

# %%
# The cross-validation curve
# ''''''''''''''''''''''''''
plt.figure(figsize=(4.5, 3.5))
f_lasso_cv.cv_plot()
plt.tight_layout()
plt.show()

# %%
# The optimum solution
# ''''''''''''''''''''
plt.figure(figsize=(4.5, 3.5))
f_lasso_cv.f.plot()
[plt.axvline(np.log10(sample), c="r", linestyle="--") for sample in T2_samples]
plt.tight_layout()
plt.show()


# %%
# Residuals
# '''''''''
residuals = f_lasso_cv.residuals(K=K, s=signal)
print(residuals.std())

plt.figure(figsize=(4.5, 3.5))
residuals.plot()
plt.tight_layout()
plt.show()
