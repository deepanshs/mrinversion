
====================================
Getting started with ``mrinversion``
====================================

We have put together a set of guidelines for using the `mrinversion` package.
We encourage our users to follow these guidelines to promote consistency.

Considering the NMR audience, what better example to start with than a spinning
sideband spectrum. For illustrative purposes, we use a synthetic one-dimensional
purely anisotropic spinning sideband spectrum. You may consider this as a
cross-section of your 2D MAT/PASS dataset.

Okay, let's start with the prediction of the nuclear shielding tensor parameters
from this spectrum.

Import the dataset
------------------

The first step is getting the sideband spectrum. In this example, we get the data
from a CSDM [#f1]_ compliant file-format.

.. note::

    The CSDM file-format is supported by most NMR software, such as SIMPSON, DMFIT, RMN.
    A python package supporting CSDM file-format, csdmpy, is also available.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> import csdmpy as cp
    ...
    >>> # the spinning sideband data file as a CSDM object.
    >>> filename = "https://osu.box.com/shared/static/xnlhecn8ifzcwx09f83gsh27rhc5i5l6.csdf"
    >>> data_object = cp.load(filename) # load the CSDM file with the csdmpy module

Here, the variable ``data_object`` is the
`CSDM <https://csdmpy.readthedocs.io/en/latest/api/CSDM.html>`_
object containing a one-dimension pure anisotropic spinning sideband spectrum.
The coordinates and the corresponding responses from this dataset are

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> # convert the dimension coordinates from `Hz` to `ppm`.
    >>> data_object.dimensions[0].to('ppm', 'nmr_frequency_ratio')
    >>> coordinates = data_object.dimensions[0].coordinates.value
    >>> responses = data_object.dependent_variables[0].components[0]

For comparison, let's also include the true probability distribution from which the
synthetic spinning sideband dataset is derived.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> # the true probability distribution dataset as a CSDM object.
    >>> datafile = "https://osu.box.com/shared/static/lufeus68orw1izrg8juthcqvp7w0cpzk.csdf"
    >>> true_data_object = cp.load(datafile) # the true solution for comparison


The following is the plot of the spinning sideband spectrum as well as the corresponding
true probability distribution.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> import matplotlib.pyplot as plt
    >>> from mrinversion.utils import get_polar_grids
    >>> import numpy as np
    ...
    >>> _, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={'projection': 'csdm'})
    >>> line = ax[0].stem(coordinates, responses, markerfmt=' ', use_line_collection=True)
    >>> plt.setp(line, color="black", linewidth=2) # doctest: +SKIP
    >>> ax[0].set_xlabel('frequency / ppm') # doctest: +SKIP
    >>> ax[0].invert_xaxis() # doctest: +SKIP
    ...
    ...
    >>> def twoD_plot(ax, csdm_object, title=''):
    ...     # convert the dimension from `Hz` to `ppm`.
    ...     csdm_object.dimensions[0].to('ppm', 'nmr_frequency_ratio')
    ...     csdm_object.dimensions[1].to('ppm', 'nmr_frequency_ratio')
    ...
    ...     levels = (np.arange(9)+1)/10
    ...     ax.contourf(csdm_object, cmap='gist_ncar', levels=levels)
    ...     ax.grid(None)
    ...     ax.set_title(title)
    ...     ax.set_aspect("equal")
    ...
    ...     # The get_polar_grids method place a polar zeta-eta grid on the background.
    ...     get_polar_grids(ax)
    ...
    >>> twoD_plot(ax[1], true_data_object, title='True distribution') # doctest: +SKIP
    >>> plt.tight_layout() # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

The figure on the left is the synthetic spinning sideband dataset for
the nuclear shielding tensor distribution shown on the right. In the figure
on the right, the parameter ζ is the radial dimension, and η is the angular
dimension, defined in Eq. :eq:`zeta_eta_def`. The region in blue and red
corresponds to the positive and negative values of ζ. The radial grid lines
are drawn at every 20 ppm increments of ζ, and the angular grid lines are drawn
at every 0.2 increments of η. The `x` and `y`-axis are η = 0, and the diagonal is
η = 1.


Setting the kernel
------------------

A kernel is a transformation matrix that transforms the single from domain to
the range space following

.. math::

    {\bf s = Kf},

where :math:`\bf K` is the transformation kernel, :math:`\bf s` is the observed signal,
and :math:`\bf f` is the unknown which resides in the domain space, respectively.

.. In `Mrinversion`, the range space is a sub-space of the signal, which is

.. we describe the domain-space with the inverse dimensions, and
.. the domain space is the part of the signal space where the data is sampled.
.. Note, the dimensionality of the inverse-dimension is not necessarily the
.. inverse of the respective direct-dimension dimensionality. This relationship
.. depends on the kernel transforming the direct-dimension to the
.. inverse-dimension.

In this example, the range-space is the signal dimension where the pure
anisotropic spinning sideband amplitudes are sampled. The domain-space
corresponds to the two dimensions relating to the two anisotropic
tensor parameters of the nuclear shielding tensor, :math:`\zeta`, and
:math:`\eta`. We express these two tensor parameters on a piece-wise polar
coordinate given as

.. math::
    :label: zeta_eta_def

    x = \left\{ \begin{array}{l r}
                |\zeta|\sin\theta, & \forall \zeta\ge0, \\
                |\zeta|\cos\theta, & \text{elsewhere}
               \end{array}
        \right. \\
    y = \left\{ \begin{array}{l r}
                |\zeta|\cos\theta, & \forall \zeta\ge0, \\
                |\zeta|\sin\theta, & \text{elsewhere}
               \end{array}
        \right.

where :math:`\theta=\pi\eta/4`.

In ``mrinversion``, the range and domain space dimensions are defined using the
`Dimension <https://csdmpy.readthedocs.io/en/latest/api/Dimensions.html>`_ objects
from the `csdmpy <https://csdmpy.readthedocs.io/en/latest/>`_ package.
For nuclear shielding tensor line-shape kernel, we refer the range space
dimensions as the `anisotropic_dimension`, and the domain space dimensions as
the `inverse_dimension`.

Anisotropic dimension
'''''''''''''''''''''

Because this example dataset is imported as a CSDM object, the `anisotropic_dimension`
is already defined as a
`CSDM Dimension <https://csdmpy.readthedocs.io/en/latest/api/Dimensions.html>`_
object. For illustration, however, we re-define the `anisotropic_dimension` as
follows,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> anisotropic_dimension = cp.LinearDimension(count=32, increment='625Hz', coordinates_offset='-10kHz')
    >>> print(anisotropic_dimension)
    LinearDimension([-10000.  -9375.  -8750.  -8125.  -7500.  -6875.  -6250.  -5625.  -5000.
      -4375.  -3750.  -3125.  -2500.  -1875.  -1250.   -625.      0.    625.
       1250.   1875.   2500.   3125.   3750.   4375.   5000.   5625.   6250.
       6875.   7500.   8125.   8750.   9375.] Hz)


Inverse dimension
'''''''''''''''''

Similarly, set up the two inverse dimensions. Here, the two inverse dimensions
are

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> inverse_dimension = [
    ...     cp.LinearDimension(count=25, increment='370 Hz', label='x'),  # the x-coordinates
    ...     cp.LinearDimension(count=25, increment='370 Hz', label='y')   # the y-coordinates
    ... ]

sampled at every 370 Hz for 25 points. The inverse dimension at index 0 and 1
are the `x` and `y` dimensions, respectively.


Setting the Kernel
------------------

Import the :class:`~mrinversion.kernel.NuclearShieldingLineshape` class and
generate the kernel as follows,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> from mrinversion.kernel import NuclearShieldingLineshape
    >>> lineshapes = NuclearShieldingLineshape(
    ...                 anisotropic_dimension=anisotropic_dimension,
    ...                 inverse_dimension=inverse_dimension,
    ...                 channel='29Si',
    ...                 magnetic_flux_density='9.4 T',
    ...                 rotor_angle='54.735 deg',
    ...                 rotor_frequency='625 Hz',
    ...                 number_of_sidebands=32
    ...             )

In the above code, the variable ``lineshapes`` is an instance of the
:class:`~mrinversion.kernel.NuclearShieldingLineshape` class. The two required arguments
of this class are the `anisotropic_dimension` and `inverse_dimension`, as defined
previously. The optional arguments are the metadata that describes the environment
under which the spectrum is acquired. In this example, these arguments describe a
:math:`^{29}\text{Si}` pure anisotropic spinning-sideband spectrum acquired at 9.4 T
magnetic flux density and spinning at the magic angle (:math:`54.735^\circ`) at 625 Hz.
The value of the `rotor_frequency` argument is the effective anisotropic modulation
frequency. For measurements like PASS [#f2]_, the anisotropic modulation frequency is
the physical rotor frequency. For other measurements like the extended chemical shift
modulation sequences (XCS) [#f3]_, or its variants, the effective anisotropic modulation
frequency is lower than the physical rotor frequency and should be set appropriately.

The argument `number_of_sidebands` is the maximum number of sidebands that will be
computed per line-shape within the kernel. For most two-dimensional isotropic v.s. pure
anisotropic spinning-sideband correlation measurements, the sampling along the sideband
dimension is the rotor or the effective anisotropic modulation frequency. Therefore, the
value of the `number_of_sidebands` argument is usually the number of points along the
sideband dimension. In this example, this value is 32.

Once the instance is created, used the
:meth:`~mrinversion.kernel.NuclearShieldingLineshape.kernel` method of the
instance to generate the spinning sideband kernel, as follows,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> K = lineshapes.kernel(supersampling=1)
    >>> print(K.shape)
    (32, 625)

Here, ``K`` is the :math:`32\times 625` kernel, where the 32 is the number of samples
(sideband amplitudes), and 625 is the number of features (line-shapes) on the
:math:`25 \times 25` `x`-`y` grid. The argument `supersampling` is the supersampling
factor. In a supersampling scheme, each grid cell is averaged over a :math:`n\times n`
sub-grid, where :math:`n` is the supersampling factor. A supersampling factor of 1 is
equivalent to no sub-grid averaging.


Data compression (optional)
---------------------------

Often when the kernel, K, is ill-conditioned, the solution becomes unstable in
the presence of the measurement noise. An ill-conditioned kernel is the one
whose singular values quickly decay to zero. In such cases, we employ the
truncated singular value decomposition method to approximately represent the
kernel K onto a smaller sub-space, called the `range space`, where the
sub-space kernel is relatively well-defined. We refer to this sub-space
kernel as the `compressed kernel`. Similarly, the measurement data on the
sub-space is referred to as the `compressed signal`. The compression also
reduces the time for further computation. To compress the kernel and the data,
import the :class:`~mrinversion.linear_model.TSVDCompression` class and follow,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> from mrinversion.linear_model import TSVDCompression
    >>> new_system = TSVDCompression(K, data_object)
    compression factor = 1.032258064516129
    >>> compressed_K = new_system.compressed_K
    >>> compressed_s = new_system.compressed_s

Here, the variable ``new_system`` is an instance of the
:class:`~mrinversion.linear_model.TSVDCompression` class. If no truncation index is
provided as the argument, when initializing the ``TSVDCompression`` class, an optimum
truncation index is chosen using the maximum entropy method, which is the default
behavior. The attributes :attr:`~mrinversion.linear_model.TSVDCompression.compressed_K`
and :attr:`~mrinversion.linear_model.TSVDCompression.compressed_s` holds the
compressed kernel and signal, respectively. The shape of the original signal `v.s.` the
compressed signal is

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> print(data_object.shape, compressed_s.shape)
    (32,) (31,)


Setting up the inverse problem
------------------------------

When setting up the inversion, we solved the smooth LASSO [#f4]_ problem of
form

.. math::
        \| {\bf Kf - s} \|^2_2 + \alpha \sum_{i=1}^{d} \| {\bf J}_i {\bf f} \|_2^2
                    + \lambda  \| {\bf f} \|_1 ,

where :math:`{\bf K}` is the kernel, :math:`{\bf s}` is the known signal
containing noise, and :math:`{\bf f}` is the desired solution. The parameters
:math:`\alpha` and :math:`\lambda` are the hyperparameters controlling the
smoothness and sparsity of the solution :math:`{\bf f}`. See the documentation
for the :class:`~mrinversion.linear_model.SmoothLasso` class for details.

Import the :class:`~mrinversion.linear_model.SmoothLasso` class and follow,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> from mrinversion.linear_model import SmoothLasso
    >>> s_lasso = SmoothLasso(alpha=0.01, lambda1=1e-04, inverse_dimension=inverse_dimension)

Here, the variable ``s_lasso`` is an instance of the
:class:`~mrinversion.linear_model.SmoothLasso` class. The required arguments
of this class are `alpha` and `lambda1`, corresponding to the hyperparameters
:math:`\alpha` and :math:`\lambda`, respectively, in the above equation. At the
moment we don't know the optimum value of the `alpha` and `lambda1` parameters.
Let's start with a guess value.
The argument `f_shape` is the shape of the solution given as the number
of points along the inverse
dimension at index 0, followed by points at index 1. In this example, this
value is (25, 25).

To solve the smooth lasso problem, use the
:meth:`~mrinversion.linear_model.SmoothLasso.fit` method of the ``s_lasso``
instance as follows,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> s_lasso.fit(K=compressed_K, s=compressed_s)

The two arguments of the :meth:`~mrinversion.linear_model.SmoothLasso.fit`
method are the kernel, `K`, the signal, `s`, and the shape of the solution `f`,
`f_shape`. In the above example, we set the value of `K` as ``compressed_K``,
and correspondingly the value of `s` as ``compressed_s``. You may also use the
uncompressed values of the kernel and signal in this method.


The solution to the smooth lasso is accessed using the
:attr:`~mrinversion.linear_model.SmoothLasso.f` attribute of the respective
``s_lasso`` object.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> f_sol = s_lasso.f

The plot of the solution is

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> _, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={'projection': 'csdm'}) # doctest: +SKIP
    >>> twoD_plot(ax[0], f_sol/f_sol.max(), title='Guess distribution') # doctest: +SKIP
    >>> twoD_plot(ax[1], true_data_object, title='True distribution') # doctest: +SKIP
    >>> plt.tight_layout() # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

The figure on the left is the solution to the nuclear shielding
tensor distribution derived from the inversion of the spinning
sideband dataset. The figure on the right is the true nuclear
shielding tensor distribution. The ζ and η coordinates are depicted
as piecewise polar, where ζ is the radial dimension, and η is the angular
dimension, defined in Eq. :eq:`zeta_eta_def`. The region in blue and red
corresponds to the positive and negative values of ζ.  The radial grid lines
are drawn at every 20 ppm increment of ζ, and the angular grid lines are
drawn at every 0.2 increment of η. The `x` and `y` axis are η = 0, and the
diagonal is η = 1.


You may also evaluate the spectrum predicted from the solution using the
:meth:`~mrinversion.linear_model.SmoothLasso.predict` method of the object as
follows,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> predicted_signal = s_lasso.predict(K)

The argument of the `predict` method is the kernel. We provide the original
kernel K because we desire the prediction of the original data and not the
compressed data.


Statistical learning of tensors
-------------------------------

The linear model trained with the combined l1 and l2 priors,
such as the smooth LASSO estimator used here, the solution depends on the
choice of the hyperparameters.
The solution shown in the above figure is when :math:`\alpha=0.1` and
:math:`\lambda=1\times 10^{-4}`. Although it's a solution, it is unknown if
this is the best solution. For this, we employ the statistical learning-based
model, such as the `n`-fold cross-validation.

The following :class:`~mrinversion.linear_model.SmoothLassoCV` class

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> from mrinversion.linear_model import SmoothLassoCV

is designed to solve the smooth-lasso problem for a range of :math:`\alpha`
and :math:`\lambda` values and determine the best solution using the `n`-fold
cross-validation. Here, we search the best model on a :math:`10 \times 10`
:math:`\alpha`-:math:`\lambda` grid, using a 10-fold cross-validation
statistical learning method. The :math:`\lambda` and :math:`\alpha` values are
sampled uniformly on a logarithmic scale as,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> lambdas = 10 ** (-4 - 2 * (np.arange(10) / 9))
    >>> alphas = 10 ** (-3 - 2 * (np.arange(10) / 9))

Setup the smooth lasso cross-validation using

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> s_lasso_cv = SmoothLassoCV(alphas=alphas, lambdas=lambdas,
    ...                            inverse_dimension=inverse_dimension,
    ...                            sigma=0.005, folds=10)
    >>> s_lasso_cv.fit(K=compressed_K, s=compressed_s)

The arguments of the :class:`~mrinversion.linear_model.SmoothLassoCV` is a list
of the `alpha` and `lambda` values, along with the standard deviation of the
noise, `sigma`. The value of the argument `folds` is the number of folds in the
cross-validation. As before, to solve the problem, use the
:meth:`~mrinversion.linear_model.SmoothLassoCV.fit` method, whose arguments are
the kernel, signal, and shape of the solution.

The optimized hyperparameters may be accessed using the
:attr:`~mrinversion.linear_model.SmoothLassoCV.hyperparameters` attribute of
the class instance,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> alpha = s_lasso_cv.hyperparameter['alpha']
    >>> lambda_1 = s_lasso_cv.hyperparameter['lambda']

and the corresponding cross-validation error surface using the
:attr:`~mrinversion.linear_model.SmoothLassoCV.cv_map` attribute.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> plt.figure(figsize=(5, 3.5)) # doctest: +SKIP
    >>> ax = plt.subplot(projection='csdm') # doctest: +SKIP
    >>> ax.contour(np.log10(s_lasso_cv.cv_map), levels=25) # doctest: +SKIP
    >>> ax.scatter(-np.log10(s_lasso_cv.hyperparameter['alpha']),
    ...         -np.log10(s_lasso_cv.hyperparameter['lambda']),
    ...         marker='x', color='k') # doctest: +SKIP
    >>> plt.tight_layout() # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

The ten-folds cross-validation prediction error surface as
a function of hyperparameters :math:`\alpha` and :math:`\beta`.

The best model selection from the cross-validation method may be accessed using
the :attr:`~mrinversion.linear_model.SmoothLassoCV.f` attribute.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> f_sol_cv = s_lasso_cv.f  # best model selected using the 10-fold cross-validation

The probability distribution of the selected model

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> _, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={'projection': 'csdm'}) # doctest: +SKIP
    >>> twoD_plot(ax[0], f_sol_cv/f_sol_cv.max(), title='Optimum distribution') # doctest: +SKIP
    >>> twoD_plot(ax[1], true_data_object, title='True distribution') # doctest: +SKIP
    >>> plt.tight_layout() # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

The figure on the left is the best model selected by the 10-folds
cross-validation method. The figure on the right is the true model of the
nuclear shielding tensor distribution. The ζ and η coordinates are depicted
as piecewise polar, where ζ is the radial dimension, and η is the angular
dimension, defined in Eq. :eq:`zeta_eta_def`. The region in blue and red
corresponds to the positive and negative values of ζ.  The radial grid lines
are drawn at every 20 ppm increment of ζ, and the angular grid lines are
drawn at every 0.2 increment of η. The `x` and `y` axis are η = 0, and the
diagonal is η = 1.


.. seealso::

    `csdmpy <https://csdmpy.readthedocs.io/en/latest/>`_,
    `Quantity <http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity>`_,
    `numpy array <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.html>`_,
    `Matplotlib library <https://matplotlib.org>`_


.. [#f1] Srivastava, D. J., Vosegaard, T., Massiot, D., Grandinetti, P. J.,
            Core Scientific Dataset Model: A lightweight and portable model and
            file format for multi-dimensional scientific data, PLOS ONE,
            **15**, 1-38, (2020).
            `DOI:10.1371/journal.pone.0225953 <https://doi.org/10.1371/journal.pone.0225953>`_

.. [#f2] Dixon, W. T., Spinning‐sideband‐free and spinning‐sideband‐only NMR
            spectra in spinning samples. J. Chem. Phys, **77**, 1800, (1982).
            `DOI:10.1063/1.444076 <https://doi.org/10.1063/1.444076>`_

.. [#f3] Gullion, T., Extended chemical-shift modulation, J. Mag. Res., **85**, 3, (1989).
            `10.1016/0022-2364(89)90253-9 <https://doi.org/10.1016/0022-2364(89)90253-9>`_

.. [#f4] Hebiri M, Sara A. Van De Geer, The Smooth-Lasso and other l1+l2-penalized
            methods, arXiv (2010). `arXiv:1003.4885v2 <https://arxiv.org/abs/1003.4885v2>`_
