
=========================================
Getting started with relaxation inversion
=========================================

Let's examine the inversion of a NMR signal decay from :math:`T_2` relaxation measurement
into a 1D distribution of :math:`T_2` parameters. For illustrative purposes,
we use a synthetic one-dimensional signal decay.

**Import relevant modules**

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib import rcParams
    >>> from mrinversion.kernel import relaxation
    ...
    >>> rcParams['pdf.fonttype'] = 42 # for exporting figures as illustrator editable pdf.


Import the dataset
------------------

The first step is getting the dataset. In this example, we get the data
from a CSDM [#f1]_ compliant file-format. Import the
`csdmpy <https://csdmpy.readthedocs.io/en/latest/>`_ module and load the dataset as
follows,

.. note::

    The CSDM file-format is a new open-source universal file format for multi-dimensional
    datasets. It is supported by NMR programs such as SIMPSON [#f2]_, DMFIT [#f3]_, and
    RMN [#f4]_. A python package supporting CSDM file-format,
    `csdmpy <https://csdmpy.readthedocs.io/en/latest/>`_, is also available.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> import csdmpy as cp
    ...
    >>> filename = "https://ssnmr.org/resources/mrinversion/test3_signal.csdf"
    >>> data_object = cp.load(filename) # load the CSDM file with the csdmpy module

Here, the variable *data_object* is a `CSDM <https://csdmpy.readthedocs.io/en/latest/api/CSDM.html>`_
object. For comparison, let's also import the true t2 distribution from which the synthetic 1D signal decay is simulated.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> datafile = "https://ssnmr.org/resources/mrinversion/test3_t2.csdf"
    >>> true_t2_dist = cp.load(datafile) # the true solution for comparison


The following is the plot of the NMR signal decay as well as the corresponding
true probability distribution.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> _, ax = plt.subplots(1, 2, figsize=(9, 3.5), subplot_kw={'projection': 'csdm'}) # doctest: +SKIP
    >>> ax[0].plot(data_object) # doctest: +SKIP
    >>> ax[0].set_xlabel('time / s') # doctest: +SKIP
    >>> ax[0].set_title('NMR signal decay') # doctest: +SKIP
    ...
    >>> ax[1].plot(true_t2_dist) # doctest: +SKIP
    >>> ax[1].set_title('True distribution') # doctest: +SKIP
    >>> ax[1].set_xlabel('log(T2 / s)') # doctest: +SKIP
    >>> plt.tight_layout() # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP


.. _fig1_getting_started_relaxation:
.. figure:: _static/null.*

    The figure on the left is the NMR signal decay corresponding
    to :math:`T_2` distribution shown on the right.

Generating the kernel
---------------------

Import the :py:class:`~mrinversion.kernel.relaxation.T2` class and
generate the kernel as follows,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> from mrinversion.kernel.relaxation import T2
    >>> relaxT2 = T2(
    ...     kernel_dimension = data_object.dimensions[0],
    ...     inverse_dimension=dict(
    ...         count=64, minimum="1e-2 s", maximum="1e3 s", scale="log", label="log (T2 / s)"
    ...     )
    ... )
    >>> inverse_dimension = relaxT2.inverse_dimension

In the above code, the variable ``relaxT2`` is an instance of the
:py:class:`~mrinversion.kernel.relaxation.T2` class. The two required
arguments of this class are the *kernel_dimension* and *inverse_dimension*.
The *kernel_dimension* is the dimension over which the signal relaxation measurements are acquired. In this case, this referes to the time at which  the relaxation measurement was performed.
The *inverse_dimension* is the dimension over which the T2 distribution is
evaluated. In this case, the inverse dimension is a log-linear scale spanning from 10 ms to 1000 s in 64 steps.

Once the *T2* instance is created, use the
:py:meth:`~mrinversion.kernel.relaxation.T2.kernel` method of the
instance to generate the relaxation kernel, as follows,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> K = relaxT2.kernel(supersampling=20)
    >>> print(K.shape)
    (25, 64)

Here, ``K`` is the :math:`25\times 64` kernel, where the 25 is the number of samples (time measurements), and 64 is the number of features (T2). The argument *supersampling* is the supersampling
factor. In a supersampling scheme, each grid cell is averaged over a :math:`n`
sub-grid, where :math:`n` is the supersampling factor.

Data compression (optional)
---------------------------

Often when the kernel, K, is ill-conditioned, the solution becomes unstable in
the presence of the measurement noise. An ill-conditioned kernel is the one
whose singular values quickly decay to zero. In such cases, we employ the
truncated singular value decomposition method to approximately represent the
kernel K onto a smaller sub-space, called the *range space*, where the
sub-space kernel is relatively well-defined. We refer to this sub-space
kernel as the *compressed kernel*. Similarly, the measurement data on the
sub-space is referred to as the *compressed signal*. The compression also
reduces the time for further computation. To compress the kernel and the data,
import the :py:class:`~mrinversion.linear_model.TSVDCompression` class and follow,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> from mrinversion.linear_model import TSVDCompression
    >>> new_system = TSVDCompression(K=K, s=data_object)
    compression factor = 1.0416666666666667
    >>> compressed_K = new_system.compressed_K
    >>> compressed_s = new_system.compressed_s

Here, the variable ``new_system`` is an instance of the
:py:class:`~mrinversion.linear_model.TSVDCompression` class. If no truncation index is
provided as the argument, when initializing the ``TSVDCompression`` class, an optimum
truncation index is chosen using the maximum entropy method [#f5]_, which is the default
behavior. The attributes :py:attr:`~mrinversion.linear_model.TSVDCompression.compressed_K`
and :py:attr:`~mrinversion.linear_model.TSVDCompression.compressed_s` holds the
compressed kernel and signal, respectively. The shape of the original signal *v.s.* the
compressed signal is

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> print(data_object.shape, compressed_s.shape)
    (25,) (24,)


Statistical learning of relaxation parameters
---------------------------------------------

The solution from a linear model trained with l1, such as the FISTA estimator used here, depends on the choice of the hyperparameters.
To find the optimum hyperparameter, we employ the statistical learning-based model, such as the
*n*-fold cross-validation.

The :py:class:`~mrinversion.linear_model.LassoFistaCV` class is designed to solve the l1 problem for a range of :math:`\lambda` values and
determine the best solution using the *n*-fold cross-validation. Here, we search the
best model using a 5-fold cross-validation statistical learning method. The :math:`\lambda` values are sampled uniformly on a logarithmic scale as,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> lambdas = 10 ** (-7 + 6 * (np.arange(64) / 63))


Fista LASSO cross-validation Setup
''''''''''''''''''''''''''''''''''

Setup the smooth lasso cross-validation as follows

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> from mrinversion.linear_model import LassoFistaCV
    >>> f_lasso_cv = LassoFistaCV(
    ...     lambdas=lambdas,
    ...     inverse_dimension=inverse_dimension,
    ...     sigma=0.0008,
    ...     folds=5,
    ... )
    >>> f_lasso_cv.fit(K=compressed_K, s=compressed_s)

The arguments of the :py:class:`~mrinversion.linear_model.LassoFistaCV` is a list
of the *lambda* values, along with the standard deviation of the
noise, *sigma*. The value of the argument *folds* is the number of folds used in the
cross-validation. As before, to solve the problem, use the
:meth:`~mrinversion.linear_model.LassoFistaCV.fit` method, whose arguments are
the kernel and signal.

The optimum hyperparameters
'''''''''''''''''''''''''''

The optimized hyperparameters may be accessed using the
:py:attr:`~mrinversion.linear_model.LassoFistaCV.hyperparameters` attribute of
the class instance,

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> lam = f_lasso_cv.hyperparameters['lambda']

The cross-validation curve
''''''''''''''''''''''''''

The cross-validation error metric is the mean square error metric. You may plot this
data using the :py:attr:`~mrinversion.linear_model.LassoFistaCV.cv_plot`
function.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> plt.figure(figsize=(5, 3.5)) # doctest: +SKIP
    >>> f_lasso_cv.cv_plot() # doctest: +SKIP
    >>> plt.tight_layout() # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. _fig3_getting_started_relaxation:
.. figure:: _static/null.*

    The five-folds cross-validation prediction error curve as a function of
    the hyperparameter :math:`\lambda`.

The optimum solution
''''''''''''''''''''

The best model selection from the cross-validation method may be accessed using
the :py:attr:`~mrinversion.linear_model.LassoFistaCV.f` attribute.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> f_sol_cv = f_lasso_cv.f  # best model selected using the 5-fold cross-validation

The plot of the selected T2 parameter distribution is shown below.

.. plot::
    :format: doctest
    :context: close-figs
    :include-source:

    >>> plt.figure(figsize=(4, 3)) # doctest: +SKIP
    >>> plt.subplot(projection='csdm') # doctest: +SKIP
    >>> plt.plot(true_t2_dist / true_t2_dist.max(), label='True distribution') # doctest: +SKIP
    >>> plt.plot(f_sol_cv / f_sol_cv.max(), label='Optimum distribution') # doctest: +SKIP
    >>> plt.legend() # doctest: +SKIP
    >>> plt.tight_layout() # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

.. _fig4_getting_started_relaxation:
.. figure:: _static/null.*

    The figure depicts the comparision of the true T2 distribution and optimal T2 distribution
    solutiom from five-fold cross-validation.


.. seealso::

    `csdmpy <https://csdmpy.readthedocs.io/en/latest/>`_,
    `Quantity <http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity>`_,
    `numpy array <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.html>`_,
    `Matplotlib library <https://matplotlib.org>`_

.. [#f1] Srivastava, D. J., Vosegaard, T., Massiot, D., Grandinetti, P. J.,
            Core Scientific Dataset Model: A lightweight and portable model and
            file format for multi-dimensional scientific data. PLOS ONE,
            **15**, 1-38, (2020).
            `DOI:10.1371/journal.pone.0225953 <https://doi.org/10.1371/journal.pone.0225953>`_

.. [#f2] Bak M., Rasmussen J. T., Nielsen N.C., SIMPSON: A General Simulation Program for
            Solid-State NMR Spectroscopy. J Magn Reson. **147**, 296–330, (2000).
            `DOI:10.1006/jmre.2000.2179 <https://doi.org/10.1006/jmre.2000.2179>`_

.. [#f3] Massiot D., Fayon F., Capron M., King I., Le Calvé S., Alonso B., et al. Modelling
            one- and two-dimensional solid-state NMR spectra. Magn Reson Chem. **40**, 70–76,
            (2002) `DOI:10.1002/mrc.984 <https://doi.org/10.1002/mrc.984>`_

.. [#f4] PhySy Ltd. RMN 2.0; 2019. Available from: https://www.physyapps.com/rmn.

.. [#f5] Varshavsky R., Gottlieb A., Linial M., Horn D., Novel unsupervised feature filtering
            of biological data. Bioinformatics, **22**, e507–e513, (2006).
            `DOI:10.1093/bioinformatics/btl214 <https://doi.org/10.1093/bioinformatics/btl214>`_.
