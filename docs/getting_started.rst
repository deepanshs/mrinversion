.. _getting_started:

Getting started with `Mrinversion` package
==========================================

Anisotropic line-shape inversion
--------------------------------

We have put together a set of guidelines for using the `Mrinversion`
package and the related methods and attributes. We encourage the users
to follow these guidelines to promote consistency, amongst others.
Import the package using

.. doctest::

    >>> import mrinversion as mrinv

MAF anisotropic line-shape
^^^^^^^^^^^^^^^^^^^^^^^^^^

We start by demonstrating the inversion of a one-dimensional pure anisotropic
spectrum.

Importing the dataset
"""""""""""""""""""""

You may use any format to import your dataset, as long as the data is as a
numpy ndarray form. For this example, we use the CSDM file format. The
following lines of code are for importing the dataset.

.. doctest::

    >>> import csdmpy as cp
    >>> data_object = cp.load('test0/0.005/MAF/MAF_spectrum.csdf')

The `data_object` is the csdm object containing the one-dimension pure
anisotropic lineshape. The data coordinates from this dataset are

.. doctest::

    >>> x = data_object.dimensions[0].coordinates
    >>> y = data_object.dependent_variables[0].components[0]

The plot depicting the line-shape is

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y)
    >>> plt.gca().invert_xaxis()
    >>> plt.xlabel(f'{x.unit.physical_type} / {x.unit}')
    >>> plt.show()


Setting the direct and inverse-dimensions
"""""""""""""""""""""""""""""""""""""""""

In the linear inversion, the direct-dimension is the dimension undergoing
an inverse tranformation, while the inverse-dimension is the dimension
accessed after the inversion.
When inverting the pure anisotropic NMR spectra into a two-dimensional
distribution of tensor paramaters, the direct-dimension is the pure anisotropic
dimension, while the inverse dimensions are the two parameters of the tensor.
Here, we express the tensor parameters as piece-wise polar coordinates on an
:math:`x`-:math:`y` grid system.

In MRInversion, the dimensions, direct or indirect, is defined using the
`Dimension <https://csdmpy.readthedocs.io/en/latest/api/Dimensions.html>`_ objects
from the `CSDMpy <https://csdmpy.readthedocs.io/en/latest/index.html>`_ package.


**Direct-dimension**

Let's first set-up the direct-dimension. Because this example dataset was
imported as a CSDM object, the direct-dimension is already defined as a
Dimension object. For illustration, however, we re-define the direct-dimension
using the `Dimension <https://csdmpy.readthedocs.io/en/latest/api/Dimensions.html>`_
object, following

.. doctest::

    >>> direct_dimensions = [
    ...     cp.Dimension(type='linear', count=96, increment='208.33 Hz', complex_fft=True)
    ... ]

where the `type` describes a linear dimension sampled at every 208.33 Hz for
a total of 96 points. The boolean, `complex_fft`, indicates that the data
is a result of a Fourier transform. For more information, refer to the
`CSDMpy <https://csdmpy.readthedocs.io/en/latest/>`_ documentation. Notice,
``direct_dimensions`` is a list of Dimension objects. In Mrinversion, this list
include all dimensions that are involved in the inversion tranformation. In
this example, however, we have a single direct-dimension, representing the
pure anisotropic NMR line-shape dimension.

**Inverse-dimension**

Similarly, set up the dimensions for the inverse dimensions. In this example,
there are two inverse dimensions,

.. doctest::

    >>> inverse_dimensions = [
    ...     cp.Dimension(type='linear', count=25, increment='370 Hz'),
    ...     cp.Dimension(type='linear', count=25, increment='370 Hz')
    ... ]

where both are linear dimensions sampled at every 370 Hz for 25 points. The
coordinates along the dimension at index 0 or 1 are

.. doctest::

    >>> inverse_dimensions[0].coordinates
    [   0.  370.  740. 1110. 1480. 1850. 2220. 2590. 2960. 3330. 3700. 4070.
    4440. 4810. 5180. 5550. 5920. 6290. 6660. 7030. 7400. 7770. 8140. 8510.
    8880.] Hz


Setting the kernel
""""""""""""""""""


.. figure:: _images/test.*
    :figclass: figure-polaroid

.. seealso::

    :ref:`csdm_api`, :ref:`dim_api`, :ref:`dv_api`,
    `Quantity <http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html#astropy.units.Quantity>`_,
    `numpy array <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.html>`_,
    `Matplotlib library <https://matplotlib.org>`_
