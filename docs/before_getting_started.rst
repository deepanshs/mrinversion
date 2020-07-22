.. _before_getting_started:


.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

Before getting started
======================

Prepping the dataset
--------------------

The following are some recommendations to help prep your dataset before subjecting to
linear inversion.

Magic angle flipping datasets
'''''''''''''''''''''''''''''

.. list-table::
  :widths: 1 25 74

  * - |uncheck|
    - **Did you shear the dataset?**
    - The inversion method assumes that the MAF dataset is sheared, such that one of
      the dimensions is a purely anisotropic frequency dimension.

      **Required**: Apply a shear transformation before proceeding.


  * - |uncheck|
    - **Did you zero-fill the time-domain dataset corresponding to the anisotropic
      dimension?**
    - Zero filling the time domain dataset is purely cosmetic. It makes the spectrum
      look visually appealing, but adds no information, that is, a zero-filled dataset
      contains the same information as a non-zero filled dataset. In terms of
      computation, however, a zero-filled spectrum will take longer to solve.

      **Recommendation**: Try to keep the total number of points along the anisotropic
      dimension in the range of 120 - 150 points.

  * - |uncheck|
    - **Did you correct for the isotropic offset along the anisotropic dimension?**
    - Ordinarily, after shear, a MAF spectrum is a 2D isotropic `v.s` pure anisotropic
      frequency correlation spectrum. In certain conditions, this is not true. In a MAF
      experiment, the sample holder (rotor) physically swaps between two angles
      (:math:`90^\circ \leftrightarrow 54.735^\circ`). It is possible to have a
      slightly different external magnetic fields at the two angles, in which case,
      there is an isotropic component along the anisotropic dimension, which is not
      removed by the shear transformation.

      **Required**: Correct for the isotropic offset along the
      anisotropic dimension by adding an appropriate coordinates-offset.

  * - |uncheck|
    - **Are there sinc wiggles in your spectrum?**
    - Kernel correction for spectrum with sinc wiggle artifacts is coming soon.

Spinning Sideband correlation datasets
''''''''''''''''''''''''''''''''''''''

.. list-table::
  :widths: 1 25 74

  * - |uncheck|
    - **Did you shear the dataset?**
    - The inversion method assumes that the spinning sideband dataset is sheared, such that
      one of the dimensions is a pure anisotropic spinning sidebands dimension.

      **Required**: Apply a shear transformation before proceeding.

  * - |uncheck|
    - **Did you data-repeat the time-domain signal corresponding to the sideband dimension?**
    - A data-repeat operation on the time-domain signal corresponding to the sideband
      dimension makes the spinning sidebands look like a stick spectrum after a
      Fourier transformation, a visual, which most NMR spectroscopists are familiar
      from the 1D magic-angle spinning spectrum. Like a zero-fill operation, a spinning
      sideband data-repeat operation is also purely cosmetic and adds no information.
      In terms of computation, however, a data-repeated spinning-sideband spectrum will
      take longer to solve.

      **Strongly recommended**: Avoid data-repeat operation.
