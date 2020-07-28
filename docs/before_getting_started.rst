.. _before_getting_started:


.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

Before getting started
======================

Prepping the 2D dataset for inversion
-------------------------------------

The following is a list of some requirements and recommendations to help prepare
the 2D dataset for inversion.

Common recommendations/requirements
'''''''''''''''''''''''''''''''''''

.. list-table::
  :widths: 2 98

  * - |uncheck|
    - **Dataset shear**

      The inversion method assumes that the 2D dataset is sheared, such that one of the
      dimensions corresponds to a pure anisotropic spectrum. The anisotropic
      cross-sections are centered at 0 Hz.

      **Required**: Apply a shear transformation before proceeding.

  * - |uncheck|
    - **Calculate the noise standard deviation**

      Use the noise region of your spectrum to calculate the standard deviation of the
      noise. You will require this value when implementing cross-validation.


Spinning Sideband correlation dataset specific recommendations
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. list-table::
  :widths: 2 98

  * - |uncheck|
    - **Data-repeat operation**

      A data-repeat operation on the time-domain signal corresponding to the sideband
      dimension makes the spinning sidebands look like a stick spectrum after a
      Fourier transformation, a visual, which most NMR spectroscopists are familiar
      from the 1D magic-angle spinning spectrum. Like a zero-fill operation, a spinning
      sideband data-repeat operation is purely cosmetic and adds no information.
      In terms of computation, however, a data-repeated spinning-sideband spectrum will
      take longer to solve.

      **Strongly recommended**: Avoid data-repeat operation.


Magic angle flipping dataset specific recommendations
'''''''''''''''''''''''''''''''''''''''''''''''''''''

.. list-table::
  :widths: 2 98

  * - |uncheck|
    - **Isotropic shift correction along the anisotropic dimension**

      Ordinarily, after shear, a MAF spectrum is a 2D isotropic `vs.` pure anisotropic
      frequency correlation spectrum. In certain conditions, this is not true. In a MAF
      experiment, the sample holder (rotor) physically swaps between two angles
      (:math:`90^\circ \leftrightarrow 54.735^\circ`). It is possible to have a
      slightly different external magnetic fields at the two angles, in which case,
      there is an isotropic component along the anisotropic dimension, which is not
      removed by the shear transformation.

      **Required**: Correct for the isotropic offset along the
      anisotropic dimension by adding an appropriate coordinates-offset.

  * - |uncheck|
    - **Zero-fill operation**

      Zero filling the time domain dataset is purely cosmetic. It makes the spectrum
      look visually appealing, but adds no information, that is, a zero-filled dataset
      contains the same information as a non-zero filled dataset. In terms of
      computation, however, a zero-filled spectrum will take longer to solve.

      **Recommendation**: If zero-filled, try to keep the total number of points along
      the anisotropic dimension in the range of 120 - 150 points.

  * -
    - **Sinc wiggles artifacts**

      Kernel correction for spectrum with sinc wiggle artifacts is coming soon.
