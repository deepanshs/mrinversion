

.. _kernel_api:


Kernel
======


.. currentmodule:: mrinversion.kernel.utils

.. autofunction:: x_y_to_zeta_eta


Pure anisotropic Nuclear Shielding
----------------------------------

Generalized Class
"""""""""""""""""

.. currentmodule:: mrinversion.kernel

.. autoclass:: NuclearShieldingLineshape
   :show-inheritance:

   .. automethod:: kernel

Specialized Classes
"""""""""""""""""""

Magic Angle Flipping
''''''''''''''''''''
.. autoclass:: MAF
   :show-inheritance:

   .. automethod:: kernel

Spinning Sidebands
''''''''''''''''''

.. autoclass:: SpinningSidebands
   :show-inheritance:

   .. automethod:: kernel
