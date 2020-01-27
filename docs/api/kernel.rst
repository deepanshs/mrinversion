

.. _kernel_api:


Kernel
======

Pure anisotropic Nuclear Shielding
----------------------------------

Generalized Class
"""""""""""""""""

.. currentmodule:: mrinversion.kernel

.. autoclass:: NuclearShieldingTensor
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
