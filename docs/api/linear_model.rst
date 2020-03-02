

.. _linear_model_api:


Linear Model
============


TSVDCompression
---------------

.. currentmodule:: mrinversion.linear_model

.. autoclass:: TSVDCompression
   :show-inheritance:

   .. rubric:: Methods Documentation

   .. automethod:: compress


Smooth Lasso
------------

.. currentmodule:: mrinversion.linear_model

.. autoclass:: SmoothLasso
   :show-inheritance:

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict
   .. automethod:: residuals
   .. automethod:: score


Smooth Lasso cross-validation
-----------------------------

.. currentmodule:: mrinversion.linear_model

.. autoclass:: SmoothLassoCV
   :show-inheritance:

   .. rubric:: Attributes Documentation

   .. autoattribute:: cross_validation_curve

   .. rubric:: Methods Documentation

   .. automethod:: fit
   .. automethod:: predict
   .. automethod:: residuals
