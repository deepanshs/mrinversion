.. _introduction:

============
Introduction
============

Linear inverse problems are frequently encountered in the scientific community and
have the following generic form

.. math::
   :label: eq_1

   {\bf K f} = {\bf s},

where :math:`{\bf K} \in \mathbb{R}^{m\times n}` is the transforming kernel (matrix),
:math:`{\bf f} \in \mathbb{R^n}` is the unknown desired solution, and
:math:`{\bf s} \in \mathbb{R^m}` is the known signal, which includes the
measurement noise. When the matrix :math:`{\bf K}` is non-singular and :math:`m=n`,
the solution to the problem in Eq. :eq:`eq_1` has a simple closed-form solution,

.. math::
    :label: eq_2

    {\bf f} = {\bf K}^{-1} {\bf s}.

But practical science isn't easy that way! Let's see how.

The deciding factor whether the solution :math:`{\bf f}` exists in Eq. :eq:`eq_2`
is whether or not the kernel :math:`{\bf K}` is invertible.
Often, most scientific problems with practical applications suffer from singular,
near-singular, or ill-conditioned kernels, where :math:`{\bf K}^{-1}` doesn't exist.
Such types of problems are termed as `ill-posed`.


Regularized problem
-------------------

A common approach in solving these ill-posed problems is to employ the regularization
methods of form

.. math::
    :label: eq_3

    {\bf f^\dagger} = \| {\bf Kf} - {\bf s}\|^2_2 + g({\bf f}),

where :math:`\|{\bf z}\|_2` is the `l-2` norm of :math:`{\bf z}`, :math:`g({\bf f})`
is the regularization term, and :math:`{\bf f}^\dagger` is the regularized solution.

these inverse problems are ill-posed and ill-conditioned, resulting in
infinit solutions which are also unstable in the presence of noise.

For example, in a more familiar linear-inverse problem, the inverse Fourier transform, the two dimensions are the frequency and time dimensions, where the frequency dimension undergoes the inverse transformation, and the time dimension is where the inversion method transforms the data.
