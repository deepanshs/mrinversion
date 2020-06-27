.. _introduction:

============
Introduction
============

Objective
---------

``mrinversion`` solves for the distribution of the second-rank traceless
symmetric tensor parameters, through an inversion of NMR spectrum, whose frequency
contributions arise predominantly from the distribution of the second-rank traceless
symmetric tensors.

**Nuclear shielding tensor parameters**

In the case of the shielding tensor, this corresponds to solving for the distribution
of anisotropy and asymmetry parameters of the second-rank traceless shielding tensor
through an inversion of a pure anisotropic frequency spectrum. The pure anisotropic
frequency spectra are the cross-sections of 2D isotropic `v.s` anisotropic correlation
spectrum, such as the magic-angle hopping (MAH) 2D One Pulse (TOP) MAS, magic-angle
turning (MAT), magic-angle flipping (MAF), Variable Angle Correlation Spectroscopy
(VACSY), phase adjusted spinning sideband (PASS), extended chemical shift modulation
(XCS), and it's variant.

Linear problem
--------------

Linear inverse problems are frequently encountered in the scientific community and
have the following generic form

.. math::
   :label: eq_1

   {\bf K f} = {\bf s},

where :math:`{\bf K} \in \mathbb{R}^{m\times n}` is the transforming kernel (matrix),
:math:`{\bf f} \in \mathbb{R}^n` is the unknown desired solution, and
:math:`{\bf s} \in \mathbb{R}^m` is the known signal, which includes the
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
The choice of the regularization term, :math:`g({\bf f})`, is often based on prior
knowledge of the system for which the linear problem is defined.



Smooth-LASSO regularization
'''''''''''''''''''''''''''

The prior assumption about the distribution tensor parameters is that it is
smooth and continuous for disordered and sparse for crystalline materials. Therefore,
we employ the smooth-lasso method, which is a linear model trained with the combined
l1 and l2 priors as the regularizer. The method minimizes the objective function,

.. math::
    \| {\bf Kf - s} \|^2_2 + \alpha \sum_{i=1}^{d} \| {\bf J}_i {\bf f} \|_2^2
                + \lambda  \| {\bf f} \|_1 ,

where :math:`\alpha` and :math:`\lambda` are the hyperparameters controlling the
smoothness and sparsity of the solution :math:`{\bf f}`. The matrix :math:`{\bf J}_i`
follows

.. math::
    {\bf J}_i = {\bf I}_{n_1} \otimes \cdots \otimes {\bf A}_{n_i}
                \otimes \cdots \otimes {\bf I}_{n_{d}},

where :math:`{\bf I}_{n_i} \in \mathbb{R}^{n_i \times n_i}` is the identity matrix, and
:math:`{\bf A}_{n_i}` is the first difference matrix given as

.. math::
    {\bf A}_{n_i} = \left(\begin{array}{ccccc}
                    1 & -1 & 0 & \cdots & \vdots \\
                    0 & 1 & -1 & \cdots & \vdots \\
                    \vdots & \vdots & \vdots & \vdots & 0 \\
                    0 & \cdots & 0 & 1 & -1
                \end{array}\right) \in \mathbb{R}^{(n_i-1)\times n_i}.

The symbol :math:`\otimes` is the Kronecker product. The terms,
:math:`\left(n_1, n_2, \cdots, n_d\right)`, are the number of points along the
respective dimensions, with the constraint that :math:`\prod_{i=1}^{d}n_i = n`,
where :math:`d` is the total number of dimensions in the solution :math:`{\bf f}`,
and :math:`n` is the total number of features in kernel, :math:`{\bf K}`.


.. For example, in a more familiar linear-inverse problem, the inverse Fourier transform, the two dimensions are the frequency and time dimensions, where the frequency dimension undergoes the inverse transformation, and the time dimension is where the inversion method transforms the data.
