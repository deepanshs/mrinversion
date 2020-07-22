# -*- coding: utf-8 -*-
from ._base_l1l2 import GeneralL2Lasso

__author__ = "Deepansh J. Srivastava"
__email__ = "srivastava.89@osu.edu"


class SparseRidgeFusion(GeneralL2Lasso):
    r"""
    Linear model trained with l1-l2 prior regularization of form,

    .. math::
        {\bf f} = \underset{{\bf f}}{\text{argmin}} \left( \| {\bf Kf - s} \|^2_2 +
                    \alpha \sum_{i=1}^{d} \| {\bf J}_i {\bf f} \|_2^2 +
                    \lambda  \| {\bf f} \|_1 \right),

    where :math:`{\bf K} \in \mathbb{R}^{m \times n}` is the kernel,
    :math:`{\bf s} \in \mathbb{R}^{m \times m_\text{count}}` is the known signal
    containing noise, and :math:`{\bf f} \in \mathbb{R}^{n \times m_\text{count}}`
    is the desired solution matrix.


    Based on the regularization literal, the above problem is constraint

    Args:
        kernel: A :math:`m \times n` kernel matrix, :math:`{\bf K}`.
        signal: A :math:`m \times m_\text{count}` signal matrix, :math:`{\bf s}`.
        inverse_dimension: A list of csdmpy Dimension objects representing the
            inverse space.
        max_iterations: An interger defining the maximum number of iterations used
            in solving the LASSO problem. The default value is 10000.
        tolerance: A float defining the tolerance at which the solution is
            considered converged. The default value is 1e-5.
        positive: A boolean. If True, the amplitudes in the solution,
            :math:`{\bf f}` is all positive, else the solution contains
            both positive and negative amplitudes. the default is True.
        sigma: The noise standard deviation. The default is 0.0
    """

    def __init__(
        self,
        alpha,
        lambda1,
        inverse_dimension,
        max_iterations=10000,
        tolerance=1e-5,
        positive=True,
    ):
        super().__init__(
            alpha=alpha,
            lambda1=lambda1,
            max_iterations=max_iterations,
            tolerance=tolerance,
            positive=positive,
            inverse_dimension=inverse_dimension,
            regularizer="sparse ridge fusion",
        )
