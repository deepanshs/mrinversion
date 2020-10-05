# -*- coding: utf-8 -*-
from ._base_l1l2 import GeneralL2Lasso
from ._base_l1l2 import GeneralL2LassoCV

__author__ = "Deepansh J. Srivastava"
__email__ = "srivastava.89@osu.edu"


class SmoothLasso(GeneralL2Lasso):
    r"""
    The linear model trained with the combined l1 and l2 priors as the regularizer.
    The method minimizes the objective function,

    .. math::
        \| {\bf Kf - s} \|^2_2 + \alpha \sum_{i=1}^{d} \| {\bf J}_i {\bf f} \|_2^2
                    + \lambda  \| {\bf f} \|_1 ,

    where :math:`{\bf K} \in \mathbb{R}^{m \times n}` is the kernel,
    :math:`{\bf s} \in \mathbb{R}^{m \times m_\text{count}}` is the known (measured)
    signal, and :math:`{\bf f} \in \mathbb{R}^{n \times m_\text{count}}`
    is the desired solution. The parameters, :math:`\alpha` and :math:`\lambda`,
    are the hyperparameters controlling the smoothness and sparsity of the
    solution :math:`{\bf f}`. The matrix :math:`{\bf J}_i` is given as

    .. math::
        {\bf J}_i = {\bf I}_{n_1} \otimes \cdots \otimes {\bf A}_{n_i}
                    \otimes \cdots \otimes {\bf I}_{n_{d}},

    where :math:`{\bf I}_{n_i} \in \mathbb{R}^{n_i \times n_i}` is the identity matrix,

    .. math::
        {\bf A}_{n_i} = \left(\begin{array}{ccccc}
                        1 & -1 & 0 & \cdots & \vdots \\
                        0 & 1 & -1 & \cdots & \vdots \\
                        \vdots & \vdots & \vdots & \vdots & 0 \\
                        0 & \cdots & 0 & 1 & -1
                    \end{array}\right) \in \mathbb{R}^{(n_i-1)\times n_i},

    and the symbol :math:`\otimes` is the Kronecker product. The terms,
    :math:`\left(n_1, n_2, \cdots, n_d\right)`, are the number of points along the
    respective dimensions, with the constraint that :math:`\prod_{i=1}^{d}n_i = n`,
    where :math:`d` is the total number of dimensions.

    Args
    ----

    alpha: float
        The hyperparameter, :math:`\alpha`.
    lambda1: float
        The hyperparameter, :math:`\lambda`.
    inverse_dimension: list
        A list of csdmpy Dimension objects representing the inverse space.
    max_iterations: int
        The maximum number of iterations allowed when solving the problem. The default
        value is 10000.
    tolerance: float
        The tolerance at which the solution is considered converged. The default value
        is 1e-5.
    positive: bool
        If True, the amplitudes in the solution, :math:`{\bf f}`, is contrained to only
        positive values, else the solution may contain positive and negative amplitudes.
        The default is True.

    Attributes
    ----------

    f: ndarray or CSDM object.
        A ndarray of shape (`m_count`, `nd`, ..., `n1`, `n0`) representing the
        solution, :math:`{\bf f} \in \mathbb{R}^{m_\text{count} \times n_d \times
        \cdots n_1 \times n_0}`.
    n_iter: int
        The number of iterations required to reach the specified tolerance.
    """

    def __init__(
        self,
        alpha,
        lambda1,
        inverse_dimension,
        max_iterations=10000,
        tolerance=1e-5,
        positive=True,
        method="gradient_decent",
    ):
        super().__init__(
            alpha=alpha,
            lambda1=lambda1,
            max_iterations=max_iterations,
            tolerance=tolerance,
            positive=positive,
            regularizer="smooth lasso",
            inverse_dimension=inverse_dimension,
            method=method,
        )


class SmoothLassoCV(GeneralL2LassoCV):
    r"""
    The linear model trained with the combined l1 and l2 priors as the
    regularizer. The method minimizes the objective function,

    .. math::
        \| {\bf Kf - s} \|^2_2 + \alpha \sum_{i=1}^{d} \| {\bf J}_i {\bf f} \|_2^2
                    + \lambda  \| {\bf f} \|_1 ,

    where :math:`{\bf K} \in \mathbb{R}^{m \times n}` is the kernel,
    :math:`{\bf s} \in \mathbb{R}^{m \times m_\text{count}}` is the known signal
    containing noise, and :math:`{\bf f} \in \mathbb{R}^{n \times m_\text{count}}`
    is the desired solution. The parameters, :math:`\alpha` and :math:`\lambda`,
    are the hyperparameters controlling the smoothness and sparsity of the
    solution :math:`{\bf f}`.
    The matrix :math:`{\bf J}_i` is given as

    .. math::
        {\bf J}_i = {\bf I}_{n_1} \otimes \cdots \otimes {\bf A}_{n_i}
                    \otimes \cdots \otimes {\bf I}_{n_{d}},

    where :math:`{\bf I}_{n_i} \in \mathbb{R}^{n_i \times n_i}` is the identity
    matrix,

    .. math::
        {\bf A}_{n_i} = \left(\begin{array}{ccccc}
                        1 & -1 & 0 & \cdots & \vdots \\
                        0 & 1 & -1 & \cdots & \vdots \\
                        \vdots & \vdots & \vdots & \vdots & 0 \\
                        0 & \cdots & 0 & 1 & -1
                    \end{array}\right) \in \mathbb{R}^{(n_i-1)\times n_i},

    and the symbol :math:`\otimes` is the Kronecker product. The terms,
    :math:`\left(n_1, n_2, \cdots, n_d\right)`, are the number of points along the
    respective dimensions, with the constraint that :math:`\prod_{i=1}^{d}n_i = n`,
    where :math:`d` is the total number of dimensions.

    The cross-validation is carried out using a stratified splitting of the signal.

    Parameters
    ----------

    alphas: ndarray
        A list of :math:`\alpha` hyperparameters.
    lambdas: ndarray
        A list of :math:`\lambda` hyperparameters.
    inverse_dimension: list
        A list of csdmpy Dimension objects representing the inverse space.
    folds: int
        The number of folds used in cross-validation.The default is 10.
    max_iterations: int
        The maximum number of iterations allowed when solving the problem. The default
        value is 10000.
    tolerance: float
        The tolerance at which the solution is considered converged. The default value
        is 1e-5.
    positive: bool
        If True, the amplitudes in the solution, :math:`{\bf f}`, is contrained to only
        positive values, else the solution may contain positive and negative amplitudes.
        The default is True.
    sigma: float
        The standard deviation of the noise in the signal. The default is 0.0.
    sigma: float
        The standard deviation of the noise in the signal. The default is 0.0.
    randomize: bool
        If true, the folds are created by randomly assigning the samples to each fold.
        If false, a stratified sampled is used to generate folds. The default is False.
    times: int
        The number of times to randomized n-folds are created. Only applicable when
        `randomize` attribute is True.
    verbose: bool
        If true, prints the process.
    n_jobs: int
        The number of CPUs used for computation. The default is -1, that is, all
        available CPUs are used.


    Attributes
    ----------
    f: ndarray or CSDM object.
        A ndarray of shape (m_count, nd, ..., n1, n0). The solution,
        :math:`{\bf f} \in \mathbb{R}^{m_\text{count} \times n_d \times \cdots n_1
        \times n_0}` or an equivalent CSDM object.
    n_iter: int.
        The number of iterations required to reach the specified tolerance.
    hyperparameters: dict.
        A dictionary with the :math:`\alpha` and :math:\lambda` hyperparameters.
    cross_validation_curve: CSDM object.
        The cross-validation error metric determined as the mean square error.
    """

    def __init__(
        self,
        alphas,
        lambdas,
        inverse_dimension,
        folds=10,
        max_iterations=10000,
        tolerance=1e-5,
        positive=True,
        sigma=0.0,
        randomize=False,
        times=2,
        verbose=False,
        n_jobs=-1,
        method="gradient_decent",
    ):
        super().__init__(
            alphas=alphas,
            lambdas=lambdas,
            inverse_dimension=inverse_dimension,
            folds=folds,
            max_iterations=max_iterations,
            tolerance=tolerance,
            positive=positive,
            sigma=sigma,
            regularizer="smooth lasso",
            randomize=randomize,
            times=times,
            verbose=verbose,
            n_jobs=n_jobs,
            method=method,
        )
