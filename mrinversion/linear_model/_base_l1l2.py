# -*- coding: utf-8 -*-
from copy import deepcopy

import csdmpy as cp
import numpy as np
from joblib import delayed
from joblib import Parallel
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from mrinversion.linear_model.tsvd_compression import TSVDCompression  # noqa: F401

__author__ = "Deepansh J. Srivastava"
__email__ = "srivastava.89@osu.edu"


class GeneralL2Lasso:
    r"""
        The Minimizer class solves the following equation,

        .. math::
            {\bf f} = \underset{{\bf f}}{\text{argmin}} \left( \frac{1}{m} \|
                        {\bf Kf - s} \|^2_2 +
                        \alpha \sum_{i=1}^{d} \| {\bf J}_i {\bf f} \|_2^2 +
                        \lambda  \| {\bf f} \|_1 \right),

        where :math:`{\bf K} \in \mathbb{R}^{m \times n}` is the kernel,
        :math:`{\bf s} \in \mathbb{R}^{m \times m_\text{count}}` is the known signal
        containing noise, and :math:`{\bf f} \in \mathbb{R}^{n \times m_\text{count}}`
        is the desired solution matrix.


        Based on the regularization literal, the above problem is constraint

        Args:
            alpha: Float, the hyperparameter, :math:`\alpha`.
            lambda1: Float, the hyperparameter, :math:`\lambda`.
            max_iterations: Interger, the maximum number of iterations allowed when
                            solving the problem. The default value is 10000.
            tolerance: Float, the tolerance at which the solution is
                       considered converged. The default value is 1e-5.
            positive: Boolean. If True, the amplitudes in the solution,
                      :math:`{\bf f}` is all positive, else the solution may contain
                      positive and negative amplitudes. The default is True.
            regularizer: String, a literal specifying the form of matrix
                         :math:`{\bf J}_i`. The allowed literals are `smooth lasso`
                         and `sparse ridge fusion`.
            f_shape: The shape of the solution, :math:`{\bf f}`, given as a tuple
                            (n1, n2, ..., nd)
        Attributes:

    """

    def __init__(
        self,
        alpha=1e-3,
        lambda1=1e-6,
        max_iterations=10000,
        tolerance=1e-5,
        positive=True,
        regularizer=None,
        inverse_dimension=None,
    ):

        self.hyperparameter = {"lambda": lambda1, "alpha": alpha}
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.regularizer = regularizer
        self.inverse_dimension = inverse_dimension
        self.f_shape = tuple([item.count for item in inverse_dimension])[::-1]

        # attributes
        self.f = None
        self.n_iter = None

    def fit(self, K, s):
        r"""
        Fit the model using the coordinate descent method from scikit-learn.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).
            s: A csdm object or an equivalent numpy array holding the signal,
                :math:`{\bf s}`, as a :math:`m \times m_\text{count}` matrix.
        """
        if isinstance(s, cp.CSDM):
            self.s = s
            s_ = s.dependent_variables[0].components[0].T
        else:
            s_ = s

        if s_.ndim == 1:
            s_ = s_[:, np.newaxis]
        prod = np.asarray(self.f_shape).prod()
        if K.shape[1] != prod:
            raise ValueError(
                "The product of the shape, `f_shape`, must be equal to the length of "
                f"the axis 1 of kernel, K, {K.shape[1]} != {prod}."
            )

        self.scale = s_.real.max()
        Ks, ss = _get_augmented_data(
            K=K,
            s=s_ / self.scale,
            alpha=s_.size * self.hyperparameter["alpha"],
            regularizer=self.regularizer,
            f_shape=self.f_shape,
        )

        # The factor 0.5 for alpha in the Lasso problem is to compensate
        # 1/(2 * n_sample) factor in OLS term
        estimator = Lasso(
            alpha=self.hyperparameter["lambda"] / 2.0,
            fit_intercept=False,
            copy_X=True,
            max_iter=self.max_iterations,
            tol=self.tolerance,
            warm_start=False,
            random_state=None,
            selection="random",
            positive=self.positive,
        )
        estimator.fit(Ks, ss)
        f = estimator.coef_.copy()
        if s_.shape[1] > 1:
            f.shape = (s_.shape[1],) + self.f_shape
            f[:, :, 0] /= 2.0
            f[:, 0, :] /= 2.0
        else:
            f.shape = self.f_shape
            f[:, 0] /= 2.0
            f[0, :] /= 2.0

        f *= self.scale

        if isinstance(s, cp.CSDM):
            f = cp.as_csdm(f)

            if len(s.dimensions) > 1:
                f.dimensions[2] = s.dimensions[1]
            f.dimensions[1] = self.inverse_dimension[1]
            f.dimensions[0] = self.inverse_dimension[0]

        self.estimator = estimator
        self.f = f
        self.n_iter = estimator.n_iter_

    def predict(self, K):
        r"""
        Predict the signal using the linear model.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).

        Return:
            A numpy array of shape (m, m_count) with the predicted values.
        """
        predict = self.estimator.predict(K) * self.scale

        return predict

    def residuals(self, K, s):
        r"""
        Return the residual as the difference the data and the prediced data(fit),
        following

        .. math::
            \text{residuals} = {\bf s - Kf^*}

        where :math:`{\bf f^*}` is the optimum solution.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).
            s: A csdm object or a :math:`m \times m_\text{count}` signal matrix,
                :math:`{\bf s}`.

        Return:
            If `s` is a csdm object, returns a csdm object with the residuals. If `s`
            is a numpy array, return a :math:`m \times m_\text{count}` residue matrix.
        """
        if isinstance(s, cp.CSDM):
            s_ = s.dependent_variables[0].components[0].T
        else:
            s_ = s
        predict = self.estimator.predict(K) * self.scale
        residue = s_ - predict

        if not isinstance(s, cp.CSDM):
            return residue

        residue = cp.as_csdm(residue.T)
        residue._dimensions = s._dimensions
        return residue

    def score(self, K, s, sample_weights=None):
        """
        Return the coefficient of determination, :math:`R^2`, of the prediction.
        For more information, read scikit-learn documentation.
        """
        return self.estimator.score(K, s / self.scale, sample_weights)


class GeneralL2LassoCV:
    def __init__(
        self,
        alphas=None,
        lambdas=None,
        folds=10,
        max_iterations=10000,
        tolerance=1e-5,
        positive=True,
        sigma=0.0,
        regularizer=None,
        randomize=False,
        times=2,
        verbose=False,
        inverse_dimension=None,
        n_jobs=-1,
    ):

        if alphas is None:
            self.cv_alphas = 10 ** ((np.arange(5) / 4) * 2 - 4)[::-1]
        else:
            self.cv_alphas = np.asarray(alphas).ravel()

        if lambdas is None:
            self.cv_lambdas = 10 ** ((np.arange(10) / 9) * 5 - 9)[::-1]
        else:
            self.cv_lambdas = np.asarray(lambdas).ravel()

        self.folds = folds

        self.n_jobs = n_jobs
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.sigma = sigma
        self.regularizer = regularizer
        self.hyperparameter = {}
        self.f = None
        self.randomize = randomize
        self.times = times
        self.verbose = verbose
        self.inverse_dimension = inverse_dimension
        self.f_shape = tuple([item.count for item in inverse_dimension])[::-1]

    def fit(self, K, s):
        r"""
        Fit the model using the coordinate descent method from scikit-learn for
        all alpha anf lambda values using the `n`-folds cross-validation technique.
        The cross-validation metric is the mean squared error.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).
            s: A :math:`m \times m_\text{count}` signal matrix, :math:`{\bf s}` as a
                csdm object or a numpy array or shape (m, m_count).
        """

        if isinstance(s, cp.CSDM):
            self.s = s
            s_ = s.dependent_variables[0].components[0].T
        else:
            s_ = s

        if s_.ndim == 1:
            s_ = s_[:, np.newaxis]
        prod = np.asarray(self.f_shape).prod()
        if K.shape[1] != prod:
            raise ValueError(
                "The product of the shape, `f_shape`, must be equal to the length of "
                f"the axis 1 of kernel, K, {K.shape[1]} != {prod}."
            )

        self.scale = s_.max().real
        s_ = s_ / self.scale
        cv_indexes = _get_cv_indexes(
            K,
            self.folds,
            self.regularizer,
            f_shape=self.f_shape,
            random=self.randomize,
            times=self.times,
        )
        self.cv_map = np.zeros((self.cv_alphas.size, self.cv_lambdas.size))

        alpha_ratio = np.ones(self.cv_alphas.size)
        alpha_ratio[1:] = np.sqrt(self.cv_alphas[1:] / self.cv_alphas[:-1])

        Ks, ss = _get_augmented_data(
            K=K,
            s=s_,
            alpha=s_.size * self.cv_alphas[0],
            regularizer=self.regularizer,
            f_shape=self.f_shape,
        )
        start_index = K.shape[0]

        # The factor 0.5 for alpha in the Lasso problem is to compensate
        # 1/(2 * n_sample) factor in OLS term.
        l1 = Lasso(
            alpha=self.cv_lambdas[0] / 2.0,
            fit_intercept=False,
            normalize=True,
            precompute=True,
            max_iter=self.max_iterations,
            tol=self.tolerance,
            copy_X=True,
            positive=self.positive,
            random_state=None,
            warm_start=True,
            selection="random",
        )
        l1.fit(Ks, ss)

        l1_array = []
        for lambda_ in self.cv_lambdas:
            l1_array.append(deepcopy(l1))
            l1_array[-1].alpha = lambda_ / 2.0

        j = 0
        for alpha_ratio_ in alpha_ratio:
            Ks[start_index:] *= alpha_ratio_
            jobs = (
                delayed(cv)(l1_array[i], Ks, ss, cv_indexes)
                for i in range(self.cv_lambdas.size)
            )
            self.cv_map[j] = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                **{
                    "backend": {
                        "threads": "threading",
                        "processes": "multiprocessing",
                        None: None,
                    }["threads"]
                },
            )(jobs)
            j += 1

        # cv_map contains negated mean square errors, therefore multiply by -1.
        self.cv_map *= -1
        # subtract the variance.
        self.cv_map -= self.sigma ** 2

        # After subtracting the variance, any negative values in the cv grid is a
        # result of fitting noise. Take the absolute value of cv to avoid such
        # models.
        self.cv_map = np.abs(self.cv_map)

        # The argmin of the minimum value is the selected model as it has the least
        # prediction error.
        index = np.unravel_index(self.cv_map.argmin(), self.cv_map.shape)
        self.hyperparameter["alpha"] = self.cv_alphas[index[0]]
        self.hyperparameter["lambda"] = self.cv_lambdas[index[1]]

        # Calculate the solution using the complete data at the optimized lambda and
        # alpha values
        self.opt = GeneralL2Lasso(
            alpha=self.hyperparameter["alpha"],
            lambda1=self.hyperparameter["lambda"],
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            positive=self.positive,
            regularizer=self.regularizer,
            inverse_dimension=self.inverse_dimension,
        )
        self.opt.fit(K, s)
        self.f = self.opt.f

        # convert cv_map to csdm
        self.cv_map = cp.as_csdm(self.cv_map.T)
        d0 = cp.as_dimension(-np.log10(self.cv_alphas), label="-log(α)")
        d1 = cp.as_dimension(-np.log10(self.cv_lambdas), label="-log(λ)")
        self.cv_map.dimensions[0] = d0
        self.cv_map.dimensions[1] = d1

    def predict(self, K):
        r"""
        Predict the signal using the linear model.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).

        Return:
            A numpy array of shape (m, m_count) with the predicted values.
        """
        return self.opt.predict(K)

    def residuals(self, K, s):
        r"""
        Return the residual as the difference the data and the prediced data(fit),
        following

        .. math::
            \text{residuals} = {\bf s - Kf^*}

        where :math:`{\bf f^*}` is the optimum solution.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).
            s: A csdm object or a :math:`m \times m_\text{count}` signal matrix,
                :math:`{\bf s}`.
        Return:
            If `s` is a csdm object, returns a csdm object with the residuals. If `s`
            is a numpy array, return a :math:`m \times m_\text{count}` residue matrix.
        """
        return self.opt.residuals(K, s)

    def score(self, K, s, sample_weights=None):
        """
        Return the coefficient of determination, :math:`R^2`, of the prediction.
        For more information, read scikit-learn documentation.
        """
        return self.opt.score(K, s, sample_weights)

    @property
    def cross_validation_curve(self):
        """Return the cross-validation error metric curve."""
        return self.cv_map


def cv(l1, X, y, cv):
    """Return the cross-validation score as negative of mean square error."""
    return cross_validate(
        l1,
        X=X,
        y=y,
        scoring="neg_mean_squared_error",  # 'neg_mean_absolute_error",
        cv=cv,
        fit_params={"check_input": False},
        n_jobs=1,
        verbose=0,
    )["test_score"].mean()


def _get_smooth_size(f_shape, regularizer, max_size):
    r"""Return the number of rows appended to for the augmented kernel.

    For smooth-lasso, the number of rows is given as
        rows = \prod_{i=1}^d n_i (\sum_{j=0}^d  (n_j-1)/n_j)

    For sparse ridge fusion, the number of rows is given as
        rows = \prod_{i=1}^d n_i (\sum_{j=0}^d  (n_j-2)/n_j)
    """
    shape = np.asarray(f_shape)
    shape_prod = shape.prod()
    if shape_prod != max_size:
        raise ValueError(
            "The product of the shape must be equal to the length of axis 1 of the "
            "kernel, K"
        )
    if regularizer == "smooth lasso":
        smooth_array_size = [int(shape_prod * (i - 1) / i) for i in shape]
        smooth_size = np.asarray(smooth_array_size).sum()
    elif regularizer == "sparse ridge fusion":
        smooth_array_size = [int(shape_prod * (i - 2) / i) for i in shape]
        smooth_size = np.asarray(smooth_array_size).sum()
    else:
        smooth_size = 0

    return smooth_size


def _get_cv_indexes(K, folds, regularizer, f_shape=None, random=False, times=1):
    """Return the indexes of the kernel and signal, corresponding to the test
    and train sets.

    """
    cv_indexes = []
    ks0, ks1 = K.shape
    if isinstance(f_shape, int):
        f_shape = (f_shape,)

    tr_ = []
    smooth_size = _get_smooth_size(f_shape, regularizer, ks1)

    tr_ = (np.arange(smooth_size) + ks0).tolist()

    for j in range(folds):
        train_ = []
        test_ = []
        for i in range(ks0):
            if (i + j - folds) % folds == 0:
                test_.append(i)
            else:
                train_.append(i)
        train_ += tr_
        cv_indexes.append([train_, test_])

    if random:
        for _ in range(times):
            kf = KFold(n_splits=folds, shuffle=True)
            kf.get_n_splits(K)
            for train_index, test_index in kf.split(K):
                cv_indexes.append([train_index.tolist() + tr_, test_index])
    return cv_indexes


def generate_J_i(Ai, alpha, f_shape):
    J = []
    sqrt_alpha = np.sqrt(alpha)
    identity = [np.eye(i) for i in f_shape]
    for i, i_count in enumerate(f_shape):
        J_array = deepcopy(identity)
        J_array[i] = Ai(i_count)
        Ji_ = 1
        for j_array_ in J_array:
            Ji_ = np.kron(Ji_, j_array_)
        J.append(Ji_ * sqrt_alpha)
    return J


def _get_augmented_data(K, s, alpha, regularizer, f_shape=None):
    """Creates a smooth kernel, K, with alpha regularization parameter."""
    ks0, ks1 = K.shape
    ss0, ss1 = s.shape

    if isinstance(f_shape, int):
        f_shape = (f_shape,)

    smooth_size = _get_smooth_size(f_shape, regularizer, ks1)

    if regularizer == "smooth lasso":

        def Ai_smooth_lasso(i):
            return (-1 * np.eye(i) + np.diag(np.ones(i - 1), k=-1))[1:]

        J = generate_J_i(Ai_smooth_lasso, alpha, f_shape)

    if regularizer == "sparse ridge fusion":

        def Ai_sparse_ridge_fusion(i):
            A_temp = -1 * np.eye(i)
            A_temp += 2 * np.diag(np.ones(i - 1), k=-1)
            A_temp += -1 * np.diag(np.ones(i - 2), k=-2)
            return A_temp[2:]
            # return (
            #     -1 * np.eye(i)
            #     + 2 * np.diag(np.ones(i - 1), k=-1)
            #     - 1 * np.diag(np.ones(i - 2), k=-2)
            # )[2:]

        J = generate_J_i(Ai_sparse_ridge_fusion, alpha, f_shape)

    K_ = np.empty((ks0 + smooth_size, ks1))

    K_[:ks0] = K
    start = ks0
    end = ks0
    for J_i in J:
        end = end + J_i.shape[0]
        K_[start:end] = J_i
        start = end

    s_ = np.zeros((ss0 + smooth_size, ss1))
    s_[:ss0] = s.real

    return np.asfortranarray(K_), np.asfortranarray(s_)