# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Any
from typing import List
from typing import Literal
from typing import Union

import csdmpy as cp
import numpy as np
from joblib import delayed
from joblib import Parallel
from pydantic import BaseModel
from pydantic import PrivateAttr
from scipy.optimize import least_squares
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import MultiTaskLasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

from mrinversion.linear_model.tsvd_compression import TSVDCompression  # noqa: F401

# from scipy.optimize import minimize

__author__ = "Deepansh J. Srivastava"
__email__ = "srivastava.89@osu.edu"


class Optimizer(BaseModel):
    def _get_minimizer(self, alpha):
        """Return the estimator for the method"""
        # The factor 0.5 for alpha in the Lasso/LassoLars problem is to compensate
        # 1/(2 * n_sample) factor in OLS term.
        if self.method == "multi-task":
            return MultiTaskLasso(
                alpha=alpha / 2.0,  # self.cv_lambdas[0] / 2.0,
                fit_intercept=False,
                normalize=False,
                # precompute=True,
                max_iter=self.max_iterations,
                tol=self.tolerance,
                copy_X=True,
                # positive=self.positive,
                random_state=None,
                warm_start=False,
                selection="random",
            )

        if self.method == "gradient_decent":
            return Lasso(
                alpha=alpha / 2.0,
                fit_intercept=False,
                normalize=False,
                precompute=True,
                max_iter=self.max_iterations,
                tol=self.tolerance,
                copy_X=True,
                positive=self.positive,
                random_state=None,
                warm_start=False,
                selection="random",
            )

        if self.method == "lars":
            return LassoLars(
                alpha=alpha / 2.0,
                fit_intercept=False,
                verbose=True,
                normalize=False,
                precompute="auto",
                max_iter=self.max_iterations,
                copy_X=True,
                fit_path=False,
                positive=self.positive,
                jitter=None,
                random_state=0,
            )

    def _pre_fit_cv(self, K, s, alpha):
        s_ = _get_proper_data(s)
        self._scale = s_.max().real
        s_ = s_ / self._scale

        # prod = np.asarray(self.f_shape).prod()
        # if K.shape[1] != prod:
        #     raise ValueError(
        #         "The product of the shape, `f_shape`, must be equal to the length of "
        #         f"the axis 1 of kernel, K, {K.shape[1]} != {prod}."
        #     )
        half = False
        if "half" in self.inverse_dimension[0].application:
            half = self.inverse_dimension[0].application["half"]

        Ks, ss = _get_augmented_data(
            K=K,
            s=s_,
            alpha=s_.size * alpha,
            regularizer=self.regularizer,
            f_shape=self.f_shape,
            half=half,
        )

        args = (self.folds, self.regularizer, self.f_shape, self.randomize, self.times)
        cv_indexes = _get_cv_indexes(K, *args)

        return Ks, ss, cv_indexes


class GeneralL2Lasso(Optimizer):
    r"""The Minimizer class solves the following equation,

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
        hyperparameters: Dict, a python dictionary of hyperparameters.
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
    """

    alpha: float = 1e-3
    lambda1: float = 1e-6
    hyperparameters: dict = None
    inverse_dimension: List[
        Union[cp.Dimension, cp.LinearDimension, cp.MonotonicDimension]
    ] = []
    max_iterations: int = 10000
    tolerance: float = 1e-5
    positive: bool = True
    regularizer: Literal["smooth lasso", "sparse ridge fusion"] = "smooth lasso"
    method: Literal["multi-task", "gradient_decent", "lars"] = "gradient_decent"
    f: Union[cp.CSDM, np.ndarray] = None
    n_iter: int = None
    _scale: float = PrivateAttr(1.0)
    _estimator: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.hyperparameters is None:
            self.hyperparameters = {"lambda": self.lambda1, "alpha": self.alpha}

    class Config:
        arbitrary_types_allowed = True

    @property
    def f_shape(self):
        return tuple([item.count for item in self.inverse_dimension])[::-1]

    def fit(self, K, s):
        r"""Fit the model using the coordinate descent method from scikit-learn.

        Args
        ----

        K: ndarray
            The :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
            shape (m, n).
        s: ndarray or CSDM object.
            A csdm object or an equivalent numpy array holding the signal,
            :math:`{\bf s}`, as a :math:`m \times m_\text{count}` matrix.
        """

        s_ = _get_proper_data(s)
        self._scale = s_.real.max()
        s_ = s_ / self._scale

        # prod = np.asarray(self.f_shape).prod()
        # if K.shape[1] != prod:
        #     raise ValueError(
        #         "The product of the shape, `f_shape`, must be equal to the length of "
        #         f"the axis 1 of kernel, K, {K.shape[1]} != {prod}."
        #     )

        half = False
        if "half" in self.inverse_dimension[0].application:
            half = self.inverse_dimension[0].application["half"]

        Ks, ss = _get_augmented_data(
            K=K,
            s=s_,
            alpha=s_.size * self.hyperparameters["alpha"],
            regularizer=self.regularizer,
            f_shape=self.f_shape,
            half=half,
        )

        _estimator = self._get_minimizer(self.hyperparameters["lambda"])
        _estimator.fit(Ks, ss)
        f = _estimator.coef_.copy()

        if half:
            index = self.inverse_dimension[0].application["index"]
            f_new = np.zeros((s_.shape[1], np.prod(self.f_shape)), dtype=float)
            f_new[:, index] = f
            f = f_new
        if s_.shape[1] > 1:
            f.shape = (s_.shape[1],) + self.f_shape
            f[:, :, 0] /= 2.0
            f[:, 0, :] /= 2.0
        else:
            f.shape = self.f_shape
            f[:, 0] /= 2.0
            f[0, :] /= 2.0

        f *= self._scale

        if isinstance(s, cp.CSDM):
            f = self.pack_as_csdm(f, s)
            # f = cp.as_csdm(f)

            # if len(s.x) > 1:
            #     f.x[2] = s.x[1]
            # f.x[1] = self.inverse_dimension[1]
            # f.x[0] = self.inverse_dimension[0]

        self._estimator = _estimator
        self.f = f
        self.n_iter = _estimator.n_iter_

    def pack_as_csdm(self, f, s):
        f = cp.as_csdm(f)

        if len(s.x) > 1:
            f.x[2] = s.x[1]
        f.x[1] = self.inverse_dimension[1]
        f.x[0] = self.inverse_dimension[0]
        return f

    def predict(self, K):
        r"""Predict the signal using the linear model.

        Args
        ----

        K: ndarray
            A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of shape
            (m, n).

        Return
        ------
        ndarray
            A numpy array of shape (m, m_count) with the predicted values
        """
        return self._estimator.predict(K) * self._scale

    def residuals(self, K, s):
        r"""Return the residual as the difference the data and the prediced data(fit),
        following

        .. math::
            \text{residuals} = {\bf s - Kf^*}

        where :math:`{\bf f^*}` is the optimum solution.

        Args
        ----
        K: ndarray.
            A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of shape
            (m, n).
        s: ndarray ot CSDM object.
            A csdm object or a :math:`m \times m_\text{count}` signal matrix,
            :math:`{\bf s}`.

        Return
        ------
        ndarray or CSDM object.
            If `s` is a csdm object, returns a csdm object with the residuals. If `s`
            is a numpy array, return a :math:`m \times m_\text{count}` residue matrix.
            csdm object
        """
        s_ = s.y[0].components[0].T if isinstance(s, cp.CSDM) else s
        predict = np.squeeze(self._estimator.predict(K)) * self._scale
        residue = s_ - predict

        if not isinstance(s, cp.CSDM):
            return residue

        residue = cp.as_csdm(residue.T)
        residue._dimensions = s._dimensions
        return residue

    def score(self, K, s, sample_weights=None):
        """
        The coefficient of determination, :math:`R^2`, of the prediction.
        For more information, read scikit-learn documentation.
        """
        return self._estimator.score(K, s / self._scale, sample_weights)


class GeneralL2LassoLS(GeneralL2Lasso):
    folds: int = 10
    sigma: float = 0.0
    randomize: bool = False
    times: int = 2
    verbose: bool = False
    n_jobs: int = -1
    _path: Any = PrivateAttr()

    def fit(self, K, s, **kwargs):
        r"""Fit the model using the coordinate descent method from scikit-learn for
        all alpha anf lambda values using the `n`-folds cross-validation technique.
        The cross-validation metric is the mean squared error.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).
            s: A :math:`m \times m_\text{count}` signal matrix, :math:`{\bf s}` as a
                csdm object or a numpy array or shape (m, m_count).
        """
        Ks, ss, cv_indexes = self._pre_fit_cv(K, s, self.hyperparameters["alpha"])

        self._path = []
        sigma_sq = self.sigma ** 2
        ks0 = K.shape[0]
        l1 = self._get_minimizer(self.hyperparameters["lambda"])

        def fnc(x0):
            self._path.append(x0)
            Ks[ks0:] *= np.sqrt(x0[0] / self.hyperparameters["alpha"])
            alpha = self.hyperparameters["alpha"] = x0[0]
            l1.alpha = x0[1] / 2.0
            mse = -cv(l1, Ks, ss, cv_indexes, alpha=alpha, n_jobs=-1)
            mse -= sigma_sq
            # mse *= 1000
            return np.sqrt(2 * mse)

            # + np.mean(
            # (l1.coef_[1:, :] - l1.coef_[:-1, :]) ** 2
            # )
            # for i in range(self.folds):
            #     # print(i)
            #     fit_ = l1.fit(Ks_[cv_indexes[i][0]], ss[cv_indexes[i][0]])
            #     y_predict = fit_.predict(Ks_[cv_indexes[i][1]])
            #     avg += np.sum((y_predict - ss[cv_indexes[i][1]]) ** 2)
            # return avg

        x0 = [self.hyperparameters["alpha"], self.hyperparameters["lambda"]]

        bounds = [(1e-15, 1e-15), (1e-4, 1e-4)]
        # method="Powell"
        res = least_squares(fnc, x0, bounds=bounds, verbose=2, **kwargs)

        self.hyperparameters["alpha"] = res.x[0]
        self.hyperparameters["lambda"] = res.x[1]

        print(res.message, res.status)

        # Calculate the solution using the complete data at the optimized lambda and
        # alpha values
        super().fit(K, s)


class GeneralL2LassoCV(Optimizer):

    alphas: Union[List[float], np.ndarray] = None
    lambdas: Union[List[float], np.ndarray] = None
    hyperparameters: dict = None
    inverse_dimension: List[
        Union[cp.Dimension, cp.LinearDimension, cp.MonotonicDimension]
    ] = []
    max_iterations: int = 10000
    tolerance: float = 1e-5
    positive: bool = True
    regularizer: Literal["smooth lasso", "sparse ridge fusion"] = "smooth lasso"
    method: Literal["multi-task", "gradient_decent", "lars"] = "gradient_decent"

    folds: int = 10
    sigma: float = 0.0
    randomize: bool = False
    times: int = 2
    verbose: bool = False
    n_jobs: int = -1

    f: Union[cp.CSDM, np.ndarray] = None
    n_iter: int = None
    _scale: float = PrivateAttr(1.0)
    _opt: Any = PrivateAttr()
    _path: Any = PrivateAttr()
    _cv_alphas: Any = PrivateAttr()
    _cv_lambdas: Any = PrivateAttr()
    _cv_map: Any = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    @property
    def f_shape(self):
        return tuple([item.count for item in self.inverse_dimension])[::-1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cv_alphas = (
            10 ** ((np.arange(5) / 4) * 2 - 4)[::-1]
            if self.alphas is None
            else np.asarray(self.alphas).ravel()
        )

        self._cv_lambdas = (
            10 ** ((np.arange(10) / 9) * 5 - 9)[::-1]
            if self.lambdas is None
            else np.asarray(self.lambdas).ravel()
        )
        if self.hyperparameters is None:
            self.hyperparameters = {"lambda": None, "alpha": None}

    def fit(self, K, s, scoring="neg_mean_squared_error"):
        r"""Fit the model using the coordinate descent method from scikit-learn for
        all alpha anf lambda values using the `n`-folds cross-validation technique.
        The cross-validation metric is the mean squared error.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).
            s: A :math:`m \times m_\text{count}` signal matrix, :math:`{\bf s}` as a
                csdm object or a numpy array or shape (m, m_count).
        """

        Ks, ss, cv_indexes = self._pre_fit_cv(K, s, self._cv_alphas[0])

        self._cv_map = np.zeros((self._cv_alphas.size, self._cv_lambdas.size))

        alpha_ratio = np.ones(self._cv_alphas.size)
        if self._cv_alphas.size != 1 and self._cv_alphas[0] != 0:
            alpha_ratio[1:] = np.sqrt(self._cv_alphas[1:] / self._cv_alphas[:-1])

        start_index = K.shape[0]

        l1 = self._get_minimizer(self._cv_lambdas[0])
        l1_array = []

        for lambda_ in self._cv_lambdas:
            l1_array.append(deepcopy(l1))
            l1_array[-1].alpha = lambda_ / 2.0

        j = 0
        for alpha_ratio_ in alpha_ratio:
            if alpha_ratio_ != 0:
                Ks[start_index:] *= alpha_ratio_
            jobs = (
                delayed(cv)(l1_, Ks, ss, cv_indexes, alpha=alpha_)
                for l1_, alpha_ in zip(l1_array, self._cv_alphas)
            )
            self._cv_map[j] = Parallel(
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

        # _cv_map contains negated mean square errors, therefore multiply by -1.
        self._cv_map *= -1
        # subtract the variance.
        if scoring == "neg_mean_squared_error":
            self._cv_map -= self.sigma ** 2
            index = np.where(self._cv_map < 0)
            self._cv_map[index] = np.nan

            # After subtracting the variance, any negative values in the cv grid is a
            # result of fitting noise. Take the absolute value of cv to avoid such
            # models.
            # self._cv_map = np.abs(self._cv_map)

        # The argmin of the minimum value is the selected model as it has the least
        # prediction error.
        index = np.unravel_index(self._cv_map.argmin(), self._cv_map.shape)
        self.hyperparameters["alpha"] = self._cv_alphas[index[0]]
        self.hyperparameters["lambda"] = self._cv_lambdas[index[1]]

        # Calculate the solution using the complete data at the optimized lambda and
        # alpha values
        self._opt = GeneralL2Lasso(
            alpha=self.hyperparameters["alpha"],
            lambda1=self.hyperparameters["lambda"],
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            positive=self.positive,
            regularizer=self.regularizer,
            inverse_dimension=self.inverse_dimension,
            method=self.method,
        )
        self._opt.fit(K, s)
        self.f = self._opt.f

        # convert _cv_map to csdm
        self._cv_map = cp.as_csdm(np.squeeze(self._cv_map.T))
        if len(self._cv_alphas) != 1:
            d0 = cp.as_dimension(-np.log10(self._cv_alphas), label="-log(α)")
            self._cv_map.x[0] = d0

        if len(self._cv_lambdas) == 1:
            return

        d1 = cp.as_dimension(-np.log10(self._cv_lambdas), label="-log(λ)")
        if len(self._cv_alphas) != 1:
            self._cv_map.x[1] = d1
        else:
            self._cv_map.x[0] = d1

    def predict(self, K):
        r"""Predict the signal using the linear model.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).

        Return:
            A numpy array of shape (m, m_count) with the predicted values.
        """
        return self._opt.predict(K)

    def residuals(self, K, s):
        r"""Return the residual as the difference the data and the prediced data(fit),
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
        return self._opt.residuals(K, s)

    def score(self, K, s, sample_weights=None):
        """
        Return the coefficient of determination, :math:`R^2`, of the prediction.
        For more information, read scikit-learn documentation.
        """
        return self._opt.score(K, s, sample_weights)

    @property
    def cross_validation_curve(self):
        """The cross-validation error metric determined as the mean square error.

        Returns: A two-dimensional CSDM object.
        """
        return self._cv_map


def cv(l1, X, y, cv, alpha=0, n_jobs=1):
    """Return the cross-validation score as negative of mean square error."""
    if isinstance(l1, (Lasso, MultiTaskLasso)):
        fit_params = {"check_input": False}
    if isinstance(l1, LassoLars):
        fit_params = None  # {"Xy": np.dot(X.T, y)}

    cv_score = cross_validate(
        l1,
        X=X,
        y=y,
        scoring="neg_mean_absolute_error",
        cv=cv,
        fit_params=fit_params,
        n_jobs=n_jobs,
        verbose=0,
        return_estimator=True,
    )
    # estimators = cv_score["estimator"]
    # print(estimators)
    # print([item.coef_.shape for item in estimators])
    # diff = 0
    # if len(estimators[0].coef_.shape) >= 2:
    #     diff = np.mean(
    #         [
    #             np.sum((item.coef_[:, 1:] - item.coef_[:, :-1]) ** 2)
    #             for item in estimators
    #         ]
    #     )
    # print(diff)
    return cv_score["test_score"].mean()  # + diff * alpha


def _get_smooth_size(f_shape, regularizer, max_size):
    r"""Return the number of rows appended to for the augmented kernel.

    For smooth-lasso, the number of rows is given as
        rows = \prod_{i=1}^d n_i (\sum_{j=0}^d  (n_j-1)/n_j)

    For sparse ridge fusion, the number of rows is given as
        rows = \prod_{i=1}^d n_i (\sum_{j=0}^d  (n_j-2)/n_j)
    """
    shape = np.asarray(f_shape)
    shape_prod = shape.prod()
    # if shape_prod != max_size:
    #     raise ValueError(
    #         "The product of the shape must be equal to the length of axis 1 of the "
    #         "kernel, K"
    #     )
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
    f_shape = (f_shape,) if isinstance(f_shape, int) else f_shape

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


def _get_augmented_data(K, s, alpha, regularizer, f_shape=None, half=False):
    """Creates a smooth kernel, K, with alpha regularization parameter."""
    if alpha == 0:
        return np.asfortranarray(K), np.asfortranarray(s)

    ks0, ks1 = K.shape
    ss0, ss1 = s.shape

    f_shape = (f_shape,) if isinstance(f_shape, int) else f_shape

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

    if half:
        row1, col, row0 = get_indexes(f_shape)
        J[0] = J[0][row0][:, col]
        J[1] = J[1][row1][:, col]
        # print(row1.size, col.size, row0.size)

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


def get_indexes(f_shape):
    s0, s1 = f_shape[0], f_shape[1]

    def arr_j1(s0, s1):
        arr = []
        lst = [np.arange(s1 - i) + (s1 + 1) * i for i in range(s0)]
        for item in lst:
            arr = np.append(arr, item)
        return np.asarray(arr, dtype=int)

    arr1 = arr_j1(s0, s1 - 1)
    arr2 = arr_j1(min(s0, s1), s1)

    def arr_j0(s0, s1):
        arr = []
        lst = [np.arange(s1 - i) + (s1 + 2) * i + 1 for i in range(s0)]
        for item in lst:
            arr = np.append(arr, item)
        return np.asarray(arr, dtype=int)

    arr3 = arr_j0(min(s0 - 1, s1), s1 - 1)

    return arr1, arr2, arr3


def _get_proper_data(s):
    """Extract the numpy array from the csdm, if csdm and returns a 2D array"""
    s_ = deepcopy(s.y[0].components[0].T) if isinstance(s, cp.CSDM) else deepcopy(s)
    s_ = s_[:, np.newaxis] if s_.ndim == 1 else s_
    return s_
