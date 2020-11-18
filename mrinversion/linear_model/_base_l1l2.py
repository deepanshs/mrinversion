# -*- coding: utf-8 -*-
from copy import deepcopy

import csdmpy as cp
import numpy as np
from joblib import delayed
from joblib import Parallel
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
        method="gradient_decent",
    ):

        self.hyperparameters = {"lambda": lambda1, "alpha": alpha}
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.regularizer = regularizer
        self.inverse_dimension = inverse_dimension
        self.f_shape = tuple([item.count for item in inverse_dimension])[::-1]
        self.method = method

        # attributes
        self.f = None
        self.n_iter = None

    def fit(self, K, s):
        r"""
        Fit the model using the coordinate descent method from scikit-learn.

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
        # if isinstance(s, cp.CSDM):
        #     self.s = s
        #     s_ = s.dependent_variables[0].components[0].T
        # else:
        #     s_ = s

        # if s_.ndim == 1:
        #     s_ = s_[:, np.newaxis]
        # prod = np.asarray(self.f_shape).prod()
        # if K.shape[1] != prod:
        #     raise ValueError(
        #         "The product of the shape, `f_shape`, must be equal to the length of "
        #         f"the axis 1 of kernel, K, {K.shape[1]} != {prod}."
        #     )

        half = False
        if "half" in self.inverse_dimension[0].application:
            half = self.inverse_dimension[0].application["half"]

        self.scale = s_.real.max()
        Ks, ss = _get_augmented_data(
            K=K,
            s=s_ / self.scale,
            alpha=s_.size * self.hyperparameters["alpha"],
            regularizer=self.regularizer,
            f_shape=self.f_shape,
            half=half,
        )

        estimator = _get_minimizer(
            self.hyperparameters["lambda"],
            self.method,
            self.max_iterations,
            self.tolerance,
            self.positive,
        )

        estimator.fit(Ks, ss)
        f = estimator.coef_.copy()

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

        f *= self.scale

        if isinstance(s, cp.CSDM):
            f = self.pack_as_csdm(f, s)
            # f = cp.as_csdm(f)

            # if len(s.dimensions) > 1:
            #     f.dimensions[2] = s.dimensions[1]
            # f.dimensions[1] = self.inverse_dimension[1]
            # f.dimensions[0] = self.inverse_dimension[0]

        self.estimator = estimator
        self.f = f
        self.n_iter = estimator.n_iter_

    def pack_as_csdm(self, f, s):
        f = cp.as_csdm(f)

        if len(s.dimensions) > 1:
            f.dimensions[2] = s.dimensions[1]
        f.dimensions[1] = self.inverse_dimension[1]
        f.dimensions[0] = self.inverse_dimension[0]
        return f

    def predict(self, K):
        r"""
        Predict the signal using the linear model.

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
        predict = self.estimator.predict(K) * self.scale

        return predict

    def residuals(self, K, s):
        r"""
        Return the residual as the difference the data and the prediced data(fit),
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
        if isinstance(s, cp.CSDM):
            s_ = s.dependent_variables[0].components[0].T
        else:
            s_ = s
        predict = np.squeeze(self.estimator.predict(K)) * self.scale
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
        return self.estimator.score(K, s / self.scale, sample_weights)


class GeneralL2LassoLS:
    def __init__(
        self,
        alpha=1e-6,
        lambda1=1e-6,
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
        method="gradient_decent",
    ):

        self.method = method
        self.folds = folds

        self.n_jobs = n_jobs
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.sigma = sigma
        self.regularizer = regularizer
        self.hyperparameters = {"alpha": alpha, "lambda": lambda1}
        self.f = None
        self.randomize = randomize
        self.times = times
        self.verbose = verbose
        self.inverse_dimension = inverse_dimension
        self.f_shape = tuple([item.count for item in inverse_dimension])[::-1]

    def fit(self, K, s, **kwargs):
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
        s_ = _get_proper_data(s)
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

        half = False
        if "half" in self.inverse_dimension[0].application:
            half = self.inverse_dimension[0].application["half"]

        ks0 = K.shape[0]
        Ks, ss = _get_augmented_data(
            K=K,
            s=s_,
            alpha=s_.size * self.hyperparameters["alpha"],
            regularizer=self.regularizer,
            f_shape=self.f_shape,
            half=half,
        )
        # start_index = K.shape[0]

        l1 = _get_minimizer(
            self.hyperparameters["lambda"],
            self.method,
            self.max_iterations,
            self.tolerance,
            self.positive,
        )

        self.path = []
        sigma_sq = self.sigma ** 2

        def fnc(x0):
            self.path.append(x0)
            # Ks_ = deepcopy(Ks)
            Ks[ks0:] *= np.sqrt(x0[0] / self.hyperparameters["alpha"])
            self.hyperparameters["alpha"] = x0[0]
            l1.alpha = x0[1] / 2.0
            mse = -cv(
                l1, Ks, ss, cv_indexes, alpha=self.hyperparameters["alpha"], n_jobs=-1
            )
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
        # res = minimize(
        #     fnc,
        #     x0,
        #     bounds=[(1e-15, 1e-4), (1e-15, 1e-4)],
        #     method="Powell",
        #     **kwargs,
        # )
        res = least_squares(
            fnc,
            x0,
            # method="Powell",
            bounds=[(1e-15, 1e-15), (1e-4, 1e-4)],
            verbose=2,
            **kwargs,
        )

        self.hyperparameters["alpha"] = res.x[0]
        self.hyperparameters["lambda"] = res.x[1]

        print(res.message, res.status)

        # Calculate the solution using the complete data at the optimized lambda and
        # alpha values
        self.opt = GeneralL2Lasso(
            alpha=self.hyperparameters["alpha"],
            lambda1=self.hyperparameters["lambda"],
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            positive=self.positive,
            regularizer=self.regularizer,
            inverse_dimension=self.inverse_dimension,
            method=self.method,
        )
        self.opt.fit(K, s)
        self.f = self.opt.f

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
        method="gradient_decent",
    ):

        if alphas is None:
            self.cv_alphas = 10 ** ((np.arange(5) / 4) * 2 - 4)[::-1]
        else:
            self.cv_alphas = np.asarray(alphas).ravel()

        if lambdas is None:
            self.cv_lambdas = 10 ** ((np.arange(10) / 9) * 5 - 9)[::-1]
        else:
            self.cv_lambdas = np.asarray(lambdas).ravel()

        self.method = method
        self.folds = folds

        self.n_jobs = n_jobs
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.sigma = sigma
        self.regularizer = regularizer
        self.hyperparameters = {}
        self.f = None
        self.randomize = randomize
        self.times = times
        self.verbose = verbose
        self.inverse_dimension = inverse_dimension
        self.f_shape = tuple([item.count for item in inverse_dimension])[::-1]

    def fit(self, K, s, scoring="neg_mean_squared_error"):
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

        s_ = _get_proper_data(s)

        # if isinstance(s, cp.CSDM):
        #     self.s = s
        #     s_ = s.dependent_variables[0].components[0].T
        # else:
        #     s_ = s

        # s_ = s_[:, np.newaxis] if s_.ndim == 1 else s_
        # prod = np.asarray(self.f_shape).prod()
        # if K.shape[1] != prod:
        #     raise ValueError(
        #         "The product of the shape, `f_shape`, must be equal to the length of "
        #         f"the axis 1 of kernel, K, {K.shape[1]} != {prod}."
        #     )

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
        if self.cv_alphas.size != 1 and self.cv_alphas[0] != 0:
            alpha_ratio[1:] = np.sqrt(self.cv_alphas[1:] / self.cv_alphas[:-1])

        half = False
        if "half" in self.inverse_dimension[0].application:
            half = self.inverse_dimension[0].application["half"]

        Ks, ss = _get_augmented_data(
            K=K,
            s=s_,
            alpha=s_.size * self.cv_alphas[0],
            regularizer=self.regularizer,
            f_shape=self.f_shape,
            half=half,
        )
        start_index = K.shape[0]

        l1 = _get_minimizer(
            self.cv_lambdas[0],
            self.method,
            self.max_iterations,
            self.tolerance,
            self.positive,
        )
        # l1.fit(Ks, ss)

        l1_array = []
        for lambda_ in self.cv_lambdas:
            l1_array.append(deepcopy(l1))
            l1_array[-1].alpha = lambda_ / 2.0

        j = 0
        for alpha_ratio_ in alpha_ratio:
            if alpha_ratio_ != 0:
                Ks[start_index:] *= alpha_ratio_
            jobs = (
                delayed(cv)(l1_, Ks, ss, cv_indexes, alpha=alpha_)
                for l1_, alpha_ in zip(l1_array, self.cv_alphas)
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
        if scoring == "neg_mean_squared_error":
            self.cv_map -= self.sigma ** 2
            index = np.where(self.cv_map < 0)
            self.cv_map[index] = np.nan

            # After subtracting the variance, any negative values in the cv grid is a
            # result of fitting noise. Take the absolute value of cv to avoid such
            # models.
            # self.cv_map = np.abs(self.cv_map)

        # The argmin of the minimum value is the selected model as it has the least
        # prediction error.
        index = np.unravel_index(self.cv_map.argmin(), self.cv_map.shape)
        self.hyperparameters["alpha"] = self.cv_alphas[index[0]]
        self.hyperparameters["lambda"] = self.cv_lambdas[index[1]]

        # Calculate the solution using the complete data at the optimized lambda and
        # alpha values
        self.opt = GeneralL2Lasso(
            alpha=self.hyperparameters["alpha"],
            lambda1=self.hyperparameters["lambda"],
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            positive=self.positive,
            regularizer=self.regularizer,
            inverse_dimension=self.inverse_dimension,
            method=self.method,
        )
        self.opt.fit(K, s)
        self.f = self.opt.f

        # self.opt = [
        #     GeneralL2Lasso(
        #         alpha=self.hyperparameters["alpha"],
        #         lambda1=self.hyperparameters["lambda"],
        #         max_iterations=self.max_iterations,
        #         tolerance=self.tolerance,
        #         positive=self.positive,
        #         regularizer=self.regularizer,
        #         inverse_dimension=self.inverse_dimension,
        #         method=self.method,
        #     )
        #     for i in range(self.folds)
        # ]

        # # print(cv_indexes)
        # ks0 = K.shape[0]
        # for i in range(self.folds):
        #     lim = ks0 - len(cv_indexes[i][1])
        #     # s_csdm = cp.as_csdm(s_[cv_indexes[i][0][:lim]])
        #     self.opt[i].fit(K[cv_indexes[i][0][:lim]], s_[cv_indexes[i][0][:lim]])

        # self.f = np.asarray([self.opt[i].f for i in range(self.folds)]).mean(axis=0)
        # self.f_std = np.asarray([self.opt[i].f for i in range(self.folds)]).std(
        # axis=0)

        # convert cv_map to csdm
        self.cv_map = cp.as_csdm(np.squeeze(self.cv_map.T))
        if len(self.cv_alphas) != 1:
            d0 = cp.as_dimension(-np.log10(self.cv_alphas), label="-log(α)")
            self.cv_map.dimensions[0] = d0

        if len(self.cv_lambdas) == 1:
            return

        d1 = cp.as_dimension(-np.log10(self.cv_lambdas), label="-log(λ)")
        if len(self.cv_alphas) != 1:
            self.cv_map.dimensions[1] = d1
        else:
            self.cv_map.dimensions[0] = d1

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
        """The cross-validation error metric determined as the mean square error.

        Returns: A two-dimensional CSDM object.
        """
        return self.cv_map


def _get_minimizer(alpha, method, max_iterations, tolerance, positive):
    """Return the estimator for the method"""
    # The factor 0.5 for alpha in the Lasso/LassoLars problem is to compensate
    # 1/(2 * n_sample) factor in OLS term.
    if method == "multi-task":
        return MultiTaskLasso(
            alpha=alpha / 2.0,  # self.cv_lambdas[0] / 2.0,
            fit_intercept=False,
            normalize=False,
            # precompute=True,
            max_iter=max_iterations,
            tol=tolerance,
            copy_X=True,
            # positive=self.positive,
            random_state=None,
            warm_start=False,
            selection="random",
        )

    if method == "gradient_decent":
        return Lasso(
            alpha=alpha / 2.0,
            fit_intercept=False,
            normalize=False,
            precompute=True,
            max_iter=max_iterations,
            tol=tolerance,
            copy_X=True,
            positive=positive,
            random_state=None,
            warm_start=False,
            selection="random",
        )

    if method == "lars":
        return LassoLars(
            alpha=alpha / 2.0,
            fit_intercept=False,
            verbose=True,
            normalize=False,
            precompute="auto",
            max_iter=max_iterations,
            copy_X=True,
            fit_path=False,
            positive=positive,
            jitter=None,
            random_state=0,
        )


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


def _get_augmented_data(K, s, alpha, regularizer, f_shape=None, half=False):
    """Creates a smooth kernel, K, with alpha regularization parameter."""
    if alpha == 0:
        return np.asfortranarray(K), np.asfortranarray(s)

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
    if isinstance(s, cp.CSDM):
        # self.s = s
        s_ = deepcopy(s.dependent_variables[0].components[0].T)
    else:
        s_ = deepcopy(s)

    s_ = s_[:, np.newaxis] if s_.ndim == 1 else s_
    return s_
