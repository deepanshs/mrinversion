# -*- coding: utf-8 -*-
import os

import csdmpy as cp
import matplotlib.pyplot as plt
import numpy as np

from mrinversion.linear_model._base_l1l2 import prepare_signal
from mrinversion.linear_model.fista import fista
from mrinversion.linear_model.fista import fista_cv

__author__ = "Deepansh Srivastava"
CPU_COUNTS = os.cpu_count()


class LassoFista:
    def __init__(
        self,
        lambda1=1.0e-3,
        max_iterations=1000,
        tolerance=2e-4,
        positive=True,
        inverse_dimension=None,
    ):
        self.hyperparameters = {"lambda": lambda1}
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.inverse_dimension = inverse_dimension

    def fit(self, K, s, warm_start=False):
        s_, self.scale = prepare_signal(s)
        # s_ = s.dependent_variables[0].components[0].T if isinstance(s, cp.CSDM) else s
        # s_ = s_[:, np.newaxis] if s_.ndim == 1 else s_

        # self.scale = np.sqrt(np.mean(np.abs(s_) ** 2))
        # s_ = s_ / self.scale

        sin_val = np.linalg.svd(K, full_matrices=False)[1]

        K_, s_ = np.asfortranarray(K), np.asfortranarray(s_)
        self.f = np.asfortranarray(np.zeros((K_.shape[1], s_.shape[1])))
        lipszit = sin_val[0] ** 2

        if warm_start:
            self.f_1 = np.asfortranarray(np.zeros((K_.shape[1], 1)))
            zf, function, chi2, iter, cpu_time, wall_time = fista.fista(
                matrix=K_,
                s=s_.mean(axis=1),
                lambd=self.hyperparameters["lambda"],
                maxiter=self.max_iterations,
                f_k=self.f_1,
                nonnegative=int(self.positive),
                linv=(1 / lipszit),
                tol=self.tolerance,
                npros=1,
            )
            self.f = np.asfortranarray(np.tile(self.f_1, s_.shape[1]))

        zf, function, chi2, iter, cpu_time, wall_time = fista.fista(
            matrix=K_,
            s=s_,
            lambd=self.hyperparameters["lambda"],
            maxiter=self.max_iterations,
            f_k=self.f,
            nonnegative=int(self.positive),
            linv=(1 / lipszit),
            tol=self.tolerance,
            npros=1,
        )

        self.f *= self.scale

        if not isinstance(s, cp.CSDM):
            return

        # CSDM pack
        self.f = cp.as_csdm(np.squeeze(self.f.T))

        app = self.inverse_dimension.application
        if "com.github.deepanshs.mrinversion" in app:
            meta = app["com.github.deepanshs.mrinversion"]
            is_log = meta.get("log", False)
            if is_log:
                # unit = self.inverse_dimension.coordinates.unit
                coords = np.log10(self.inverse_dimension.coordinates.value)
                self.inverse_dimension = cp.as_dimension(
                    array=coords, label=meta["label"]
                )

        if len(s.dimensions) > 1:
            self.f.dimensions[1] = s.dimensions[1]
        self.f.dimensions[0] = self.inverse_dimension

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
        f = self.f.y[0].components[0].T if isinstance(self.f, cp.CSDM) else self.f
        return np.dot(K, f)

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
        predict = np.squeeze(self.predict(K))
        residue = s_ - predict

        if not isinstance(s, cp.CSDM):
            return residue

        residue = cp.as_csdm(residue.T.copy())
        residue._dimensions = s._dimensions
        return residue


class LassoFistaCV:
    def __init__(
        self,
        lambdas=None,
        folds=10,
        max_iterations=1000,
        tolerance=1e-4,
        positive=True,
        sigma=0.0,
        randomize=False,
        times=1,
        inverse_dimension=None,
        n_jobs=CPU_COUNTS,
    ):

        if lambdas is None:
            self.cv_lambdas = 10 ** ((np.arange(10) / 9) * 5 - 9)[::-1]
        else:
            self.cv_lambdas = np.asarray(lambdas).ravel()

        self.folds = folds
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.sigma = sigma
        self.randomize = randomize
        self.times = times
        self.inverse_dimension = inverse_dimension
        self.n_jobs = n_jobs
        # self.warm_start = warm_start

        self.hyperparameters = {}
        self.f = None

    def fit(self, K, s):
        r"""Fit the model using the coordinate descent method from scikit-learn for
        all alpha anf lambda values using the `n`-folds cross-validation technique.
        The cross-validation metric is the mean squared error.

        Args:
            K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`. A numpy array of
                shape (m, n).
            s: A :math:`m \times m_\text{count}` signal matrix, :math:`{\bf s}` as a
                csdm object or a numpy array or shape (m, m_count).
        """
        s_, self.scale = prepare_signal(s)
        # s_ = s.dependent_variables[0].components[0].T if isinstance(s, cp.CSDM) else s
        # s_ = s_[:, np.newaxis] if s_.ndim == 1 else s_

        # self.scale = np.sqrt(np.mean(np.abs(s_) ** 2))
        # s_ = s_ / self.scale

        sin_val = np.linalg.svd(K, full_matrices=False)[1]

        K_, s_ = np.asfortranarray(K), np.asfortranarray(s_)
        lipszit = sin_val[0] ** 2
        # test train CV set
        k_train, s_train, k_test, s_test, m = test_train_set(
            K_, s_, self.folds, self.randomize, self.times
        )

        self.cv_map, self.std, _, _, _, self.predictionerror = fista_cv.fista(
            matrix=k_train,
            s=s_train,
            matrixtest=k_test,
            stest=s_test,
            lambdaval=self.cv_lambdas,
            maxiter=self.max_iterations,
            nonnegative=int(self.positive),
            linv=(1 / lipszit),
            tol=self.tolerance,
            npros=self.n_jobs,
            m=m,
            var=self.sigma**2,
        )
        # subtract the variance.
        # self.cv_map -= (self.sigma / self.scale) ** 2
        self.cv_map = np.abs(self.cv_map)

        lambdas = np.log10(self.cv_lambdas)
        l1_index, l2_index = calculate_opt_lambda(self.cv_map, self.std)
        lambda1, lambda2 = lambdas[l1_index], lambdas[l2_index]
        self.hyperparameters["lambda"] = 10 ** ((lambda1 + 0.0 * lambda2) / 2.0)

        # Calculate the solution using the complete data at the optimized lambda and
        # alpha values
        self.opt = LassoFista(
            lambda1=self.hyperparameters["lambda"],
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            positive=self.positive,
            inverse_dimension=self.inverse_dimension,
        )
        self.opt.fit(K, s)
        self.f = self.opt.f

        # convert cv_map to csdm
        self.cv_map = cp.as_csdm(np.squeeze(self.cv_map.T.copy()))
        self.cv_map.y[0].component_labels = ["log10"]
        d0 = cp.as_dimension(np.log10(self.cv_lambdas), label="log(λ)")
        self.cv_map.dimensions[0] = d0
        self.cross_validation_curve = self.cv_map

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
        return self.opt.predict(K)

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
        return self.opt.residuals(K, s)

    def cv_plot(self):
        cv = self.cv_map.y[0].components[0]
        predictionerror = self.predictionerror
        std = self.std

        l1_idx, l2_idx = calculate_opt_lambda(cv, std)
        lambdas = np.log10(self.cv_lambdas)
        opt_lambda = 0.5 * (lambdas[l1_idx] + lambdas[l2_idx])

        plt.axhline(y=std[l1_idx] + cv[l1_idx], linestyle="--", c="r")
        plt.plot(lambdas, predictionerror, alpha=0.5, linestyle="dotted")

        kwargs = {"s": 70, "edgecolors": "k", "linewidth": 1.5}
        plt.scatter(lambdas[l1_idx], cv[l1_idx], facecolors="b", **kwargs)
        plt.scatter(lambdas[l2_idx], cv[l2_idx], facecolors="r", **kwargs)
        plt.axvline(x=opt_lambda, linestyle="--", c="g", label="$\\lambda^*$")

        plt.plot(lambdas, cv, c="k", alpha=1, label="CV curve")
        plt.yscale("log")
        # plt.errorbar(lambdas, cv, std, c='k', alpha=0.2)
        # plt.ylim([0, cv.max()])
        plt.legend()
        plt.xlabel(r"log10 ($\lambda$)")
        plt.ylabel("test error")


def calculate_opt_lambda(cv, std):
    l1_index = np.unravel_index(cv.argmin(), cv.shape)[0]
    cv_std = cv[l1_index] + std[l1_index]
    temp = np.where(cv < cv_std)[0]
    if temp.size != 0:
        index = np.where(temp > l1_index)[0]
        if index.size != 0:
            index = index.max()
            l2_index = temp[index]
            return l1_index, l2_index
    return l1_index, l1_index


def test_train_set(X, y, folds, random=False, repeat_folds=1):
    # test_indexSize = np.empty(folds)
    # chi2_test = np.empty((folds, lambdaPoints))
    # cv = np.empty(lambdaPoints)
    # cvk = np.empty(lambdaPoints)

    index = np.arange(X.shape[0])
    # print('index', index)

    test_size = np.int(index.size / folds)
    m = index.size % folds
    train_size = index.size - test_size
    # print('test_size', test_size, 'train_size', train_size)

    shape_k_train = (train_size, X.shape[1], folds * repeat_folds)
    k_train = np.zeros(shape_k_train)

    shape_s_train = (train_size, y.shape[1], folds * repeat_folds)
    s_train = np.zeros(shape_s_train)

    shape_k_test = (test_size + 1, X.shape[1], folds * repeat_folds)
    k_test = np.zeros(shape_k_test)

    shape_s_test = (test_size + 1, y.shape[1], folds * repeat_folds)
    s_test = np.zeros(shape_s_test)

    for j in range(repeat_folds):
        if random:
            np.random.shuffle(index)
        for i in range(folds):
            if random:
                if i < m:
                    test_index = index[i * (test_size + 1) : (i + 1) * (test_size + 1)]
                    set_index = np.arange(
                        i * (test_size + 1), (i + 1) * (test_size + 1)
                    )
                else:
                    test_index = index[i * test_size + m : (i + 1) * test_size + m]
                    set_index = np.arange(i * test_size + m, (i + 1) * test_size + m)
                train_index = np.delete(index, set_index)
            else:
                if i < m:
                    test_index = index[i:None:folds][: test_size + 1]
                else:
                    test_index = index[i:None:folds][:test_size]
                train_index = np.delete(index, test_index)

            k_train[: train_index.size, :, j * folds + i] = X[train_index]
            s_train[: train_index.size, :, j * folds + i] = y[train_index]

            k_test[: test_index.size, :, j * folds + i] = X[test_index]
            s_test[: test_index.size, :, j * folds + i] = y[test_index]

    return k_train, s_train, k_test, s_test, m
