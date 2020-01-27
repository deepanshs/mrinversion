from copy import deepcopy

import numpy as np
from joblib import delayed
from joblib import Parallel
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


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
            return (
                -1 * np.eye(i)
                + 2 * np.diag(np.ones(i - 1), k=-1)
                - 1 * np.diag(np.ones(i - 2), k=-2)
            )[2:]

        J = generate_J_i(Ai_sparse_ridge_fusion, alpha, f_shape)

    K_ = np.empty((ks0 + smooth_size, ks1))

    K_[:ks0] = K
    start = ks0
    end = ks0
    for J_i in J:
        end = end + J_i.shape[0]
        K_[start:end] = J_i
        start = end

    ss0, ss1 = s.shape
    s_ = np.zeros((ss0 + smooth_size, ss1))
    s_[:ss0] = s

    return np.asfortranarray(K_), np.asfortranarray(s_)


class GeneralL2Lasso:
    r"""
        The Minimizer class solves the following equation,

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
    ):

        self.hyperparameter = {"lambda": lambda1, "alpha": alpha}
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.regularizer = regularizer

        # attributes
        self.f = None
        self.n_iter = None

    def fit(self, K, s, f_shape=None):
        r"""
            Fit the model using the coordinate descent method from scikit-learn.

            Args:
                K: A :math:`m \times n` kernel matrix, :math:`{\bf K}`.
                s: A :math:`m \times m_\text{count}` signal matrix, :math:`{\bf s}`.
                f_shape: The shape of the solution, :math:`{\bf f}`, given as a tuple
                            (n1, n2, ..., nd)
        """
        if s.ndim == 1:
            s = s[:, np.newaxis]
        if f_shape is None:
            f_shape = K.shape[1]
        prod = np.asarray(f_shape).prod()
        if K.shape[1] != prod:
            raise ValueError(
                "The product of the shape, `f_shape`, must be equal to the length of "
                f"the axis 1 of kernel, K, {K.shape[1]} != {prod}."
            )

        self.scale = s.max()
        Ks, ss = _get_augmented_data(
            K=K,
            s=s / self.scale,
            alpha=self.hyperparameter["alpha"],
            regularizer=self.regularizer,
            f_shape=f_shape,
        )

        self.estimator = Lasso(
            alpha=self.hyperparameter["lambda"],
            fit_intercept=False,
            copy_X=True,
            max_iter=self.max_iterations,
            tol=self.tolerance,
            warm_start=False,
            random_state=None,
            selection="random",
            positive=self.positive,
        )
        self.estimator.fit(Ks, ss)
        self.f = self.estimator.coef_
        self.f.shape = (s.shape[1],) + f_shape
        self.n_iter = self.estimator.n_iter_

    def predict(self, K):
        """
        Predict using the linear model.

        Args:
            K: The kernel.

        Return:
            y_predict: The predicted values.
        """
        shape = self.estimator.coef_.shape
        self.estimator.coef_.shape = (1, self.estimator.coef_.size)
        predict = self.estimator.predict(K) * self.scale
        self.estimator.coef_.shape = shape
        return predict

    def score(self, K, s, sample_weights=None):
        """
        Return the coefficient of determination, :math:`R^2`, of the prediction.
        For more information, read scikit-learn documentation.
        """
        return self.estimator.score(K, s / self.scale, sample_weights)


class SmoothLasso(GeneralL2Lasso):
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

        .. rubric:: Attribute documentation

        Attributes:
            f: A ndarray of shape (m_count, nd, ..., n1, n0). The solution,
                        :math:`{\bf f} \in \mathbb{R}^{m_\text{count}
                        \times n_d \times \cdots n_1 \times n_0}`.
            n_iter: Integer, the number of iterations required to reach the
                    specified tolerance.
    """

    def __init__(
        self,
        alpha=1e-3,
        lambda1=1e-6,
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
            regularizer="smooth lasso",
        )


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
        alpha=1e-3,
        lambda1=1e-6,
        max_iterations=10000,
        tolerance=1e-5,
        positive=True,
        f_shape=None,
    ):
        super().__init__(
            alpha=alpha,
            lambda1=lambda1,
            max_iterations=max_iterations,
            tolerance=tolerance,
            positive=positive,
            f_shape=f_shape,
            regularizer="sparse ridge fusion",
        )


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

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.positive = positive
        self.sigma = sigma
        self.regularizer = regularizer
        self.hyperparameter = {}
        self.f = None
        self.randomize = randomize
        self.times = times

    def fit(self, K, s, f_shape=None):
        """
            Solves the lasso variant problem with the give range of alphas
            and lambdas and determine the optimum alpha and lambda hyperparameters
            using k-fold cross-validation.
            Args:
                alphas: A list of alpha > 0 values.
                lambdas: A list of lambda > 0 values.
                regularizer: Regularization method. Options are ['lasso',
                        'smooth lasso', and 'sparse ridge fusion']
                folds: Integer, k, for k-fold cross-validation.
        """

        if s.ndim == 1:
            s = s[:, np.newaxis]
        if f_shape is None:
            f_shape = K.shape[1]
        prod = np.asarray(f_shape).prod()
        if K.shape[1] != prod:
            raise ValueError(
                "The product of the shape, `f_shape`, must be equal to the length of "
                f"the axis 1 of kernel, K, {K.shape[1]} != {prod}."
            )

        self.scale = s.max()
        s = s / self.scale
        cv_indexes = _get_cv_indexes(
            K,
            self.folds,
            self.regularizer,
            f_shape=f_shape,
            random=self.randomize,
            times=self.times,
        )
        self.cv_map = np.zeros((self.cv_alphas.size, self.cv_lambdas.size))

        alpha_ratio = np.ones(self.cv_alphas.size)
        alpha_ratio[1:] = np.sqrt(self.cv_alphas[1:] / self.cv_alphas[:-1])

        # self.hyperparameter["alpha"] = self.cv_alphas[0]
        Ks, ss = _get_augmented_data(
            K=K,
            s=s,
            alpha=self.cv_alphas[0],
            regularizer=self.regularizer,
            f_shape=f_shape,
        )
        start_index = K.shape[0]

        l1 = Lasso(
            alpha=self.cv_lambdas[0],
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
            l1_array[-1].alpha = lambda_

        i = 0
        for alpha_ratio_ in alpha_ratio:
            Ks[start_index:] *= alpha_ratio_
            jobs = (
                delayed(cv)(l1_array[i], Ks, ss, cv_indexes)
                for i in range(self.cv_lambdas.size)
            )
            self.cv_map[i] = Parallel(
                n_jobs=-1,
                verbose=1,
                **{
                    "backend": {
                        "threads": "threading",
                        "processes": "multiprocessing",
                        None: None,
                    }["threads"]
                },
            )(jobs)
            i += 1

        # cv_map contains negated mean square errors, therefore
        #  add the variance.
        self.cv_map += self.sigma ** 2
        self.cv_map = np.abs(self.cv_map)
        # self.cv_optimum_hyperparamters(self.regularizer, folds)

        index = np.unravel_index(self.cv_map.argmin(), self.cv_map.shape)
        self.hyperparameter["alpha"] = self.cv_alphas[index[0]]
        self.hyperparameter["lambda"] = self.cv_lambdas[index[1]]

        # self.minimize(regularizer=regularizer)
        self.opt = GeneralL2Lasso(
            alpha=self.hyperparameter["alpha"],
            lambda1=self.hyperparameter["lambda"],
            # tsvd=True,
            # tsvd_index=None,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            positive=self.positive,
            regularizer=self.regularizer,
        )

        self.opt.fit(K, s, f_shape)
        self.f = self.opt.estimator.coef_

    def predict(self, K):
        return self.opt.predict(K) * self.scale

        #  print("optimum hyperparameters")
        # print("SVD truncation =", self.hyperparameter["r"])
        # print("α =", self.hyperparameter["alpha"])
        # print("λ =", self.hyperparameter["lambda"])
        # diff = s.ravel() - predict.ravel()
        # data_size = s.size
        # sigma = np.sqrt((diff ** 2).sum() / data_size) * s_max
        # mean = diff.sum() * s_max / data_size

        # print("non-zero features in f =", np.where(self.estimator.coef_ > 0)[0].size)
        # print("residue statistics")
        # print("mean, μ =", mean)
        # print("standard deviation, σ =", sigma)

        # self.statistic = {
        #     "hyperparameter, SVD truncation": self.hyperparameter["r"],
        #     "hyperparameter, α": self.hyperparameter["alpha"],
        #     "hyperparameter, λ": self.hyperparameter["lambda"],
        #     "residual mean": mean,
        #     "residual standard deviation": sigma,
        # }

    def cross_validation_error(self):
        """Return the cross-validation error metric."""
        return self.cv_map
