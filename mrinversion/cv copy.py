# coding: utf-8
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>
# In[ ]:
# import os
import matplotlib.pyplot as plt
import numpy as np

from .minimizer.l1 import fista_lambda_cv


def cross_validate(
    folds,
    lambda_range,
    lambda_points,
    solver,
    random=False,
    repeat_folds=1,
    show_details=False,
):

    # os.environ["MKL_NUM_THREADS"] = "2"
    # os.environ["OMP_NUM_THREADS"] = "4"
    lambdaval = np.linspace(lambda_range[0], lambda_range[1], lambda_points)
    lambdaval = 10 ** lambdaval[::-1]

    k_train, s_train, k_test, s_test, m = test_train_set(
        solver, folds, random, repeat_folds
    )
    # return K_train, s_train, K_test, s_test, m
    return get_lambda(
        solver, k_train, s_train, k_test, s_test, lambdaval, m, folds, show_details
    )


def test_train_set(solver, folds, random=False, repeat_folds=1):
    # test_indexSize = np.empty(folds)
    # chi2_test = np.empty((folds, lambdaPoints))
    # cv = np.empty(lambdaPoints)
    # cvk = np.empty(lambdaPoints)

    index = np.arange(solver.k_tilde.shape[0])
    # print('index', index)

    test_size = np.int(index.size / folds)
    m = index.size % folds
    train_size = index.size - test_size
    # print('test_size', test_size, 'train_size', train_size)

    shape_k_train = (train_size, solver.k_tilde.shape[1], folds * repeat_folds)
    k_train = np.zeros(shape_k_train)

    shape_s_train = (train_size, solver.s_tilde.shape[1], folds * repeat_folds)
    s_train = np.zeros(shape_s_train)

    shape_k_test = (test_size + 1, solver.k_tilde.shape[1], folds * repeat_folds)
    k_test = np.zeros(shape_k_test)

    shape_s_test = (test_size + 1, solver.s_tilde.shape[1], folds * repeat_folds)
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

            # print('test', test_index)

            k_train[: train_index.size, :, j * folds + i] = solver.k_tilde[train_index]
            s_train[: train_index.size, :, j * folds + i] = solver.s_tilde[train_index]

            k_test[: test_index.size, :, j * folds + i] = solver.k_tilde[test_index]
            s_test[: test_index.size, :, j * folds + i] = solver.s_tilde[test_index]

    return k_train, s_train, k_test, s_test, m


def get_lambda(
    solver, k_train, s_train, k_test, s_test, lambdaval, m, folds, show_getails=False
):
    s = solver.maxSingularValue ** 2
    # print (K_train.shape, s_train.shape, K_test.shape, s_test.shape)

    non = 0
    if solver.nonnegative:
        non = 1
    cv, std, iteri, cpu_time, wall_time, predictionerror = fista_lambda_cv(
        matrix=k_train,
        s=s_train,
        matrixtest=k_test,
        stest=s_test,
        lambdaval=lambdaval,
        maxiter=solver.max_iteration,
        nonnegative=non,
        linv=(1 / s),
        tol=solver.tolerance,
        npros=solver.npros,
        m=m,
    )

    print("Computing {0}-fold cross-validation...".format(folds))
    print("cpu time = {0} s ".format(cpu_time + cpu_time))
    print("wall time = {0} s".format(wall_time + wall_time))

    opt_lambda = np.argmin(cv)
    min_index_1s_dev = np.where(cv < cv[opt_lambda] + std[opt_lambda])[0].min()
    print(("\nOptimum lamda values from {0} folds cross-valudation are").format(folds))
    print("lambda1 = {0}".format(lambdaval[min_index_1s_dev]))
    print("lambda2 = {0}".format(lambdaval[opt_lambda]))

    if show_getails:
        plot(std, opt_lambda, min_index_1s_dev, predictionerror, lambdaval, cv)
    return lambdaval[min_index_1s_dev], lambdaval[opt_lambda]


# def inverse_laplace(solver, lambda_value, warm_start=True):
#     s = solver.maxSingularValue**2
#     if warm_start:
#         warm_f_k = np.asfortranarray(np.zeros((solver.k_tilde.shape[1], 1)))
#         zf, function, chi_square, iteri, cpu_time0, wall_time0 = fista(
#                                         matrix=solver.k_tilde,
#                                         s=solver.s_tilde.mean(axis=1),
#                                         lambdaval=lambda_value,
#                                         maxiter=solver.max_iteration,
#                                         f_k=warm_f_k,
#                                         nonnegative=int(solver.nonnegative),
#                                         linv=(1/s),
#                                         tol=solver.tolerance*2,
#                                         npros=solver.npros
#                                     )

#         f_k_std = np.asfortranarray(
#             np.tile(warm_f_k[:, 0], (solver.s_tilde.shape[1], 1)).T
#         )
#     else:
#         cpu_time0 = 0.0
#         wall_time0 = 0.0
#         f_k_std = np.asfortranarray(
#             np.zeros((solver.k_tilde.shape[1], solver.s_tilde.shape[1]))
#         )

#     zf, function, chi_square, iteri, cpu_time, wall_time = fista(
#                                         matrix=solver.k_tilde,
#                                         s=solver.s_tilde,
#                                         lambdaval=lambda_value,
#                                         maxiter=solver.max_iteration,
#                                         f_k=f_k_std,
#                                         nonnegative=int(solver.nonnegative),
#                                         linv=(1/s),
#                                         tol=solver.tolerance,
#                                         npros=solver.npros
#                                     )

#     print('Solving for T2 vector at lambda = {0}'.format(lambda_value))
#     print('cpu time = {0} s '.format(cpu_time + cpu_time0))
#     print('wall time = {0} s'.format(wall_time + wall_time0))

#     fit = np.dot(solver.U, zf)

#     plot_data(f_k_std, solver)

#     solver.inverse_vector = f_k_std
#     solver.fit = fit
#     solver.chi_square = chi_square
#     solver.function = function


def plot(std, opt_lambda, min_index_1s_dev, predictionerror, lambdaval, cv):
    plt.figure()
    plt.axhline(y=std[opt_lambda] + cv[opt_lambda], linestyle="--", c="r")

    plt.plot(np.log10(lambdaval), predictionerror, alpha=0.5, linestyle="dotted")

    plt.scatter(
        np.log10(lambdaval[opt_lambda]),
        cv[opt_lambda],
        s=70,
        edgecolors="k",
        facecolors="",
        linewidth=1.5,
    )

    plt.scatter(
        np.log10(lambdaval[min_index_1s_dev]),
        cv[min_index_1s_dev],
        s=70,
        edgecolors="k",
        facecolors="r",
        linewidth=1.5,
    )

    # plt.scatter(np.log10(lambdaval[midlab]), cv[midlab], \
    #             s=70, edgecolors='k', facecolors='b', linewidth=1.5)
    plt.plot(np.log10(lambdaval), cv, c="k", alpha=1)
    plt.yscale("log")
    # plt.errorbar(np.log10(lambdaval), cv, std, c='k', alpha=0.2)
    # plt.ylim([0, cv.max()])
    plt.xlabel(r"log10 ($\lambda$)")
    plt.ylabel("test error")
    plt.show()
