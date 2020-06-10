# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np

from mrinversion.linear_model import _get_augmented_data
from mrinversion.linear_model import _get_cv_indexes


def test01():
    K = np.empty((8, 16))
    indexes = _get_cv_indexes(K, 4, "lasso", f_shape=(4, 4))

    index_test = [
        [[1, 2, 3, 5, 6, 7], [0, 4]],
        [[0, 1, 2, 4, 5, 6], [3, 7]],
        [[0, 1, 3, 4, 5, 7], [2, 6]],
        [[0, 2, 3, 4, 6, 7], [1, 5]],
    ]

    assert indexes == index_test, "test01"


def test02():
    K = np.empty((8, 16))
    lst = [
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
    ]

    index_test = [
        [[1, 2, 3, 5, 6, 7], [0, 4]],
        [[0, 1, 2, 4, 5, 6], [3, 7]],
        [[0, 1, 3, 4, 5, 7], [2, 6]],
        [[0, 2, 3, 4, 6, 7], [1, 5]],
    ]

    indexes = _get_cv_indexes(K, folds=4, regularizer="smooth lasso", f_shape=(4, 4))

    index_test_1 = deepcopy(index_test)
    for tr_, _ in index_test_1:
        tr_ += lst

    assert indexes == index_test_1, "test02 - 1"

    indexes = _get_cv_indexes(K, 4, "smooth lasso", f_shape=16)

    index_test_2 = deepcopy(index_test)
    for tr_, _ in index_test_2:
        tr_ += lst[:15]

    assert indexes == index_test_2, "test02 - 2"


def test03():
    # 1d - explicit
    K = np.empty((5, 5))
    s = np.empty((5, 1))
    KK, _ = _get_augmented_data(K, s, 1, "smooth lasso", f_shape=(5))

    A = [[1, -1, 0, 0, 0], [0, 1, -1, 0, 0], [0, 0, 1, -1, 0], [0, 0, 0, 1, -1]]
    assert np.allclose(KK[5:], A)

    # 2d - explicit symmetric
    K = np.empty((5, 4))
    s = np.empty((5, 1))
    KK, _ = _get_augmented_data(K, s, 1, "smooth lasso", f_shape=(2, 2))

    J1 = [[1, 0, -1, 0], [0, 1, 0, -1]]
    J2 = [[1, -1, 0, 0], [0, 0, 1, -1]]

    assert np.allclose(KK[5:7], J1)
    assert np.allclose(KK[7:9], J2)

    # 2d - explicit asymmetric
    K = np.empty((5, 6))
    s = np.empty((5, 1))
    KK, _ = _get_augmented_data(K, s, 1, "smooth lasso", f_shape=(3, 2))

    J1 = [
        [1, 0, -1, 0, 0, 0],
        [0, 1, 0, -1, 0, 0],
        [0, 0, 1, 0, -1, 0],
        [0, 0, 0, 1, 0, -1],
    ]
    J2 = [[1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0], [0, 0, 0, 0, 1, -1]]

    assert np.allclose(KK[5:9], J1)
    assert np.allclose(KK[9:12], J2)

    # 1d - function
    K = np.empty((5, 12))
    KK, _ = _get_augmented_data(K, s, 1, "smooth lasso", f_shape=(12))

    A1 = (-1 * np.eye(12) + np.diag(np.ones(11), k=-1))[1:]

    assert np.allclose(KK[5:], A1)

    # 2d - function symmetric
    K = np.empty((5, 16))
    s = np.empty((5, 1))
    KK, _ = _get_augmented_data(K, s, 1, "smooth lasso", f_shape=(4, 4))

    J = -1 * np.eye(4) + np.diag(np.ones(3), k=-1)
    I_eye = np.eye(4)
    J1 = np.kron(J[1:], I_eye)
    J2 = np.kron(I_eye, J[1:])

    assert np.allclose(KK[5 : 5 + 12], J1)
    assert np.allclose(KK[5 + 12 :], J2)

    # 2d - function asymmetric
    K = np.empty((5, 12))
    KK, _ = _get_augmented_data(K, s, 1, "smooth lasso", f_shape=(4, 3))

    A1 = -1 * np.eye(4) + np.diag(np.ones(3), k=-1)
    I2 = np.eye(3)
    J1 = np.kron(A1[1:], I2)

    A2 = -1 * np.eye(3) + np.diag(np.ones(2), k=-1)
    I1 = np.eye(4)
    J2 = np.kron(I1, A2[1:])

    assert np.allclose(KK[5 : 5 + 9], J1)
    assert np.allclose(KK[5 + 9 :], J2)


def test04():
    # 1d - explicit
    K = np.empty((5, 5))
    s = np.empty((5, 1))
    KK, _ = _get_augmented_data(K, s, 1, "sparse ridge fusion", f_shape=(5))

    A = [[-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1]]

    assert np.allclose(KK[5:], A)

    # 2d - explicit symmetric
    K = np.empty((5, 9))
    s = np.empty((5, 1))
    KK, _ = _get_augmented_data(K, s, 1, "sparse ridge fusion", f_shape=(3, 3))

    J1 = [
        [-1, 0, 0, 2, 0, 0, -1, 0, 0],
        [0, -1, 0, 0, 2, 0, 0, -1, 0],
        [0, 0, -1, 0, 0, 2, 0, 0, -1],
    ]
    J2 = [
        [-1, 2, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 2, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 2, -1],
    ]

    assert np.allclose(KK[5:8], J1)
    assert np.allclose(KK[8:11], J2)

    # 2d - explicit asymmetric
    K = np.empty((5, 12))
    s = np.empty((5, 1))
    KK, _ = _get_augmented_data(K, s, 1, "sparse ridge fusion", f_shape=(3, 4))

    J1 = [
        [-1, 0, 0, 0, 2, 0, 0, 0, -1, 0, 0, 0],
        [0, -1, 0, 0, 0, 2, 0, 0, 0, -1, 0, 0],
        [0, 0, -1, 0, 0, 0, 2, 0, 0, 0, -1, 0],
        [0, 0, 0, -1, 0, 0, 0, 2, 0, 0, 0, -1],
    ]
    J2 = [
        [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 2, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, -1],
    ]

    assert np.allclose(KK[5:9], J1)
    assert np.allclose(KK[9:15], J2)
