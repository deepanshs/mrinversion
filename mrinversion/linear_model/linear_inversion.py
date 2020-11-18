# -*- coding: utf-8 -*-
import numpy as np

__author__ = "Deepansh J. Srivastava"
__email__ = "srivastava.89@osu.edu"


def find_optimum_singular_value(s):
    length = s.size
    s2 = s ** 2.0
    sj = s2 / s2.sum()
    T = sj * np.log10(sj)
    T[np.where(np.isnan(T))] = 0
    logn = np.log10(length)
    lognm1 = np.log10(length - 1.0)
    entropy = (-1.0 / logn) * T.sum()

    deltaEntropy = entropy - (entropy * logn + T) / lognm1

    c = deltaEntropy.mean()
    d = deltaEntropy.std()

    r = np.argmin(deltaEntropy - c + d)
    return r


def TSVD(K):
    U, S, VT = np.linalg.svd(K, full_matrices=False)
    r = find_optimum_singular_value(S)
    return U, S, VT, r


# standard deviation of noise remains unchanged after unitary tranformation.
def reduced_subspace_kernel_and_data(U, S, VT, signal, sigma=None):
    diagS = np.diag(S)
    K_tilde = np.dot(diagS, VT)
    s_tilde = np.dot(U.T, signal)

    projectedSignal = np.dot(U, s_tilde)
    guess_solution = np.dot(np.dot(VT.T, np.diag(1 / S)), s_tilde)

    K_tilde = np.asfortranarray(K_tilde)
    s_tilde = np.asfortranarray(s_tilde)
    projectedSignal = np.asfortranarray(projectedSignal)
    return K_tilde, s_tilde, projectedSignal, guess_solution
