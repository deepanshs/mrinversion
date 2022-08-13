import csdmpy as cp
import numpy as np

__author__ = "Deepansh J. Srivastava"
__email__ = "srivastava.89@osu.edu"


def find_optimum_singular_value(s):
    length = s.size
    s2 = s**2.0
    sj = s2 / s2.sum()
    T = sj * np.log10(sj)
    T[np.where(np.isnan(T))] = 0
    log_n = np.log10(length)
    log_nm1 = np.log10(length - 1.0)
    entropy = (-1.0 / log_n) * T.sum()

    deltaEntropy = entropy - (entropy * log_n + T) / log_nm1

    c = deltaEntropy.mean()
    d = deltaEntropy.std()

    r = np.argmin(deltaEntropy - c + d)
    return r


def TSVD(K):
    U, S, VT = np.linalg.svd(K, full_matrices=False)
    r = find_optimum_singular_value(S)
    return U, S, VT, r


def TSVD_denoise(dataset, r=-1):
    dataset_ = dataset.y[0].components[0] if isinstance(dataset, cp.CSDM) else dataset

    # original_shape = dataset_.shape
    # size = dataset_.size
    # half = int(np.sqrt(size)) + 1
    # size_2 = half**2
    # diff = size_2 - size
    # new_dat_ = np.append(dataset_.ravel(), dataset_.ravel()[:diff])
    # print(half)
    # new_dat_ = new_dat_.reshape(half, half)

    U, S, VT = np.linalg.svd(dataset_, full_matrices=False)
    r_vec = S / S[0]
    r = np.where(r_vec < 0.04)[0][0]
    # r = find_optimum_singular_value(S)

    print(r)
    S[r:] = 0
    new_dataset_ = (U * S) @ VT

    # new_dataset_ = new_dataset_.ravel()[:size].reshape(original_shape)

    if isinstance(dataset, cp.CSDM):
        new = dataset.copy()
        new.y[0].components[0] = new_dataset_
    else:
        new = new_dataset_
    return new


# standard deviation of noise remains unchanged after unitary transformation.
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
