import numba as nb
import numpy as np
from numpy.linalg import norm


@nb.njit(fastmath=True)
def l1_soft_threshold(a: np.ndarray, c: float):
    """Soft threshold without non-negative constraint."""
    b = np.abs(a) - c
    b = np.where(b >= 0, b * np.sign(a), 0.0)
    return b


@nb.njit(fastmath=True)
def l1_soft_threshold_nonnegative(a: np.ndarray, c: float):
    """Soft threshold with non-negative constraint."""
    b = a - c
    b += np.abs(b)
    b /= 2.0
    return b


@nb.njit(fastmath=True)
def fista(
    matrix: np.ndarray,
    s: np.ndarray,
    f_k: np.ndarray,
    lam: float,
    max_iter: int,
    tol: float,
    l_inv: float,
    nonnegative: bool = False,
):
    matrix = np.asarray(matrix)
    s = np.asarray(s)

    residue = np.zeros(max_iter)
    data_consistency = np.zeros(max_iter)

    gradient = matrix.T @ matrix
    c = matrix.T @ s

    last_data_consistency = 0.0
    normalization_factor = norm(s) ** 2
    last_data_consistency = float(normalization_factor)

    last_fk_l1 = 0.0
    t_kp1 = 1.0
    y_k = f_k.copy()

    for k in range(max_iter):
        t_k = t_kp1
        f_km1 = f_k.copy()

        t_kp1 = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) * 0.5
        constant_factor = (t_k - 1.0) / t_kp1

        temp_c = gradient @ y_k
        temp_c = l_inv * (c - temp_c)
        temp_c += y_k

        if nonnegative:
            f_k[:] = l1_soft_threshold_nonnegative(temp_c, lam * l_inv)
        else:
            f_k[:] = l1_soft_threshold(temp_c, lam * l_inv)

        temp = (matrix @ f_k) - s
        residue_temp = norm(temp) ** 2
        fk_l1 = np.sum(np.abs(f_k))

        residue[k] = residue_temp
        data_consistency[k] = residue[k] + lam * fk_l1
        residue[k] = residue[k] / normalization_factor

        if k >= 5:
            recent_avg = np.mean(residue[k - 5 : k])
            if abs(1.0 - (recent_avg / residue[k])) <= tol:
                break

        # data consistency check
        if data_consistency[k] > last_data_consistency:
            f_k[:] = f_km1.copy()
            fk_l1 = last_fk_l1
            data_consistency[k] = data_consistency[k - 1]
            residue[k] = residue[k - 1]

        last_data_consistency = data_consistency[k]
        last_fk_l1 = fk_l1

        y_k = f_k + constant_factor * (f_k - f_km1)

    zf = matrix @ f_k
    iter = k

    return zf, f_k, residue[: iter + 1], data_consistency[: iter + 1], iter


@nb.njit(fastmath=True)
def fista_cv_nb(
    matrix: np.ndarray,
    s: np.ndarray,
    matrix_test: np.ndarray,
    s_test: np.ndarray,
    max_iter: int,
    lambda_vals: np.ndarray,
    nonnegative: bool,
    l_inv: float,
    tol: float,
):
    n_fold = matrix.shape[-1]
    n_lambda = len(lambda_vals)
    n_targets = s.shape[1]
    n_features = matrix.shape[1]
    prediction_error = np.zeros((n_lambda, n_fold))
    iter_arr = np.zeros(n_lambda)

    residue = np.zeros(max_iter)
    data_consistency = np.zeros(max_iter)

    for fold in range(n_fold):
        x_train = matrix[..., fold]
        y_train = s[..., fold]
        x_test = matrix_test[..., fold]
        y_test = s_test[..., fold]
        y_points = y_test.shape[0] * y_test.shape[1]

        gradient = x_train.T @ x_train
        c = x_train.T @ y_train

        norm_factor = norm(y_train) ** 2
        f_k = np.zeros((n_features, n_targets))
        y_k = f_k.copy()

        for j, lam in enumerate(lambda_vals):
            t_kp1 = 1.0
            y_k[:] = 0.0
            f_k[:] = 0.0
            residue[:] = 0
            fk_l1 = 0.0
            last_fk_l1 = 0.0
            data_consistency[:] = 0
            last_data_consistency = norm_factor

            for k in range(max_iter):
                t_k = t_kp1
                f_km1 = f_k.copy()
                t_kp1 = (1.0 + np.sqrt(1.0 + 4.0 * t_k**2)) / 2.0
                constant_factor = (t_k - 1.0) / t_kp1

                grad_yk = gradient @ y_k
                temp_c = y_k - l_inv * (grad_yk - c)

                if nonnegative:
                    f_k[:] = l1_soft_threshold_nonnegative(temp_c, l_inv * lam)
                else:
                    f_k[:] = l1_soft_threshold(temp_c, l_inv * lam)

                residue[k] = norm(x_train @ f_k - y_train) ** 2
                fk_l1 = np.sum(np.abs(f_k))
                data_consistency[k] = residue[k] + lam * fk_l1

                if k >= 5:
                    recent_avg = np.mean(residue[k - 5 : k])
                    if abs(1.0 - (recent_avg / residue[k])) <= tol:
                        break

                # data consistency check
                if data_consistency[k] > last_data_consistency:
                    f_k[:] = f_km1.copy()
                    fk_l1 = last_fk_l1
                    data_consistency[k] = data_consistency[k - 1]
                    residue[k] = residue[k - 1]

                last_data_consistency = data_consistency[k]
                last_fk_l1 = fk_l1

                y_k = f_k + constant_factor * (f_k - f_km1)

            err = np.linalg.norm(x_test @ f_k - y_test) ** 2
            prediction_error[j, fold] = err / y_points
            iter_arr[j] = k

    return prediction_error, iter_arr


def fista_cv(
    matrix: np.ndarray,
    s: np.ndarray,
    matrix_test: np.ndarray,
    s_test: np.ndarray,
    max_iter: int,
    lambda_vals: np.ndarray,
    nonnegative: bool,
    l_inv: float,
    tol: float,
):
    prediction_error, iter_arr = fista_cv_nb(
        matrix, s, matrix_test, s_test, max_iter, lambda_vals, nonnegative, l_inv, tol
    )

    cv = prediction_error.mean(axis=1)
    cvstd = prediction_error.std(axis=1)
    return cv, cvstd, prediction_error, iter_arr
