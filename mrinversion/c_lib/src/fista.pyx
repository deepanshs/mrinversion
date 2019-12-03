

import cython
cimport numpy as np
import numpy as np
cimport fista as c_fista
from libcpp cimport bool as bool_t

@cython.boundscheck(False)
@cython.wraparound(False)
def fista(
        np.ndarray[double, ndim=2, mode="fortran"] matrix,
        np.ndarray[double, ndim=2, mode="fortran"] s,
        double hyperparameter,
        double Linv,
        int maxiter = 15000,
        bool_t nonnegative = True,
        double tol=1e-7,
        int npros=1,
        f_k=None):

    cdef int iter_
    cdef double cpu_time_

    cdef np.ndarray[double, ndim=1, mode="c"] f_k_c
    cdef np.ndarray[double, ndim=1, mode="c"] zf = np.zeros( s.shape[1]*matrix.shape[0])

    if f_k is not None:
        f_k_c = f_k
    else:
        f_k_c = np.zeros( s.shape[1]*matrix.shape[1])

    c_fista.fista(
        &matrix[0,0], #_c.data.as_doubles,
        &s[0,0], #_c.data.as_doubles,
        &zf[0], #.data.as_doubles,
        hyperparameter,
        maxiter,
        &f_k_c[0], # .data.as_doubles,
        matrix.shape[0], # matrixrow,
        matrix.shape[1], # matrixcolumn,
        nonnegative,
        Linv,
        s.shape[1], # totalappend
        &iter_,
        &cpu_time_,
        tol,
        &npros
	)

    return zf, f_k_c, iter_, cpu_time_


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def fista_cv_cython(
#         np.ndarray[double, ndim=2, mode="fortran"] matrix,
#         np.ndarray[double, ndim=2, mode="fortran"] s,
#         np.ndarray[double, ndim=2, mode="fortran"] matrixTest,
#         np.ndarray[double, ndim=2, mode="fortran"] stest,
#         int nfold,
#         int m,
#         int lambda_size,
#         double Linv,
#         int maxiter = 15000,
#         int nonnegative = 1,
#         double tol=1e-7,
#         int npros=1):

#     cdef int iter
#     cdef double cpu_time_

#     cdef np.ndarray[double, ndim=1, mode="c"] lambdaVal = np.zeros( lambda_size)
#     cdef np.ndarray[double, ndim=1, mode="c"] PredictionError = np.zeros( lambda_size * nfold)
#     cdef np.ndarray[double, ndim=1, mode="c"] CV = np.zeros( lambda_size)
#     cdef np.ndarray[double, ndim=1, mode="c"] CVstd = np.zeros( lambda_size)
#     cdef np.ndarray[double, ndim=1, mode="c"] f_k = np.zeros( s.shape[1]*matrix.shape[1])

#     c_fista.fista_cv(
#             &matrix[0,0],
#             &s[0,0],
#             &matrixTest[0,0],
#             &stest[0,0],
#             stest.shape[0],
#             nfold,
#             &PredictionError[0],
#             m,
#             &lambdaVal[0],
#             lambda_size,
#             &CV[0],
#             &CVstd[0],
#             maxiter,
#             &f_k[0],
#             matrix.shape[0],
#             matrix.shape[1],
#             nonnegative,
#             Linv,
#             s.shape[1],
#             &iter,
#             &cpu_time_,
#             tol,
#             &npros)