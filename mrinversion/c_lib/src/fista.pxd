

cdef extern from "fista_lapack.h":
    void fista(double * matrix, \
           double * s, \
           double * zf, \
           double lambd, \
           int maxiter, \
           double * f_k, \
      #   //    double * minimizer_function,
      #   //    double * residue,
           int matrixrow, \
           int matrixcolumn, \
           int nonnegative, \
           double Linv, \
           int totalappend, \
           int * iter, \
           double * cpu_time_, \
      #   //    double wall_time_,
           double tol, \
           int * npros)

# cdef extern from "fista_lapack_cv.h":
#    void fista_cv(double * matrix,
#             double * s,
#             double * matrixTest,
#             double * stest,
#             int testrow,
#             int nfold,
#             double *PredictionError,
#             int m,
#             double *lambdaVal,
#             int n_lambda,
#             double *CV,
#             double *CVstd,
#             int maxiter,
#             double * f_k,
#             int matrixrow,
#             int matrixcolumn,
#             int nonnegative,
#             double Linv,
#             int totalappend,
#             int * iter,
#             double * cpu_time_,
#             double tol,
#             int * npros)