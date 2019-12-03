
#include "c_array.h"
#include "math.h"
#include <time.h>
#include "mkl.h"

void l1_softTresholding(double *B, double c, int n);
void l1_softTresholdingWithNonNegative(double *B, double c, int n);
int sign(double x);

void fista_cv(double * matrix,
            double * s,
            double * matrixTest,
            double * stest,
            int testrow,
            int nfold,
            double *PredictionError,
            int m,
            double *lambdaVal,
            int n_lambda,
            double *CV,
            double *CVstd,
            int maxiter,
            double * f_k,
            int matrixrow,
            int matrixcolumn,
            int nonnegative,
            double Linv,
            int totalappend,
            int * iter,
            double * cpu_time_,
            double tol,
            int * npros);