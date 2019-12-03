
#include "config.h"

/**
 * Evalute the sign of the number
 * @param x A double scalar
 **/
static inline int sign(double x) { return (x > 0.0) - (x < 0.0); }

/**
 * Evaluate the l1 penalty (soft thresholding),
 *      convex function h(A) = c * || B ||_1
 * using the proximal function
 *    prox_h (B)_i = max(abs(B_i) - c, 0)*sign(A_i)
 *
 * @param B A pointer to a double vector of size n.
 * @param c A double scalar.
 * @param n An integer with the size of vector B.
 **/
static inline void l1_soft_thresholding(double *B, double c, int n) {
  double sign_;

  for (int i = 0; i < n; i++) {
    sign_ = sign(B[i]);
    B[i] = fabs(B[i]) - c;
    if (B[i] >= 0.) {
      B[i] *= sign_;
    } else {
      B[i] = 0.;
    }
  }
}

/**
 * Evaluate the l1 penalty (soft thresholding) with non-negative constraints
 *      convex function h(A) = c * || B ||_1
 * using the proximal function
 *      prox_h (B)_i = max(abs(B_i) - c, 0)*sign(A_i)
 *
 * @param B A pointer to a double vector of size n.
 * @param c A double scalar.
 * @param n An integer with the size of vector B.
 **/
static inline void
l1_soft_thresholding_with_non_negative_constrains(double *B, double c, int n) {
  for (int i = 0; i < n; i++) {
    B[i] -= c;
    if (B[i] < 0.0) {
      B[i] = 0.0; //!* SIGN(aa, A)
    }
  }
}

void fista(double *matrix, double *s, double *zf, double hyperparameter,
           int maxiter, double *f_k, int matrixrow, int matrixcolumn,
           int nonnegative, double Linv, int totalappend, int *iter,
           double *cpu_time_, double tol, int *npros);