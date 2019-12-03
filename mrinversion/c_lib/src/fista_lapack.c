
#include "fista_lapack.h"
#include "omp.h"

void fista(double *matrix, double *s, double *zf, double hyperparameter,
           int maxiter, double *f_k, int matrixrow, int matrixcolumn,
           int nonnegative, double Linv, int totalappend, int *iter,
           double *cpu_time_, double tol, int *npros)
//    double * minimizer_function,
//    double * residue,
//    double wall_time_,
{

  int k, i, j;
  double t_k, sumabs, previous_minimizer_function, t_kp1, residue_temp;
  double sumabsm1, constantFactor, normalizationFactor, hyperparameter_linv;
  double constantFactor_p1, temp_res;

  double *temp = malloc_double(matrixrow);
  double *c = malloc_double(totalappend * matrixcolumn);
  double *f_km1 = malloc_double(matrixcolumn * totalappend);
  double *y_k = malloc_double(totalappend * matrixcolumn);

  // The gradient matrix.
  double *Gradient = malloc_double(matrixcolumn * matrixcolumn);

  // double start_wall_time, end_wall_time;
  clock_t start, end;

  start = clock();

  double *residue = malloc_double(maxiter);
  double *minimizer_function = malloc_double(maxiter);

  sumabsm1 = 0.0;
  t_kp1 = 1.0;
  hyperparameter_linv = hyperparameter * Linv;

  int c_total = matrixrow * totalappend;
  int f_total = matrixcolumn * totalappend;

  // copy vector f_k to vector y_k => y_k = f_k;
  cblas_dcopy(f_total, &f_k[0], 1, &y_k[0], 1);

  // Gradient = -Linv * MATMUL(TRANSPOSE(matrix), matrix)
  // Here matrix is the kernel.
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, matrixcolumn,
              matrixcolumn, matrixrow, -Linv, &matrix[0], matrixrow, &matrix[0],
              matrixrow, 0.0, &Gradient[0], matrixcolumn);

  // c = Linv * MATMUL(TRANSPOSE(matrix), s)
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, matrixcolumn,
              totalappend, matrixrow, Linv, &matrix[0], matrixrow, &s[0],
              matrixrow, 0.0, &c[0], matrixcolumn);

  previous_minimizer_function = pow(cblas_dnrm2(c_total, &s[0], 1), 2);

  normalizationFactor = previous_minimizer_function;

  for (k = 0; k < maxiter; k++) {

    t_k = t_kp1;

    // copy vector f_k to vector f_km1 => f_km1 = f_k;
    cblas_dcopy(f_total, &f_k[0], 1, &f_km1[0], 1);

    t_kp1 = (1.0 + sqrt(1.0 + 4.0 * t_k * t_k)) * 0.5;
    constantFactor = ((t_k - 1.0) / t_kp1);

    constantFactor_p1 = constantFactor + 1.0;

    sumabs = 0.0;
    residue_temp = 0.0;
    j = 0;

    // !$omp parallel num_threads(npros) private(i, y_k_i, f_k_i, s_temp, c_i, &
    // !$omp        temp_c, temp) &
    // !$omp shared(totalappend, y_k, f_k, s, c, matrixcolumn, Gradient, Linv, &
    // !$omp        nonnegative, hyperparameter, matrixrow, matrix, residue) &
    // !$omp reduction(+:residue_temp, sumabs)
    // !$omp do
    // #pragma omp parallel for
    for (i = 0; i < totalappend; i++) {
      // temp = MATMUL(Gradient, y_k_i)
      // temp = y_k_i + (temp + c)

      // f_k[j] = MATMUL(Gradient, y_k[j]) + 0.0 f_k[j]
      // Because gradient is a symmetric matrix use dsymv
      cblas_dsymv(CblasColMajor, CblasUpper, matrixcolumn, 1.0, &Gradient[0],
                  matrixcolumn, &y_k[j], 1, 0.0, &f_k[j], 1);

      // f_k[j] = c + f_k[j]
      cblas_daxpy(matrixcolumn, 1.0, &c[j], 1, &f_k[j], 1);

      // f_k[j] = y_k[j] + f_k[j]
      cblas_daxpy(matrixcolumn, 1.0, &y_k[j], 1, &f_k[j], 1);

      // Apply l1 prox 'soft thresholding' with non-negative constrains, if
      // requested.
      if (nonnegative == 1) {
        l1_soft_thresholding_with_non_negative_constrains(
            &f_k[j], hyperparameter_linv, matrixcolumn);
      } else {
        l1_soft_thresholding(&f_k[j], hyperparameter_linv, matrixcolumn);
      }

      // residue(k) = (Norm2(MATMUL(matrix, f_k[j]) - s[a])**2.0)
      // temp = MATMUL(matrix, f_k[j]) + 0.0 temp
      cblas_dgemv(CblasColMajor, CblasNoTrans, matrixrow, matrixcolumn, 1.0,
                  &matrix[0], matrixrow, &f_k[j], 1, 0.0, temp, 1);
      // temp = - s + temp
      cblas_daxpy(matrixrow, -1.0, &s[i * matrixrow], 1, temp, 1);
      // residue = Norm2(temp)**2
      residue_temp += pow(cblas_dnrm2(matrixrow, temp, 1), 2);

      // sumabs = SUM(ABS(f_k_i))
      sumabs += cblas_dasum(matrixcolumn, &f_k[j], 1);
      j += matrixcolumn;
    }
    // !$omp end do
    // !$omp end parallel

    residue[k] = residue_temp;
    minimizer_function[k] += residue[k] + hyperparameter * sumabs;
    residue[k] /= normalizationFactor;

    if (k >= 3) {
      temp_res = 0.0;
      for (i = 1; i <= 3; i++) {
        temp_res += residue[k - i];
      }
      if (temp_res / 3.0 - residue[k] < tol)
        break;
    }

    // ! Checking for decency !
    if (minimizer_function[k] > previous_minimizer_function) {
      // copy vector f_km1 to vector f_k => f_k = f_km1;
      cblas_dcopy(f_total, &f_km1[0], 1, &f_k[0], 1);

      sumabs = sumabsm1; // !SUM(ABS(f_km1))
      minimizer_function[k] = minimizer_function[k - 1];
      residue[k] = residue[k - 1];
    }

    previous_minimizer_function = minimizer_function[k];
    sumabsm1 = sumabs;

    // copy vector f_km1 to vector y_k
    // dcopy_(&f_total, &f_km1[0], &incr, &y_k[0], &incr);
    // dscal_(&f_total, &m_constantFactor, &y_k[0], &incr);
    // daxpy_(&matrixcolumn, &constantFactor_p1, &f_k[0], &incr, &y_k[0],
    // &incr);

    j = 0;
    // !$omp parallel private(i, y_k_i, f_k_i) &
    // !$omp shared(totalappend, y_k, f_k, f_km1, matrixcolumn, constantFactor)
    // !$omp do
    for (i = 0; i <= totalappend - 1; i++) {
      // y_k[j] = f_km1[j];
      cblas_dcopy(matrixcolumn, &f_km1[j], 1, &y_k[j], 1);
      // y_k[j] = (constantFactor + 1.0) * _k[j] - constantFactor * y_k[j])
      cblas_daxpby(matrixcolumn, constantFactor_p1, &f_k[j], 1, -constantFactor,
                   &y_k[j], 1);
      j += matrixcolumn;
    }
    // !$omp end do
    // !$omp end parallel

    // !totalcheckFunction(k) = totalcheckFunction(k) + minimizer_function(k)
    // !totalresidue(k) = totalresidue(k) + residue(k)
  }

  // !$omp parallel private(i, f_k_i, temp) &
  // !$omp shared(totalappend, f_k, matrixrow, matrixcolumn, matrix, zf)
  // !$omp do
  for (i = 0; i <= totalappend - 1; i++) {
    // ! zf = MATMUL(matrix, f_k[j]) + 0.0 * zf
    cblas_dgemv(CblasColMajor, CblasNoTrans, matrixrow, matrixcolumn, 1.0,
                &matrix[0], matrixrow, &f_k[i * matrixcolumn], 1, 0.0,
                &zf[i * matrixrow], 1);
  }
  // !$omp end do
  // !$omp end parallel

  free(c);
  free(f_km1);
  free(y_k);
  free(Gradient);
  free(temp);

  end = clock();
  cpu_time_[0] = ((double)(end - start)) / CLOCKS_PER_SEC;
  // call cpu_time(end_cpu_time)
  // end_wall_time = omp_get_wtime()
  // cpu_time_ = end_cpu_time - start_cpu_time
  // wall_time_ = end_wall_time - start_wall_time
  iter[0] = k - 1;
}

// int main() { return 0; }