
#include "Fista_lapack_cv.h"

void fista_cv(double * matrix,
            double * s,
            double * matrixTest,
            double * stest,
            int testrow,
            int nfold,
            double * PredictionError,
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
            int * npros) {

int k, i, j, l, endIndex, endIndexTest, j_fold_matrix, j_fold_s, c_total;
int j_fold_matrix_test, j_fold_s_test;
double t_k, sumabs, previous_minimizer_function, t_kp1, residue_temp;
double sumabsm1, constantFactor, normalizationFactor, lambd_linv;
double constantFactor_p1, temp_res, sqrtTemp;

double* temp = createDouble1DArray(matrixrow);
double* c = createDouble1DArray(totalappend * matrixcolumn);
double* f_km1 = createDouble1DArray(matrixcolumn * totalappend);
double* y_k = createDouble1DArray(totalappend * matrixcolumn);

double* temp_test = createDouble1DArray(testrow);

// double precision, dimension(0:matrixcolumn-1, 0:matrixcolumn-1) :: Gradient !&
double* Gradient = createDouble1DArray(matrixcolumn * matrixcolumn);

// double start_wall_time, end_wall_time;
clock_t start, end;

start = clock();

double* residue = createDouble1DArray(maxiter);
double* minimizer_function = createDouble1DArray(maxiter);
int f_total = matrixcolumn*totalappend;
int j_fold_s_sub = 0;

for (int fold=0; fold < nfold; fold++){
    if (fold <= m-1) {
        endIndex = matrixrow-1;
        endIndexTest = testrow;
    }
    else {
        endIndex = matrixrow;
        endIndexTest = testrow-1;
    }

    c_total = endIndex*totalappend;
    j_fold_matrix = fold*(endIndex * matrixcolumn);
    j_fold_s = fold*(c_total);
    j_fold_matrix_test = fold*(endIndexTest * matrixcolumn);
    j_fold_s_test = 0;

    // copy vector f_k to vector f_k
    // y_k = f_k;
    // cblas_dcopy(f_total, &f_k[0], 1, &y_k[0], 1);

    // !Gradient = MATMUL(TRANSPOSE(matrix), matrix)
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, matrixcolumn, \
                matrixcolumn, endIndex, -Linv, &matrix[j_fold_matrix], \
                endIndex, &matrix[j_fold_matrix], endIndex, 0.0, \
                &Gradient[0], matrixcolumn);

    // !c = MATMUL(TRANSPOSE(matrix), s)
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, matrixcolumn, \
                totalappend, endIndex, Linv, &matrix[j_fold_matrix], \
                endIndex, &s[j_fold_s], endIndex, 0.0, &c[0], matrixcolumn);

    previous_minimizer_function = pow(cblas_dnrm2(c_total, &s[j_fold_s], 1),2);

    normalizationFactor = previous_minimizer_function;

    for (l = 0; l < n_lambda; l++){
        sumabsm1 = 0.0;
        t_kp1 = 1.0;
        lambd_linv = lambdaVal[l] * Linv;
        // residue = 0.0;
        cblas_dscal(f_total, 0.0, &f_k[0], 1);
        // f_k = 0.0;
        previous_minimizer_function = normalizationFactor;

        for (k = 0; k<= maxiter-1; k++){

            t_k = t_kp1;

            // copy vector f_k to vector f_km1
            cblas_dcopy(f_total, &f_k[0], 1, &f_km1[0], 1);
            // f_km1 = f_k;

            t_kp1 = (1.0 + sqrt(1.0 + 4.0 * t_k * t_k)) * 0.5;
            constantFactor = ((t_k - 1.0) / t_kp1);

            constantFactor_p1 = constantFactor+1.0;

            sumabs = 0.0;
            residue_temp = 0.0;
            j = 0;

            // !$omp parallel num_threads(npros) private(i, y_k_i, f_k_i, s_temp, c_i, &
            // !$omp        temp_c, temp) &
            // !$omp shared(totalappend, y_k, f_k, s, c, matrixcolumn, Gradient, Linv, &
            // !$omp        nonnegative, lambd, matrixrow, matrix, residue) &
            // !$omp reduction(+:residue_temp, sumabs)
            // !$omp do
            // #pragma omp parallel for
            for(i = 0; i<=totalappend-1; i++){
            // !temp = MATMUL(Gradient, y_k_i)
            // !temp = y_k_i - Linv * (temp - c)

                // ! f_k[j] = MATMUL(Gradient*y_k[j]) + 0.0 f_k[j]
                // ! since gradient is a symmetric matric use dsymv
                cblas_dsymv(CblasColMajor, CblasUpper, matrixcolumn, 1.0, &Gradient[0], \
                            matrixcolumn, &y_k[j], 1, 0.0, &f_k[j], 1);

                // ! f_k[j] = c + f_k[j]
                cblas_daxpy(matrixcolumn, 1.0, &c[j], 1, &f_k[j], 1);

                // ! f_k[j] = y_k[j] + f_k[j]
                cblas_daxpy(matrixcolumn, 1.0, &y_k[j], 1, &f_k[j], 1);

            // ! Applying l1 prox 'soft thresholding'
            // !call l1_softTresholding(f_k_i, temp_c, lambda_i*Linv, matrixcolumn)

                // ! Apply non-negative constrain
                if (nonnegative == 1) {
                    l1_softTresholdingWithNonNegative(&f_k[j], lambd_linv, matrixcolumn);
                }
                else {
                    l1_softTresholding(&f_k[j], lambd_linv, matrixcolumn);
                    // !call nonNegativeConstrain(f_k_i, matrixcolumn)
                }

            // !residue(k) = (Norm2(MATMUL(matrix, f_k_i) - s_temp)**2.0)

                // ! temp = MATMUL(matrix, f_k[j]) + 0.0 temp
                cblas_dgemv(CblasColMajor, CblasNoTrans, endIndex, matrixcolumn, \
                            1.0, &matrix[j_fold_matrix], endIndex, &f_k[j], 1, \
                            0.0, temp, 1);
                // ! temp = - s + temp
                cblas_daxpy(endIndex, -1.0, &s[j_fold_s_sub], 1, temp, 1);
                // ! residue = Norm2(temp)**2
                residue_temp += pow(cblas_dnrm2(endIndex, temp, 1), 2);

                // !sumabs = SUM(ABS(f_k_i))
                sumabs += cblas_dasum(matrixcolumn, &f_k[j], 1);
                j += matrixcolumn;
                j_fold_s_sub += endIndex;
            }
            // !$omp end do
            // !$omp end parallel

            residue[k] = residue_temp;
            minimizer_function[k] += residue[k] + lambdaVal[l] * sumabs;
            residue[k] /= normalizationFactor;

            if (k >= 3) {
                temp_res = 0.0;
                for (i=1; i<=3; i++){
                    temp_res += residue[k-i];
                }
                if (temp_res/3.0 - residue[k] < tol) break;
            }

            // ! Checking for decency !
            if (minimizer_function[k] > previous_minimizer_function) {

                // copy vector f_km1 to vector f_k
                // f_k = f_km1;
                cblas_dcopy(f_total, &f_km1[0], 1, &f_k[0], 1);

                sumabs = sumabsm1; // !SUM(ABS(f_km1))
                minimizer_function[k] = minimizer_function[k-1];
                residue[k] = residue[k-1];
                // !print *, k, 'true'
            }

            previous_minimizer_function = minimizer_function[k];
            sumabsm1 = sumabs;

            // !t_kp1_i = (1.0d+0 + sqrt(1.0d+0 + 4.0d+0 * t_k * t_k)) * 0.5d+0
            // !constantFactor = ((t_k - 1.0d+0) / t_kp1_i)

            // ! y_k_i = f_k_i + constantFactor * (f_k_i - f_km1)
            // ! y_k_i = f_km1


            // copy vector f_km1 to vector y_k
            // dcopy_(&f_total, &f_km1[0], &incr, &y_k[0], &incr);
            // dscal_(&f_total, &m_constantFactor, &y_k[0], &incr);
            // daxpy_(&matrixcolumn, &constantFactor_p1, &f_k[0], &incr, &y_k[0], &incr);

            j = 0;
            // !$omp parallel private(i, y_k_i, f_k_i) &
            // !$omp shared(totalappend, y_k, f_k, f_km1, matrixcolumn, constantFactor)
            // !$omp do
            for (i = 0; i<= totalappend-1; i++){
                // y_k[j] = f_km1[j];
                cblas_dcopy(matrixcolumn, &f_km1[j], 1, &y_k[j], 1);
                // y_k[j] = (constantFactor + 1.0) * _k[j] - constantFactor * y_k[j])
                cblas_daxpby(matrixcolumn, constantFactor_p1, &f_k[j], 1, \
                            -constantFactor, &y_k[j], 1);
                j += matrixcolumn;
            }
            // !$omp end do
            // !$omp end parallel


            // !totalcheckFunction(k) = totalcheckFunction(k) + minimizer_function(k)
            // !totalresidue(k) = totalresidue(k) + residue(k)

        } // end k

        j = 0;
        for (i=0; i<totalappend; i++){
            cblas_dcopy(endIndexTest, &stest[j_fold_s_test], 1, temp_test, 1);
            cblas_dgemv(CblasColMajor, CblasNoTrans, endIndexTest, matrixcolumn, 1.0, \
                        &matrixTest[j_fold_matrix_test], endIndexTest, \
                        &f_k[j], 1, -1.0, temp_test, 1);
            j += matrixcolumn;
            j_fold_s_test += endIndexTest;
        }
    }
}


sqrtTemp = sqrt(real(nfold));
for (i = 0; i<n_lambda; i++){
    CV[i] = sum(&PredictionError[i])/nfold;
    CVstd[i] = sqrt( pow( sum((&PredictionError[i] - &CV[i]) ), 2))/ sqrtTemp;
}

// !enddo
destroyDouble1DArray(c);
destroyDouble1DArray(f_km1);
destroyDouble1DArray(y_k);
destroyDouble1DArray(Gradient);
destroyDouble1DArray(temp);

end = clock();
cpu_time_[0] = ((double) (end - start)) / CLOCKS_PER_SEC;
// call cpu_time(end_cpu_time)
// end_wall_time = omp_get_wtime()
// cpu_time_ = end_cpu_time - start_cpu_time
// wall_time_ = end_wall_time - start_wall_time
iter[0] = k-1;
}


// !!! l1 penalty !!!
void l1_softTresholding(double *B, double c, int n){
// ! A is a vector of size n
// ! c is a scalar
// ! convex function h(A) = c * || A ||_1

// ! prox_h (A)_i = max(abs(A_i) - c, 0)*sign(A_i)
// implicit none

// double precision, intent(in), dimension(0:n-1) :: A
// double precision, intent(in) :: c

// double precision, intent(out), dimension(0:n-1) :: B
// double precision :: aa

// integer*4 , intent(in) :: n
//
    double sign_;


    for (int i=0; i<n; i++){
        sign_ = sign(B[i]);
        B[i] = fabs(B[i]) - c;
        if (B[i] >= 0.){
            B[i] *= sign_;
        }
        else{
            B[i] = 0.;
        }
    }
}

int sign(double x) {
    return (x > 0.0) - (x < 0.0);
}

// !!! l1 penalty !!!
void l1_softTresholdingWithNonNegative(double *B, double c, int n){
// ! A is a vector of size n
// ! c is a scalar
// ! convex function h(A) = c * || A ||_1

// ! prox_h (A)_i = max(abs(A_i) - c, 0)*sign(A_i)
// implicit none

// double precision, intent(in), dimension(0:n-1) :: A
// double precision, intent(in) :: c

// double precision, intent(out), dimension(0:n-1) :: B
// !double precision :: aa

// integer*4 , intent(in) :: n

    for (int i=0; i<n; i++){
        B[i] -= c;
        if (B[i] < 0.0){
            B[i] = 0.0; //!* SIGN(aa, A)
        }
    }
}

// !subroutine nonNegativeConstrain(A, n)
// ! A is a vector of size n
// ! A = max(A, 0)
// !implicit none
// !
// !double precision, intent(inout), dimension(0:n-1) :: A
// !integer*4 , intent(in) :: n
// !WHERE (A > 0.)
// !    A = A
// !ELSEWHERE
// !    A = 0.
// !END WHERE
// !
// !end subroutine nonNegativeConstrain


int main(){
    return 0;
}