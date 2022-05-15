subroutine fista(matrix, s, matrixTest, sTest, maxiter, lambdaVal, CV, CVstd, &
                 matrixrow, matrixcolumn, nonnegative, Linv, totalappend, &
                 iter, cpu_time_, wall_time_, tol, lambda, testrow, npros, &
                 nfold, PredictionError, m)

use omp_lib

    implicit none
    integer*4, intent(in) :: matrixrow, matrixcolumn, lambda, npros, m
!f2py integer*4, intent(in) :: matrixrow, matrixcolumn, lambda, npros, m

    integer*4, intent(out), dimension(0:lambda-1) :: iter
!f2py integer*4, intent(out) :: iter

    double precision, intent(out) :: cpu_time_, wall_time_
!f2py double precision, intent(out) :: cpu_time_, wall_time_

    integer*4, intent(in) :: maxiter, nonnegative, totalappend, testrow, nfold
!f2py integer*4, intent(in) :: maxiter, nonnegative, totalappend, testrow, nfold

    double precision, intent(in) :: Linv, tol
!f2py double precision, intent(in) :: Linv, tol

double precision, intent(out), dimension(0:lambda-1, nfold) :: PredictionError
!f2py double precision, intent(out) :: PredictionError

double precision, intent(out), dimension(0:lambda-1) :: CV, CVstd
!f2py double precision, intent(out) :: CV, CVstd

    double precision, intent(in), dimension(0:lambda-1) :: lambdaVal
!f2py double precision, intent(in) :: lambdaVal

    double precision, intent(in), dimension(0:testrow-1, 0:matrixcolumn-1, nfold) :: matrixTest
!f2py double precision, intent(in) :: matrixTest

    double precision, intent(in), dimension(0:testrow-1, 0:totalappend-1, nfold) :: sTest
!f2py double precision, intent(in) :: sTest

    double precision, intent(in), dimension(0:matrixrow-1, 0:matrixcolumn-1, nfold) :: matrix
!f2py double precision, intent(in) :: matrix

    double precision, intent(in), dimension(0:matrixrow-1, 0:totalappend-1, nfold) :: s
!f2py double precision, intent(in) :: s


    integer*4 :: k, i, j, fold, endIndex, endIndexTest
    !double precision, dimension(0:lambda-1, nfold) :: chi2
    double precision :: t_k, sumabs, t_kp1, previousCheckFunction, chi2_temp, residue_temp
    double precision :: dnrm2, sumabsm1, dasum, constantFactor, normalizationFactor
    double precision, dimension(0:matrixrow-1) :: s_temp, temp
    double precision, dimension(0:testrow-1) :: s_test_i, temp_test
    double precision, dimension(0:matrixcolumn-1) :: f_k_i, y_k_i, temp_c, c_i
    double precision, dimension(0:matrixcolumn-1, 0:totalappend-1) :: c, f_km1,  y_k, f_k
    double precision, dimension(0:matrixcolumn-1, 0:matrixcolumn-1) :: Gradient !&

    double precision, dimension(0:maxiter-1) :: checkFunction, residue
    double precision :: start_cpu_time, end_cpu_time, start_wall_time, end_wall_time, sqrtTemp

    call cpu_time(start_cpu_time)
    start_wall_time = omp_get_wtime()
    call omp_set_num_threads(npros)

PredictionError = 0.0d0
!chi2 = 0.0d0
CV = 0.0d0
iter = 0.0

do fold = 1, nfold

    if (fold <= m) then
        endIndex = matrixrow-2
        endIndexTest = testrow-1
    else
        endIndex = matrixrow-1
        endIndexTest = testrow-2
    endif

    !$omp parallel sections shared(matrixcolumn, matrix, endIndex, &
    !$omp           totalappend, fold, s)
    !$omp section

    !Gradient = MATMUL(TRANSPOSE(matrix), matrix)
    Gradient = 0.0
    call dgemm('T', 'N', matrixcolumn, matrixcolumn, endIndex+1, 1.0d+0, matrix(:endIndex,:, fold),  &
                endIndex+1, matrix(:endIndex,:, fold), endIndex+1, 0.0d+0, Gradient, matrixcolumn)

    !$omp section
    !c = MATMUL(TRANSPOSE(matrix), s)
    c = 0.0
    call dgemm('T', 'N', matrixcolumn, totalappend, endIndex+1, 1.0d+0, matrix(:endIndex,:, fold),  &
                endIndex+1, s(:endIndex,:,fold), endIndex+1, 0.0d+0, c, matrixcolumn)

    !$omp section
    checkFunction = 0.0
    normalizationFactor = 0.0
    do i = 0, totalappend-1
        s_temp = s(:endIndex,i,fold)
        normalizationFactor = normalizationFactor + dnrm2(endIndex+1, s_temp(:endIndex), 1)**2
    enddo
    previousCheckFunction = normalizationFactor
    !$omp end parallel sections

    f_k = 0.0
! !$omp parallel num_threads(npros) shared(matrixcolumn, matrixrow, matrix, &
! !$omp            totalappend, s, lambda, maxiter, Linv, nonnegative, lambdaVal, tol, &
! !$omp            sTest, matrixTest, testrow, fold, Gradient, c, &
! !$omp            normalizationFactor, endIndexTest, endIndex, PredictionError) &
! !$omp private(checkFunction, i, s_temp, previousCheckFunction, j, y_k, sumabsm1, &
! !$omp         sumabs, t_kp1, f_k, residue, k, t_k, f_km1, constantFactor, y_k_i, f_k_i, &
! !$omp         c_i, temp_c, temp, f_km1_i, s_test_i, temp_test, chi2_temp)
! !$omp do
    do j = 0, lambda-1
        y_k = 0.0
        sumabsm1 = 0.0
        sumabs = 0.0
        t_kp1 = 1.0
        f_k = 0.0
        residue = 0.0
        previousCheckFunction = normalizationFactor

        !! Train !!
        do k = 0, maxiter-1
            t_k = t_kp1
            f_km1 = f_k

            t_kp1 = (1.0d+0 + sqrt(1.0d+0 + 4.0d+0 * t_k * t_k)) * 0.5d+0
            constantFactor = ((t_k - 1.0d+0) / t_kp1)

            sumabs = 0.0d0
            residue_temp = 0.0d0

            !$omp parallel num_threads(npros) private(i, y_k_i, f_k_i, s_temp, c_i, &
            !$omp        temp_c, temp) &
            !$omp shared(totalappend, y_k, f_k, s, c, matrixcolumn, Gradient, Linv, &
            !$omp        nonnegative, lambdaVal, matrixrow, matrix, residue, endIndex, fold) &
            !$omp reduction(+:residue_temp, sumabs)
            !$omp do

            do i = 0, totalappend-1
                y_k_i = y_k(:,i)
                f_k_i = f_k(:,i)
                s_temp = s(:endIndex,i, fold)
                c_i = c(:, i)

                !temp = MATMUL(Gradient, y_k_i)
                !temp = y_k_i - Linv * (temp - c)

                !temp_c = 0
                !call dsymm('L', 'U', matrixcolumn, totalappend, 1.0d+0, Gradient, &
                !            matrixcolumn, y_k, matrixcolumn, 0.0d+0, temp_c, matrixcolumn)
                call dscal(matrixcolumn, 0.0d+0, temp_c, 1)
                ! temp_c = (MATMUL(Gradient, y_k_i) + 0
                ! call dgemv('N', matrixcolumn, matrixcolumn, 1.0d+0, Gradient, &
                !                matrixcolumn, y_k_i, 1, 0.0d+0, temp_c, 1)
                ! since gradient is a symmetric matric use dsymv
                call dsymv('U', matrixcolumn, 1.0d+0, Gradient, &
                            matrixcolumn, y_k_i, 1, 0.0d+0, temp_c, 1)

                ! temp_c = Linv*(c - temp_c)
                call dscal(matrixcolumn, -Linv, temp_c, 1)
                call daxpy(matrixcolumn, Linv, c_i, 1, temp_c, 1)

                ! temp_c = y_k_i + temp_c
                call daxpy(matrixcolumn, 1.0d+0, y_k_i, 1, temp_c, 1)



                ! Applying l1 prox 'soft thresholding'
                !call l1_softTresholding(f_k_i, temp_c, lambda_i*Linv, matrixcolumn)

                ! Apply non-negative constrain
                if (nonnegative == 1) then
                    call l1_softTresholdingWithNonNegative(f_k_i, temp_c, &
                                                lambdaVal(j)*Linv, matrixcolumn)
                else
                    call l1_softTresholding(f_k_i, temp_c, lambdaVal(j)*Linv, matrixcolumn)
                    !call nonNegativeConstrain(f_k_i, matrixcolumn)
                endif

                !residue(k) = (Norm2(MATMUL(matrix, f_k_i) - s_temp)**2.0)

                ! temp = 0.0d+0
                call dscal(endIndex+1, 0.0d+0, temp(:endIndex), 1)
                ! temp = MATMUL(matrix, f_k_i)
                call dgemv('N', endIndex+1, matrixcolumn, 1.0d+0,&
                            matrix(:endIndex,:, fold), endIndex+1, &
                            f_k_i, 1, 0.0d+0, temp(:endIndex), 1)
                ! temp = - s_temp + temp
                call daxpy(endIndex+1, -1.0d+0, s_temp(:endIndex), &
                            1, temp(:endIndex), 1)
                ! residue = Norm2(temp)**2
                residue_temp = residue_temp + dnrm2(endIndex+1, temp(:endIndex), 1)**2

                !sumabs = SUM(ABS(f_k_i))
                sumabs = sumabs + dasum(matrixcolumn, f_k_i, 1)

                ! if (i < totalappend-1) then
                !     f_k(:,i+1) = f_k_i
                ! endif
                ! y_k(:,i) = y_k_i
                f_k(:,i) = f_k_i
            enddo
            !$omp end do
            !$omp end parallel

            residue(k) = residue_temp
            checkFunction(k) = residue(k) + lambdaVal(j) * sumabs
            residue(k) = residue(k)/normalizationFactor !(1.0d0 * totalappend * (endIndex+1))

            if (k .ge. 5) then
                if (sum(residue(k-5:k-1))/5.0 - residue(k) < tol) exit

                !if (dnrm2(matrixcolumn, (f_k_i - f_km1_i), 1)**2/ &
                !        dnrm2(matrixcolumn, f_k_i, 1)**2 < tol) exit
            endif

            ! Checking for decency !
            if (checkFunction(k) > previousCheckFunction) then
                f_k = f_km1
                sumabs = sumabsm1 !SUM(ABS(f_km1))
                checkFunction(k) = checkFunction(k-1)
                residue(k) = residue(k-1)
                !print *, k, 'true'
            endif

            previousCheckFunction = checkFunction(k)
            sumabsm1 = sumabs

            !t_kp1_i = (1.0d+0 + sqrt(1.0d+0 + 4.0d+0 * t_k * t_k)) * 0.5d+0
            !constantFactor = ((t_k - 1.0d+0) / t_kp1_i)

            ! y_k_i = f_k_i + constantFactor * (f_k_i - f_km1)
            ! y_k_i = f_km1

            !$omp parallel private(i, y_k_i, f_k_i) &
            !$omp shared(totalappend, y_k, f_k, f_km1, matrixcolumn, constantFactor)
            !$omp do
            do i = 0, totalappend-1
                ! y_k_i = y_k(:,i)
                f_k_i = f_k(:,i)
                y_k_i = f_km1(:,i)
                ! call scopy(matrixcolumn, f_km1_i, 1, y_k_i, 1)
                ! y_k_i = (constantFactor + 1.0d+0) * f_k_i - constantFactor * f_km1)
                call dscal(matrixcolumn, -constantFactor, y_k_i, 1)
                call daxpy(matrixcolumn, constantFactor+1.0d+0, f_k_i, 1, &
                           y_k_i, 1)
                y_k(:,i) = y_k_i
                !f_k(:,i) = f_k_i
            enddo
            !$omp end do
            !$omp end parallel

        enddo ! k


        !! Test
        chi2_temp = 0.0d0

        !$omp parallel private(i, f_k_i, s_test_i, temp_test) &
        !$omp shared(totalappend, f_k, sTest, endIndexTest, fold, &
        !$omp       matrixcolumn, matrixTest) &
        !$omp reduction(+:chi2_temp)
        !$omp do
        do i = 0, totalappend-1
            f_k_i = f_k(:,i)
            s_test_i = sTest(:endIndexTest,i, fold)
            call dscal(endIndexTest+1, 0.0d+0, temp_test(:endIndexTest), 1)
            ! temp = MATMUL(matrix, f_k_i)
            call dgemv('N', endIndexTest+1, matrixcolumn, 1.0d+0, &
                        matrixTest(:endIndexTest,:,fold), endIndexTest+1, &
                        f_k_i, 1, 0.0d+0, temp_test(:endIndexTest), 1)
            ! temp_test = - s_test_i + temp_test
            call daxpy(endIndexTest+1, -1.0d+0, s_test_i(:endIndexTest), &
                        1, temp_test(:endIndexTest), 1)
            chi2_temp = chi2_temp + dnrm2(endIndexTest+1, temp_test(:endIndexTest), 1)**2
        enddo
        !$omp end do
        !$omp end parallel

        PredictionError(j, fold) = chi2_temp/(endIndexTest+1)
        !chi2(j, fold) = chi2_temp/(endIndexTest+1)
        !CV(j) = CV(j) + chi2_temp/(endIndexTest+endIndex+2)
        !iter(j) = k-1
    enddo
!!$omp end do
!!$omp end parallel
enddo  ! fold



sqrtTemp = sqrt(real(nfold))
!$omp parallel shared(lambda, CV, PredictionError, sqrtTemp, nfold, CVstd) &
!$omp  private(i)
!$omp do
do i = 0, lambda-1
    CV(i) = sum(PredictionError(i, :))/nfold
    CVstd(i) = sqrt(sum((PredictionError(i,:) - CV(i))**2))/sqrtTemp
enddo
!$omp enddo
!$omp end parallel

call cpu_time(end_cpu_time)
end_wall_time = omp_get_wtime()
cpu_time_ = end_cpu_time - start_cpu_time
wall_time_ = end_wall_time - start_wall_time

end subroutine fista



!!! l1 penalty !!!
subroutine l1_softTresholding(B, A, c, n)
! A is a vector of size n
! c is a scalar
! convex function h(A) = c * || A ||_1

! prox_h (A)_i = max(abs(A_i) - c, 0)*sign(A_i)
implicit none

double precision, intent(in), dimension(0:n-1) :: A
double precision, intent(in) :: c

double precision, intent(out), dimension(0:n-1) :: B
double precision :: aa

integer*4 , intent(in) :: n

aa = 1.0

B = ABS(A) - c
WHERE (B >= 0.)
    B = B * SIGN(aa, A)
ELSEWHERE
    B = 0.
END WHERE

end subroutine l1_softTresholding


!!! l1 penalty !!!
subroutine l1_softTresholdingWithNonNegative(B, A, c, n)
! A is a vector of size n
! c is a scalar
! convex function h(A) = c * || A ||_1

! prox_h (A)_i = max(abs(A_i) - c, 0)*sign(A_i)
implicit none

double precision, intent(in), dimension(0:n-1) :: A
double precision, intent(in) :: c

double precision, intent(out), dimension(0:n-1) :: B
!double precision :: aa

integer*4 , intent(in) :: n

B = A - c
WHERE (B >= 0.0)
B = B !* SIGN(aa, A)
ELSEWHERE
B = 0.
END WHERE

end subroutine l1_softTresholdingWithNonNegative


!subroutine nonNegativeConstrain(A, n)
! A is a vector of size n
! A = max(A, 0)
!implicit none
!
!double precision, intent(inout), dimension(0:n-1) :: A
!integer*4 , intent(in) :: n
!WHERE (A > 0.)
!    A = A
!ELSEWHERE
!    A = 0.
!END WHERE
!
!end subroutine nonNegativeConstrain
