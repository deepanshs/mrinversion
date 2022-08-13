

subroutine fista(matrix, s, zf, lambd, maxiter, f_k, &
                checkFunction, residue, matrixrow, matrixcolumn, &
                nonnegative, Linv, totalappend, iter, cpu_time_, wall_time_, tol, npros)

use omp_lib

    implicit none
    integer*4, intent(in) :: matrixrow, matrixcolumn, npros
!f2py integer*4, intent(in) :: matrixrow, matrixcolumn, npros

    integer*4, intent(out) :: iter
!f2py integer*4, intent(out) :: iter

    double precision, intent(out) :: cpu_time_, wall_time_
!f2py double precision, intent(out) :: cpu_time_, wall_time_

    integer*4, intent(in) :: maxiter, nonnegative, totalappend
!f2py integer*4, intent(in) :: maxiter, nonnegative, totalappend

    double precision, intent(in) :: Linv, lambd, tol
!f2py double precision, intent(in) :: Linv, lambd, tol

    double precision, intent(in), dimension(0:matrixrow-1, 0:matrixcolumn-1) :: matrix
!f2py double precision, intent(in) :: matrix

    double precision, intent(in), dimension(0:matrixrow-1, 0:totalappend-1) :: s
!f2py double precision, intent(in) :: s

    double precision, intent(out), dimension(0:matrixrow-1, 0:totalappend-1) :: zf
!f2py double precision, intent(out) :: zf

    double precision, intent(inout), dimension(0:matrixcolumn-1, 0:totalappend-1) :: f_k
!f2py double precision, intent(inout) :: f_k

    double precision, intent(out), dimension(0:maxiter-1) :: checkFunction, residue
!f2py double precision, intent(out) :: checkFunction, residue


    integer*4 :: k, i
    double precision :: t_k, sumabs, previousCheckFunction, t_kp1, residue_temp
    double precision :: dnrm2, sumabsm1, dasum, constantFactor, normalizationFactor, temp_val
    double precision, dimension(0:matrixrow-1) :: s_temp, temp
    double precision, dimension(0:matrixcolumn-1) :: f_k_i, y_k_i, temp_c, c_i
    double precision, dimension(0:matrixcolumn-1, 0:totalappend-1) :: c, f_km1, y_k
    double precision, dimension(0:matrixcolumn-1, 0:matrixcolumn-1) :: Gradient !&

    double precision :: start_cpu_time, end_cpu_time, start_wall_time, end_wall_time

    call cpu_time(start_cpu_time)

    start_wall_time = omp_get_wtime()
    call omp_set_num_threads(npros)

    !inc = 1
    checkFunction = 0.0
    zf = 0.0
    residue = 0.0
    sumabs = 0.0
    Gradient = 0.0
    c = 0.0
    sumabsm1 = 0.0
    previousCheckFunction = 0.0
    t_kp1 = 1.0

    y_k = f_k
    !Gradient = MATMUL(TRANSPOSE(matrix), matrix)
    call dgemm('T', 'N', matrixcolumn, matrixcolumn, matrixrow, 1.0d+0,  matrix,  &
                matrixrow, matrix, matrixrow, 0.0d+0, Gradient, matrixcolumn)

    !c = MATMUL(TRANSPOSE(matrix), s)
    call dgemm('T', 'N', matrixcolumn, totalappend, matrixrow, 1.0d+0,  matrix,  &
                matrixrow, s, matrixrow, 0.0d+0, c, matrixcolumn)

    do i = 0, totalappend-1
        s_temp = s(:,i)
        previousCheckFunction = previousCheckFunction + dnrm2(matrixrow, s_temp, 1)**2
    enddo
    normalizationFactor = previousCheckFunction

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
        !$omp        nonnegative, lambd, matrixrow, matrix, residue) &
        !$omp reduction(+:residue_temp, sumabs)
        !$omp do

        do i = 0, totalappend-1
            y_k_i = y_k(:,i)
            f_k_i = f_k(:,i)
            s_temp = s(:,i)
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
                                            lambd*Linv, matrixcolumn)
            else
                call l1_softTresholding(f_k_i, temp_c, lambd*Linv, matrixcolumn)
                !call nonNegativeConstrain(f_k_i, matrixcolumn)
            endif

            !residue(k) = (Norm2(MATMUL(matrix, f_k_i) - s_temp)**2.0)

            ! temp = 0.0d+0
            call dscal(matrixrow, 0.0d+0, temp, 1)
            ! temp = MATMUL(matrix, f_k_i)
            call dgemv('N', matrixrow, matrixcolumn, 1.0d+0, matrix, matrixrow, &
                        f_k_i, 1, 0.0d+0, temp, 1)
            ! temp = - s_temp + temp
            call daxpy(matrixrow, -1.0d+0, s_temp, 1, temp, 1)
            ! residue = Norm2(temp)**2
            residue_temp = residue_temp + dnrm2(matrixrow, temp, 1)**2

            !sumabs = SUM(ABS(f_k_i))
            sumabs = sumabs + dasum(matrixcolumn, f_k_i, 1)

            ! y_k(:,i) = y_k_i
            f_k(:,i) = f_k_i

            ! if (i < totalappend-1) then
            !     f_k(:,i+1) = f_k_i
            !     ! y_k(:,i+1) = y_k_i
            ! endif
        enddo
        !$omp end do
        !$omp end parallel

        residue(k) = residue_temp
        checkFunction(k) = residue(k) + lambd * sumabs
        residue(k) = residue(k)/normalizationFactor

        if (k .ge. 5) then
            temp_val = 1.0 - ((sum(residue(k-5:k-1))/5.0) / residue(k))
            if (abs(temp_val) <= tol) exit
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


        !totalcheckFunction(k) = totalcheckFunction(k) + checkFunction(k)
        !totalresidue(k) = totalresidue(k) + residue(k)

    enddo

    !$omp parallel private(i, f_k_i, temp) &
    !$omp shared(totalappend, f_k, matrixrow, matrixcolumn, matrix, zf)
    !$omp do
    do i = 0, totalappend-1
        f_k_i = f_k(:,i)
        ! temp = 0.0d+0
        call dscal(matrixrow, 0.0d+0, temp, 1)
        ! temp = MATMUL(matrix, f_k_i)
        call dgemv('N', matrixrow, matrixcolumn, 1.0d+0, matrix, matrixrow, &
                        f_k_i, 1, 0.0d+0, temp, 1)
        zf(:,i) = temp
    enddo
    !$omp end do
    !$omp end parallel


!enddo
    call cpu_time(end_cpu_time)
    end_wall_time = omp_get_wtime()
    cpu_time_ = end_cpu_time - start_cpu_time
    wall_time_ = end_wall_time - start_wall_time
    iter = k-1
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
