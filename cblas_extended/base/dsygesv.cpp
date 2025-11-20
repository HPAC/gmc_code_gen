#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

/**
 * @brief Symmetric(A)-General(B) system solver.
 *
 * @param SIDE 'L' or 'R'.
 * @param UPLO 'U' or 'L'.
 *
 * If SIDE == 'L':
 *    A is an MxM symmetric matrix.
 * If SIDE == 'R':
 *    A is an NxN matrix.
 * B is a general MxN matrix in every case.
 *
 * On exit:
 *    A contains the permuted LDLT factorisation.
 *    B contains the result of the computation.
 */
void dsygesv(const char SIDE, const char UPLO, const int M, const int N,
             double* A, const int LDA, double* B, const int LDB) {
  const bool LHS_LEFT = SIDE == 'L';
  const bool UPPER_SYM = UPLO == 'U';
  const int NROWA = (SIDE == 'L') ? M : N;

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLO != 'U') && (UPLO != 'L'))
    INFO = 2;
  else if (M < 0)
    INFO = 3;
  else if (N < 0)
    INFO = 4;
  else if (LDA < std::max<int>(1, NROWA))
    INFO = 6;
  else if (LDB < std::max<int>(1, M))
    INFO = 8;

  if (INFO != 0) {
    // @todo check this is correct.
    // cblas_xerbla(INFO, "dsygesv", "Illegal setting, %d\n");
    return;
  }

  /*
   *  Quick return if possible.
   */
  if (M == 0 || N == 0) return;

  /*
   *  Start the operations.
   */
  int* XPIV = (int*)malloc(NROWA * sizeof(int));
  LAPACKE_dsytrf(CblasColMajor, UPLO, NROWA, A, LDA, XPIV);

  double* E = (double*)malloc(NROWA * sizeof(double));
  LAPACKE_dsyconv(CblasColMajor, UPLO, 'C', NROWA, A, LDA, XPIV, E);

  if (LHS_LEFT) {
    /*
     *  Form: A * X = B
     */
    if (UPPER_SYM) {
      /*
       *  A = P * U * D * U^T * P^T
       *  X = P * U^{-T} * D^{-1} * U^{-1} * P^T * B
       */

      // Compute B := P^T * B (row exchanges).
      dlaswp3('U', M, N, B, LDB, XPIV, -1);

      // Compute B := U^{-1} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasUnit,
                  M, N, 1.0, A, LDA, B, LDB);

      // Compute B := D^{-1} * B.
      dbdigesv('L', UPLO, M, N, A, LDA, E, 1, XPIV, 1, B, LDB);

      // Compute B := U^{-T} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasUnit,
                  M, N, 1.0, A, LDA, B, LDB);

      // Compute B := P * B (row exchanges).
      dlaswp3('U', M, N, B, LDB, XPIV, 1);
    } else {
      /*
       *  A = P^T * L * D * L^T * P
       *  X = P^T * L^{-T} * D^{-1} * L^{-1} * P * B
       */

      // Compute B := P * B (row exchanges).
      dlaswp3('L', M, N, B, LDB, XPIV, 1);

      // Compute B := L^{-1} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                  M, N, 1.0, A, LDA, B, LDB);

      // Compute B := D^{-1} * B.
      dbdigesv('L', UPLO, M, N, A, LDA, E, 1, XPIV, 1, B, LDB);

      // Compute B := L^{-T} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                  M, N, 1.0, A, LDA, B, LDB);

      // Compute B:= P^T * B (row exchanges).
      dlaswp3('L', M, N, B, LDB, XPIV, -1);
    }
  } else {
    /*
     *  Form: X * A = B.
     */
    if (UPPER_SYM) {
      /*
       *  A = P * U * D * U^T * P^T
       *  X = B * P * U^{-T} * D^{-1} * U^{-1} * P^T
       */

      // Compute B := B * P (column exchanges).
      dlaswp4('U', M, N, B, LDB, XPIV, -1);

      // Compute B := B * U^{-T}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasUnit,
                  M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * D^{-1}.
      dbdigesv('R', UPLO, M, N, A, LDA, E, 1, XPIV, 1, B, LDB);

      // Compute B := B * U^{-1}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                  CblasUnit, M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * P^T (column exchanges).
      dlaswp4('U', M, N, B, LDB, XPIV, 1);

    } else {
      /*
       *  A = P^T * L * D * L^T * P
       *  X = B * P^T * L^{-T} * D^{-1} * L^{-1} * P
       */

      // Compute B := B * P^T (column exchanges).
      dlaswp4('L', M, N, B, LDB, XPIV, 1);  // col-exchange needs reverse incx

      // Compute B := B * L^{-T}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                  M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * D^{-1}.
      dbdigesv('R', UPLO, M, N, A, LDA, E, 1, XPIV, 1, B, LDB);

      // Compute B := B * L^{-1}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                  CblasUnit, M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * P (column exchanges).
      dlaswp4('L', M, N, B, LDB, XPIV, -1);  // col-exchange needs reverse incx
    }
  }

  // Perform the permutations backwards and restore the sup/sub-diagonal.
  LAPACKE_dsyconv(CblasColMajor, UPLO, 'R', NROWA, A, LDA, XPIV, E);

  free(XPIV);
  free(E);
}