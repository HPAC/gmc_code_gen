#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

/**
 * @brief Positive Definite(A)-General(B) system solver.
 *
 * @param SIDE 'L' or 'R'.
 * @param UPLO 'U' or 'L'.
 *
 * If SIDE == 'L':
 *    A is an MxM positive-definite matrix.
 * If SIDE == 'R':
 *    A is an NxN matrix.
 * B is a general MxN matrix in every case.
 *
 * On exit:
 *    A contains the Cholesky factorisation.
 *    B contains the result of the computation.
 */
void dpogesv(const char SIDE, const char UPLO, const int M, const int N,
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
    // cblas_xerbla(INFO, "dpogesv", "Illegal setting, %d\n");
    return;
  }

  /*
   *  Quick return if possible.
   */
  if (M == 0 || N == 0) return;

  /*
   *  Start the operations.
   */
  LAPACKE_dpotrf(CblasColMajor, UPLO, NROWA, A, LDA);

  if (LHS_LEFT) {
    /*
     *  Form: A * X = B
     */
    if (UPPER_SYM) {
      /*
       *  A = U^T * U
       *  X = U^{-1} * U^{-T} * B
       */

      // Compute B := U^{-T} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);

      // Compute B := U^{-1} * B
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);
    } else {
      /*
       *  A = L * L^T
       *  X = L^{-T} * L^{-1} * B.
       */

      // Compute B := L^{-1} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);

      // Compute B := L^{-T} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);
    }
  } else {
    /*
     *  Form: X * A = B.
     */
    if (UPPER_SYM) {
      /*
       *  A = U^T * U
       *  X = B * U^{-1} * U^{-T}
       */

      // Compute B := B * U^{-1}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * U^{-T}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);
    } else {
      /*
       *  A = L * L^T
       *  X = B * L^{-T} * L^{-1}.
       */

      // Compute B := B * L^{-T}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * L^{-1}
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);
    }
  }
}  // END OF DPOGESV