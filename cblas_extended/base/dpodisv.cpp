#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

/**
 * @brief SPD(A)-Diagonal(B) system solver.
 *
 * @param SIDE 'L' or 'R'.
 * @param UPLO 'U' or 'L'.
 *
 * A is an MxM symmetric matrix.
 * D is an M-array containing the diagonal of B.
 * B is an initially empty MxM matrix.
 *
 * On exit:
 *    A contains the permuted LDLT factorisation.
 *    D is untouched.
 *    B contains the result of the computation.
 */
void dpodisv(const char SIDE, const char UPLO, const int M, double* A,
             const int LDA, const double* D, const int INCD, double* B,
             const int LDB) {
  const bool LHS_LEFT = SIDE == 'L';
  const bool UPPER_PO = UPLO == 'U';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLO != 'U') && (UPLO != 'L'))
    INFO = 2;
  else if (M < 0)
    INFO = 3;
  else if (LDA < std::max<int>(1, M))
    INFO = 5;
  else if (INCD < 0)
    INFO = 7;
  else if (LDB < std::max<int>(1, M))
    INFO = 9;

  if (INFO != 0) {
    // @todo check this is correct.
    // cblas_xerbla(INFO, "dpodisv", "Illegal setting, %d\n");
    return;
  }

  /*
   *  Quick return if possible.
   */
  if (M == 0) return;

  /*
   *  Start the operations.
   */
  LAPACKE_dpotrf(CblasColMajor, UPLO, M, A, LDA);

  if (LHS_LEFT) {
    /*
     *  Form: A * X = B
     */
    if (UPPER_PO) {
      /*
       *  A = U^T * U
       *  X = U^{-1} * U^{-T} * B
       */
      dtrdisv(SIDE, UPLO, 'T', 'N', M, 1.0, A, LDA, D, INCD, B, LDB);

      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                  CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
    } else {
      /*
       *  A = L * L^T
       *  X = L^{-T} * L^{-1} * B.
       */
      dtrdisv(SIDE, UPLO, 'N', 'N', M, 1.0, A, LDA, D, INCD, B, LDB);

      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                  CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
    }
  } else {
    /*
     *  Form: X * A = B.
     */
    if (UPPER_PO) {
      /*
       *  A = U^T * U
       *  X = B * U^{-1} * U^{-T}
       */
      dtrdisv(SIDE, UPLO, 'N', 'N', M, 1.0, A, LDA, D, INCD, B, LDB);

      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                  CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
    } else {
      /*
       *  A = L * L^T
       *  X = B * L^{-T} * L^{-1}.
       */
      dtrdisv(SIDE, UPLO, 'T', 'N', M, 1.0, A, LDA, D, INCD, B, LDB);

      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                  CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
    }
  }
}  // END OF DPODISV