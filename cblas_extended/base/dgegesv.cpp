#include <openblas/cblas.h>
#include <openblas/lapacke.h>

#include <cmath>

#include "extended_base.hpp"

/**
 * if SIDE = 'L':  Form is X = inv(A) * B
 * A is M x M -- B is M x N
 *
 * Otherwise: Form is X = B * inv(A)
 * B is M x N -- A is N x N
 */
void dgegesv(const char SIDE, const char TRANSA, const int M, const int N,
             double* A, const int LDA, double* B, const int LDB) {
  bool LHS_LEFT = SIDE == 'L';
  bool NOTA = TRANSA == 'N';

  int NROWA;

  if (LHS_LEFT)
    NROWA = M;
  else
    NROWA = N;

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((TRANSA != 'N') && (TRANSA != 'C') && (TRANSA != 'T'))
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
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dgegesv", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0 || N == 0) return;

  /*
   * Start the operations.
   */
  int* XPIV = (int*)malloc(NROWA * sizeof(int));
  LAPACKE_dgetrf(CblasColMajor, NROWA, NROWA, A, LDA, XPIV);

  if (LHS_LEFT) {
    /*
     * Form X = op(A^-1) * B
     */
    if (NOTA) {
      /*
       *  A = P^T * L * U
       *  X = U^{-1} * L^{-1} * P * B
       */

      // Compute B := P * B.
      LAPACKE_dlaswp(CblasColMajor, N, B, LDB, 1, M, XPIV, 1);

      // Compute B := L^{-1} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                  NROWA, N, 1.0, A, LDA, B, LDB);

      // Compute B := U^{-1} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                  CblasNonUnit, NROWA, N, 1.0, A, LDA, B, LDB);
    } else {
      /*
       *  A = P^T * L * U
       *  X = P^T * L^{-T} * U^{-T} * B
       */

      // Compute B := U^{-T} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                  CblasNonUnit, NROWA, N, 1.0, A, LDA, B, LDB);

      // Compute B := L^{-T} * B.
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                  NROWA, N, 1.0, A, LDA, B, LDB);

      // Compute B := P^T * B.
      LAPACKE_dlaswp(CblasColMajor, N, B, LDB, 1, M, XPIV, -1);
    }
  } else {
    /*
     * Form X = B * op(A^-1)
     */
    if (NOTA) {
      /*
       *  A = P^T * L * U
       *  X = B * U^{-1} * L^{-1} * P
       */

      // Compute B := B * U^{-1}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * L^{-1}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                  CblasUnit, M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * P.
      dlaswp2(M, B, LDB, 0, N - 1, XPIV, -1);
    } else {
      /*
       *  A = P^T * L * U
       *  X = B * P * L^{-T} * U^{-T}
       */

      // Compute B := B * P^T.
      dlaswp2(M, B, LDB, 0, N - 1, XPIV, 1);

      // Compute B := B * L^{-T}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                  M, N, 1.0, A, LDA, B, LDB);

      // Compute B := B * U^{-T}.
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                  CblasNonUnit, M, N, 1.0, A, LDA, B, LDB);
    }
  }
  free(XPIV);
}  // END OF DGEGESV