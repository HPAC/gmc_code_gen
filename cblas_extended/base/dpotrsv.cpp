#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

/**
 * @brief Positive Definite(A)-Triangular(B) system solver.
 *
 * @param SIDE 'L' or 'R'.
 * @param UPLOA 'U' or 'L'.
 * @param UPLOB 'U' or 'L'.
 * @param DIAGB 'U' or 'N'.
 *
 * A is an MxM positive-definite matrix.
 * B is a triangular MxM matrix.
 *
 * On exit:
 *    A contains the Cholesky factorisation.
 *    B contains the result of the computation. Both the upper and lower parts
 *      of B are referenced and overwritten.
 */
void dpotrsv(const char SIDE, const char UPLOA, const char UPLOB,
             const char DIAGB, const int M, double* A, const int LDA, double* B,
             const int LDB) {
  const bool LHS_LEFT = SIDE == 'L';
  const bool UPPER_PO = UPLOA == 'U';
  const bool UPPER_TR = UPLOB == 'U';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLOA != 'U') && (UPLOA != 'L'))
    INFO = 2;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 3;
  else if ((DIAGB != 'U') && (DIAGB != 'N'))
    INFO = 4;
  else if (M < 0)
    INFO = 5;
  else if (LDA < std::max<int>(1, M))
    INFO = 7;
  else if (LDB < std::max<int>(1, M))
    INFO = 9;

  if (INFO != 0) {
    // @todo check this is correct.
    // cblas_xerbla(INFO, "dpotrsv", "Illegal setting, %d\n");
    return;
  }

  /*
   *  Quick return if possible.
   */
  if (M == 0) return;

  /*
   *  Start the operations.
   */

  LAPACKE_dpotrf(CblasColMajor, UPLOA, M, A, LDA);

  if (LHS_LEFT) {
    /*
     *  Form: A * X = B
     */
    if (UPPER_PO) {
      /*
       *  A = U^T * U
       *  X = U^{-1} * U^{-T} * B
       */
      if (UPPER_TR) {
        // B is upper. Cost computation: 7/3 n**3.
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);

        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
      } else {
        // B is lower. Cost computation: 5/3 n**3.
        dtrtrsv('L', 'U', 'L', 'T', 'N', DIAGB, M, 1.0, A, LDA, B, LDB);

        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
      }
    } else {
      /*
       *  A = L * L^T
       *  X = L^{-T} * L^{-1} * B.
       */
      if (UPPER_TR) {
        // B is upper. Cost computation: 7/3 n**3.
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);

        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
      } else {
        // B is lower. Cost computation: 5/3 n**3.
        dtrtrsv('L', 'L', 'L', 'N', 'N', DIAGB, M, 1.0, A, LDA, B, LDB);

        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
      }
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
      if (UPPER_TR) {
        // B is upper. Cost computation: 5/3 n**3.
        dtrtrsv('R', 'U', 'U', 'N', 'N', DIAGB, M, 1.0, A, LDA, B, LDB);

        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
      } else {
        // B is lower. Cost computation: 7/3 n**3.
        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
      }
    } else {
      /*
       *  A = L * L^T
       *  X = B * L^{-T} * L^{-1}.
       */
      if (UPPER_TR) {
        // B is upper. Cost computation: 5/3 n**3.
        dtrtrsv('R', 'L', 'U', 'T', 'N', DIAGB, M, 1.0, A, LDA, B, LDB);

        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
      } else {
        // B is lower. Cost computation: 7/3 n**3.
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);

        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                    CblasNonUnit, M, M, 1.0, A, LDA, B, LDB);
      }
    }
  }
}  // END OF DPOTRSV