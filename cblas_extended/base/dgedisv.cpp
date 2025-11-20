#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

void dgedisv(const char SIDE, const char TRANSA, const int M, double* A,
             const int LDA, const double* D, const int INCD, double* B,
             const int LDB) {
  bool LHS_LEFT = SIDE == 'L';
  bool NOTA = TRANSA == 'N';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((TRANSA != 'N') && (TRANSA != 'C') && (TRANSA != 'T'))
    INFO = 2;
  else if (M < 0)
    INFO = 3;
  else if (LDA < std::max<int>(1, M))
    INFO = 5;
  else if (LDB < std::max<int>(1, M))
    INFO = 9;

  if (INFO != 0) {
    // @todo check if this is correct.
    // cblas_xerbla(INFO, "dgedisv", "Illegal setting, %d\n");
    return;
  }

  /*
   *  Quick return if possible.
   */
  if (M == 0) return;

  /*
   *  Start the operations.
   */
  int* XPIV = (int*)malloc(M * sizeof(int));

  if (LHS_LEFT) {
    /*
     *  Form X = op(A^{-1}) * B
     */
    if (NOTA) {
      // A is not transposed.
      double* A_TR = (double*)malloc(M * M * sizeof(double));
      dge_trans(M, M, A, LDA, A_TR, M);
      LAPACKE_dgetrf(CblasColMajor, M, M, A_TR, M, XPIV);
      dtrdisv('L', 'U', 'T', 'N', M, 1.0, A_TR, M, D, INCD, B, LDB);
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                  M, M, 1.0, A_TR, M, B, LDB);
      LAPACKE_dlaswp(CblasColMajor, M, B, LDB, 1, M, XPIV, -1);
      free(A_TR);
    } else {
      // A is transposed.
      LAPACKE_dgetrf(CblasColMajor, M, M, A, LDA, XPIV);
      dtrdisv('L', 'U', 'T', 'N', M, 1.0, A, LDA, D, INCD, B, LDB);
      cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                  M, M, 1.0, A, LDA, B, LDB);
      LAPACKE_dlaswp(CblasColMajor, M, B, LDB, 1, M, XPIV, -1);
    }
  } else {
    /*
     *  Form X = B * op(A^{-1})
     */
    if (NOTA) {
      // A is not transposed.
      LAPACKE_dgetrf(CblasColMajor, M, M, A, LDA, XPIV);
      dtrdisv('R', 'U', 'N', 'N', M, 1.0, A, LDA, D, INCD, B, LDB);
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                  CblasUnit, M, M, 1.0, A, LDA, B, LDB);
      dlaswp2(M, B, LDB, 0, M - 1, XPIV, -1);
    } else {
      // A is transposed.
      double* A_TR = (double*)malloc(M * M * sizeof(double));
      dge_trans(M, M, A, LDA, A_TR, M);
      LAPACKE_dgetrf(CblasColMajor, M, M, A_TR, M, XPIV);
      dtrdisv('R', 'U', 'N', 'N', M, 1.0, A_TR, M, D, INCD, B, LDB);
      cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                  CblasUnit, M, M, 1.0, A_TR, M, B, LDB);
      dlaswp2(M, B, LDB, 0, M - 1, XPIV, -1);
      free(A_TR);
    }
  }
  free(XPIV);
}