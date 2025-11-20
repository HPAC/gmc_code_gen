#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

void GETRSV_NU_L(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* IPIV);

void GETRSV_NL_L(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* IPIV);

void GETRSV_TU_L(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* IPIV);

void GETRSV_TL_L(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* IPIV);

void GETRSV_NU_R(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* JPIV);

void GETRSV_NL_R(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* JPIV);

void GETRSV_TU_R(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* JPIV);

void GETRSV_TL_R(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* JPIV);

/**
 * @brief Dense(A)-Triangular(B) solver.
 *
 * Compute op(A) * X = B    or    X * op(A) = B
 *
 * A is not guaranteed to remain unmodified. B will hold the result. B needs to
 * be a triangular matrix in memory: if the matrix is upper-triangular, the
 * lower part must contain zeros and viceversa. This can be circumvented adding
 * a third operand, C, that will hold the result and performing a copy of B onto
 * C.
 */
void dgetrsv(const char SIDE, const char UPLOB, const char TRANSA,
             const char DIAGB, const int M, double* A, const int LDA, double* B,
             const int LDB) {
  bool LHS_LEFT = SIDE == 'L';
  bool NOTA = TRANSA == 'N';
  bool UPPER_B = UPLOB == 'U';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 2;
  else if ((TRANSA != 'N') && (TRANSA != 'C') && (TRANSA != 'T'))
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
    // @todo check if this is correct.
    // cblas_xerbla(INFO, "dgetrsv", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  /*
   * Start the operations.
   */
  int* XPIV = (int*)malloc(M * sizeof(int));

  if (LHS_LEFT) {
    /*
     *  Form X = op(A^{-1}) * B
     */
    if (NOTA) {
      // A is not transposed.
      if (UPPER_B) {
        // B is upper.
        GETRSV_NU_L(DIAGB, M, A, LDA, B, LDB, XPIV);
      } else {
        // B is lower.
        GETRSV_NL_L(DIAGB, M, A, LDA, B, LDB, XPIV);
      }
    } else {
      // A is transposed.
      if (UPPER_B) {
        // B is upper.
        GETRSV_TU_L(DIAGB, M, A, LDA, B, LDB, XPIV);
      } else {
        // B is lower.
        GETRSV_TL_L(DIAGB, M, A, LDA, B, LDB, XPIV);
      }
    }
  } else {
    /*
     *  Form X = B * op(A^{-1})
     */
    if (NOTA) {
      // A is not transposed.
      if (UPPER_B) {
        // B is upper.
        GETRSV_NU_R(DIAGB, M, A, LDA, B, LDB, XPIV);
      } else {
        // B is lower.
        GETRSV_NL_R(DIAGB, M, A, LDA, B, LDB, XPIV);
      }
    } else {
      // A is transposed.
      if (UPPER_B) {
        // B is upper.
        GETRSV_TU_R(DIAGB, M, A, LDA, B, LDB, XPIV);
      } else {
        // B is lower.
        GETRSV_TL_R(DIAGB, M, A, LDA, B, LDB, XPIV);
      }
    }
  }

  free(XPIV);
}  // END OF DGETRSV

void GETRSV_NU_L(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* IPIV) {
  LAPACKE_dgetrf(CblasColMajor, M, M, A, LDA, IPIV);

  LAPACKE_dlaswp(CblasColMajor, M, B, LDB, 1, M, IPIV, 1);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, M,
              M, 1.0, A, LDA, B, LDB);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
              M, M, 1.0, A, LDA, B, LDB);
}

void GETRSV_NL_L(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* IPIV) {
  double* A_TR = (double*)malloc(M * M * sizeof(double));
  const int LDA_TR = M;
  dge_trans(M, M, A, LDA, A_TR, LDA_TR);
  LAPACKE_dgetrf(CblasColMajor, M, M, A_TR, LDA_TR, IPIV);

  dtrtrsv('L', 'U', 'L', 'T', 'N', DIAGB, M, 1.0, A_TR, LDA_TR, B, LDB);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit, M, M,
              1.0, A_TR, LDA_TR, B, LDB);
  LAPACKE_dlaswp(CblasColMajor, M, B, LDB, 1, M, IPIV, -1);

  free(A_TR);
}

void GETRSV_TU_L(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* IPIV) {
  LAPACKE_dgetrf(CblasColMajor, M, M, A, LDA, IPIV);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, M,
              M, 1.0, A, LDA, B, LDB);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit, M, M,
              1.0, A, LDA, B, LDB);
  LAPACKE_dlaswp(CblasColMajor, M, B, LDB, 1, M, IPIV, -1);
}

void GETRSV_TL_L(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* IPIV) {
  LAPACKE_dgetrf(CblasColMajor, M, M, A, LDA, IPIV);
  dtrtrsv('L', 'U', 'L', 'T', 'N', DIAGB, M, 1.0, A, LDA, B, LDB);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit, M, M,
              1.0, A, LDA, B, LDB);
  LAPACKE_dlaswp(CblasColMajor, M, B, LDB, 1, M, IPIV, -1);
}

void GETRSV_NU_R(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* JPIV) {
  LAPACKE_dgetrf(CblasColMajor, M, M, A, LDA, JPIV);
  dtrtrsv('R', 'U', 'U', 'N', 'N', DIAGB, M, 1.0, A, LDA, B, LDB);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, M,
              M, 1.0, A, LDA, B, LDB);
  dlaswp2(M, B, LDB, 0, M - 1, JPIV, -1);
}

void GETRSV_NL_R(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* JPIV) {
  LAPACKE_dgetrf(CblasColMajor, M, M, A, LDA, JPIV);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
              M, M, 1.0, A, LDA, B, LDB);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, M,
              M, 1.0, A, LDA, B, LDB);
  dlaswp2(M, B, LDB, 0, M - 1, JPIV, -1);
}

void GETRSV_TU_R(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* JPIV) {
  double* A_TR = (double*)malloc(M * M * sizeof(double));
  const int LDA_TR = M;
  dge_trans(M, M, A, LDA, A_TR, LDA_TR);
  LAPACKE_dgetrf(CblasColMajor, M, M, A_TR, LDA_TR, JPIV);

  dtrtrsv('R', 'U', 'U', 'N', 'N', DIAGB, M, 1.0, A_TR, LDA_TR, B, LDB);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, M,
              M, 1.0, A_TR, LDA_TR, B, LDB);
  dlaswp2(M, B, LDB, 0, M - 1, JPIV, -1);
  free(A_TR);
}

void GETRSV_TL_R(const char DIAGB, const int M, double* A, const int LDA,
                 double* B, const int LDB, int* JPIV) {
  LAPACKE_dgetrf(CblasColMajor, M, M, A, LDA, JPIV);
  dlaswp2(M, B, LDB, 0, M - 1, JPIV, 1);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit, M,
              M, 1.0, A, LDA, B, LDB);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit,
              M, M, 1.0, A, LDA, B, LDB);
}