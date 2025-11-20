#include <openblas/cblas.h>
#include <openblas/lapacke.h>

#include <algorithm>

#include "extended_base.hpp"

/**
 * @brief B := ALPHA * A + BETA * B
 *
 * @param M       rows of A and B
 * @param N       cols of A and B
 * @param ALPHA
 * @param A
 * @param LDA
 * @param BETA
 * @param B
 * @param LDB
 */
void ADD_IN_PLACE(const int M, const int N, const double ALPHA, const double* A,
                  const int LDA, const double BETA, double* B, const int LDB) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      // B[j * LDB + i] += ALPHA * A[j * LDA + i];
      B[j * LDB + i] = ALPHA * A[j * LDA + i] + BETA * B[j * LDB + i];
    }
  }
}

/**
 * @brief B := ALPHA * A^T + BETA * B
 *
 * @param M rows of B and colums of A
 * @param N colums of B and rows of A
 */
void ADD_IN_PLACE_TRANSA(const int M, const int N, const double ALPHA,
                         const double* A, const int LDA, const double BETA,
                         double* B, const int LDB) {
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) {
      B[j * LDB + i] = ALPHA * A[i * LDA + j] + BETA * B[j * LDB + i];
    }
  }
}

void TRTRSV_1x1(const char DIAGA, const char DIAGB, const double ALPHA,
                const double A_0, double& B_0);

void TRTRSV_NUU_L_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB);

void TRTRSV_NLL_L_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB);

void TRTRSV_TUL_L_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB);

void TRTRSV_TLU_L_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB);

void TRTRSV_NUU_R_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB);

void TRTRSV_NLL_R_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB);

void TRTRSV_TUL_R_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB);

void TRTRSV_TLU_R_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB);

/**
 * @brief Solves a triangular linear system with a triangular rhs of the same
 * triangularity after transposition is applied. This is:
 * B := ALPHA * op(A^{-1}) * B  or  B := ALPHA * B * op(A^{-1})
 *
 * The B matrix holds the result. The A matrix is GUARANTEED to remain INTACT.
 *
 */
void dtrtrsv(const char SIDE, const char UPLOA, const char UPLOB,
             const char TRANSA, const char DIAGA, const char DIAGB, const int M,
             const double ALPHA, double* A, const int LDA, double* B,
             const int LDB) {
  constexpr double ZERO = 0.0;

  bool LHS_LEFT = SIDE == 'L';
  bool UPPER_A = UPLOA == 'U';
  bool UPPER_B = UPLOB == 'U';
  bool NOTA = TRANSA == 'N';
  CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLOA != 'U') && (UPLOA != 'L'))
    INFO = 2;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 3;
  else if ((TRANSA != 'N') && (TRANSA != 'C') && (TRANSA != 'T'))
    INFO = 4;
  else if ((DIAGA != 'U') && (DIAGA != 'N'))
    INFO = 5;
  else if ((DIAGB != 'U') && (DIAGB != 'N'))
    INFO = 6;
  else if (M < 0)
    INFO = 7;
  else if (LDA < std::max<int>(1, M))
    INFO = 10;
  else if (LDB < std::max<int>(1, M))
    INFO = 12;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dtrtrsv", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  /*
   * And when ALPHA == 0
   */
  if (ALPHA == ZERO) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < M; ++i) B[j * LDB + i] = ZERO;
    }
    return;
  }

  /*
   * Start the operations.
   */
  if (LHS_LEFT) {
    // Form B := ALPHA * op(A^{-1}) * B
    if (NOTA) {
      // A is not transposed.
      if (UPPER_A) {
        // A is upper and not transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRSV_NUU_L_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB);
        } else {
          // B is lower.
          cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                      diag_a, M, M, ALPHA, A, LDA, B, LDB);
        }
      } else {
        // A is lower and not transposed.
        if (UPPER_B) {
          // B is upper.
          cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                      diag_a, M, M, ALPHA, A, LDA, B, LDB);
        } else {
          // B is lower.
          TRTRSV_NLL_L_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB);
        }
      }
    } else {
      // A is transposed.
      if (UPPER_A) {
        // A is upper and transposed.
        if (UPPER_B) {
          // B is upper.
          cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, diag_a,
                      M, M, ALPHA, A, LDA, B, LDB);
        } else {
          // B is lower.
          TRTRSV_TUL_L_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB);
        }
      } else {
        // A is lower and transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRSV_TLU_L_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB);
        } else {
          // B is lower.
          cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, diag_a,
                      M, M, ALPHA, A, LDA, B, LDB);
        }
      }
    }
  } else {
    // Form B := ALPHA * B * op(A^{-1})
    if (NOTA) {
      // A is not transposed.
      if (UPPER_A) {
        // A is upper and not transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRSV_NUU_R_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB);
        } else {
          // B is lower.
          cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                      diag_a, M, M, ALPHA, A, LDA, B, LDB);
        }
      } else {
        // A is lower and not tranposed.
        if (UPPER_B) {
          // B is upper.
          cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                      diag_a, M, M, ALPHA, A, LDA, B, LDB);
        } else {
          // B is lower.
          TRTRSV_NLL_R_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB);
        }
      }
    } else {
      // A is transposed.
      if (UPPER_A) {
        // A is upper and transposed.
        if (UPPER_B) {
          // B is upper.
          cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, diag_a,
                      M, M, ALPHA, A, LDA, B, LDB);
        } else {
          // B is lower.
          TRTRSV_TUL_R_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB);
        }
      } else {
        // A is lower and transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRSV_TLU_R_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB);
        } else {
          // B is lower.
          cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, diag_a,
                      M, M, ALPHA, A, LDA, B, LDB);
        }
      }
    }
  }
}  // END OF TRTRSV

void TRTRSV_1x1(const char DIAGA, const char DIAGB, const double ALPHA,
                const double A_0, double& B_0) {
  const double b = (DIAGB == 'N') ? B_0 : 1.0;
  const double a = (DIAGA == 'N') ? A_0 : 1.0;
  B_0 = ALPHA * b / a;
}

void TRTRSV_NUU_L_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB) {
  if (M == 1) {
    // base case
    TRTRSV_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0]);
  } else {
    // Partition and recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    double* A12 = A + LDA * H1;
    double* A22 = A + LDA * H1 + H1;
    double* B12 = B + LDB * H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X22
    TRTRSV_NUU_L_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB);

    // Computation X12
    double* CPY_A12 = (double*)malloc(H1 * H2 * sizeof(double));
    dlacpy(H1, H2, A12, LDA, CPY_A12, H1);

    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, H1, H2, 1.0, B22, LDB, CPY_A12, H1);
    ADD_IN_PLACE(H1, H2, -1.0, CPY_A12, H1, ALPHA, B12, LDB);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, diag_a, H1,
                H2, 1.0, A, LDA, B12, LDB);

    free(CPY_A12);

    // Computation X11
    TRTRSV_NUU_L_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB);
  }
}
/*
         H1    H2
 H1 | [ A11 | A12] [ X11 | X12]  =  [ B11 | B12]
 H2 | [   0 | A22] [   0 | X22]  =  [   0 | B22]

1. A11 * X11 = B11 --> REC(A11, B11)
2. A11 * X12 + A12 * X22 = B12; A11 * X12 = B12 - A12 * X22; --> TRMM(X22, A12)
+ TRSM(A11, B12)
3. X21 = 0
4. A22 * X22 = B22 --> REC(A22, B22)
*/

void TRTRSV_NLL_L_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB) {
  if (M == 1) {
    // base case
    TRTRSV_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0]);
  } else {
    // Partition and recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    double* A21 = A + H1;
    double* A22 = A + LDA * H1 + H1;
    double* B21 = B + H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11
    TRTRSV_NLL_L_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB);

    // Computation X21
    double* CPY_A21 = (double*)malloc(H2 * H1 * sizeof(double));
    dlacpy(H2, H1, A21, LDA, CPY_A21, H2);

    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                CblasNonUnit, H2, H1, 1.0, B, LDB, CPY_A21, H2);
    ADD_IN_PLACE(H2, H1, -1.0, CPY_A21, H2, ALPHA, B21, LDB);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, diag_a, H2,
                H1, 1.0, A22, LDA, B21, LDB);

    free(CPY_A21);

    // Computation X22
    TRTRSV_NLL_L_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB);
  }
}
/*
         H1    H2
 H1 | [ A11 |   0] [ X11 |   0]  =  [ B11 |   0]
 H2 | [ A21 | A22] [ X21 | X22]  =  [ B21 | B22]

1. A11 * X11 = ALPHA * B11 --> REC(A11, B11)
2. X12 = 0
3. A21 * X11 + A22 * X21 = ALPHA * B21; A22 * X21 = ALPHA * B21 - A21 * X11 -->
TRMM(X11, A21) + TRSM(A22, B21)
4. A22 * X22 = ALPHA * B22; --> REC(A22, B22);
*/

void TRTRSV_TUL_L_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB) {
  if (M == 1) {
    // base case
    TRTRSV_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0]);
  } else {
    // Partition and recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    double* A12 = A + LDA * H1;
    double* A22 = A + LDA * H1 + H1;
    double* B21 = B + H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11
    TRTRSV_TUL_L_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB);

    // Computation X21
    double* CPY_A12 = (double*)malloc(H1 * H2 * sizeof(double));
    dlacpy(H1, H2, A12, LDA, CPY_A12, H1);

    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                H1, H2, 1.0, B, LDB, CPY_A12, H1);
    ADD_IN_PLACE_TRANSA(H2, H1, -1.0, CPY_A12, H1, ALPHA, B21, LDB);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, diag_a, H2,
                H1, 1.0, A22, LDA, B21, LDB);

    free(CPY_A12);

    // Computation X22
    TRTRSV_TUL_L_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB);
  }
}
/*
         H1    H2
 H1 | [ A11 | A12]^T   [ X11 |   0]  =  [ B11 |   0]
 H2 | [   0 | A22]     [ X21 | X22]  =  [ B21 | B22]

 1. A11^T * X11 = ALPHA * B11; --> REC(A11, B11);
 2. X12 = 0;
 3. A12^T * X11 + A22^T * X21 = B21; A22^T * X21 = ALPHA * B21 - A12^T * X11;
 what if I transpose the computation of A12^T * X11 and add it transpose onto
 B21?
 TRMM(X11, A12^T) + TRSM(A22^T, B21);
 4. A22^T * X22 = ALPHA * B22; --> REC(A22, B22);
*/

void TRTRSV_TLU_L_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB) {
  if (M == 1) {
    // base case
    TRTRSV_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0]);
  } else {
    // Partition and recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    double* A21 = A + H1;
    double* A22 = A + LDA * H1 + H1;
    double* B12 = B + LDB * H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X22
    TRTRSV_TLU_L_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB);

    // Computation X12
    double* CPY_A21 = (double*)malloc(H2 * H1 * sizeof(double));
    dlacpy(H2, H1, A21, LDA, CPY_A21, H2);

    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                H2, H1, 1.0, B22, LDB, CPY_A21, H2);
    ADD_IN_PLACE_TRANSA(H1, H2, -1.0, CPY_A21, H2, ALPHA, B12, LDB);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, diag_a, H1,
                H2, 1.0, A, LDA, B12, LDB);

    free(CPY_A21);

    // Computation X11
    TRTRSV_TLU_L_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB);
  }
}

/*
         H1    H2
 H1 | [ A11 |   0]^T [ X11 | X12]  =  [ B11 | B12]
 H2 | [ A21 | A22]   [   0 | X22]  =  [   0 | B22]

1. A11^T * X11 = ALPHA * B11; ==> REC(A11, B11);
2. A11^T * X12 + A21^T * X22 = ALPHA * B12;
   A11^T * X12 = ALPHA * B12 - A21^T * X22;
   TRMM(X22^T, A21, TR_LEFT);
   ADD_IN_PLACE_TRANSA(A21, B12);
   TRSM(A11^T, B12);
3. X21 = 0;
4. A22^T * X22 = ALPHA * B22; ==> REC(A22, B22);
*/

void TRTRSV_NUU_R_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB) {
  if (M == 1) {
    // base case
    TRTRSV_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0]);
  } else {
    // Partition and recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    double* A12 = A + LDA * H1;
    double* A22 = A + LDA * H1 + H1;
    double* B12 = B + LDB * H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11
    TRTRSV_NUU_R_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB);

    // Computation X12
    double* CPY_A12 = (double*)malloc(H1 * H2 * sizeof(double));
    dlacpy(H1, H2, A12, LDA, CPY_A12, H1);

    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, H1, H2, 1.0, B, LDB, CPY_A12, H1);
    ADD_IN_PLACE(H1, H2, -1.0, CPY_A12, H1, ALPHA, B12, LDB);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, diag_a, H1,
                H2, 1.0, A22, LDA, B12, LDB);

    free(CPY_A12);

    // Computation X22
    TRTRSV_NUU_R_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB);
  }
}

void TRTRSV_NLL_R_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB) {
  if (M == 1) {
    // base case
    TRTRSV_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0]);
  } else {
    // Partition and recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    double* A21 = A + H1;
    double* A22 = A + LDA * H1 + H1;
    double* B21 = B + H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X22
    TRTRSV_NLL_R_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB);

    // Computation X21
    double* CPY_A21 = (double*)malloc(H2 * H1 * sizeof(double));
    dlacpy(H2, H1, A21, LDA, CPY_A21, H2);

    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, H2, H1, 1.0, B22, LDB, CPY_A21, H2);
    ADD_IN_PLACE(H2, H1, -1.0, CPY_A21, H2, ALPHA, B21, LDB);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, diag_a, H2,
                H1, 1.0, A, LDA, B21, LDB);

    free(CPY_A21);

    // Computation X11
    TRTRSV_NLL_R_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB);
  }
}

void TRTRSV_TUL_R_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB) {
  if (M == 1) {
    // base case
    TRTRSV_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0]);
  } else {
    // Partition and recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    double* A12 = A + LDA * H1;
    double* A22 = A + LDA * H1 + H1;
    double* B21 = B + H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X22
    TRTRSV_TUL_R_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB);

    // Computation X21
    double* CPY_A12 = (double*)malloc(H1 * H2 * sizeof(double));
    dlacpy(H1, H2, A12, LDA, CPY_A12, H1);

    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                H1, H2, 1.0, B22, LDB, CPY_A12, H1);
    ADD_IN_PLACE_TRANSA(H2, H1, -1.0, CPY_A12, H1, ALPHA, B21, LDB);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, diag_a, H2,
                H1, 1.0, A, LDA, B21, LDB);

    free(CPY_A12);

    // Computation X11
    TRTRSV_TUL_R_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB);
  }
}

void TRTRSV_TLU_R_REC(const char DIAGA, const char DIAGB, const int M,
                      const double ALPHA, double* A, const int LDA, double* B,
                      const int LDB) {
  if (M == 1) {
    // base case
    TRTRSV_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0]);
  } else {
    // Partition and recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    double* A21 = A + H1;
    double* A22 = A + LDA * H1 + H1;
    double* B12 = B + LDB * H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11
    TRTRSV_TLU_R_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB);

    // Computation X12
    double* CPY_A21 = (double*)malloc(H2 * H1 * sizeof(double));
    dlacpy(H2, H1, A21, LDA, CPY_A21, H2);

    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit,
                H2, H1, 1.0, B, LDB, CPY_A21, H2);
    ADD_IN_PLACE_TRANSA(H1, H2, -1.0, CPY_A21, H2, ALPHA, B12, LDB);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, diag_a, H1,
                H2, 1.0, A22, LDA, B12, LDB);

    free(CPY_A21);

    // Computation X22
    TRTRSV_TLU_R_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB);
  }
}
