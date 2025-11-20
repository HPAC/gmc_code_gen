#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

void TRDISV_NU_L_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB);

void TRDISV_NL_L_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB);

void TRDISV_TU_L_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB);

void TRDISV_TL_L_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB);

void TRDISV_NU_R_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB);

void TRDISV_NL_R_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB);

void TRDISV_TU_R_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB);

void TRDISV_TL_R_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB);

void dtrdisv(const char SIDE, const char UPLOA, const char TRANSA,
             const char DIAGA, const int M, const double ALPHA, const double* A,
             const int LDA, const double* D, const int INCD, double* B,
             const int LDB) {
  bool LHS_LEFT = SIDE == 'L';
  bool NOTA = TRANSA == 'N';
  bool UPPER_A = UPLOA == 'U';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLOA != 'U') && (UPLOA != 'L'))
    INFO = 2;
  else if ((TRANSA != 'N') && (TRANSA != 'C') && (TRANSA != 'T'))
    INFO = 3;
  else if ((DIAGA != 'U') && (DIAGA != 'N'))
    INFO = 4;
  else if (M < 0)
    INFO = 5;
  else if (LDA < std::max<int>(1, M))
    INFO = 8;
  else if (LDB < std::max<int>(1, M))
    INFO = 12;

  if (INFO != 0) {
    // @todo check if this is correct.
    // cblas_xerbla(INFO, "dtrdisv", "Illegal setting, %d\n");
    return;
  }

  /*
   *  Quick return if possible.
   */
  if (M == 0) return;

  /*
   *  If ALPHA == 0
   */
  if (ALPHA == 0.0) {
    for (int j = 0; j < M; j++) {
      for (int i = 0; i < M; i++) {
        B[j * LDB + i] = 0.0;
      }
    }
  }

  /*
   *  Start the operations. Put the diagonal in place and scale by ALPHA.
   */
  for (int i = 0; i < M; i++) {
    B[i * LDB + i] = ALPHA * D[i * INCD];
  }

  if (LHS_LEFT) {
    /*
     *  Form X = op(A^{-1}) * ALPHA * B
     */
    if (NOTA) {
      // A is not transposed.
      if (UPPER_A) {
        // A is upper.
        TRDISV_NU_L_REC(DIAGA, M, A, LDA, B, LDB);
      } else {
        // A is lower.
        TRDISV_NL_L_REC(DIAGA, M, A, LDA, B, LDB);
      }
    } else {
      // A is transposed.
      if (UPPER_A) {
        // A is upper.
        TRDISV_TU_L_REC(DIAGA, M, A, LDA, B, LDB);
      } else {
        // A is lower.
        TRDISV_TL_L_REC(DIAGA, M, A, LDA, B, LDB);
      }
    }
  } else {
    /*
     *  Form X = ALPHA * B * op(A^{-1})
     */
    if (NOTA) {
      // A is not transposed.
      if (UPPER_A) {
        // A is upper.
        TRDISV_NU_R_REC(DIAGA, M, A, LDA, B, LDB);
      } else {
        // A is lower.
        TRDISV_NL_R_REC(DIAGA, M, A, LDA, B, LDB);
      }
    } else {
      // A is transposed.
      if (UPPER_A) {
        // A is upper.
        TRDISV_TU_R_REC(DIAGA, M, A, LDA, B, LDB);
      } else {
        // A is lower.
        TRDISV_TL_R_REC(DIAGA, M, A, LDA, B, LDB);
      }
    }
  }
}  // END OF TRDISV

void TRDISV_NU_L_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB) {
  if (M == 1) {
    // Base case.
    if (DIAGA == 'N') {
      B[0] = B[0] / A[0];
    }
  } else {
    // Partition, computation, and recursive calls.
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    double* B12 = B + LDB * H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG DIAG_A = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11.
    TRDISV_NU_L_REC(DIAGA, H1, A, LDA, B, LDB);

    // Computation X22.
    TRDISV_NU_L_REC(DIAGA, H2, A22, LDA, B22, LDB);

    // Computation X12.
    dlacpy(H1, H2, A12, LDA, B12, LDB);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, H1, H2, -1.0, B22, LDB, B12, LDB);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, DIAG_A, H1,
                H2, 1.0, A, LDA, B12, LDB);
  }
}

void TRDISV_NL_L_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB) {
  if (M == 1) {
    // Base case.
    if (DIAGA == 'N') {
      B[0] = B[0] / A[0];
    }
  } else {
    // Partition, computation, and recursive calls.
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    double* B21 = B + H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG DIAG_A = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11.
    TRDISV_NL_L_REC(DIAGA, H1, A, LDA, B, LDB);

    // Computation X21.
    dlacpy(H2, H1, A21, LDA, B21, LDB);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                CblasNonUnit, H2, H1, -1.0, B, LDB, B21, LDB);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, DIAG_A, H2,
                H1, 1.0, A22, LDA, B21, LDB);

    // Computation X22.
    TRDISV_NL_L_REC(DIAGA, H2, A22, LDA, B22, LDB);
  }
}

void TRDISV_TU_L_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB) {
  if (M == 1) {
    // Base case.
    if (DIAGA == 'N') {
      B[0] = B[0] / A[0];
    }
  } else {
    // Partition, computation, and recursive calls.
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    double* B21 = B + H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG DIAG_A = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11.
    TRDISV_TU_L_REC(DIAGA, H1, A, LDA, B, LDB);

    // Computation X21.
    dge_trans(H1, H2, A12, LDA, B21, LDB);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                CblasNonUnit, H2, H1, -1.0, B, LDB, B21, LDB);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, DIAG_A, H2,
                H1, 1.0, A22, LDA, B21, LDB);

    // Computation X22.
    TRDISV_TU_L_REC(DIAGA, H2, A22, LDA, B22, LDB);
  }
}

void TRDISV_TL_L_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB) {
  if (M == 1) {
    // Base case.
    if (DIAGA == 'N') {
      B[0] = B[0] / A[0];
    }
  } else {
    // Partition, computation, and recursive calls.
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    double* B12 = B + LDA * H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG DIAG_A = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11.
    TRDISV_TL_L_REC(DIAGA, H1, A, LDA, B, LDB);

    // Computation X22.
    TRDISV_TL_L_REC(DIAGA, H2, A22, LDA, B22, LDB);

    // Computation X12.
    dge_trans(H2, H1, A21, LDA, B12, LDB);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, H1, H2, -1.0, B22, LDB, B12, LDB);
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, DIAG_A, H1,
                H2, 1.0, A, LDA, B12, LDB);
  }
}

void TRDISV_NU_R_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB) {
  if (M == 1) {
    // Base case.
    if (DIAGA == 'N') {
      B[0] = B[0] / A[0];
    }
  } else {
    // Partition, computation, and recursive calls.
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    double* B12 = B + LDB * H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG DIAG_A = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11.
    TRDISV_NU_R_REC(DIAGA, H1, A, LDA, B, LDB);

    // Computation X12.
    dlacpy(H1, H2, A12, LDA, B12, LDB);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, H1, H2, -1.0, B, LDB, B12, LDB);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, DIAG_A, H1,
                H2, 1.0, A22, LDA, B12, LDB);

    // Computation X22.
    TRDISV_NU_R_REC(DIAGA, H2, A22, LDA, B22, LDB);
  }
}

void TRDISV_NL_R_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB) {
  if (M == 1) {
    // Base case.
    if (DIAGA == 'N') {
      B[0] = B[0] / A[0];
    }
  } else {
    // Partition, computation, and recursive calls.
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    double* B21 = B + H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG DIAG_A = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X22.
    TRDISV_NL_R_REC(DIAGA, H2, A22, LDA, B22, LDB);

    // Computation X21.
    dlacpy(H2, H1, A21, LDA, B21, LDB);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, H2, H1, -1.0, B22, LDB, B21, LDB);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, DIAG_A, H2,
                H1, 1.0, A, LDA, B21, LDB);

    // Computation X11.
    TRDISV_NL_R_REC(DIAGA, H1, A, LDA, B, LDB);
  }
}

void TRDISV_TU_R_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB) {
  if (M == 1) {
    // Base case.
    if (DIAGA == 'N') {
      B[0] = B[0] / A[0];
    }
  } else {
    // Partition, computation, and recursive calls.
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    double* B21 = B + H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG DIAG_A = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X22.
    TRDISV_TU_R_REC(DIAGA, H2, A22, LDA, B22, LDB);

    // Computation X21.
    dge_trans(H1, H2, A12, LDA, B21, LDB);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, H2, H1, -1.0, B22, LDB, B21, LDB);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, DIAG_A, H2,
                H1, 1.0, A, LDA, B21, LDB);

    // Computation X11.
    TRDISV_TU_R_REC(DIAGA, H1, A, LDA, B, LDB);
  }
}

void TRDISV_TL_R_REC(const char DIAGA, const int M, const double* A,
                     const int LDA, double* B, const int LDB) {
  if (M == 1) {
    // Base case.
    if (DIAGA == 'N') {
      B[0] = B[0] / A[0];
    }
  } else {
    // Partition, computation, and recursive calls.
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    double* B12 = B + LDB * H1;
    double* B22 = B + LDB * H1 + H1;
    CBLAS_DIAG DIAG_A = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;

    // Computation X11.
    TRDISV_TL_R_REC(DIAGA, H1, A, LDA, B, LDB);

    // Computation X12.
    dge_trans(H2, H1, A21, LDA, B12, LDB);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, H1, H2, -1.0, B, LDB, B12, LDB);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, DIAG_A, H1,
                H2, 1.0, A22, LDA, B12, LDB);

    // Computation X22.
    TRDISV_TL_R_REC(DIAGA, H2, A22, LDA, B22, LDB);
  }
}