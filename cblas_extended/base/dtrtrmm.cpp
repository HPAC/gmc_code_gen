#include <omp.h>
#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

/**
 * @brief B := A + B
 *
 * @param M rows of A and B
 * @param N cols of A and B
 */
void ADD_IN_PLACE(const int M, const int N, const double* A, const int LDA,
                  double* B, const int LDB) {
  unsigned n_threads = std::stoi(std::getenv("OMP_NUM_THREADS"));
  unsigned PW = N / n_threads;  // Panel Width
  if (PW <= 2) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        B[j * LDB + i] += A[j * LDA + i];
      }
    }
  } else {
#pragma omp parallel for schedule(static, PW)
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        B[j * LDB + i] += A[j * LDA + i];
      }
    }
  }
}

void TRTRMM_NNUU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_NNUL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_NTUU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_NTUL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_NNLU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_NNLL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_NTLU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_NTLL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_TNUU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_TNUL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_TTUU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_TTUL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_TNLU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_TNLL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_TTLU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void TRTRMM_TTLL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC);

void dtrtrmm(const char UPLOA, const char UPLOB, const char TRANSA,
             const char TRANSB, const char DIAGA, const char DIAGB, const int M,
             const double ALPHA, const double* A, const int LDA,
             const double* B, const int LDB, double* C, const int LDC) {
  constexpr double ZERO = 0.0;

  bool UPPER_A = UPLOA == 'U';
  bool UPPER_B = UPLOB == 'U';
  bool NOTA = TRANSA == 'N';
  bool NOTB = TRANSB == 'N';

  int INFO = 0;
  if ((UPLOA != 'U') && (UPLOA != 'L'))
    INFO = 1;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 2;
  else if ((TRANSA != 'N') && (TRANSA != 'C') && (TRANSA != 'T'))
    INFO = 3;
  else if ((TRANSB != 'N') && (TRANSB != 'C') && (TRANSB != 'T'))
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
  else if (LDC < std::max<int>(1, M))
    INFO = 15;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of
    // cases. cblas_xerbla(INFO, "dtrtrmm", "Illegal setting, %d\n");
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
      for (int i = 0; i < M; ++i) C[j * LDC + i] = ZERO;
    }
    return;
  }

  /*
   * Start the operations.
   */
  if (NOTA) {
    // A is not transposed.
    if (UPPER_A) {
      // A is upper.
      if (NOTB) {
        // B is not transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRMM_NNUU_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        } else {
          // B is lower.
          TRTRMM_NNUL_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        }
      } else {
        // B is transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRMM_NTUU_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        } else {
          // B is lower.
          TRTRMM_NTUL_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        }
      }
    } else {
      // A is lower.
      if (NOTB) {
        // B is not transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRMM_NNLU_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        } else {
          // B is lower.
          TRTRMM_NNLL_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        }
      } else {
        // B is transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRMM_NTLU_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        } else {
          // B is lower.
          TRTRMM_NTLL_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        }
      }
    }
  } else {
    // A is transposed.
    if (UPPER_A) {
      // A is upper.
      if (NOTB) {
        // B is not transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRMM_TNUU_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        } else {
          // B is lower.
          TRTRMM_TNUL_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        }
      } else {
        // B is transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRMM_TTUU_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        } else {
          // B is lower.
          TRTRMM_TTUL_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        }
      }
    } else {
      // A is lower.
      if (NOTB) {
        // B is not transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRMM_TNLU_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        } else {
          // B is lower.
          TRTRMM_TNLL_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        }
      } else {
        // B is transposed.
        if (UPPER_B) {
          // B is upper.
          TRTRMM_TTLU_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        } else {
          // B is lower.
          TRTRMM_TTLL_REC(DIAGA, DIAGB, M, ALPHA, A, LDA, B, LDB, C, LDC);
        }
      }
    }
  }
}  // END OF TRTRMM

void TRTRMM_1x1(const char DIAGA, const char DIAGB, const double ALPHA,
                const double A_0, const double B_0, double& C_0) {
  const double b = (DIAGB == 'N') ? B_0 : 1.0;
  const double a = (DIAGA == 'N') ? A_0 : 1.0;
  C_0 = ALPHA * a * b;
}

void TRTRMM_NNUU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B12 = B + LDB * H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Compute C11
    TRTRMM_NNUU_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Compute C12
    dlacpy(H1, H2, A12, LDA, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, diag_b, H1,
                H2, ALPHA, B22, LDB, C12, LDC);

    double* cpy_B12 = (double*)malloc(H1 * H2 * sizeof(double));
    dlacpy(H1, H2, B12, LDB, cpy_B12, H1);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, diag_a, H1,
                H2, ALPHA, A, LDA, cpy_B12, H1);

    ADD_IN_PLACE(H1, H2, cpy_B12, H1, C12, LDC);

    // Compute C22
    TRTRMM_NNUU_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);

    free(cpy_B12);
  }
}

void TRTRMM_NNUL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B21 = B + H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_NNUL_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, H1, H1, H2, ALPHA,
                A12, LDA, B21, LDB, 1.0, C, LDC);

    // Computation C12
    dlacpy(H1, H2, A12, LDA, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, diag_b, H1,
                H2, ALPHA, B22, LDB, C12, LDC);

    // Computation C21
    dlacpy(H2, H1, B21, LDB, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, diag_a, H2,
                H1, ALPHA, A22, LDA, C21, LDC);

    // Computation C22
    TRTRMM_NNUL_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);
  }
}

void TRTRMM_NTUU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B12 = B + LDB * H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_NTUU_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, H1, H1, H2, ALPHA, A12,
                LDA, B12, LDB, 1.0, C, LDC);

    // Computation C12
    dlacpy(H1, H2, A12, LDA, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, diag_b, H1,
                H2, ALPHA, B22, LDB, C12, LDC);

    // Computation C21 -- This is TRMM(A22, B12^T)
    dge_trans(H1, H2, B12, LDB, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, diag_a, H2,
                H1, 1.0, A22, LDA, C21, LDC);

    // Computation C22
    TRTRMM_NTUU_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);
  }
}

void TRTRMM_NTUL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B21 = B + H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_NTUL_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C12
    double* cpy_B21T = (double*)malloc(H1 * H2 * sizeof(double));
    dge_trans(H2, H1, B21, LDB, cpy_B21T, H1);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, diag_a, H1,
                H2, ALPHA, A, LDA, cpy_B21T, H1);

    dlacpy(H1, H2, A12, LDA, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, diag_b, H1,
                H2, ALPHA, B22, LDB, C12, LDC);
    ADD_IN_PLACE(H1, H2, cpy_B21T, H1, C12, LDC);

    // Computation C22
    TRTRMM_NTUL_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);

    free(cpy_B21T);
  }
}

void TRTRMM_NNLU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B12 = B + LDB * H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_NNLU_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C12
    dlacpy(H1, H2, B12, LDB, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, diag_a, H1,
                H2, ALPHA, A, LDA, C12, LDC);

    // Computation C21
    dlacpy(H2, H1, A21, LDA, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, diag_b, H2,
                H1, ALPHA, B, LDB, C21, LDC);

    // Computation C22
    TRTRMM_NNLU_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, H2, H2, H1, ALPHA,
                A21, LDA, B12, LDB, 1.0, C22, LDC);
  }
}

void TRTRMM_NNLL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B21 = B + H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_NNLL_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C21
    dlacpy(H2, H1, A21, LDA, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, diag_b, H2,
                H1, ALPHA, B, LDB, C21, LDC);

    double* cpy_B21 = (double*)malloc(H2 * H1 * sizeof(double));
    dlacpy(H2, H1, B21, LDB, cpy_B21, H2);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, diag_a, H2,
                H1, ALPHA, A22, LDA, cpy_B21, H2);
    ADD_IN_PLACE(H2, H1, cpy_B21, H2, C21, LDC);

    // Computation C22
    TRTRMM_NNLL_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);

    free(cpy_B21);
  }
}

void TRTRMM_NTLU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B12 = B + LDB * H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_NTLU_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C21
    dlacpy(H2, H1, A21, LDA, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, diag_b, H2,
                H1, ALPHA, B, LDB, C21, LDC);

    double* cpy_B12T = (double*)malloc(H2 * H1 * sizeof(double));
    dge_trans(H1, H2, B12, LDB, cpy_B12T, H2);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, diag_a, H2,
                H1, ALPHA, A22, LDA, cpy_B12T, H2);
    ADD_IN_PLACE(H2, H1, cpy_B12T, H2, C21, LDC);

    // Computation C22
    TRTRMM_NTLU_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);

    free(cpy_B12T);
  }
}

void TRTRMM_NTLL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B21 = B + H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_NTLL_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C12
    dge_trans(H2, H1, B21, LDB, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, diag_a, H1,
                H2, ALPHA, A, LDA, C12, LDC);

    // Computation C21
    dlacpy(H2, H1, A21, LDA, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, diag_b, H2,
                H1, ALPHA, B, LDB, C21, LDC);

    // Computation C22
    TRTRMM_NTLL_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, H2, H2, H1, ALPHA, A21,
                LDA, B21, LDB, 1.0, C22, LDC);
  }
}

void TRTRMM_TNUU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B12 = B + LDB * H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_TNUU_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C12
    dlacpy(H1, H2, B12, LDB, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, diag_a, H1,
                H2, ALPHA, A, LDA, C12, LDC);

    // Computation C21
    dge_trans(H1, H2, A12, LDA, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, diag_b, H2,
                H1, ALPHA, B, LDB, C21, LDC);

    // Computation C22
    TRTRMM_TNUU_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, H2, H2, H1, ALPHA, A12,
                LDA, B12, LDB, 1.0, C22, LDC);
  }
}

void TRTRMM_TNUL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B21 = B + H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_TNUL_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C21
    dlacpy(H2, H1, B21, LDB, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, diag_a, H2,
                H1, ALPHA, A22, LDA, C21, LDC);

    double* cpy_A12T = (double*)malloc(H2 * H1 * sizeof(double));
    dge_trans(H1, H2, A12, LDA, cpy_A12T, H2);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, diag_b, H2,
                H1, ALPHA, B, LDB, cpy_A12T, H2);

    ADD_IN_PLACE(H2, H1, cpy_A12T, H2, C21, LDC);

    // Computation C22
    TRTRMM_TNUL_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);

    free(cpy_A12T);
  }
}

void TRTRMM_TTUU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B12 = B + LDB * H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_TTUU_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C21
    // TRMM(B11^T, A12^T)
    double* cpy_A12T = (double*)malloc(H2 * H1 * sizeof(double));
    dge_trans(H1, H2, A12, LDA, cpy_A12T, H2);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, diag_b, H2,
                H1, ALPHA, B, LDB, cpy_A12T, H2);

    // TRMM(A22^T, B12^T)
    dge_trans(H1, H2, B12, LDB, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, diag_a, H2,
                H1, ALPHA, A22, LDA, C21, LDC);

    ADD_IN_PLACE(H2, H1, cpy_A12T, H2, C21, LDC);

    // Computation C22
    TRTRMM_TTUU_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);

    free(cpy_A12T);
  };
}

void TRTRMM_TTUL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A12 = A + LDA * H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B21 = B + H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_TTUL_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C12
    // TRMM(A11^T, B21^T)
    dge_trans(H2, H1, B21, LDB, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, diag_a, H1,
                H2, ALPHA, A, LDA, C12, LDC);

    // Computation C21
    // TRMM(B11^T, A12^T)
    dge_trans(H1, H2, A12, LDA, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, diag_b, H2,
                H1, ALPHA, B, LDB, C21, LDC);

    // Computation C22
    TRTRMM_TTUL_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, H2, H2, H1, ALPHA, A12,
                LDA, B21, LDB, 1.0, C22, LDC);
  }
}

void TRTRMM_TNLU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B12 = B + LDB * H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_TNLU_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C12
    // TRMM(A11^T, B12)
    dlacpy(H1, H2, B12, LDB, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, diag_a, H1,
                H2, ALPHA, A, LDA, C12, LDC);

    // TRMM(B22, A21^T)
    double* cpy_A21T = (double*)malloc(H1 * H2 * sizeof(double));
    dge_trans(H2, H1, A21, LDA, cpy_A21T, H1);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, diag_b, H1,
                H2, ALPHA, B22, LDB, cpy_A21T, H1);

    ADD_IN_PLACE(H1, H2, cpy_A21T, H1, C12, LDC);

    // Computation C22
    TRTRMM_TNLU_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);

    free(cpy_A21T);
  }
}

void TRTRMM_TNLL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B21 = B + H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_TNLL_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, H1, H1, H2, ALPHA, A21,
                LDA, B21, LDB, 1.0, C, LDC);

    // Computation C12 ==> TRMM(B22, A21^T)
    dge_trans(H2, H1, A21, LDA, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, diag_b, H1,
                H2, ALPHA, B22, LDB, C12, LDC);

    // Computation C21 ==> TRMM(A22^T, B21)
    dlacpy(H2, H1, B21, LDB, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, diag_a, H2,
                H1, ALPHA, A22, LDA, C21, LDC);

    // Computation C22
    TRTRMM_TNLL_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);
  }
}

void TRTRMM_TTLU_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B12 = B + LDB * H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C21 = C + H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_TTLU_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, H1, H1, H2, ALPHA, A21,
                LDA, B12, LDB, 1.0, C, LDC);

    // Computation C12 ==> TRMM(B22^T, A21^T)
    dge_trans(H2, H1, A21, LDA, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, diag_b, H1,
                H2, ALPHA, B22, LDB, C12, LDC);

    // Computation C21 ==> TRMM(A22^T, B12^T)
    dge_trans(H1, H2, B12, LDB, C21, LDC);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, diag_a, H2,
                H1, ALPHA, A22, LDA, C21, LDC);

    // Computation C22
    TRTRMM_TTLU_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);
  }
}

void TRTRMM_TTLL_REC(const char DIAGA, const char DIAGB, const int M,
                     const double ALPHA, const double* A, const int LDA,
                     const double* B, const int LDB, double* C, const int LDC) {
  if (M == 1) {
    // base case
    TRTRMM_1x1(DIAGA, DIAGB, ALPHA, A[0], B[0], C[0]);
  } else {
    // recursive call
    int H1 = M / 2;
    int H2 = M - H1;
    const double* A21 = A + H1;
    const double* A22 = A + LDA * H1 + H1;
    const double* B21 = B + H1;
    const double* B22 = B + LDB * H1 + H1;
    double* C12 = C + LDC * H1;
    double* C22 = C + LDC * H1 + H1;
    CBLAS_DIAG diag_a = (DIAGA == 'N') ? CblasNonUnit : CblasUnit;
    CBLAS_DIAG diag_b = (DIAGB == 'N') ? CblasNonUnit : CblasUnit;

    // Computation C11
    TRTRMM_TTLL_REC(DIAGA, DIAGB, H1, ALPHA, A, LDA, B, LDB, C, LDC);

    // Computation C12
    // TRMM(A11^T, B21^T)
    double* cpy_B21T = (double*)malloc(H1 * H2 * sizeof(double));
    dge_trans(H2, H1, B21, LDB, cpy_B21T, H1);
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, diag_a, H1,
                H2, ALPHA, A, LDA, cpy_B21T, H1);

    // TRMM(B22^T, A21^T)
    dge_trans(H2, H1, A21, LDA, C12, LDC);
    cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, diag_b, H1,
                H2, ALPHA, B22, LDB, C12, LDC);

    ADD_IN_PLACE(H1, H2, cpy_B21T, H1, C12, LDC);

    // Computation C22
    TRTRMM_TTLL_REC(DIAGA, DIAGB, H2, ALPHA, A22, LDA, B22, LDB, C22, LDC);

    free(cpy_B21T);
  }
}
