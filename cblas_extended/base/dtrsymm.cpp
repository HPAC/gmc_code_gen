#include <openblas/cblas.h>

#include <algorithm>

#include "extended_base.hpp"
// #include "instrumentation.hpp"

void dtrsymm(const char SIDE, const char UPLOA, const char UPLOB,
             const char TRANSA, const char DIAG, const int M,
             const double ALPHA, const double* A, const int LDA, double* B,
             const int LDB) {
  constexpr double ZERO = 0.0;

  CBLAS_SIDE TR_SIDE = (SIDE == 'L') ? CblasLeft : CblasRight;
  CBLAS_UPLO UPLO_A = (UPLOA == 'U') ? CblasUpper : CblasLower;
  CBLAS_TRANSPOSE TRANS_A = (TRANSA == 'N') ? CblasNoTrans : CblasTrans;
  CBLAS_DIAG DIAG_A = (DIAG == 'U') ? CblasUnit : CblasNonUnit;

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLOA != 'U') && (UPLOA != 'L'))
    INFO = 2;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 3;
  else if ((TRANSA != 'N') && (TRANSA != 'C') && (TRANSA != 'T'))
    INFO = 4;
  else if ((DIAG != 'U') && (DIAG != 'N'))
    INFO = 5;
  else if (M < 0)
    INFO = 6;
  else if (LDA < std::max<int>(1, M))
    INFO = 9;
  else if (LDB < std::max<int>(1, M))
    INFO = 11;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dtrsymm", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  /*
   * And when ALPHA == ZERO
   */
  if (ALPHA == ZERO) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < M; ++i) B[j * LDB + i] = ZERO;
    }
  }

  /*
   * Start the operations.
   */
  sy2full(UPLOB, M, B, LDB);  // convert B to a full matrix

  cblas_dtrmm(CblasColMajor, TR_SIDE, UPLO_A, TRANS_A, DIAG_A, M, M, ALPHA, A,
              LDA, B, LDB);
}  // END OF DTRSYMM
