#include <openblas/cblas.h>

#include <algorithm>

#include "extended_base.hpp"
// #include "instrumentation.hpp"

void dsysymm(const char UPLOA, const char UPLOB, const int M,
             const double ALPHA, double* A, const int LDA, double* B,
             const int LDB, const double BETA, double* C, const int LDC) {
  constexpr double ZERO = 0.0;

  bool UPPER_A = UPLOA == 'U';
  bool UPPER_B = UPLOB == 'U';

  int INFO = 0;
  if ((UPLOA != 'U') && (UPLOA != 'L'))
    INFO = 1;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 2;
  else if (M < 0)
    INFO = 3;
  else if (LDA < std::max<int>(1, M))
    INFO = 6;
  else if (LDB < std::max<int>(1, M))
    INFO = 8;
  else if (LDC < std::max<int>(1, M))
    INFO = 11;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dsysymm", "Illegal setting, %d\n");
    return;
  }

  // RESET_COUNT

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  /*
   * And when ALPHA == 0
   */
  if (ALPHA == ZERO) {
    if (BETA == ZERO) {
      for (int j = 0; j < M; ++j) {
        for (int i = 0; i < M; ++i) C[j * LDC + i] = ZERO;
      }
    } else {
      for (int j = 0; j < M; ++j) {
        for (int i = 0; i < M; ++i) C[j * LDC + i] *= BETA;
      }
    }
    return;
  }

  /*
   * Start the operations.
   */
  if (UPPER_A) {
    sy2full('U', M, A, LDA);
  } else {
    sy2full('L', M, A, LDA);
  }

  if (UPPER_B) {
    sy2full('U', M, B, LDB);
  } else {
    sy2full('L', M, B, LDB);
  }
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, M, M, ALPHA, A, LDA,
              B, LDB, BETA, C, LDC);
}  // END OF DSYSYMM
