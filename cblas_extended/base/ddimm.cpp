#include <algorithm>

#include "extended_base.hpp"
// #include "instrumentation.hpp"

void ddimm(const char SIDE, const int M, const int N, const double ALPHA,
           const double* A, const int INCA, double* B, const int LDB) {
  constexpr double ZERO = 0.0;

  double TEMP;

  bool LSIDE = SIDE == 'L';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if (M < 0)
    INFO = 2;
  else if (N < 0)
    INFO = 3;
  else if (INCA < 0)
    INFO = 6;
  else if (LDB < std::max<int>(1U, M))
    INFO = 8;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "ddimm", "Illegal setting, %d\n");
    return;
  }

  // RESET_COUNT

  /*
   * Quick return if possible.
   */
  if (M == 0 || N == 0) return;

  /*
   * And when ALPHA == 0
   */
  if (ALPHA == ZERO) {
    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < M; ++i) B[j * LDB + i] = ZERO;
    }
    return;
  }

  /*
   * Start the operations.
   */
  if (LSIDE) {
    /*
     * Form B := ALPHA*A*B
     */
    for (int i = 0; i < M; ++i) {
      TEMP = ALPHA * A[INCA * i];
      // COUNT(1)
      for (int j = 0; j < N; ++j) {
        B[j * LDB + i] *= TEMP;
        // COUNT(1)
      }
    }
  } else {
    /*
     * Form B := ALPHA*B*A
     */
    for (int j = 0; j < N; ++j) {
      TEMP = ALPHA * A[INCA * j];
      // COUNT(1)
      for (int i = 0; i < M; ++i) {
        B[j * LDB + i] *= TEMP;
        // COUNT(1)
      }
    }
  }
}  // End of DDIMM