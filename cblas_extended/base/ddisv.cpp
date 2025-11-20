#include <algorithm>

#include "extended_base.hpp"

void ddisv(const char SIDE, const int M, const int N, const double ALPHA,
           const double* A, const int INCA, double* B, const int LDB) {
  constexpr double ZERO = 0.0;

  double temp;

  bool DIAG_LSIDE = SIDE == 'L';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if (M < 0)
    INFO = 2;
  else if (N < 0)
    INFO = 3;
  else if (INCA < 0)
    INFO = 6;
  else if (LDB < std::max<int>(1, M))
    INFO = 8;

  if (INFO != 0) {
    // @todo check if this is correct.
    // cblas_xerbla(INFO, "ddimm", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0 || N == 0) return;

  /*
   * And when ALPHA == ZERO
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
  if (DIAG_LSIDE) {
    /*
     * Form B := ALPHA * A^{-1} * B
     */
    for (int i = 0; i < M; ++i) {
      temp = ALPHA / A[INCA * i];
      for (int j = 0; j < N; ++j) B[j * LDB + i] *= temp;
    }
  } else {
    /*
     * Form B := ALPHA * B * A^{-1}
     */
    for (int j = 0; j < N; ++j) {
      temp = ALPHA / A[INCA * j];
      for (int i = 0; i < M; ++i) B[j * LDB + i] *= temp;
    }
  }
}  // END OF DDISM