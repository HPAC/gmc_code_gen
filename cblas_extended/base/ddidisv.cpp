#include <algorithm>

#include "extended_base.hpp"

void ddidisv(const int M, const double ALPHA, const double* A, const int INCA,
             double* B, const int INCB) {
  constexpr double ZERO = 0.0;

  int INFO = 0;
  if (M < 0)
    INFO = 1;
  else if (INCA < 0)
    INFO = 4;
  else if (INCB < 0)
    INFO = 6;

  if (INFO != 0) {
    // @todo check if this is correct.
    // cblas_xerbla(INFO, "ddidism", "Illegal setting, %d\n");
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
    for (int i = 0; i < M; ++i) B[i * INCB] = ZERO;
    return;
  }

  /*
   * Start the operations.
   */
  for (int i = 0; i < M; ++i) B[i * INCB] *= ALPHA / A[i * INCA];
}