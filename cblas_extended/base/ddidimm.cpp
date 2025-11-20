#include <algorithm>

#include "extended_base.hpp"

void ddidimm(const int M, const double ALPHA, const double* A, const int INCA,
             double* B, const int INCB) {
  constexpr double ZERO = 0.0;

  int INFO = 0;
  if (M < 0) INFO = 1;
  if (INCA < 0) INFO = 4;
  if (INCB < 0) INFO = 6;

  if (INFO != 0) {
    // cblas_xerbla(INFO, "ddidimm", "Illegal setting, %d\n");
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
    for (int j = 0; j < M; ++j) B[INCB * j] = ZERO;
    return;
  }

  /*
   * Start the operations.
   */
  for (int j = 0; j < M; ++j) {
    B[INCB * j] *= ALPHA * A[INCA * j];
  }

}  // END OF DDIDIMM