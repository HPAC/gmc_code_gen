#include <algorithm>

#include "extended_base.hpp"

void dditrsv(const char SIDE, const char UPLOB, const char DIAGB, const int M,
             const double ALPHA, const double* A, const int INCA, double* B,
             const int LDB) {
  constexpr double ZERO = 0.0;

  bool DIAG_LSIDE = SIDE == 'L';
  bool UPPER_B = UPLOB == 'U';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 2;
  else if ((DIAGB != 'U') && (DIAGB != 'N'))
    INFO = 3;
  else if (M < 0)
    INFO = 4;
  else if (INCA < 0)
    INFO = 7;
  else if (LDB < std::max<int>(1, M))
    INFO = 9;

  if (INFO != 0) {
    // @todo check if this is correct.
    // cblas_xerbla(INFO, "dditrsm", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  /*
   * And when ALPHA == ZERO
   * @warning: zeroing the whole matrix
   */
  if (ALPHA == ZERO) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < M; ++i) B[j * LDB + i] = ZERO;
    }
    return;
  }

  /*
   * Allocate memory for A^{-1}
   */
  double* INV_A = (double*)malloc(M * sizeof(double));

  /*
   * Compute INV_A := ALPHA * A^{-1}
   */
  for (int i = 0; i < M; ++i) INV_A[i] = ALPHA / A[INCA * i];

  /*
   * Start the operations.
   */
  if (DIAG_LSIDE) {
    /*
     * Form: B := ALPHA * A^{-1} * B
     */
    if (UPPER_B) {
      // B is upper.
      for (int j = 0; j < M; ++j) {
        for (int i = 0; i <= j; ++i) B[j * LDB + i] *= INV_A[i];
      }
    } else {
      // B is lower.
      for (int j = 0; j < M; ++j) {
        for (int i = j; i < M; ++i) B[j * LDB + i] *= INV_A[i];
      }
    }
  } else {
    /*
     * Form: B := ALPHA * B * A^{-1}
     */
    if (UPPER_B) {
      // B is on the left-side, upper, and not transposed.
      for (int j = 0; j < M; ++j) {
        for (int i = 0; i <= j; ++i) B[j * LDB + i] *= INV_A[j];
      }
    } else {
      // B is on the left-side, lower, and not transposed.
      for (int j = 0; j < M; ++j) {
        for (int i = j; i < M; ++i) B[j * LDB + i] *= INV_A[j];
      }
    }
  }

  free(INV_A);
}  // END OF DDITRSM