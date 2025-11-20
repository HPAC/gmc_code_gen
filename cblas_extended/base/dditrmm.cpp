#include <algorithm>

#include "extended_base.hpp"

void dditrmm(const char SIDE, const char UPLOB, const char DIAGB, const int M,
             const double ALPHA, const double* A, const int INCA, double* B,
             const int LDB) {
  constexpr double ZERO = 0.0;

  double TEMP;

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
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dditrmm", "Illegal setting, %d\n");
    return;
  }

  // RESET_COUNT

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  /*
   * And when ALPHA == 0
   * @warning: zeroing the whole matrix
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
  if (DIAG_LSIDE) {
    /*
     * Form B := ALPHA * A * B
     */
    if (UPPER_B) {
      // B is upper.
      for (int i = 0; i < M; ++i) {
        TEMP = ALPHA * A[INCA * i];
        for (int j = i; j < M; ++j) {
          B[j * LDB + i] *= TEMP;
        }
      }
    } else {
      // B is lower.
      for (int i = 0; i < M; ++i) {
        TEMP = ALPHA * A[INCA * i];
        for (int j = 0; j <= i; ++j) {
          B[j * LDB + i] *= TEMP;
        }
      }
    }
  } else {
    /*
     * Form B := ALPHA * B * A
     */
    if (UPPER_B) {
      // B is upper.
      for (int j = 0; j < M; ++j) {
        TEMP = ALPHA * A[INCA * j];
        for (int i = 0; i <= j; ++i) {
          B[j * LDB + i] *= TEMP;
        }
      }
    } else {
      // B is lower.
      for (int j = 0; j < M; ++j) {
        TEMP = ALPHA * A[INCA * j];
        for (int i = j; i < M; ++i) {
          B[j * LDB + i] *= TEMP;
        }
      }
    }
  }
}  // END OF DDITRMM