#include <algorithm>

#include "extended_base.hpp"

void ddisymm(const char SIDE, const char UPLO, const int M, const double ALPHA,
             const double* A, const int INCA, double* B, const int LDB) {
  constexpr double ZERO = 0.0;

  bool DIAG_LSIDE = SIDE == 'L';
  bool UPPER_B = UPLO == 'U';

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLO != 'U') && (UPLO != 'L'))
    INFO = 2;
  else if (M < 0)
    INFO = 3;
  else if (INCA < 0)
    INFO = 6;
  else if (LDB < std::max<int>(1, M))
    INFO = 8;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases
    // cblas_xerbla(INFO, "ddisymm", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  /*
   * And when ALPHA == ZERO all elements in B are set to ZERO (up and low parts)
   */
  if (ALPHA == ZERO) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < M; ++i) B[j * LDB + i] = ZERO;
    }
  }

  /*
   * Start the operations.
   */
  double* CPY_A = (double*)malloc(M * sizeof(double));
  for (int i = 0; i < M; i++) {
    CPY_A[i] = ALPHA * A[i * INCA];
  }

  if (DIAG_LSIDE) {
    /*
     * Form C := ALPHA*A*B
     */
    if (UPPER_B) {
      // A is on the left and B is upper.
      for (int j = 0; j < M; j++) {
        for (int i = 0; i <= j; i++) {
          B[j * LDB + i] *= CPY_A[i];
        }
        for (int i = j + 1; i < M; i++) {
          B[j * LDB + i] = B[i * LDB + j] * CPY_A[i];
        }
      }
    } else {
      // A is on the left and B is lower.
      for (int i = 0; i < M; i++) {
        for (int j = 0; j <= i; j++) {
          B[j * LDB + i] *= CPY_A[i];
        }
        for (int j = i + 1; j < M; j++) {
          B[j * LDB + i] = B[i * LDB + j] * CPY_A[i];
        }
      }
    }
  } else {
    /*
     * Form C := ALPHA*B*A
     */
    if (UPPER_B) {
      // A is on the right and B is upper.
      for (int j = 0; j < M; j++) {
        for (int i = 0; i <= j; i++) {
          B[j * LDB + i] *= CPY_A[j];
        }
        for (int i = j + 1; i < M; ++i) {
          B[j * LDB + i] = B[i * LDB + j] * CPY_A[j];
        }
      }
    } else {
      // A is on the right and B is lower.
      for (int i = 0; i < M; i++) {
        for (int j = 0; j <= i; j++) {
          B[j * LDB + i] *= CPY_A[j];
        }
        for (int j = i + 1; j < M; j++) {
          B[j * LDB + i] = B[i * LDB + j] * CPY_A[j];
        }
      }
    }
  }

  free(CPY_A);
}  // END OF DDISYMM