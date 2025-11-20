#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_ddisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const int M,
                 const int N, const double alpha, const double* A,
                 const int IncA, double* B, const int ldB) {
  char SD;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_ddism", "Illegal Side setting, %d\n", Side);
      return;
    }

    ddisv(SD, M, N, alpha, A, IncA, B, ldB);
  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_ddism", "Illegal Layout setting, %d\n", Layout);
  }
}