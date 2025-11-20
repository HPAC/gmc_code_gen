#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dpodisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const char UploA, const int M, double* A, const int ldA,
                   const double* D, const int IncD, double* B, const int ldB) {
  char SD, ULA;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_dpodisv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(3, "cblas_dpodisv", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    dpodisv(SD, ULA, M, A, ldA, D, IncD, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dpodisv", "Illegal Layout setting, %d\n", Layout);
  }
}
