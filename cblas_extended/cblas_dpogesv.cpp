#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dpogesv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const int M, const int N, double* A,
                   const int ldA, double* B, const int ldB) {
  char SD, ULA;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_dpogesv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(3, "cblas_dpogesv", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    dpogesv(SD, ULA, M, N, A, ldA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dpogesv", "Illegal Layout setting, %d\n", Layout);
  }
}
