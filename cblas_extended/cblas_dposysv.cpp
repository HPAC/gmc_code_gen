#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dposysv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB, const int M,
                   double* A, const int ldA, double* B, const int ldB) {
  char SD, ULA, ULB;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_dposysv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(3, "cblas_dposysv", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(4, "cblas_dposysv", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    dposysv(SD, ULA, ULB, M, A, ldA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dposysv", "Illegal Layout setting, %d\n", Layout);
  }
}
