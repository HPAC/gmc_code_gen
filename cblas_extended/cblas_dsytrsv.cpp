#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dsytrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB,
                   const CBLAS_DIAG DiagB, const int M, double* A,
                   const int ldA, double* B, const int ldB) {
  char SD, ULA, ULB, DIB;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_dsytrsv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(3, "cblas_dsytrsv", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(4, "cblas_dsytrsv", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    if (DiagB == CblasUnit)
      DIB = 'U';
    else if (DiagB == CblasNonUnit)
      DIB = 'N';
    else {
      // cblas_xerbla(5, "cblas_dsytrsv", "Illegal DiagB setting, %d\n", DiagB);
      return;
    }

    dsytrsv(SD, ULA, ULB, DIB, M, A, ldA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dsytrsv", "Illegal Layout setting, %d\n", Layout);
  }
}
