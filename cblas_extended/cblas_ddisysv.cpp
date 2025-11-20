#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_ddisysv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const int M, const double alpha,
                   const double* A, const int IncA, double* B, const int ldB) {
  char SD, ULB;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_ddisysm", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(3, "cblas_ddisysm", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    ddisysv(SD, ULB, M, alpha, A, IncA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_ddisysm", "Illegal Layout setting, %d\n", Layout);
  }
}