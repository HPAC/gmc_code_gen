#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dditrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const CBLAS_DIAG DiagB, const int M,
                   const double alpha, const double* A, const int IncA,
                   double* B, const int ldB) {
  char SD, ULB, DIB;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_dditrsm", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(3, "cblas_dditrsm", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    if (DiagB == CblasUnit)
      DIB = 'U';
    else if (DiagB == CblasNonUnit)
      DIB = 'N';
    else {
      // cblas_xerbla(5, "cblas_dditrsm", "Illegal DiagB setting, %d\n", DiagB);
      return;
    }

    dditrsv(SD, ULB, DIB, M, alpha, A, IncA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dditrsm", "Illegal Layout setting, %d\n", Layout);
  }
}