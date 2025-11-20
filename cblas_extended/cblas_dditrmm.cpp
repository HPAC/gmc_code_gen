#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dditrmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const CBLAS_DIAG DiagB, const int M,
                   const double alpha, const double* A, const int INCA,
                   double* B, const int ldB) {
  char SD, ULB, DIB;

  if (Layout == CblasColMajor) {
    if (Side == CblasLeft)
      SD = 'L';
    else if (Side == CblasRight)
      SD = 'R';
    else {
      // cblas_xerbla(2, "cblas_dditrmm", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploB == CblasLower)
      ULB = 'L';
    else if (UploB == CblasUpper)
      ULB = 'U';
    else {
      // cblas_xerbla(3, "cblas_dditrmm", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    if (DiagB == CblasUnit)
      DIB = 'U';
    else if (DiagB == CblasNonUnit)
      DIB = 'N';
    else {
      // cblas_xerbla(5, "cblas_dditrmm", "Illegal DiagB setting, %d\n", DiagB);
      return;
    }

    dditrmm(SD, ULB, DIB, M, alpha, A, INCA, B, ldB);
  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dditrmm", "Illegal Layout setting, %d\n", Layout);
  }
}