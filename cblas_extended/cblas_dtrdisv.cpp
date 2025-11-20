#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dtrdisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_TRANSPOSE TransA,
                   const CBLAS_DIAG DiagA, const int M, const double Alpha,
                   const double* A, const int ldA, const double* D,
                   const int IncD, double* B, const int ldB) {
  char SD, ULA, TA, DIA;

  if (Layout == CblasColMajor) {
    if (Side == CblasLeft)
      SD = 'L';
    else if (Side == CblasRight)
      SD = 'R';
    else {
      // cblas_xerbla(2, "cblas_dtrdisv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(3, "cblas_dtrdisv", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    if (TransA == CblasTrans)
      TA = 'T';
    else if (TransA == CblasConjTrans)
      TA = 'C';
    else if (TransA == CblasNoTrans)
      TA = 'N';
    else {
      // cblas_xerbla(4, "cblas_dtrdisv", "Illegal TransA setting, %d\n",
      // TransA);
      return;
    }

    if (DiagA == CblasUnit)
      DIA = 'U';
    else if (DiagA == CblasNonUnit)
      DIA = 'N';
    else {
      // cblas_xerbla(5, "cblas_dtrdisv", "Illegal DiagA setting, %d\n", DiagA);
      return;
    }

    dtrdisv(SD, ULA, TA, DIA, M, Alpha, A, ldA, D, IncD, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dtrdisv", "Illegal Layout setting, %d\n", Layout);
  }
}
