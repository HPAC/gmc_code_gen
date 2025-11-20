#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dtrtrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB,
                   const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG DiagA,
                   const CBLAS_DIAG DiagB, const int M, const double alpha,
                   double* A, const int ldA, double* B, const int ldB) {
  char SD, ULA, ULB, TA, DIA, DIB;

  if (Layout == CblasColMajor) {
    if (Side == CblasLeft)
      SD = 'L';
    else if (Side == CblasRight)
      SD = 'R';
    else {
      // cblas_xerbla(2, "cblas_dtrtrsv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(3, "cblas_dtrtrsv", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(4, "cblas_dtrtrsv", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    if (TransA == CblasTrans)
      TA = 'T';
    else if (TransA == CblasConjTrans)
      TA = 'C';
    else if (TransA == CblasNoTrans)
      TA = 'N';
    else {
      // cblas_xerbla(5, "cblas_dtrtrsv", "Illegal TransA setting, %d\n",
      // TransA);
      return;
    }

    if (DiagA == CblasUnit)
      DIA = 'U';
    else if (DiagA == CblasNonUnit)
      DIA = 'N';
    else {
      // cblas_xerbla(6, "cblas_dtrtrsv", "Illegal DiagA setting, %d\n", DiagA);
      return;
    }

    if (DiagB == CblasUnit)
      DIB = 'U';
    else if (DiagB == CblasNonUnit)
      DIB = 'N';
    else {
      // cblas_xerbla(7, "cblas_dtrtrsv", "Illegal DiagB setting, %d\n", DiagB);
      return;
    }

    dtrtrsv(SD, ULA, ULB, TA, DIA, DIB, M, alpha, A, ldA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dtrtrsv", "Illegal Layout setting, %d\n", Layout);
  }
}