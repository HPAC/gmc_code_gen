#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dtrtrmm(const CBLAS_LAYOUT Layout, const CBLAS_UPLO UploA,
                   const CBLAS_UPLO UploB, const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB, const CBLAS_DIAG DiagA,
                   const CBLAS_DIAG DiagB, const int M, const double alpha,
                   const double* A, const int ldA, const double* B,
                   const int ldB, double* C, const int ldC) {
  char ULA, ULB, TA, TB, DIA, DIB;

  if (Layout == CblasColMajor) {
    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(2, "cblas_dtrtrmm", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(3, "cblas_dtrtrmm", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    if (TransA == CblasTrans)
      TA = 'T';
    else if (TransA == CblasConjTrans)
      TA = 'C';
    else if (TransA == CblasNoTrans)
      TA = 'N';
    else {
      // cblas_xerbla(4, "cblas_dtrtrmm", "Illegal TransA setting, %d\n",
      // TransA);
      return;
    }

    if (TransB == CblasTrans)
      TB = 'T';
    else if (TransB == CblasConjTrans)
      TB = 'C';
    else if (TransB == CblasNoTrans)
      TB = 'N';
    else {
      // cblas_xerbla(5, "cblas_dtrtrmm", "Illegal TransB setting, %d\n",
      // TransB);
      return;
    }

    if (DiagA == CblasUnit)
      DIA = 'U';
    else if (DiagA == CblasNonUnit)
      DIA = 'N';
    else {
      // cblas_xerbla(6, "cblas_dtrtrmm", "Illegal DiagA setting, %d\n", DiagA);
      return;
    }

    if (DiagB == CblasUnit)
      DIB = 'U';
    else if (DiagB == CblasNonUnit)
      DIB = 'N';
    else {
      // cblas_xerbla(7, "cblas_dtrtrmm", "Illegal DiagB setting, %d\n", DiagB);
      return;
    }

    dtrtrmm(ULA, ULB, TA, TB, DIA, DIB, M, alpha, A, ldA, B, ldB, C, ldC);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dtrtrmm", "Illegal Layout setting, %d\n", Layout);
  }
}