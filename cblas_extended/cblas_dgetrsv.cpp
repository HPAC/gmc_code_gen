#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dgetrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const CBLAS_TRANSPOSE TransA,
                   const CBLAS_DIAG DiagB, const int M, double* A,
                   const int ldA, double* B, const int ldB) {
  char SD, ULB, TA, DIB;

  if (Layout == CblasColMajor) {
    if (Side == CblasLeft)
      SD = 'L';
    else if (Side == CblasRight)
      SD = 'R';
    else {
      // cblas_xerbla(2, "cblas_dgetrsv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(3, "cblas_dgetrsv", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    if (TransA == CblasTrans)
      TA = 'T';
    else if (TransA == CblasConjTrans)
      TA = 'C';
    else if (TransA == CblasNoTrans)
      TA = 'N';
    else {
      // cblas_xerbla(4, "cblas_dgetrsv", "Illegal TransA setting, %d\n",
      // TransA);
      return;
    }

    if (DiagB == CblasUnit)
      DIB = 'U';
    else if (DiagB == CblasNonUnit)
      DIB = 'N';
    else {
      // cblas_xerbla(5, "cblas_dgetrsv", "Illegal DiagB setting, %d\n", DiagB);
      return;
    }

    dgetrsv(SD, ULB, TA, DIB, M, A, ldA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dgetrsv", "Illegal Layout setting, %d\n", Layout);
  }
}

// void dgetrsv(const char SIDE, const char UPLOB, const char TRANSA,
//              const char DIAGB, const int M, double* A, const int LDA, double*
//              B, const int LDB)