#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dtrsymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB,
                   const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                   const int M, const double alpha, const double* A,
                   const int ldA, double* B, const int ldB) {
  char SD, ULA, ULB, TA, DIA;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_dtrsymm", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(3, "cblas_dtrsymm", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(4, "cblas_dtrsymm", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    if (TransA == CblasTrans)
      TA = 'T';
    else if (TransA == CblasConjTrans)
      TA = 'C';
    else if (TransA == CblasNoTrans)
      TA = 'N';
    else {
      // cblas_xerbla(5, "cblas_dtrsymm", "Illegal TransA setting, %d\n",
      // TransA);
      return;
    }

    if (Diag == CblasUnit)
      DIA = 'U';
    else if (Diag == CblasNonUnit)
      DIA = 'N';
    else {
      // cblas_xerbla(6, "cblas_dtrsymm", "Illegal Diag setting, %d\n", Diag);
      return;
    }

    dtrsymm(SD, ULA, ULB, TA, DIA, M, alpha, A, ldA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dtrsymm", "Illegal Layout setting, %d\n", Layout);
  }
}