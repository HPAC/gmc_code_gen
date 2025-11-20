#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dgesysv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const CBLAS_TRANSPOSE TransA,
                   const int M, double* A, const int ldA, double* B,
                   const int ldB) {
  char SD, ULB, TA;

  if (Layout == CblasColMajor) {
    if (Side == CblasLeft)
      SD = 'L';
    else if (Side == CblasRight)
      SD = 'R';
    else {
      // cblas_xerbla(2, "cblas_dgesysv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(3, "cblas_dgesysv", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    if (TransA == CblasNoTrans)
      TA = 'N';
    else if (TransA == CblasTrans)
      TA = 'T';
    else if (TransA == CblasConjTrans)
      TA = 'C';
    else {
      // cblas_xerbla(4, "cblas_dgesysv", "Illegal TransA setting, %d\n",
      // TransA)
      return;
    }

    dgesysv(SD, ULB, TA, M, A, ldA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dgesysv", "Illegal Layout Setting, %d\n", Layout);
  }
}