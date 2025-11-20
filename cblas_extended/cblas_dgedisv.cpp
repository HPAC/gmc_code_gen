#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dgedisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_TRANSPOSE TransA, const int M, double* A,
                   const int ldA, const double* D, const int IncD, double* B,
                   const int ldB) {
  char SD, TA;

  if (Layout == CblasColMajor) {
    if (Side == CblasLeft)
      SD = 'L';
    else if (Side == CblasRight)
      SD = 'R';
    else {
      // cblas_xerbla(2, "cblas_dtrdisv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (TransA == CblasTrans)
      TA = 'T';
    else if (TransA == CblasConjTrans)
      TA = 'C';
    else if (TransA == CblasNoTrans)
      TA = 'N';
    else {
      // cblas_xerbla(3, "cblas_dtrdisv", "Illegal TransA setting, %d\n",
      // TransA);
      return;
    }

    dgedisv(SD, TA, M, A, ldA, D, IncD, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dgedisv", "Illegal Layout setting, %d\n", Layout);
  }
}
