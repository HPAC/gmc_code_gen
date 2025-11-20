#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dgegesv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_TRANSPOSE TransA, const int M, const int N,
                   double* A, const int ldA, double* B, const int ldB) {
  char SD, TA;

  if (Layout == CblasColMajor) {
    if (Side == CblasLeft)
      SD = 'L';
    else if (Side == CblasRight)
      SD = 'R';
    else {
      // cblas_xerbla(2, "cblas_dgegesv", "Illegal Side setting, %d\n", Side);
      return;
    }

    if (TransA == CblasNoTrans)
      TA = 'N';
    else if (TransA == CblasTrans)
      TA = 'T';
    else if (TransA == CblasConjTrans)
      TA = 'C';
    else {
      // cblas_xerbla(3, "cblas_dgegesv", "Illegal TransA setting, %d\n",
      // TransA)
      return;
    }

    dgegesv(SD, TA, M, N, A, ldA, B, ldB);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dgegesv", "Illegal Layout Setting, %d\n", Layout);
  }
}