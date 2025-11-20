#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_ddimm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const int M,
                 const int N, const double alpha, const double* A,
                 const int IncA, double* B, const int ldB) {
  char SD;

  if (Layout == CblasColMajor) {
    if (Side == CblasRight)
      SD = 'R';
    else if (Side == CblasLeft)
      SD = 'L';
    else {
      // cblas_xerbla(2, "cblas_ddimm", "Illegal Side setting, %d\n", Side);
      return;
    }

    ddimm(SD, M, N, alpha, A, IncA, B, ldB);
  } else if (Layout == CblasRowMajor) {
    // @todo

    // ddimm();
  } else {
    // cblas_xerbla(1, "cblas_ddimm", "Illegal Layout setting, %d\n", Layout);
    return;
  }
}