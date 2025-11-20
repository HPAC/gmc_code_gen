#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_ddidisv(const CBLAS_LAYOUT Layout, const int M, const double alpha,
                   const double* A, const int IncA, double* B, const int IncB) {
  if (Layout == CblasColMajor || Layout == CblasRowMajor) {
    ddidisv(M, alpha, A, IncA, B, IncB);
  } else {
    // cblas_xerbla(1, "cblas_ddidism", "Illegal Layout setting, %d\n", Layout);
  }
}