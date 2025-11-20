#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_ddidimm(const CBLAS_LAYOUT Layout, const int M, const double alpha,
                   const double* A, const int IncA, double* B, const int IncB) {
  if (Layout == CblasColMajor) {
    ddidimm(M, alpha, A, IncA, B, IncB);
  } else if (Layout == CblasRowMajor) {
    ddidimm(M, alpha, A, IncA, B, IncB);
  } else {
    // cblas_xerbla(1, "cblas_ddidimm", "Illegal Layout setting, %d\n", Layout);
  }
}