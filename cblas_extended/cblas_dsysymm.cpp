#include <openblas/cblas.h>

#include "base/extended_base.hpp"

void cblas_dsysymm(const CBLAS_LAYOUT Layout, const CBLAS_UPLO UploA,
                   const CBLAS_UPLO UploB, const int M, const double alpha,
                   double* A, const int ldA, double* B, const int ldB,
                   const double beta, double* C, const int ldC) {
  char ULA, ULB;

  if (Layout == CblasColMajor) {
    if (UploA == CblasUpper)
      ULA = 'U';
    else if (UploA == CblasLower)
      ULA = 'L';
    else {
      // cblas_xerbla(2, "cblas_dsysymm", "Illegal UploA setting, %d\n", UploA);
      return;
    }

    if (UploB == CblasUpper)
      ULB = 'U';
    else if (UploB == CblasLower)
      ULB = 'L';
    else {
      // cblas_xerbla(3, "cblas_dsysymm", "Illegal UploB setting, %d\n", UploB);
      return;
    }

    dsysymm(ULA, ULB, M, alpha, A, ldA, B, ldB, beta, C, ldC);

  } else if (Layout == CblasRowMajor) {
    // @todo
  } else {
    // cblas_xerbla(1, "cblas_dsysymm", "Illegal Layout setting, %d\n", Layout);
  }
}