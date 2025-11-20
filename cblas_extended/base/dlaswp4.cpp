#include <openblas/cblas.h>

#include <cmath>

void dlaswp4(const char UPLO, const int M, const int N, double* A,
             const int LDA, const int* IPIV, const int INCX) {
  int kp, K1, K2;
  if (INCX == 1) {
    K1 = 0;
    K2 = N;
  } else if (INCX == -1) {
    K1 = N - 1;
    K2 = -1;
  } else
    return;

  if (UPLO == 'U') {
    // The permutation comes from a factorisation where the symmetric matrix is
    // upper-stored.
    for (int k = K1; k != K2; k += INCX) {
      if (IPIV[k] > 0) {
        // 1x1 diagonal block.
        kp = (IPIV[k] - 1);
        if (kp != k) {
          cblas_dswap(M, A + LDA * k, 1, A + LDA * kp, 1);
        }

      } else {
        // 2x2 diagonal block.
        kp = -(IPIV[k] + 1);
        if (kp == -(IPIV[k + INCX] + 1)) {
          cblas_dswap(M, A + LDA * std::min<int>(k, k + INCX), 1, A + LDA * kp,
                      1);
        }
        k += INCX;
      }
    }
  }

  else if (UPLO == 'L') {
    // The permutation comes from a factorisation where the symmetric matrix is
    // lower-stored.
    for (int k = K1; k != K2; k += INCX) {
      if (IPIV[k] > 0) {
        // 1x1 diagonal block.
        kp = (IPIV[k] - 1);
        if (kp != k) {
          cblas_dswap(M, A + LDA * k, 1, A + LDA * kp, 1);
        }
      } else {
        // 2x2 diagonal block.
        kp = -(IPIV[k] + 1);
        if (kp == -(IPIV[k + INCX] + 1)) {
          cblas_dswap(M, A + LDA * std::max<int>(k, k + INCX), 1, A + LDA * kp,
                      1);
        }
        k += INCX;
      }
    }
  }
}