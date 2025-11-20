#include <openblas/cblas.h>

#include <cmath>

void dlaswp3(const char UPLO, const int M, const int N, double* A,
             const int LDA, const int* IPIV, const int INCX) {
  int K1, K2;
  int kp;
  if (INCX == 1) {
    K1 = 0;
    K2 = M;
  } else if (INCX == -1) {
    K1 = M - 1;
    K2 = -1;
  } else
    return;

  if (UPLO == 'U') {
    // the permutation comes from a factorisation where the symmetric matrix is
    // upper-stored.
    for (int k = K1; k != K2; k += INCX) {
      if (IPIV[k] > 0) {
        // 1x1 diagonal block.
        kp = (IPIV[k] - 1);
        if (kp != k) {
          cblas_dswap(N, A + k, LDA, A + kp, LDA);
        }
      } else {
        // 2x2 diagonal block.
        kp = -(IPIV[k] + 1);
        if (kp == -(IPIV[k + INCX] + 1)) {
          cblas_dswap(N, A + std::min<int>(k, k + INCX), LDA, A + kp, LDA);
        }
        k += INCX;
      }
    }
  } else {
    // the permutation comes from a factorisation where the symmetric matrix is
    // lower-stored.
    for (int k = K1; k != K2; k += INCX) {
      if (IPIV[k] > 0) {
        // 1x1 diagonal block.
        kp = (IPIV[k] - 1);
        if (kp != k) {
          cblas_dswap(N, A + k, LDA, A + kp, LDA);
        }
      } else {
        // 2x2 diagonal block.
        kp = -(IPIV[k] + 1);
        if (kp == -(IPIV[k + INCX] + 1)) {
          cblas_dswap(N, A + std::max<int>(k, k + INCX), LDA, A + kp, LDA);
        }
        k += INCX;
      }
    }
  }
}