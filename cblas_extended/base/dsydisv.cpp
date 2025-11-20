#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

void dsydisv(const char SIDE, const char UPLO, const int M, double* A,
             const int LDA, const double* D, const int INCD, double* B,
             const int LDB) {
  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLO != 'U') && (UPLO != 'L'))
    INFO = 2;
  else if (M < 0)
    INFO = 3;
  else if (LDA < std::max<int>(1, M))
    INFO = 5;
  else if (INCD < 0)
    INFO = 7;
  else if (LDB < std::max<int>(1, M))
    INFO = 9;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dsydisv", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  for (int i = 0; i < M; i++) B[i * LDB + i] = D[i * INCD];
  dsygesv(SIDE, UPLO, M, M, A, LDA, B, LDB);
}