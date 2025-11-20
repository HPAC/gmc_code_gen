#include <openblas/cblas.h>
#include <openblas/lapacke.h>

#include <cmath>

#include "extended_base.hpp"

/**
 * if SIDE = 'L':  Form is X = inv(A) * B
 * A is M x M -- B (symmetric) is M x M
 *
 * Otherwise: Form is X = B * inv(A)
 * B (symm) is M x M -- A is M x M
 */

void dgesysv(const char SIDE, const char UPLOB, const char TRANSA, const int M,
             double* A, const int LDA, double* B, const int LDB) {
  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 2;
  else if ((TRANSA != 'N') && (TRANSA != 'C') && (TRANSA != 'T'))
    INFO = 3;
  else if (M < 0)
    INFO = 4;
  else if (LDA < std::max<int>(1, M))
    INFO = 6;
  else if (LDB < std::max<int>(1, M))
    INFO = 8;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dgesysv", "Illegal setting, %d\n");
    return;
  }

  /*
   * Quick return if possible.
   */
  if (M == 0) return;

  /*
   * Start the operations.
   */
  sy2full(UPLOB, M, B, LDB);  // convert B to a full matrix
  dgegesv(SIDE, TRANSA, M, M, A, LDA, B, LDB);
}  // END OF DGESYSV