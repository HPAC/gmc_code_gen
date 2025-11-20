#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

/**
 * @brief Symmetric(A)-Symmetric(B) system solver.
 *
 * @param SIDE   'L' or 'R'.
 * @param UPLOA  'U' or 'L'.
 * @param UPLOB  'U' or 'L'.
 * @param M      rows and columns of A and B
 *
 * Every element of B will be referenced.
 *
 * On exit:
 *    A contains the permuted LDLT factorisation.
 *    B holds the result of the computation.
 */
void dsysysv(const char SIDE, const char UPLOA, const char UPLOB, const int M,
             double* A, const int LDA, double* B, const int LDB) {
  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLOA != 'U') && (UPLOA != 'L'))
    INFO = 2;
  else if ((UPLOB != 'U') && (UPLOB != 'L'))
    INFO = 3;
  else if (M < 0)
    INFO = 4;
  else if (LDA < std::max<int>(1, M))
    INFO = 6;
  else if (LDB < std::max<int>(1, M))
    INFO = 8;

  if (INFO != 0) {
    // @todo check this is correct.
    // cblas_xerbla(INFO, "dsysysv", "Illegal setting, %d\n");
    return;
  }

  /*
   *  Quick return if possible.
   */
  if (M == 0) return;

  /*
   *  Start the operations.
   */
  sy2full(UPLOB, M, B, LDB);
  dsygesv(SIDE, UPLOA, M, M, A, LDA, B, LDB);
}