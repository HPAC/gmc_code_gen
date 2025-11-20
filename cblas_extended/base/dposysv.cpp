#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <algorithm>

#include "extended_base.hpp"

/**
 * @brief Positive Definite(A)-Symmetric(B) system solver.
 *
 * @param SIDE 'L' or 'R'.
 * @param UPLO 'U' or 'L'.
 *
 * A is an MxM positive-definite matrix.
 * B is an MxN symmetric matrix.
 *
 * On exit:
 *    A contains the Cholesky factorisation.
 *    B is a full matrix containing the result of the computation.
 */
void dposysv(const char SIDE, const char UPLOA, const char UPLOB, const int M,
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
    // cblas_xerbla(INFO, "dposysv", "Illegal setting, %d\n");
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
  dpogesv(SIDE, UPLOA, M, M, A, LDA, B, LDB);
}  // END OF DPOSYSV