#include <openblas/cblas.h>

#include <cmath>

/**
 * @brief Block-Diagonal-General matrix solver.
 *
 * @param A Upper or lower matrix that contains the main diagonal of the
 * block-diagonal matrix in its own diagonal.
 * @param E Array that contains the sup/sub-diagonal of the block-diagonal.
 * @param B General matrix.
 *
 * if SIDE == 'L':
 *    A is an MxM matrix.
 *    XPIV is an M-array
 * if SIDE == 'R':
 *    A is an NxN matrix.
 *    XPIV is an N-array
 * B is an MxN matrix.
 */
void dbdigesv(const char SIDE, const char UPLO, const int M, const int N,
              const double* A, const int LDA, const double* E, const int INCE,
              const int* XPIV, const int INCX, double* B, const int LDB) {
  const bool LHS_LEFT = SIDE == 'L';
  const bool UPPER_A = UPLO == 'U';

  const int NROWA = (SIDE == 'L') ? M : N;

  double a, b, d, denom, temp;

  int INFO = 0;
  if ((SIDE != 'L') && (SIDE != 'R'))
    INFO = 1;
  else if ((UPLO != 'U') && (UPLO != 'L'))
    INFO = 2;
  else if (M < 0)
    INFO = 3;
  else if (N < 0)
    INFO = 4;
  else if (LDA < std::max<int>(1, NROWA))
    INFO = 6;
  else if (INCE < 0)
    INFO = 8;
  else if (INCX < 0)
    INFO = 10;
  else if (LDB < std::max<int>(1, M))
    INFO = 12;

  if (INFO != 0) {
    // @todo check this is correct.
    // cblas_xerbla(INFO, "dbdigesv", "Illegal setting, %d\n");
    return;
  }

  /*
   *  Quick return if possible.
   */
  if (M == 0 || N == 0) return;

  /*
   * A = [a  b]
   *     [b  d]
   *
   * A^{-1} = 1 / (a*d - b*b) * [d  -b]
   *                            [-b  a]
   */

  if (LHS_LEFT) {
    /*
     *  Form: B := D^{-1} * B.
     */
    for (unsigned i = 0; i < static_cast<unsigned>(M); i++) {
      if (XPIV[i * INCX] > 0) {
        // 1x1 diagonal block -- in A(i,i). Invert and scale B(i,:)
        cblas_dscal(N, 1.0 / A[i * LDA + i], B + i, LDB);
      } else {
        if (XPIV[i * INCX] == XPIV[(i + 1) * INCX]) {
          // 2x2 diagonal block -- in A(i,i), A(i+1, i+1), and E(i+1) for
          // upper or E(i) for lower.
          a = A[i * LDA + i];
          d = A[(i + 1) * LDA + i + 1];
          b = UPPER_A ? E[i + 1] : E[i];
          denom = a * d - b * b;
          for (unsigned j = 0; j < static_cast<unsigned>(N); j++) {
            temp = B[j * LDB + i];
            B[j * LDB + i] =
                (d * B[j * LDB + i] - b * B[j * LDB + i + 1]) / denom;
            B[j * LDB + i + 1] = (a * B[j * LDB + i + 1] - b * temp) / denom;
          }
          i++;
        }
      }
    }
  } else {
    /*
     *  Form: B := B * D^{-1}.
     */
    for (unsigned j = 0; j < static_cast<unsigned>(N); j++) {
      if (XPIV[j * INCX] > 0) {
        // 1x1 diagonal block -- in A(j,j). Invert and scale B(:,j)
        cblas_dscal(M, 1.0 / A[j * LDA + j], B + LDB * j, 1);
      } else {
        if (XPIV[j * INCX] == XPIV[(j + 1) * INCX]) {
          // 2x2 diagonal block -- in A(j,j), A(j+1,j+1), and E(j+1) for upper
          // or E(j) for lower.
          a = A[j * LDA + j];
          d = A[(j + 1) * LDA + j + 1];
          b = UPPER_A ? E[j + 1] : E[j];
          denom = a * d - b * b;
          for (unsigned i = 0; i < static_cast<unsigned>(M); i++) {
            temp = B[j * LDB + i];
            B[j * LDB + i] =
                (d * B[j * LDB + i] - b * B[(j + 1) * LDB + i]) / denom;
            B[(j + 1) * LDB + i] =
                (a * B[(j + 1) * LDB + i] - b * temp) / denom;
          }
          j++;
        }
      }
    }
  }
}