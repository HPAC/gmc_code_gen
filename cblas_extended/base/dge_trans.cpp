#include <omp.h>

#include <algorithm>

/*
 * Performs an out-of-place explicit transpose.
 *
 * M: integer. Number of rows of A.
 *
 * N: integer. Number of columns of A.
 *
 * A: double precision array, dimension (LDA, N). The input matrix, whose
 * transpose we place onto B.
 *
 * LDA: integer. The leading dimension of the array A.
 *
 * B: double precision array, dimension (LDB, M). The output matrix, with the
 * result of the transpose of A.
 *
 * LDB: integer. The leading dimension of the array B.
 */
void dge_trans(const int M, const int N, const double* const A, const int LDA,
               double* B, const int LDB) {
  int INFO = 0;
  if (M < 0)
    INFO = 1;
  else if (N < 0)
    INFO = 2;
  else if (LDA < std::max<int>(1, M))
    INFO = 4;
  else if (LDB < std::max<int>(1, N))
    INFO = 6;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dge_trans", "Illegal setting, %d\n");
    return;
  }

/*
 * Start operations
 */
#pragma omp parallel for
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++) B[i * LDB + j] = A[j * LDA + i];
  }
}