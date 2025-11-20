#include <omp.h>

#include <cstdlib>
#include <string>

/*
 * Performs a parallel copy of a matrix.
 *
 * M: integer. Number of rows of A and B.
 *
 * N: integer. Number of columns of A and B.
 *
 * A: double precision array, dimension (LDA, N). The input matrix, whose copy
 * we place onto B.
 *
 * LDA: integer. The leading dimension of the array A.
 *
 * B: double precision array, dimension (LDB, N). The output matrix, with the
 * copy of A.
 *
 * LDB: integer. The leading dimension of the array B.
 */

void dlacpy(const int M, const int N, const double* const A, const int LDA,
            double* const B, const int LDB) {
  int INFO = 0;
  if (M < 0)
    INFO = 1;
  else if (N < 0)
    INFO = 2;
  else if (LDA < std::max<int>(1, M))
    INFO = 4;
  else if (LDB < std::max<int>(1, M))
    INFO = 6;

  if (INFO != 0) {
    // @todo check if this is correct. cblas_xerbla has a fixed set of cases.
    // cblas_xerbla(INFO, "dge_trans", "Illegal setting, %d\n");
  }

  /*
   * Start operations
   */
  unsigned n_threads = std::stoi(std::getenv("OMP_NUM_THREADS"));
  unsigned PW = N / n_threads;  // PW = Panel Width
  if (PW <= 2) {
    for (int j = 0U; j < N; j++) {
      for (int i = 0U; i < M; i++) B[j * LDB + i] = A[j * LDA + i];
    }
  } else {
#pragma omp parallel for schedule(static, PW)
    for (int j = 0U; j < N; j++) {
      for (int i = 0U; i < M; i++) B[j * LDB + i] = A[j * LDA + i];
    }
  }
}