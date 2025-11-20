#include <omp.h>

/*
 * SY2FULL copies the contents of the symmetric matrix onto the 'empty' half,
 * making it a full matrix with symmetry.
 *
 * UPLO: char. UPLO specifies whether the upper or lower triangular part of
 * the symmetric matrix A contains the elements of A.
 *
 * M: integer. Number of rows/columns of A.
 *
 * A: double precision array, dimension (LDA, M). On entry, the symmetric matrix
 * whose elements we want to duplicate and copy onto the 'emtpy' triangular
 * half.
 *
 * LDA: integer. The leading dimension of the array A.
 */
void sy2full(const char UPLO, const int M, double* A, const int LDA) {
  bool UPPER = UPLO == 'U';

  int INFO = 0;
  if ((UPLO != 'U') && (UPLO != 'L')) INFO = 1;

  if (INFO != 0) {
    // cblas_xerbla(INFO, "sy2full", "Illegal setting, %d\n");
    return;
  }

  /*
   * Start operations
   */
  if (UPPER) {
// #pragma omp parallel for schedule(dynamic, 8)
#pragma omp parallel for
    for (int j = 1U; j < M; ++j) {
      for (int i = 0U; i < j; ++i) A[i * LDA + j] = A[j * LDA + i];
    }
  } else {
// #pragma omp parallel for schedule(dynamic, 8)
#pragma omp parallel for
    for (int j = 0; j < M; ++j) {
      for (int i = j + 1; i < M; ++i) A[i * LDA + j] = A[j * LDA + i];
    }
  }
}