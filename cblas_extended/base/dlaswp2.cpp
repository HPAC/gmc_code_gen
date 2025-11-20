/*
 * DLASWP2 performs a series of column interchanges on the matrix A.
 * One column interchange is initiated for each of columns K1 through K2 of A.
 *
 * Important: when the array of pivots (JPIV) comes from an LU factorization,
 * to perform the multiplication A := A * P, the column interchanges
 * must be performed in reverse order (this is, INCX = -1). To perform the
 * multiplication A := A * P**T, the column interchanges must be performed in
 * forward order (this is, INCX = 1).
 *
 * M: number of rows of the matrix A.
 *
 * A: double precision array, dimension (LDA, *). On entry, the matrix of row
 * dimension M to which the column interchanges will be applied. On exit, the
 * permuted matrix.
 *
 * LDA: integer. The leading dimension of the array A.
 *
 * K1: integer. The first element of JPIV for which a column interchange will
 * be done.
 *
 * K2: integer. (K2-K1+1) is the number of elements of JPIV for which a column
 * interchange will be done.
 *
 * JPIV: integer array, dimension (K1+(K2-K1)*abs(INCX)). The vector of pivot
 * indices. Only the elements in positions K1 through K1+(K2-K1)*abs(INCX) of
 * JPIV are accessed. JPIV(K1+(K-K1)*abs(INCX)) = L implies columns K and L are
 * to be interchanged.
 *
 * INCX: integer. The increment between successive values of JPIV. If INCX is
 * negative, the pivots are applied in reverse order.
 *
 */
void dlaswp2(const int M, double* A, const int LDA, const int K1, const int K2,
             const int* JPIV, const int INCX) {
  double temp;

  int j0, jp;

  if (INCX > 0) {
    j0 = K1;

    for (int j = 0; j <= (K2 - K1); ++j) {
      jp = JPIV[j0 + j * INCX] - 1;  // adjust offset
      if (jp != (K1 + j)) {
        for (int k = 0; k < M; ++k) {
          temp = A[(K1 + j) * LDA + k];
          A[(K1 + j) * LDA + k] = A[jp * LDA + k];
          A[jp * LDA + k] = temp;
        }
      }
    }
  } else {
    j0 = K1 + (K1 - K2) * INCX;

    for (int j = 0; j <= (K2 - K1); ++j) {
      jp = JPIV[j0 + j * INCX] - 1;  // adjust offset
      if (jp != (K2 - j)) {
        for (int k = 0; k < M; ++k) {
          temp = A[(K2 - j) * LDA + k];
          A[(K2 - j) * LDA + k] = A[jp * LDA + k];
          A[jp * LDA + k] = temp;
        }
      }
    }
  }
}
