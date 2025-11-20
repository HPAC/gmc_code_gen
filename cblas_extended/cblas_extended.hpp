#ifndef EXTENDED_CBLAS_H
#define EXTENDED_CBLAS_H

#include <openblas/cblas.h>

/**
 * @brief Diagonal(A)-Dense(B) matrix multiply.
 *
 * Compute B := alpha * A * B   or    B := alpha * B * A
 */
void cblas_ddimm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const int M,
                 const int N, const double alpha, const double* A,
                 const int IncA, double* B, const int ldB);

/**
 * @brief Symmetric-Symmetric matrix multiply.
 *
 * Compute C := alpha * A * B + beta * C
 */
void cblas_dsysymm(const CBLAS_LAYOUT Layout, const CBLAS_UPLO UploA,
                   const CBLAS_UPLO UploB, const int M, const double alpha,
                   double* A, const int ldA, double* B, const int ldB,
                   const double beta, double* C, const int ldC);

/**
 * @brief Triangular(A)-Symmetric(B) matrix multiply.
 *
 * Compute B := alpha * op(A) * B    or    B := alpha * B * op(A)
 */
void cblas_dtrsymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB,
                   const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
                   const int M, const double alpha, const double* A,
                   const int ldA, double* B, const int ldB);

/**
 * @brief Diagonal(A)-Symmetric(B) matrix multiply.
 *
 * Compute C := alpha * A * B    or    C := alpha * B * A
 */
void cblas_ddisymm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO Uplo, const int M, const double alpha,
                   const double* A, const int IncA, double* B, const int ldB);

/**
 * @brief Triangular-Triangular matrix multiply.
 *
 * Compute C := alpha * op(A) * op(B) + beta * C
 */
void cblas_dtrtrmm(const CBLAS_LAYOUT Layout, const CBLAS_UPLO UploA,
                   const CBLAS_UPLO UploB, const CBLAS_TRANSPOSE TransA,
                   const CBLAS_TRANSPOSE TransB, const CBLAS_DIAG DiagA,
                   const CBLAS_DIAG DiagB, const int M, const double alpha,
                   const double* A, const int ldA, const double* B,
                   const int ldB, double* C, const int ldC);

/**
 * @brief Diagonal(A)-Triangular(B) matrix multiply.
 *
 * Compute B := alpha * A * B   or    B := alpha * B * A
 */
void cblas_dditrmm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const CBLAS_DIAG DiagB, const int M,
                   const double alpha, const double* A, const int IncA,
                   double* B, const int ldB);

/**
 * @brief Diagonal-Diagonal matrix multiply.
 *
 * Compute B := alpha * A * B
 */
void cblas_ddidimm(const CBLAS_LAYOUT Layout, const int M, const double alpha,
                   const double* A, const int IncA, double* B, const int IncB);

/* ==================================================
   ================= SYSTEM SOLVERS =================
   ================================================== */

/**
 * @brief Triangular(A)-Symmetric(B) solver.
 *
 * Compute op(A) * X = alpha * B    or    X * op(A) = alpha * B
 */
void cblas_dtrsysv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB,
                   const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG DiagA,
                   const int M, const double alpha, const double* A,
                   const int ldA, double* B, const int ldB);

/**
 * @brief Triangular-Triangular solver.
 *
 * Compute op(A) * C = alpha * op(B)   or    C * op(A) = alpha * op(B)
 */
void cblas_dtrtrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB,
                   const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG DiagA,
                   const CBLAS_DIAG DiagB, const int M, const double alpha,
                   double* A, const int ldA, double* B, const int ldB);

/**
 * @brief Triangular(A)-Diagonal(B) solver. Stores computation in B.
 *
 * Compute op(A) * X = alpha * B    or    X * op(A) = alpha * B
 */
void cblas_dtrdisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_TRANSPOSE TransA,
                   const CBLAS_DIAG DiagA, const int M, const double Alpha,
                   const double* A, const int ldA, const double* D,
                   const int IncD, double* B, const int ldB);

/**
 * @brief Diagonal(A)-Dense solver.
 *
 * Compute A * X = alpha * B    or    X * A = alpha * B
 *
 * result stored in B.
 */
void cblas_ddisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side, const int M,
                 const int N, const double alpha, const double* A,
                 const int IncA, double* B, const int ldB);

/**
 * @brief Diagonal(A)-Symmetric(B) solver.
 *
 * Compute  A * X = alpha * B   or    X * A = alpha * B
 *
 * result stored in B.
 */
void cblas_ddisysv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const int M, const double alpha,
                   const double* A, const int IncA, double* B, const int ldB);

/**
 * @brief Diagonal(A)-Triangular(B) solver.
 *
 * Compute A * X = alpha * B    or    X * A = alpha * B
 *
 * result stored in B.
 */
void cblas_dditrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const CBLAS_DIAG DiagB, const int M,
                   const double alpha, const double* A, const int IncA,
                   double* B, const int ldB);

/**
 * @brief Diagonal-Diagonal solver.
 *
 * Compute A * X = alpha * B    or    X * A = alpha * B
 */
void cblas_ddidisv(const CBLAS_LAYOUT Layout, const int M, const double alpha,
                   const double* A, const int IncA, double* B, const int IncB);

/**
 * @brief Dense-Dense solver.
 *
 * Compute op(A) * X = B      or     X * op(A) = B
 */
void cblas_dgegesv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_TRANSPOSE TransA, const int M, const int N,
                   double* A, const int ldA, double* B, const int ldB);

/**
 * @brief Dense(A)-Symmetric(B) solver.
 *
 * Compute op(A) * X = B      or     X * op(A) = B
 */
void cblas_dgesysv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const CBLAS_TRANSPOSE TransA,
                   const int M, double* A, const int ldA, double* B,
                   const int ldB);

/**
 * @brief Dense(A)-Triangular(B) solver.
 *
 * Compute op(A) * X = B      or     X * op(A) = B
 */
void cblas_dgetrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploB, const CBLAS_TRANSPOSE TransA,
                   const CBLAS_DIAG DiagB, const int M, double* A,
                   const int ldA, double* B, const int ldB);

/**
 * @brief Dense(A)-Diagonal(B) solver.
 *
 * Compute op(A) * X = B      or     X * op(A) = B
 */
void cblas_dgedisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_TRANSPOSE TransA, const int M, double* A,
                   const int ldA, const double* D, const int IncD, double* B,
                   const int ldB);

/**
 * @brief Symmetric(A)-Dense(B) solver.
 *
 * Compute A * X = B      or     X * A = B
 */
void cblas_dsygesv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const int M, const int N, double* A,
                   const int ldA, double* B, const int ldB);

/**
 * @brief Symmetric(A)-Symmetric(B) solver.
 *
 * Compute A * X = B      or     X * A = B
 */
void cblas_dsysysv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB, const int M,
                   double* A, const int ldA, double* B, const int ldB);

/**
 * @brief Symmetric(A)-Triangular(B) solver.
 *
 * Compute A * X = B      or     X * A = B
 */
void cblas_dsytrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB,
                   const CBLAS_DIAG DiagB, const int M, double* A,
                   const int ldA, double* B, const int ldB);

/**
 * @brief Symmetric(A)-Diagonal(B) system solver.
 *
 * Compute A * X = B      or     X * A = B
 */
void cblas_dsydisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const char UploA, const int M, double* A, const int ldA,
                   const double* D, const int IncD, double* B, const int ldB);

/**
 * @brief SPD(A)-Dense(B) solver.
 *
 * Compute A * X = B       or     X * A = B
 */
void cblas_dpogesv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const int M, const int N, double* A,
                   const int ldA, double* B, const int ldB);

/**
 * @brief SPD(A)-Symmetric(B) solver.
 *
 * Compute A * X = B      or      X * A = B
 */
void cblas_dposysv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB, const int M,
                   double* A, const int ldA, double* B, const int ldB);

/**
 * @brief SPD(A)-Triangular(B) solver.
 *
 * Compute A * X = B      or     X * A = B
 */
void cblas_dpotrsv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const CBLAS_UPLO UploA, const CBLAS_UPLO UploB,
                   const CBLAS_DIAG DiagB, const int M, double* A,
                   const int ldA, double* B, const int ldB);

/**
 * @brief SPD(A)-Diagonal(B) system solver.
 *
 * Compute A * X = B      or     X * A = B
 */
void cblas_dpodisv(const CBLAS_LAYOUT Layout, const CBLAS_SIDE Side,
                   const char UploA, const int M, double* A, const int ldA,
                   const double* D, const int IncD, double* B, const int ldB);

/* ==================================================
   ============== AUXILIARY FUNCTIONS ===============
   ================================================== */

#endif