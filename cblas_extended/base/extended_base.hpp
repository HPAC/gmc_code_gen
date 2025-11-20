#ifndef EXTENDED_BASE_H
#define EXTENDED_BASE_H

void dlaswp2(const int M, double* A, const int LDA, const int K1, const int K2,
             const int* JPIV, const int INCX);

void dlaswp3(const char UPLO, const int M, const int N, double* A,
             const int LDA, const int* IPIV, const int INCX);

void dlaswp4(const char UPLO, const int M, const int N, double* A,
             const int LDA, const int* IPIV, const int INCX);

void dbdigesv(const char SIDE, const char UPLO, const int M, const int N,
              const double* A, const int LDA, const double* E, const int INCE,
              const int* XPIV, const int INCX, double* B, const int LDB);

void sy2full(const char UPLO, const int M, double* A, const int LDA);

void dge_trans(const int M, const int N, const double* const A, const int LDA,
               double* B, const int LDB);

void dlacpy(const int M, const int N, const double* const A, const int LDA,
            double* const B, const int LDB);

void ddimm(const char SIDE, const int M, const int N, const double ALPHA,
           const double* A, const int INCA, double* B, const int LDB);

void dsysymm(const char UPLOA, const char UPLOB, const int M,
             const double ALPHA, double* A, const int LDA, double* B,
             const int LDB, const double BETA, double* C, const int LDC);

void dtrsymm(const char SIDE, const char UPLOA, const char UPLOB,
             const char TRANSA, const char DIAG, const int M,
             const double ALPHA, const double* A, const int LDA, double* B,
             const int LDB);

void ddisymm(const char SIDE, const char UPLO, const int M, const double ALPHA,
             const double* A, const int INCA, double* B, const int LDB);

void dtrtrmm(const char UPLOA, const char UPLOB, const char TRANSA,
             const char TRANSB, const char DIAGA, const char DIAGB, const int M,
             const double ALPHA, const double* A, const int LDA,
             const double* B, const int LDB, double* C, const int LDC);

void dditrmm(const char SIDE, const char UPLOB, const char DIAGB, const int M,
             const double ALPHA, const double* A, const int INCA, double* B,
             const int LDB);

void ddidimm(const int M, const double ALPHA, const double* A, const int INCA,
             double* B, const int INCB);

void ddisv(const char SIDE, const int M, const int N, const double ALPHA,
           const double* A, const int INCA, double* B, const int LDB);

void ddisysv(const char SIDE, const char UPLOB, const int M, const double ALPHA,
             const double* A, const int INCA, double* B, const int LDB);

void dditrsv(const char SIDE, const char UPLOB, const char DIAGB, const int M,
             const double ALPHA, const double* A, const int INCA, double* B,
             const int LDB);

void ddidisv(const int M, const double ALPHA, const double* A, const int INCA,
             double* B, const int INCB);

void dgegesv(const char SIDE, const char TRANSA, const int M, const int N,
             double* A, const int LDA, double* B, const int LDB);

void dgesysv(const char SIDE, const char UPLOB, const char TRANSA, const int M,
             double* A, const int LDA, double* B, const int LDB);

void dgetrsv(const char SIDE, const char UPLOB, const char TRANSA,
             const char DIAGB, const int M, double* A, const int LDA, double* B,
             const int LDB);

void dgedisv(const char SIDE, const char TRANSA, const int M, double* A,
             const int LDA, const double* D, const int INCD, double* B,
             const int LDB);

void dsygesv(const char SIDE, const char UPLO, const int M, const int N,
             double* A, const int LDA, double* B, const int LDB);

void dsysysv(const char SIDE, const char UPLOA, const char UPLOB, const int M,
             double* A, const int LDA, double* B, const int LDB);

void dsytrsv(const char SIDE, const char UPLOA, const char UPLOB,
             const char DIAGB, const int M, double* A, const int LDA, double* B,
             const int LDB);

void dsydisv(const char SIDE, const char UPLO, const int M, double* A,
             const int LDA, const double* D, const int INCD, double* B,
             const int LDB);

void dtrsysv(const char SIDE, const char UPLOA, const char UPLOB,
             const char TRANSA, const char DIAG, const int M,
             const double ALPHA, const double* A, const int LDA, double* B,
             const int LDB);

void dtrtrsv(const char SIDE, const char UPLOA, const char UPLOB,
             const char TRANSA, const char DIAGA, const char DIAGB, const int M,
             const double ALPHA, double* A, const int LDA, double* B,
             const int LDB);

void dtrdisv(const char SIDE, const char UPLOA, const char TRANSA,
             const char DIAGA, const int M, const double ALPHA, const double* A,
             const int LDA, const double* D, const int INCD, double* B,
             const int LDB);

void dpogesv(const char SIDE, const char UPLO, const int M, const int N,
             double* A, const int LDA, double* B, const int LDB);

void dposysv(const char SIDE, const char UPLOA, const char UPLOB, const int M,
             double* A, const int LDA, double* B, const int LDB);

void dpotrsv(const char SIDE, const char UPLOA, const char UPLOB,
             const char DIAGB, const int M, double* A, const int LDA, double* B,
             const int LDB);

void dpodisv(const char SIDE, const char UPLOA, const int M, double* A,
             const int LDA, const double* D, const int INCD, double* B,
             const int LDB);

#endif