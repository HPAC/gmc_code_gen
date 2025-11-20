#include "matrix_generator.hpp"

#include <openblas/cblas.h>
#include <openblas/lapacke.h>
#include <openblas/lapacke_utils.h>

#include <random>

#include "../definitions.hpp"
#include "../matrix.hpp"
#include "dMatrix.hpp"

std::random_device rd;
inline std::mt19937 random_generator(rd());
inline std::uniform_real_distribution<double> dist(-0.5, 0.5);
inline double cond_number = 1.2;

namespace cg {

dMatrix MatrixGenerator::generateMatrix(const cg::Features& features,
                                        const unsigned m, const unsigned n) {
  char uplo, diag;

  if (features.isDense()) {
    if (!features.isInvertible())
      return generateDense(m, n);
    else
      return generateDenseFR(m);
  }

  else if (features.isSymmetric()) {
    uplo = (features.isSymmetricLower()) ? 'L' : 'U';
    if (!features.isInvertible())
      return generateSymmetric(uplo, m);
    else if (features.isSPD())
      return generateSPD(uplo, m);
    else
      return generateSymmetricFR(uplo, m);
  }

  else if (features.isTriangular()) {
    uplo = (features.isLower()) ? 'L' : 'U';
    diag = (features.isUnit()) ? 'U' : 'N';
    if (!features.isInvertible())
      return generateTriangular(uplo, diag, m);
    else
      return generateTriangularFR(uplo, diag, m);
  }

  else {
    if (!features.isInvertible())
      return generateDiagonal(m);
    else
      return generateDiagonalFR(m);
  }
}

std::vector<dMatrix> MatrixGenerator::generateChain(
    const MatrixChain& symb_chain, const Instance& instance) {
  std::vector<dMatrix> chain;
  chain.reserve(symb_chain.size());
  unsigned idx_r, idx_c;

  for (unsigned k = 0; k < symb_chain.size(); k++) {
    if (symb_chain[k].isTransposed()) {
      idx_r = k + 1;
      idx_c = k;
    } else {
      idx_r = k;
      idx_c = k + 1;
    }
    chain.emplace_back(
        generateMatrix(symb_chain[k], instance[idx_r], instance[idx_c]));
  }

  return chain;
}

dMatrix MatrixGenerator::generateDense(const unsigned m, const unsigned n) {
  dMatrix matrix = generateRandomDense(m, n);

  // copy the penultimate column onto the last column
  for (unsigned i = 0; i < m; i++) {
    matrix.DATA[(n - 1) * matrix.STRIDE + i] =
        matrix.DATA[(n - 2) * matrix.STRIDE + i];
  }

  return matrix;
}

dMatrix MatrixGenerator::generateDenseFR(const unsigned m) {
  dMatrix S = generateSigma(m);
  dMatrix U = generateOrthonormal(m);
  dMatrix V = generateOrthonormal(m);
  dMatrix TEMP(m, m);
  dMatrix A(m, m);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, U.DATA,
              U.STRIDE, S.DATA, S.STRIDE, 0.0, TEMP.DATA, TEMP.STRIDE);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, TEMP.DATA,
              TEMP.STRIDE, V.DATA, V.STRIDE, 0.0, A.DATA, A.STRIDE);

  return A;
}

dMatrix MatrixGenerator::generateSymmetric(const char UPLO, const unsigned m) {
  dMatrix matrix(m, m);
  matrix.zero();

  if (UPLO == 'U') {
    for (unsigned j = 0U; j < m; j++) {
      for (unsigned i = 0U; i <= j; i++) {
        matrix.DATA[j * matrix.STRIDE + i] = dist(random_generator);
      }
    }

    // set last two diagonal entries to zero (not diagonally-dominant).
    matrix.DATA[(m - 2) * matrix.STRIDE + (m - 2)] = 0.0;
    matrix.DATA[(m - 1) * matrix.STRIDE + (m - 1)] = 0.0;
    // copy penultimate column onto last column (not full-rank).
    for (unsigned i = 0U; i < m - 1; i++) {
      matrix.DATA[(m - 1) * matrix.STRIDE + i] =
          matrix.DATA[(m - 2) * matrix.STRIDE + i];
    }
  }

  else if (UPLO == 'L') {
    for (unsigned j = 0U; j < m; j++) {
      for (unsigned i = j; i < m; i++) {
        matrix.DATA[j * matrix.STRIDE + i] = dist(random_generator);
      }
    }

    // set last two diagonal entries to zero.
    matrix.DATA[(m - 2) * matrix.STRIDE + (m - 2)] = 0.0;
    matrix.DATA[(m - 1) * matrix.STRIDE + (m - 1)] = 0.0;
    // copy penultimate row onto last row.
    for (unsigned j = 0; j < m - 1; j++) {
      matrix.DATA[j * matrix.STRIDE + (m - 1)] =
          matrix.DATA[j * matrix.STRIDE + (m - 2)];
    }
  }
  return matrix;
}

dMatrix MatrixGenerator::generateSymmetricFR(const char UPLO,
                                             const unsigned m) {
  dMatrix Sigma = generateSigma(m);
  // to make it non-spd -- make a singular value negative.
  unsigned idx = m / 2;
  Sigma.DATA[idx * Sigma.STRIDE + idx] = -Sigma.DATA[idx * Sigma.STRIDE + idx];
  dMatrix Q = generateOrthonormal(m);
  dMatrix TEMP(m, m);
  dMatrix S(m, m);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, Q.DATA,
              Q.STRIDE, Sigma.DATA, Sigma.STRIDE, 0.0, TEMP.DATA, TEMP.STRIDE);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, TEMP.DATA,
              TEMP.STRIDE, Q.DATA, Q.STRIDE, 0.0, S.DATA, S.STRIDE);

  if (UPLO == 'U') {
    for (unsigned j = 0U; j < m; j++) {
      for (unsigned i = j + 1; i < m; i++) {
        S.DATA[j * S.STRIDE + i] = 0.0;
      }
    }
  }

  else if (UPLO == 'L') {
    for (unsigned j = 0U; j < m; j++) {
      for (unsigned i = 0U; i < j; i++) {
        S.DATA[j * S.STRIDE + i] = 0.0;
      }
    }
  }
  return S;
}

dMatrix MatrixGenerator::generateSPD(const char UPLO, const unsigned m) {
  dMatrix Sigma = generateSigma(m);
  dMatrix Q = generateOrthonormal(m);
  dMatrix TEMP(m, m);
  dMatrix S(m, m);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, Q.DATA,
              Q.STRIDE, Sigma.DATA, Sigma.STRIDE, 0.0, TEMP.DATA, TEMP.STRIDE);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, m, m, 1.0, TEMP.DATA,
              TEMP.STRIDE, Q.DATA, Q.STRIDE, 0.0, S.DATA, S.STRIDE);

  if (UPLO == 'U') {
    for (unsigned j = 0U; j < m; j++) {
      for (unsigned i = j + 1; i < m; i++) {
        S.DATA[j * S.STRIDE + i] = 0.0;
      }
    }
  }

  else if (UPLO == 'L') {
    for (unsigned j = 0U; j < m; j++) {
      for (unsigned i = 0U; i < j; i++) {
        S.DATA[j * S.STRIDE + i] = 0.0;
      }
    }
  }
  return S;
}

dMatrix MatrixGenerator::generateTriangular(const char UPLO,
                                            const char Diagonal,
                                            const unsigned m) {
  dMatrix matrix = generateTriangularFR(UPLO, Diagonal, m);
  // make one entry on diagonal zero.
  matrix.DATA[(m - 2) * matrix.STRIDE + (m - 2)] = 0.0;

  return matrix;
}

dMatrix MatrixGenerator::generateTriangularFR(const char UPLO,
                                              const char Diagonal,
                                              const unsigned m) {
  dMatrix TEMP = generateDenseFR(m);
  double* tau = (double*)malloc(m * sizeof(double));
  LAPACKE_dgeqrf(CblasColMajor, m, m, TEMP.DATA, TEMP.STRIDE, tau);
  free(tau);

  for (unsigned j = 0; j < m; j++) {
    for (unsigned i = j + 1; i < m; i++) {
      TEMP.DATA[j * TEMP.STRIDE + i] = 0.0;
    }
  }

  dMatrix A(m, m);
  A.zero();
  if (UPLO == 'U') {
    LAPACKE_dlacpy(CblasColMajor, 'U', m, m, TEMP.DATA, TEMP.STRIDE, A.DATA,
                   A.STRIDE);
  } else if (UPLO == 'L') {
    LAPACKE_dge_trans(CblasColMajor, m, m, TEMP.DATA, TEMP.STRIDE, A.DATA,
                      A.STRIDE);
  }

  if (Diagonal == 'U') {
    for (unsigned i = 0; i < m; i++) {
      A.DATA[i * A.STRIDE + i] = 1.0;
    }
  }

  return A;
}

dMatrix MatrixGenerator::generateDiagonal(const unsigned m) {
  dMatrix matrix = generateDiagonalFR(m);

  matrix.DATA[(m - 1) * matrix.STRIDE] = 0.0;

  return matrix;
}

dMatrix MatrixGenerator::generateDiagonalFR(const unsigned m) {
  dMatrix matrix = generateSigma(m).extractDiag();

  return matrix;
}

dMatrix MatrixGenerator::generateRandomDense(const unsigned m,
                                             const unsigned n) {
  dMatrix matrix(m, n);

  for (unsigned j = 0; j < n; j++) {
    for (unsigned i = 0; i < m; i++) {
      matrix.DATA[j * matrix.STRIDE + i] = dist(random_generator);
    }
  }
  return matrix;
}

dMatrix MatrixGenerator::generateSigma(const unsigned m) {
  double max_sv = m * std::abs(dist(random_generator));  // max singular value
  double min_sv = max_sv / cond_number;                  // min singular value
  double np = static_cast<double>(m);                    // number of rows
  double step = (max_sv - min_sv) / (np - 1.0);

  dMatrix S(m, m);
  S.zero();
  for (unsigned i = 0; i < np; i++) {
    S.DATA[i * S.STRIDE + i] = max_sv - step * i;
  }

  return S;
}

dMatrix MatrixGenerator::generateOrthonormal(const unsigned m) {
  dMatrix Q = generateRandomDense(m, m);
  double* tau = (double*)malloc(m * sizeof(double));
  LAPACKE_dgeqrf(CblasColMajor, m, m, Q.DATA, m, tau);
  LAPACKE_dorgqr(CblasColMajor, m, m, m, Q.DATA, m, tau);
  free(tau);

  return Q;
}

double MatrixGenerator::compareMatrix(const dMatrix& A, const dMatrix& B) {
  double diff = 0.0;
  double acc = 0.0;
  for (unsigned j = 0; j < A.COLS; ++j) {
    for (unsigned i = 0; i < A.ROWS; ++i) {
      diff = A.DATA[j * A.STRIDE + i] - B.DATA[j * B.STRIDE + i];
      acc += diff * diff;
    }
  }
  return sqrt(acc);
}

}  // namespace cg