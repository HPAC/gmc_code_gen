#ifndef MATRIX_GENERATOR_H
#define MATRIX_GENERATOR_H

#include <random>
#include <vector>

#include "../definitions.hpp"
#include "../matrix.hpp"
#include "dMatrix.hpp"

namespace cg {

class MatrixGenerator {
 public:
  static dMatrix generateMatrix(const cg::Features& features, const unsigned m,
                                const unsigned n);

  static std::vector<dMatrix> generateChain(const MatrixChain& symb_chain,
                                            const Instance& instance);

  static dMatrix generateDense(const unsigned m, const unsigned n);

  static dMatrix generateDenseFR(const unsigned m);

  static dMatrix generateSymmetric(const char UPLO, const unsigned m);

  static dMatrix generateSymmetricFR(const char UPLO, const unsigned m);

  static dMatrix generateSPD(const char UPLO, const unsigned m);

  static dMatrix generateTriangular(const char UPLO, const char DIAG,
                                    const unsigned m);

  static dMatrix generateTriangularFR(const char UPLO, const char DIAG,
                                      const unsigned m);

  static dMatrix generateDiagonal(const unsigned m);

  static dMatrix generateDiagonalFR(const unsigned m);

  static dMatrix generateRandomDense(const unsigned m, const unsigned n);

  static dMatrix generateSigma(const unsigned m);

  static dMatrix generateOrthonormal(const unsigned m);

  static double compareMatrix(const dMatrix& A, const dMatrix& B);
};

}  // namespace cg

#endif