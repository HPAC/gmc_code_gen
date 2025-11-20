#ifndef DMATRIX_H
#define DMATRIX_H

#include <functional>
#include <string>

#include "../macros.hpp"

class MATRIX {
 private:
  bool isDiag = false;

 public:
  unsigned ROWS{0U};
  unsigned COLS{0U};
  unsigned STRIDE{0U};
  double* DATA{nullptr};

  /*< Member functions >*/
  MATRIX() = default;

  ~MATRIX();

  MATRIX(const unsigned nrows, const unsigned ncols,
         const bool is_diag = false);

  MATRIX(const MATRIX& other);

  MATRIX(MATRIX&& rhs) noexcept;

  MATRIX& operator=(const MATRIX& rhs);

  MATRIX& operator=(MATRIX&& rhs) noexcept;

  friend void swap(MATRIX& first, MATRIX& second);

  double& operator()(unsigned row, unsigned col) noexcept;

  MATRIX& TRANSPOSE();

  void DEALLOCATE();

  friend std::ostream& operator<<(std::ostream& os, const MATRIX& matrix);

  void printMatrix() const;

  void writeToDisk(const std::string& filename) const;

  void write(std::ostream& os) const;

  void readMatrix(const std::string& filename);

  MATRIX extractDiag() const;

  MATRIX buildFromDiag() const;

  // MATRIX buildFromDiag(const unsigned n_rows, const unsigned n_cols) const;

  // The functions below to fill a matrix are deprecated. They do not guarantee,
  // amongst others, good condition numbers. Better functions to initialize
  // matrices can be found in src/utils/matrix_generator.{hpp,cpp}.
  MATRIX& randomDense(const std::function<double()>&& generator);

  MATRIX& randomSymmetric(const std::function<double()>&& generator);

  MATRIX& randomLower(const std::function<double()>&& generator);

  MATRIX& randomUnitLower(const std::function<double()>&& generator);

  MATRIX& randomUpper(const std::function<double()>&& generator);

  MATRIX& randomUnitUpper(const std::function<double()>&& generator);

  MATRIX& randomDiag(const std::function<double()>&& generator);

  MATRIX& zero();
};

#endif