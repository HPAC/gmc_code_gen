#include "dMatrix.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

#include "../macros.hpp"

MATRIX::MATRIX(const unsigned nrows, const unsigned ncols, const bool is_diag)
    : isDiag{is_diag}, ROWS{nrows}, COLS{ncols} {
  if (is_diag) {
    STRIDE = 1U;
    DATA = new double[ROWS];
  } else {
    STRIDE = nrows;
    DATA = new double[ROWS * COLS];
  }
}

MATRIX::~MATRIX() { delete[] DATA; }

MATRIX::MATRIX(const MATRIX& other)
    : isDiag{other.isDiag}, ROWS{other.ROWS}, COLS{other.COLS} {
  if (isDiag) {
    STRIDE = 1U;
    DATA = new double[ROWS]();
    std::copy(other.DATA, other.DATA + ROWS, DATA);
  } else {
    STRIDE = ROWS;
    DATA = new double[ROWS * COLS]();
    std::copy(other.DATA, other.DATA + ROWS * COLS, DATA);
  }
}

MATRIX::MATRIX(MATRIX&& rhs) noexcept : MATRIX() { swap(*this, rhs); }

MATRIX& MATRIX::operator=(const MATRIX& rhs) {
  MATRIX tmp = rhs;
  swap(*this, tmp);
  return *this;
}

MATRIX& MATRIX::operator=(MATRIX&& rhs) noexcept {
  swap(*this, rhs);
  return *this;
}

void swap(MATRIX& first, MATRIX& second) {
  // enable ADL (Argument-Dependant Lookup)
  using std::swap;

  swap(first.isDiag, second.isDiag);
  swap(first.ROWS, second.ROWS);
  swap(first.COLS, second.COLS);
  swap(first.STRIDE, second.STRIDE);
  swap(first.DATA, second.DATA);
}

double& MATRIX::operator()(unsigned row, unsigned col) noexcept {
  return isDiag ? DATA[row] : DATA[col * STRIDE + row];
}

// out-of-place transpose. Not ideal. Do not use in time-critical applications.
MATRIX& MATRIX::TRANSPOSE() {
  double* n_data = new double[ROWS * COLS]();
  for (unsigned j = 0; j < COLS; ++j) {
    for (unsigned i = 0; i < ROWS; ++i) {
      n_data[j + i * COLS] = DATA[i + j * STRIDE];
    }
  }
  delete[] DATA;
  DATA = n_data;
  std::swap(ROWS, COLS);
  STRIDE = ROWS;
  return *this;
}

void MATRIX::DEALLOCATE() {
  delete[] DATA;
  ROWS = 0U;
  COLS = 0U;
  STRIDE = 0U;
  DATA = nullptr;
}

std::ostream& operator<<(std::ostream& os, const MATRIX& matrix) {
  if (matrix.isDiag) {
    for (unsigned i = 0; i < matrix.ROWS; ++i) {
      os << std::setprecision(6) << matrix.DATA[i * matrix.STRIDE];
      if (i < matrix.ROWS - 1)
        os << ' ';
      else
        os << '\n';
    }
  } else {
    for (unsigned i = 0; i < matrix.ROWS; ++i) {
      for (unsigned j = 0; j < matrix.COLS; ++j) {
        os << std::setprecision(5) << matrix.DATA[j * matrix.STRIDE + i];
        if (j < matrix.COLS - 1)
          os << ' ';
        else
          os << '\n';
      }
    }
  }
  return os;
}

void MATRIX::printMatrix() const {
  if (isDiag) {
    for (unsigned i = 0; i < ROWS; ++i)
      std::cout << std::setw(11) << std::setprecision(6) << DATA[i * STRIDE];
    std::cout << '\n';
  } else {
    for (unsigned i = 0; i < ROWS; ++i) {
      for (unsigned j = 0; j < COLS; ++j) {
        std::cout << std::setw(10) << std::setprecision(4)
                  << DATA[j * STRIDE + i];
      }
      std::cout << '\n';
    }
  }
}

void MATRIX::writeToDisk(const std::string& filename) const {
  std::ofstream ofile;
  ofile.open(filename, std::ofstream::out);
  write(ofile);
  ofile.close();
}

void MATRIX::write(std::ostream& os) const {
  if (isDiag) {
    for (unsigned i = 0; i < ROWS; i++) os << DATA[i * STRIDE] << '\n';

  } else {
    for (unsigned i = 0; i < ROWS; i++) {
      for (unsigned j = 0; j < COLS; j++) {
        os << DATA[j * STRIDE + i];
        if (j == COLS - 1)
          os << '\n';
        else
          os << ' ';
      }
    }
  }
}

void MATRIX::readMatrix(const std::string& filename) {
  std::string line;
  std::ifstream ifile;
  ifile.open(filename);
  if (ifile.fail()) {
    std::cerr << "couldn't open the file " << filename << '\n';
    exit(-1);
  }

  if (isDiag) {
    for (unsigned i = 0; i < ROWS; i++) {
      std::getline(ifile, line);
      DATA[i * STRIDE] = std::stod(line);
    }
  } else {
    unsigned start = 0U;
    unsigned end = 0U;
    unsigned k = 0U;
    for (unsigned i = 0; i < ROWS; i++) {
      std::getline(ifile, line);
      start = 0U;
      end = 0U;

      for (unsigned j = 0; j < COLS; j++) {
        k = start;
        while (k < line.size() && line[k] != ',') {
          k++;
        }
        end = k;
        DATA[j * STRIDE + i] = std::stod(line.substr(start, end - start));
        start = end + 1;
      }
    }
  }

  ifile.close();
}

MATRIX MATRIX::extractDiag() const {
  MATRIX mat;
  mat.ROWS = std::min(ROWS, COLS);
  mat.COLS = mat.ROWS;
  mat.STRIDE = 1;
  mat.DATA = new double[mat.ROWS]();
  mat.isDiag = true;

  for (unsigned i = 0; i < mat.ROWS; ++i) {
    mat.DATA[i] = DATA[i + i * STRIDE];
  }

  return mat;
}

MATRIX MATRIX::buildFromDiag() const {
  MATRIX mat;
  mat.ROWS = this->ROWS;
  mat.COLS = this->COLS;
  mat.STRIDE = mat.ROWS;
  mat.DATA = new double[mat.ROWS * mat.COLS]();
  mat.isDiag = false;

  for (unsigned i = 0; i < mat.ROWS; ++i)
    mat.DATA[i + i * mat.STRIDE] = this->DATA[i];

  return mat;
}

// MATRIX MATRIX::buildFromDiag(const unsigned n_rows,
//                              const unsigned n_cols) const {
//   MATRIX mat;
//   mat.ROWS = n_rows;
//   mat.COLS = n_cols;
//   mat.STRIDE = mat.ROWS;
//   mat.DATA = new double[mat.ROWS * mat.COLS]();

//   // todo fix accessing this->DATA -- depends on which dimension is smaller
//   for (unsigned i = 0; i < mat.ROWS; ++i)
//     mat.DATA[i + i * STRIDE] = this->DATA[i];

//   return mat;
// }

// The following functions give no guarantees wrt condition number. Functions to
// generate "nice" matrices can be found in src/matrix_generator.hpp.

MATRIX& MATRIX::randomDense(const std::function<double()>&& generator) {
  for (unsigned j = 0; j < COLS; ++j) {
    for (unsigned i = 0; i < ROWS; ++i) {
      DATA[i + j * STRIDE] = generator();
    }
  }
  return *this;
}

MATRIX& MATRIX::randomSymmetric(const std::function<double()>&& generator) {
  for (unsigned j = 0; j < COLS; ++j) {
    for (unsigned i = 0; i <= j; ++i) {
      DATA[i + j * STRIDE] = generator();
      DATA[j + i * STRIDE] = DATA[i + j * STRIDE];  // self-assignment in diag
    }
  }
  return *this;
}

MATRIX& MATRIX::randomLower(const std::function<double()>&& generator) {
  for (unsigned j = 0; j < COLS; ++j) {
    for (unsigned i = j; i < ROWS; ++i) {
      DATA[i + j * STRIDE] = generator();
    }
  }
  return *this;
}

MATRIX& MATRIX::randomUnitLower(const std::function<double()>&& generator) {
  for (unsigned j = 0; j < COLS; ++j) {
    for (unsigned i = j + 1; i < ROWS; ++i) {
      DATA[i + j * STRIDE] = generator();
    }
  }
  for (unsigned j = 0; j < COLS; ++j) DATA[j + j * STRIDE] = 1.0;
  return *this;
}

MATRIX& MATRIX::randomUpper(const std::function<double()>&& generator) {
  for (unsigned j = 0; j < COLS; ++j) {
    for (unsigned i = 0; i <= j; ++i) {
      DATA[i + j * STRIDE] = generator();
    }
  }
  return *this;
}

MATRIX& MATRIX::randomUnitUpper(const std::function<double()>&& generator) {
  for (unsigned j = 1; j < COLS; ++j) {
    for (unsigned i = 0; i < j; ++i) {
      DATA[i + j * STRIDE] = generator();
    }
  }
  for (unsigned j = 0; j < COLS; ++j) DATA[j + j * STRIDE] = 1.0;
  return *this;
}

MATRIX& MATRIX::randomDiag(const std::function<double()>&& generator) {
  for (unsigned j = 0; j < COLS; ++j) DATA[j + j * STRIDE] = generator();
  return *this;
}

MATRIX& MATRIX::zero() {
  for (unsigned j = 0U; j < COLS; ++j) {
    for (unsigned i = 0U; i < ROWS; ++i) {
      DATA[i + j * STRIDE] = 0.0;
    }
  }
  return *this;
}
