/**
 * @file experiment_util.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Common functions for programs that execute experiments with matrices
 * whose features are one of the 10 combinations of features.
 * @version 0.1
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "experiment_util.hpp"

#include <array>
#include <cmath>
#include <string>

#include "definitions.hpp"
#include "matrix.hpp"

namespace cg {

cg::Matrix ID2matrix(const unsigned digit, const unsigned id_matrix) {
  cg::Matrix matrix;
  char name = 'A' + id_matrix;

  if (digit == 0U) {
    matrix =
        cg::Matrix(std::string{name}, cg::Structure::Dense, cg::Property::None,
                   cg::Trans::N, cg::Inversion::N, true);
  } else if (digit == 1U) {
    matrix = cg::Matrix(std::string{name}, cg::Structure::Dense,
                        cg::Property::FullRank, cg::Trans::N, cg::Inversion::Y,
                        true);
  } else if (digit == 2U) {
    matrix =
        cg::Matrix(std::string{name}, cg::Structure::Symmetric_L,
                   cg::Property::SPD, cg::Trans::N, cg::Inversion::N, true);
  } else if (digit == 3U) {
    matrix =
        cg::Matrix(std::string{name}, cg::Structure::Symmetric_L,
                   cg::Property::SPD, cg::Trans::N, cg::Inversion::Y, true);
  } else if (digit == 4U) {
    matrix =
        cg::Matrix(std::string{name}, cg::Structure::Upper, cg::Property::None,
                   cg::Trans::N, cg::Inversion::N, true);
  } else if (digit == 5U) {
    matrix =
        cg::Matrix(std::string{name}, cg::Structure::Lower, cg::Property::None,
                   cg::Trans::N, cg::Inversion::N, true);
  } else if (digit == 6U) {
    matrix = cg::Matrix(std::string{name}, cg::Structure::Upper,
                        cg::Property::FullRank, cg::Trans::N, cg::Inversion::N,
                        true);
  } else if (digit == 7U) {
    matrix = cg::Matrix(std::string{name}, cg::Structure::Lower,
                        cg::Property::FullRank, cg::Trans::N, cg::Inversion::N,
                        true);
  } else if (digit == 8U) {
    matrix = cg::Matrix(std::string{name}, cg::Structure::Upper,
                        cg::Property::FullRank, cg::Trans::N, cg::Inversion::Y,
                        true);
  } else if (digit == 9U) {
    matrix = cg::Matrix(std::string{name}, cg::Structure::Lower,
                        cg::Property::FullRank, cg::Trans::N, cg::Inversion::Y,
                        true);
  }
  return matrix;
}

cg::MatrixChain ID2chain(const unsigned chain_N, const unsigned id_shape) {
  cg::MatrixChain chain;
  chain.reserve(chain_N);

  unsigned digit;
  float power_10;
  for (unsigned i = 0U; i < chain_N; i++) {
    power_10 = std::pow(10.0, static_cast<float>(chain_N - 1 - i));
    digit = id_shape / static_cast<unsigned>(power_10);
    digit %= 10U;
    chain.push_back(ID2matrix(digit, i));
  }
  return chain;
}

std::array<unsigned, 2U> getProcRange(const unsigned max_value,
                                      const unsigned proc_id,
                                      const unsigned proc_num) {
  unsigned block = static_cast<unsigned>(max_value / proc_num);
  std::array<unsigned, 2U> range;
  range[0] = block * proc_id;
  range[1] = block * (proc_id + 1U);
  if (proc_id == proc_num - 1U) range[1] = max_value;
  return range;
}

bool isSquareShape(const unsigned n, const unsigned id_shape) {
  bool square_shape = true;
  unsigned digit;
  float power_10;

  for (unsigned i = 0U; i < n; i++) {
    power_10 = std::pow(10.0, static_cast<float>(i));
    digit = id_shape / static_cast<unsigned>(power_10);  // fix this
    digit %= 10U;
    if (digit == 0U) square_shape = false;
  }

  return square_shape;
}

unsigned getMaxValue(const unsigned n) {
  return static_cast<unsigned>(std::pow(10.0, static_cast<double>(n)));
}

}  // namespace cg