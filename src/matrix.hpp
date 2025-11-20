/**
 * @file matrix.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Header for `Matrix`. See the class documentation for more info.
 * @version 0.1
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <string>

#include "features.hpp"
#include "macros.hpp"

namespace cg {

/**
 * @brief Representation of a symbolic matrix.
 *
 * Inherits directly from `Features` and adds a number of data-members and its
 * managing.
 *
 * nrows and ncols are used by `Algorithm` when the number of FLOPs is computed
 * and when the `Algorithm` is executed to obtain its execution time.
 *
 * is_modifiable is mainly used with matrices that are input to an `Algorithm`.
 * Intermediate operands are modifiable by default. This data-member indicates
 * whether the corresponding live matrix in the produced code could be modified
 * or not. This has consequences when generating code. For instance, TRMM
 * overwrites one of the operands. If such operand were not modifiable, it
 * should be copied before performing the overwrite.
 *
 */
class Matrix : public Features {
 public:
  std::string name{};          // name of the symbolic matrix
  std::string row_dim_name{};  // name of the row dimension
  std::string col_dim_name{};  // name of the column dimension
  unsigned nrows{};            // number of rows - used to compute FLOPs of algs
  unsigned ncols{};            // number of cols - used to compute FLOPs of algs
  bool is_modifiable = false;  // whether the matrix can be modified

 public:
  // Default ctor.
  Matrix() = default;

  // User-supplied ctor.
  Matrix(const std::string& name, const Structure& structure,
         const Property& property, const Trans& trans,
         const Inversion& inversion, const bool is_temp = false)
      : Features(structure, property, trans, inversion),
        name{name},
        is_modifiable{is_temp} {
    row_dim_name = name + "." + STR(ROWS);
    col_dim_name = name + "." + STR(COLS);
  }

  // User-supplied ctor.
  Matrix(const std::string& name, const Features& features,
         const bool is_temp = false)
      : Features{features}, name{name}, is_modifiable{is_temp} {
    row_dim_name = name + "." + STR(ROWS);
    col_dim_name = name + "." + STR(COLS);
  }

  // User-supplied ctor. Defaults to a general dense matrix with no property, no
  // transposition, and no inversion.
  Matrix(const std::string& name, const bool is_modifiable)
      : name{name}, is_modifiable{is_modifiable} {}

  bool operator==(const Matrix& rhs) const;

  bool isModifiable() const;

  /* Getters */
  std::string getName() const;

  std::string getRowName() const;

  std::string getColName() const;

  unsigned getNrows() const;

  unsigned getNcols() const;

  /* Setters */
  void setName(const std::string& name);

  void setRowName(const std::string& row_name);

  void setColName(const std::string& col_name);

  void setNrows(const unsigned nrows);

  void setNcols(const unsigned ncols);

  void setModifiable(const bool is_modifiable);
};

/**
 * @brief This struct is used in conjunction with `Variant`, which, in turn, is
 * used by `Kernel` and, eventually, the `Matcher`.
 *
 * A `RangeMatrix` is a set of particular features that, when combined, form
 * a proper `Features`.
 *
 */
struct RangeMatrix {
  std::vector<Structure> structure;
  std::vector<Property> property;
  std::vector<Trans> trans;
  std::vector<Inversion> inv;
};

}  // namespace cg

#endif