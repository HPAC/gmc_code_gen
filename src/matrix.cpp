/**
 * @file matrix.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Implementation of `Matrix` class. See the header file for more info.
 * @version 0.1
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "matrix.hpp"

#include <string>

#include "features.hpp"

namespace cg {

bool Matrix::operator==(const Matrix& rhs) const {
  return (this->name == rhs.name) and
         (this->row_dim_name == rhs.row_dim_name) and
         (this->col_dim_name == rhs.col_dim_name) and
         (this->is_modifiable == rhs.is_modifiable) and
         Features::operator==(rhs);
}

bool Matrix::isModifiable() const { return is_modifiable; }

std::string Matrix::getName() const { return name; }

std::string Matrix::getRowName() const { return row_dim_name; }

std::string Matrix::getColName() const { return col_dim_name; }

unsigned Matrix::getNrows() const { return nrows; }

unsigned Matrix::getNcols() const { return ncols; }

void Matrix::setName(const std::string& name) { this->name = name; }

void Matrix::setRowName(const std::string& row_name) {
  row_dim_name = row_name;
}

void Matrix::setColName(const std::string& col_name) {
  col_dim_name = col_name;
}

void Matrix::setNrows(const unsigned nrows) { this->nrows = nrows; }

void Matrix::setNcols(const unsigned ncols) { this->ncols = ncols; }

void Matrix::setModifiable(const bool is_modifiable) {
  this->is_modifiable = is_modifiable;
}

}  // namespace cg
