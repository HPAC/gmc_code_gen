#include "gegesv.hpp"

#include <fmt/core.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../../cblas_extended/cblas_extended.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "../matrix.hpp"
#include "../models/model2d.hpp"
#include "../models/model_common.hpp"
#include "../variant.hpp"

using std::string;

namespace cg {

bool KernelGegesv::tweakTransposition(Matrix& left, Matrix& right) const {
  Matrix& rhs = (left.isInverted()) ? right : left;
  return (rhs.isTransposed());
}

std::array<bool, 2U> KernelGegesv::needsNewMatrix() const {
  return {false, false};
}

void KernelGegesv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = (left.isInverted()) ? right : left;
  result.setName(rhs.isModifiable() ? rhs.getName()
                                    : PREFIX_COPY + rhs.getName());
}

std::string KernelGegesv::generateCode(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  string side, trans_lhs;
  string m, n;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;
  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  trans_lhs = lhs.isTransposed() ? CG_TRANS : CG_NO_TRANS;

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  m = result.getRowName();
  n = result.getColName();

  code +=
      fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {});\n", CBLAS_DGEGESV,
                  LAYOUT, side, trans_lhs, m, n, dataMatrix(lhs),
                  strideMatrix(lhs), dataMatrix(result), strideMatrix(result));
  if (lhs.isModifiable()) code += freeMatrix(lhs);

  return code;
}

std::string KernelGegesv::generateCost(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  string rhs_dim = left.isInverted() ? rhs.getColName() : rhs.getRowName();

  return fmt::format(
      "{0} += (2.0 / 3.0) * {1} * {1} * {1} + 2.0 * {1} * {1} * {2}; {3}",
      COST_VAR, lhs.getRowName(), rhs_dim, infoInvocation(left, right, result));
}

double KernelGegesv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  const double lhs_rows = static_cast<double>(lhs.getNrows());
  const double rhs_dim = static_cast<double>(
      (left.isInverted() ? rhs.getNcols() : rhs.getNrows()));

  return (2.0 / 3.0 * lhs_rows * lhs_rows * lhs_rows) +
         (2.0 * lhs_rows * lhs_rows * rhs_dim);
}

std::vector<Variant> KernelGegesv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Dense}, {Property::FullRank}, all_trans, {Inversion::Y}},
      {{Structure::Dense}, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

std::string KernelGegesv::infoInvocation(const Matrix& left,
                                         const Matrix& right,
                                         const Matrix& result) const {
  return Kernel::infoInvocation(CG_GEGESV, left, right, result);
}

void KernelGegesv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;
  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_TRANSPOSE TransA = _lhs.isTransposed() ? CblasTrans : CblasNoTrans;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;
  int M = left.ROWS;
  int N = right.COLS;

  cblas_dgegesv(CblasColMajor, Side, TransA, M, N, lhs.DATA, lhs.STRIDE,
                rhs.DATA, rhs.STRIDE);

  result = std::move(rhs);
}

void KernelGegesv::loadModel() { model.read(getPathModel()); }

double KernelGegesv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  const Matrix& lhs = (left.isInverted()) ? left : right;
  if (!left.isInverted()) mdl::setBitLR(key);
  if (lhs.isTransposed()) mdl::setBitTransA(key);

  unsigned m = left.getNrows();
  unsigned n = right.getNcols();
  return computeFLOPs(left, right, result) / model.predict(key, m, n);
}

}  // namespace cg
