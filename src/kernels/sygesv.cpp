#include "sygesv.hpp"

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
bool KernelSygesv::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  return rhs.isTransposed();
}

std::array<bool, 2U> KernelSygesv::needsNewMatrix() const {
  return {false, false};
}

void KernelSygesv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  result.setName(rhs.isModifiable() ? rhs.getName()
                                    : PREFIX_COPY + rhs.getName());
}

string KernelSygesv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo, m, n;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  uplo = lhs.isSymmetricLower() ? CG_LOWER : CG_UPPER;

  m = rhs.getRowName();
  n = rhs.getColName();

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  code +=
      fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {});\n", CBLAS_DSYGESV,
                  LAYOUT, side, uplo, m, n, dataMatrix(lhs), strideMatrix(lhs),
                  dataMatrix(result), strideMatrix(result));
  if (lhs.isModifiable()) code += freeMatrix(lhs);

  return code;
}

string KernelSygesv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  /*
   * Cost = LDLT + 2TRSM
   *      LDLT = 1/3 * {m,n}**3
   *      TRSM = {m,n}**2 * {n,m}
   */
  string rows_sy = left.isInverted() ? left.getRowName() : right.getRowName();
  string rhs_dim = left.isInverted() ? right.getColName() : left.getRowName();

  return fmt::format("{0} += {1} * {1} * {1} / 3.0 + 2.0 * {1} * {1} * {2};{3}",
                     COST_VAR, rows_sy, rhs_dim,
                     infoInvocation(left, right, result));
}

double KernelSygesv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const double rows_sy = static_cast<double>(
      left.isInverted() ? left.getNrows() : right.getNrows());
  const double rhs_dim = static_cast<double>(
      left.isInverted() ? right.getNcols() : left.getNrows());

  return rows_sy * rows_sy * rows_sy / 3.0 + 2.0 * rows_sy * rows_sy * rhs_dim;
}

std::vector<Variant> KernelSygesv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Symmetric_L, Structure::Symmetric_U},
       {Property::FullRank},
       {Trans::N},
       {Inversion::Y}},
      {{Structure::Dense}, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

string KernelSygesv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_SYGESV, left, right, result);
}

void KernelSygesv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _lhs.isSymmetricLower() ? CblasLower : CblasUpper;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;
  int N = right.COLS;

  cblas_dsygesv(CblasColMajor, Side, UploA, M, N, lhs.DATA, lhs.STRIDE,
                rhs.DATA, rhs.STRIDE);

  result = std::move(rhs);
}

void KernelSygesv::loadModel() { model.read(getPathModel()); }

double KernelSygesv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  const Matrix& lhs = (left.isInverted()) ? left : right;
  if (right.isInverted()) mdl::setBitLR(key);
  if (lhs.isSymmetricLower()) mdl::setBitUploA(key);

  unsigned m = result.getNrows();
  unsigned n = result.getNcols();
  return computeFLOPs(left, right, result) / model.predict(key, m, n);
}

}  // namespace cg
