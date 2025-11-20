#include "trtrsv.hpp"

#include <fmt/core.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../../cblas_extended/cblas_extended.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "../matrix.hpp"
#include "../models/model1d.hpp"
#include "../models/model_common.hpp"
#include "../variant.hpp"

using std::string;

namespace cg {

bool KernelTrtrsv::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  return rhs.isTransposed();
}

std::array<bool, 2U> KernelTrtrsv::needsNewMatrix() const {
  return {false, false};
}

void KernelTrtrsv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  result.setName(rhs.isModifiable() ? rhs.getName()
                                    : PREFIX_COPY + rhs.getName());
}

string KernelTrtrsv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo_lhs, uplo_rhs, trans_lhs, diag_lhs, diag_rhs, m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;
  side = left.isInverted() ? CG_LEFT : CG_RIGHT;

  uplo_lhs = lhs.isLower() ? CG_LOWER : CG_UPPER;
  uplo_rhs = rhs.isLower() ? CG_LOWER : CG_UPPER;
  trans_lhs = lhs.isTransposed() ? CG_TRANS : CG_NO_TRANS;
  diag_lhs = lhs.isUnit() ? CG_UNIT : CG_NO_UNIT;
  diag_rhs = rhs.isUnit() ? CG_UNIT : CG_NO_UNIT;

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  m = result.getRowName();

  code +=
      fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, 1.0, {}, {}, {}, {});\n",
                  CBLAS_DTRTRSV, LAYOUT, side, uplo_lhs, uplo_rhs, trans_lhs,
                  diag_lhs, diag_rhs, m, dataMatrix(lhs), strideMatrix(lhs),
                  dataMatrix(result), strideMatrix(result));
  if (lhs.isModifiable()) code += freeMatrix(lhs);

  return code;
}

string KernelTrtrsv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const string m = result.getRowName();
  string cost{""};

  if (result.isTriangular()) {
    cost = fmt::format("{0} += {1} * {1} * {1} / 3.0; {2}", COST_VAR, m,
                       infoInvocation(left, right, result));
  } else {
    cost = fmt::format("{0} += {1} * {1} * {1}; {2}", COST_VAR, m,
                       infoInvocation(left, right, result));
  }
  return cost;
}

double KernelTrtrsv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const double m = static_cast<double>(result.getNrows());
  double cost = m * m * m;

  if (result.isTriangular()) cost /= 3.0;
  return cost;
}

std::vector<Variant> KernelTrtrsv::getCoveredVariants() const {
  RangeVariant range{
      {triangular, {Property::FullRank}, all_trans, {Inversion::Y}},
      {triangular, all_properties, all_trans, {Inversion::N}}};
  return range.generateVariants();
}

string KernelTrtrsv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_TRTRSV, left, right, result);
}

void KernelTrtrsv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;
  const Matrix& _rhs = _left.isInverted() ? _right : _left;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _lhs.isLower() ? CblasLower : CblasUpper;
  CBLAS_UPLO UploB = _rhs.isLower() ? CblasLower : CblasUpper;
  CBLAS_TRANSPOSE TransA = _lhs.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG DiagA = _lhs.isUnit() ? CblasUnit : CblasNonUnit;
  CBLAS_DIAG DiagB = _rhs.isUnit() ? CblasUnit : CblasNonUnit;
  int M = left.ROWS;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  cblas_dtrtrsv(CblasColMajor, Side, UploA, UploB, TransA, DiagA, DiagB, M, 1.0,
                lhs.DATA, lhs.STRIDE, rhs.DATA, rhs.STRIDE);

  result = std::move(rhs);
}

void KernelTrtrsv::loadModel() { model.read(getPathModel()); }

double KernelTrtrsv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = (0x00);

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;
  if (right.isInverted()) mdl::setBitLR(key);
  if (lhs.isLower()) mdl::setBitUploA(key);
  if (rhs.isLower()) mdl::setBitUploB(key);
  if (lhs.isTransposed()) mdl::setBitTransA(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
