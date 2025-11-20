#include "trdisv.hpp"

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

bool KernelTrdisv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelTrdisv::needsNewMatrix() const {
  return {true, true};
}

void KernelTrdisv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  return;
}

string KernelTrdisv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo_lhs, trans_lhs, unit_lhs, m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  uplo_lhs = lhs.isLower() ? CG_LOWER : CG_UPPER;
  trans_lhs = lhs.isTransposed() ? CG_TRANS : CG_NO_TRANS;
  unit_lhs = lhs.isUnit() ? CG_UNIT : CG_NO_UNIT;
  m = result.getRowName();

  string code = infoInvocation(left, right, result);
  code += createMatrix(result);

  code +=
      fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});\n",
                  CBLAS_DTRDISV, LAYOUT, side, uplo_lhs, trans_lhs, unit_lhs, m,
                  1.0, dataMatrix(lhs), strideMatrix(lhs), dataMatrix(rhs),
                  strideMatrix(rhs), dataMatrix(result), strideMatrix(result));

  if (lhs.isModifiable()) code += freeMatrix(lhs);
  if (rhs.isModifiable()) code += freeMatrix(rhs);

  return code;
}

string KernelTrdisv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return fmt::format("{0} += {1} * {1} * {1} / 3.0;{2}", COST_VAR,
                     result.getRowName(), infoInvocation(left, right, result));
}

double KernelTrdisv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  double m = static_cast<double>(result.getNrows());
  return m * m * m / 3.0;
}

std::vector<Variant> KernelTrdisv::getCoveredVariants() const {
  RangeVariant range{
      {triangular, {Property::FullRank}, all_trans, {Inversion::Y}},
      {{Structure::Diagonal}, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

string KernelTrdisv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_TRDISV, left, right, result);
}

void KernelTrdisv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _lhs.isLower() ? CblasLower : CblasUpper;
  CBLAS_TRANSPOSE TransA = _lhs.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG DiagA = _lhs.isUnit() ? CblasUnit : CblasNonUnit;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;

  cblas_dtrdisv(CblasColMajor, Side, UploA, TransA, DiagA, M, 1.0, lhs.DATA,
                lhs.STRIDE, rhs.DATA, rhs.STRIDE, result.DATA, result.STRIDE);
}

void KernelTrdisv::loadModel() { model.read(getPathModel()); }

double KernelTrdisv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = (0x00);

  const Matrix& lhs = (left.isInverted()) ? left : right;
  if (right.isInverted()) mdl::setBitLR(key);
  if (lhs.isLower()) mdl::setBitUploA(key);
  if (lhs.isTransposed()) mdl::setBitTransA(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
