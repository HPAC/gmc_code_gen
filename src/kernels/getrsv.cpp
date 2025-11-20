#include "getrsv.hpp"

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

bool KernelGetrsv::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  return rhs.isTransposed();
}

std::array<bool, 2U> KernelGetrsv::needsNewMatrix() const {
  return {false, false};
}

void KernelGetrsv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  result.setName(rhs.isModifiable() ? rhs.getName()
                                    : PREFIX_COPY + rhs.getName());
}

string KernelGetrsv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo_rhs, trans_lhs, diag_rhs, m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  uplo_rhs = rhs.isLower() ? CG_LOWER : CG_UPPER;
  trans_lhs = lhs.isTransposed() ? CG_TRANS : CG_NO_TRANS;
  diag_rhs = rhs.isUnit() ? CG_UNIT : CG_NO_UNIT;
  m = result.getRowName();

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  code += fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {}, {});\n",
                      CBLAS_DGETRSV, LAYOUT, side, uplo_rhs, trans_lhs,
                      diag_rhs, m, dataMatrix(lhs), strideMatrix(lhs),
                      dataMatrix(result), strideMatrix(result));

  if (lhs.isModifiable()) code += freeMatrix(lhs);

  return code;
}

string KernelGetrsv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  bool lhs_right = !left.isInverted();
  bool lower = rhs.isLower();

  string multiplier = (lhs_right == lower) ? "8.0 / 3.0" : "2.0";
  string m = result.getRowName();

  return fmt::format("{0} += {1} * {2} * {2} * {2};{3}", COST_VAR, multiplier,
                     m, infoInvocation(left, right, result));
}

double KernelGetrsv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  bool lhs_right = !left.isInverted();
  bool lower = rhs.isLower();

  double multiplier = (lhs_right == lower) ? 8.0 / 3.0 : 2.0;
  double m = static_cast<double>(result.getNrows());

  return multiplier * m * m * m;
}

std::vector<Variant> KernelGetrsv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Dense}, {Property::FullRank}, all_trans, {Inversion::Y}},
      {triangular, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

string KernelGetrsv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_GETRSV, left, right, result);
}

void KernelGetrsv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;
  const Matrix& _rhs = _left.isInverted() ? _right : _left;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploB = _rhs.isLower() ? CblasLower : CblasUpper;
  CBLAS_TRANSPOSE TransA = _lhs.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG DiagB = _rhs.isUnit() ? CblasUnit : CblasNonUnit;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;

  cblas_dgetrsv(CblasColMajor, Side, UploB, TransA, DiagB, M, lhs.DATA,
                lhs.STRIDE, rhs.DATA, rhs.STRIDE);

  result = std::move(rhs);
}

void KernelGetrsv::loadModel() { model.read(getPathModel()); }

double KernelGetrsv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = (0x00);

  const Matrix& lhs = (left.isInverted()) ? left : right;
  const Matrix& rhs = (left.isInverted()) ? right : left;
  if (!left.isInverted()) mdl::setBitLR(key);
  if (rhs.isLower()) mdl::setBitUploB(key);
  if (lhs.isTransposed()) mdl::setBitTransA(key);

  unsigned m = left.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
