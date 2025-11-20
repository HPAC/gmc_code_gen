#include "potrsv.hpp"

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

bool KernelPotrsv::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  return rhs.isTransposed();
}

std::array<bool, 2U> KernelPotrsv::needsNewMatrix() const {
  return {false, false};
}

void KernelPotrsv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  result.setName(rhs.isModifiable() ? rhs.getName()
                                    : PREFIX_COPY + rhs.getName());
}

string KernelPotrsv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo_lhs, uplo_rhs, diag_rhs, m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  uplo_lhs = lhs.isSymmetricLower() ? CG_LOWER : CG_UPPER;
  uplo_rhs = rhs.isLower() ? CG_LOWER : CG_UPPER;
  diag_rhs = rhs.isUnit() ? CG_UNIT : CG_NO_UNIT;
  m = left.getRowName();

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  code += fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {}, {});\n",
                      CBLAS_DPOTRSV, LAYOUT, side, uplo_lhs, uplo_rhs, diag_rhs,
                      m, dataMatrix(lhs), strideMatrix(lhs), dataMatrix(result),
                      strideMatrix(result));

  if (lhs.isModifiable()) code += freeMatrix(lhs);

  return code;
}

string KernelPotrsv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  bool lhs_right = !left.isInverted();
  const Matrix& rhs = left.isInverted() ? right : left;
  bool rhs_lower = rhs.isLower();

  string multiplier = (lhs_right == rhs_lower) ? "7.0 / 3.0" : "5.0 / 3.0";
  string m = left.getRowName();

  return fmt::format("{0} += {1} * {2} * {2} * {2};{3}", COST_VAR, multiplier,
                     m, infoInvocation(left, right, result));
}

double KernelPotrsv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  bool lhs_right = !left.isInverted();
  const Matrix& rhs = left.isInverted() ? right : left;
  bool rhs_lower = rhs.isLower();

  const double multiplier = (lhs_right == rhs_lower) ? 7.0 / 3.0 : 5.0 / 3.0;
  const double m = static_cast<double>(left.getNrows());

  return multiplier * m * m * m;
}

std::vector<Variant> KernelPotrsv::getCoveredVariants() const {
  RangeVariant range{{{Structure::Symmetric_L, Structure::Symmetric_U},
                      {Property::SPD},
                      {Trans::N},
                      {Inversion::Y}},
                     {triangular, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

string KernelPotrsv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_POTRSV, left, right, result);
}

void KernelPotrsv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;
  const Matrix& _rhs = _left.isInverted() ? _right : _left;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _lhs.isSymmetricLower() ? CblasLower : CblasUpper;
  CBLAS_UPLO UploB = _rhs.isLower() ? CblasLower : CblasUpper;
  CBLAS_DIAG DiagB = _rhs.isUnit() ? CblasUnit : CblasNonUnit;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;

  cblas_dpotrsv(CblasColMajor, Side, UploA, UploB, DiagB, M, lhs.DATA,
                lhs.STRIDE, rhs.DATA, rhs.STRIDE);

  result = std::move(rhs);
}

void KernelPotrsv::loadModel() { model.read(getPathModel()); }

double KernelPotrsv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  const Matrix& lhs = (left.isInverted()) ? left : right;
  const Matrix& rhs = (left.isInverted()) ? right : left;
  if (!left.isInverted()) mdl::setBitLR(key);
  if (lhs.isSymmetricLower()) mdl::setBitUploA(key);
  if (rhs.isLower()) mdl::setBitUploB(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg