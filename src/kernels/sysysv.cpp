#include "sysysv.hpp"

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

bool KernelSysysv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelSysysv::needsNewMatrix() const {
  return {false, false};
}

void KernelSysysv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  result.setName(rhs.isModifiable() ? rhs.getName()
                                    : PREFIX_COPY + rhs.getName());
}

string KernelSysysv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo_lhs, uplo_rhs, m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  uplo_lhs = lhs.isSymmetricLower() ? CG_LOWER : CG_UPPER;
  uplo_rhs = rhs.isSymmetricLower() ? CG_LOWER : CG_UPPER;

  m = rhs.getRowName();

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  code +=
      fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {});\n", CBLAS_DSYSYSV,
                  LAYOUT, side, uplo_lhs, uplo_rhs, m, dataMatrix(lhs),
                  strideMatrix(lhs), dataMatrix(result), strideMatrix(result));
  if (lhs.isModifiable()) code += freeMatrix(lhs);

  return code;
}

string KernelSysysv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string m = left.getRowName();
  return fmt::format("{0} += 7.0 / 3.0 * {1} * {1} * {1};{2}", COST_VAR, m,
                     infoInvocation(left, right, result));
}

double KernelSysysv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const double m = static_cast<double>(left.getNrows());
  return 7.0 / 3.0 * m * m * m;
}

std::vector<Variant> KernelSysysv::getCoveredVariants() const {
  RangeVariant range{{{Structure::Symmetric_L, Structure::Symmetric_U},
                      {Property::FullRank},
                      {Trans::N},
                      {Inversion::Y}},
                     {{Structure::Symmetric_L, Structure::Symmetric_U},
                      all_properties,
                      {Trans::N},
                      {Inversion::N}}};

  return range.generateVariants();
}

string KernelSysysv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_SYSYSV, left, right, result);
}

void KernelSysysv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;
  const Matrix& _rhs = _left.isInverted() ? _right : _left;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _lhs.isSymmetricLower() ? CblasLower : CblasUpper;
  CBLAS_UPLO UploB = _rhs.isSymmetricLower() ? CblasLower : CblasUpper;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;

  cblas_dsysysv(CblasColMajor, Side, UploA, UploB, M, lhs.DATA, lhs.STRIDE,
                rhs.DATA, rhs.STRIDE);

  result = std::move(rhs);
}

void KernelSysysv::loadModel() { model.read(getPathModel()); }

double KernelSysysv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = (0x00);

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;
  if (right.isInverted()) mdl::setBitLR(key);
  if (lhs.isSymmetricLower()) mdl::setBitUploA(key);
  if (rhs.isSymmetricLower()) mdl::setBitUploB(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
