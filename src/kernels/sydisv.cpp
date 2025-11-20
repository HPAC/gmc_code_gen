#include "sydisv.hpp"

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

bool KernelSydisv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelSydisv::needsNewMatrix() const {
  return {true, true};
}

void KernelSydisv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  return;
}

string KernelSydisv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo, m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  uplo = lhs.isSymmetricLower() ? CG_LOWER : CG_UPPER;
  m = left.getRowName();

  string code = infoInvocation(left, right, result);
  code += createMatrix(result);

  code += fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {}, {});\n",
                      CBLAS_DSYDISV, LAYOUT, side, uplo, m, dataMatrix(lhs),
                      strideMatrix(lhs), dataMatrix(rhs), strideMatrix(rhs),
                      dataMatrix(result), strideMatrix(result));

  if (lhs.isModifiable()) code += freeMatrix(lhs);
  if (rhs.isModifiable()) code += freeMatrix(rhs);

  return code;
}

string KernelSydisv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string m = left.getRowName();
  return fmt::format("{0} += 7.0 / 3.0 * {1} * {1} * {1};{2}", COST_VAR, m,
                     infoInvocation(left, right, result));
}

double KernelSydisv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const double m = static_cast<double>(left.getNrows());
  return 7.0 / 3.0 * m * m * m;
}

std::vector<Variant> KernelSydisv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Symmetric_L, Structure::Symmetric_U},
       {Property::FullRank},
       {Trans::N},
       {Inversion::Y}},
      {{Structure::Diagonal}, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

string KernelSydisv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_SYDISV, left, right, result);
}

void KernelSydisv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _lhs.isSymmetricLower() ? CblasLower : CblasUpper;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;

  cblas_dsydisv(CblasColMajor, Side, UploA, M, lhs.DATA, lhs.STRIDE, rhs.DATA,
                rhs.STRIDE, result.DATA, result.STRIDE);
}

void KernelSydisv::loadModel() { model.read(getPathModel()); }

double KernelSydisv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  const Matrix& lhs = (left.isInverted()) ? left : right;
  if (!left.isInverted()) mdl::setBitLR(key);
  if (lhs.isSymmetricLower()) mdl::setBitUploA(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
