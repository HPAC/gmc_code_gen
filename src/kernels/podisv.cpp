#include "podisv.hpp"

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

bool KernelPodisv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelPodisv::needsNewMatrix() const {
  return {true, true};
}

void KernelPodisv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  return;
}

string KernelPodisv::generateCode(const Matrix& left, const Matrix& right,
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
                      CBLAS_DPODISV, LAYOUT, side, uplo, m, dataMatrix(lhs),
                      strideMatrix(lhs), dataMatrix(rhs), strideMatrix(rhs),
                      dataMatrix(result), strideMatrix(result));

  if (lhs.isModifiable()) code += freeMatrix(lhs);
  if (rhs.isModifiable()) code += freeMatrix(rhs);

  return code;
}

string KernelPodisv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string m = left.getRowName();
  return fmt::format("{0} += 5.0 / 3.0 * {1} * {1} * {1};{2}", COST_VAR, m,
                     infoInvocation(left, right, result));
}

double KernelPodisv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const double m = static_cast<double>(left.getNrows());
  return 5.0 / 3.0 * m * m * m;
}

std::vector<Variant> KernelPodisv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Symmetric_L, Structure::Symmetric_U},
       {Property::SPD},
       {Trans::N},
       {Inversion::Y}},
      {{Structure::Diagonal}, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

string KernelPodisv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_PODISV, left, right, result);
}

void KernelPodisv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _lhs.isSymmetricLower() ? CblasLower : CblasUpper;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;

  cblas_dpodisv(CblasColMajor, Side, UploA, M, lhs.DATA, lhs.STRIDE, rhs.DATA,
                rhs.STRIDE, result.DATA, result.STRIDE);
}

void KernelPodisv::loadModel() { model.read(getPathModel()); }

double KernelPodisv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  const Matrix& lhs = (left.isInverted()) ? left : right;
  if (!left.isInverted()) mdl::setBitLR(key);
  if (lhs.isSymmetricLower()) mdl::setBitUploA(key);

  unsigned m = left.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg