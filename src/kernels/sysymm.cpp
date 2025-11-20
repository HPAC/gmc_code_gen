#include "sysymm.hpp"

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

bool KernelSysymm::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelSysymm::needsNewMatrix() const {
  return {true, false};
}

void KernelSysymm::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {}

string KernelSysymm::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string uplo_left, uplo_right;

  uplo_left = (left.isSymmetricLower()) ? CG_LOWER : CG_UPPER;
  uplo_right = (right.isSymmetricLower()) ? CG_LOWER : CG_UPPER;

  string code = infoInvocation(left, right, result);
  code += createMatrix(result);
  code += fmt::format("{}({}, {}, {}, {}, 1.0, {}, {}, {}, {}, 0.0, {}, {});\n",
                      CBLAS_DSYSYMM, LAYOUT, uplo_left, uplo_right,
                      result.getRowName(), dataMatrix(left), strideMatrix(left),
                      dataMatrix(right), strideMatrix(right),
                      dataMatrix(result), strideMatrix(result));
  if (left.isModifiable()) code += freeMatrix(left);
  if (right.isModifiable()) code += freeMatrix(right);

  return code;
}

string KernelSysymm::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return fmt::format("{} += 2.0 * {} * {} * {}; {}", COST_VAR,
                     result.getRowName(), left.getColName(),
                     result.getColName(), infoInvocation(left, right, result));
}

double KernelSysymm::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(left.getNcols()) *
         static_cast<double>(result.getNcols()) * 2.0;
}

std::vector<Variant> KernelSysymm::getCoveredVariants() const {
  RangeVariant range{{{Structure::Symmetric_L, Structure::Symmetric_U},
                      all_properties,
                      {Trans::N},
                      {Inversion::N}},
                     {{Structure::Symmetric_L, Structure::Symmetric_U},
                      all_properties,
                      {Trans::N},
                      {Inversion::N}}};
  return range.generateVariants();
}

string KernelSysymm::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_SYSYMM, left, right, result);
}

void KernelSysymm::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  CBLAS_UPLO UploA = _left.isSymmetricLower() ? CblasLower : CblasUpper;
  CBLAS_UPLO UploB = _right.isSymmetricLower() ? CblasLower : CblasUpper;

  int M = left.ROWS;

  cblas_dsysymm(CblasColMajor, UploA, UploB, M, 1.0, left.DATA, left.STRIDE,
                right.DATA, right.STRIDE, 0.0, result.DATA, result.STRIDE);
}

void KernelSysymm::loadModel() { model.read(getPathModel()); }

double KernelSysymm::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  if (left.isSymmetricLower()) mdl::setBitUploA(key);
  if (right.isSymmetricLower()) mdl::setBitUploB(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
