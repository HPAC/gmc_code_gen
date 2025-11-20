#include "disysv.hpp"

#include <fmt/core.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../../cblas_extended/cblas_extended.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "../matrix.hpp"
#include "../models/model_common.hpp"
#include "../variant.hpp"

using std::string;

namespace cg {

bool KernelDisysv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelDisysv::needsNewMatrix() const {
  return {false, false};
}

void KernelDisysv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {}

std::string KernelDisysv::generateCode(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  string side_diag, uplo_sy, m;

  const Matrix& diag = (left.isDiagonal()) ? left : right;
  const Matrix& symm = (left.isDiagonal()) ? right : left;

  side_diag = (left.isDiagonal()) ? CG_LEFT : CG_RIGHT;
  uplo_sy = (symm.isSymmetricLower()) ? CG_LOWER : CG_UPPER;
  m = result.getRowName();

  string code = infoInvocation(left, right, result);
  if (!symm.isModifiable()) code += createCopy(result, symm);

  code +=
      fmt::format("{}({}, {}, {}, {}, 1.0, {}, {}, {}, {});\n", CBLAS_DDISYSV,
                  LAYOUT, side_diag, uplo_sy, m, dataMatrix(diag),
                  strideMatrix(diag), dataMatrix(result), strideMatrix(result));
  if (diag.isModifiable()) code += freeMatrix(diag);
  if (symm.isModifiable()) code += freeMatrix(symm);

  return code;
}

std::string KernelDisysv::generateCost(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  return fmt::format("{} += {} * {}; {}", COST_VAR, result.getRowName(),
                     result.getColName(), infoInvocation(left, right, result));
}

double KernelDisysv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(result.getNcols());
}

std::vector<Variant> KernelDisysv::getCoveredVariants() const {
  RangeVariant range{{{Structure::Diagonal},
                      {Property::FullRank, Property::SPD},
                      {Trans::N},
                      {Inversion::Y}},
                     {{Structure::Symmetric_L, Structure::Symmetric_U},
                      all_properties,
                      {Trans::N},
                      {Inversion::N}}};
  return range.generateVariants();
}

std::string KernelDisysv::infoInvocation(const Matrix& left,
                                         const Matrix& right,
                                         const Matrix& result) const {
  return Kernel::infoInvocation(CG_DISYSV, left, right, result);
}

void KernelDisysv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _symm = _left.isDiagonal() ? _right : _left;
  CBLAS_SIDE Side = _left.isDiagonal() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploB = _symm.isSymmetricLower() ? CblasLower : CblasUpper;

  int M = left.ROWS;

  dMatrix& diag = _left.isDiagonal() ? left : right;
  dMatrix& symm = _left.isDiagonal() ? right : left;

  cblas_ddisysv(CblasColMajor, Side, UploB, M, 1.0, diag.DATA, diag.STRIDE,
                symm.DATA, symm.STRIDE);

  result = std::move(symm);
}

void KernelDisysv::loadModel() { model.read(getPathModel()); }

double KernelDisysv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  if (!left.isDiagonal()) mdl::setBitLR(key);
  const Matrix& symm = (left.isDiagonal()) ? right : left;
  if (symm.isSymmetricLower()) mdl::setBitUploB(key);

  unsigned m = left.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
