#include "disymm.hpp"

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

bool KernelDisymm::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelDisymm::needsNewMatrix() const {
  return {false, false};
}

void KernelDisymm::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& symm = (left.isSymmetric()) ? left : right;
  result.setName((symm.isModifiable()) ? symm.getName()
                                       : PREFIX_COPY + symm.getName());
}

string KernelDisymm::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side_diag, uplo_sy;

  const Matrix& diag = (left.isDiagonal()) ? left : right;
  const Matrix& symm = (left.isDiagonal()) ? right : left;

  side_diag = (left.isDiagonal()) ? CG_LEFT : CG_RIGHT;
  uplo_sy = (symm.isSymmetricLower()) ? CG_LOWER : CG_UPPER;

  string code = infoInvocation(left, right, result);
  if (!symm.isModifiable()) code += createCopy(result, symm);

  code += fmt::format("{}({}, {}, {}, {}, 1.0, {}, {}, {}, {});\n",
                      CBLAS_DDISYMM, LAYOUT, side_diag, uplo_sy,
                      result.getRowName(), dataMatrix(diag), strideMatrix(diag),
                      dataMatrix(result), strideMatrix(result));
  if (diag.isModifiable()) code += freeMatrix(diag);
  if (symm.isModifiable()) code += freeMatrix(symm);

  return code;
}

string KernelDisymm::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return fmt::format("{} += {} * {}; {}", COST_VAR, result.getRowName(),
                     result.getColName(), infoInvocation(left, right, result));
}

double KernelDisymm::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(result.getNcols());
}

std::vector<Variant> KernelDisymm::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Diagonal}, all_properties, {Trans::N}, {Inversion::N}},
      {{Structure::Symmetric_L, Structure::Symmetric_U},
       all_properties,
       {Trans::N},
       {Inversion::N}}};
  return range.generateVariants();
}

string KernelDisymm::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_DISYMM, left, right, result);
}

void KernelDisymm::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  dMatrix& diag = _left.isDiagonal() ? left : right;
  dMatrix& symm = _left.isDiagonal() ? right : left;

  const Matrix& _symm = _left.isDiagonal() ? _right : _left;
  CBLAS_SIDE Side = _left.isDiagonal() ? CblasLeft : CblasRight;
  CBLAS_UPLO Uplo = _symm.isSymmetricLower() ? CblasLower : CblasUpper;

  int M = left.ROWS;

  cblas_ddisymm(CblasColMajor, Side, Uplo, M, 1.0, diag.DATA, diag.STRIDE,
                symm.DATA, symm.STRIDE);

  result = std::move(symm);
}

void KernelDisymm::loadModel() { model.read(getPathModel()); }

double KernelDisymm::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  if (!left.isDiagonal()) mdl::setBitLR(key);
  const Matrix& symm = (left.isDiagonal()) ? right : left;
  if (symm.isSymmetricLower()) mdl::setBitUploB(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
