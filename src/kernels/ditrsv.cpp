#include "ditrsv.hpp"

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

bool KernelDitrsv::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& tr = (left.isTriangular()) ? left : right;
  return tr.isTransposed();
}

std::array<bool, 2U> KernelDitrsv::needsNewMatrix() const {
  return {false, false};
}

void KernelDitrsv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& tr = (left.isTriangular()) ? left : right;
  result.setName((tr.isModifiable()) ? tr.getName()
                                     : PREFIX_COPY + tr.getName());
}

std::string KernelDitrsv::generateCode(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  string side_diag, uplo_tr, diag_tr;

  side_diag = (left.isDiagonal()) ? CG_LEFT : CG_RIGHT;
  const Matrix& diag = (left.isDiagonal()) ? left : right;
  const Matrix& tr = (left.isDiagonal()) ? right : left;

  uplo_tr = (tr.isLower()) ? CG_LOWER : CG_UPPER;
  diag_tr = (tr.isUnit()) ? CG_UNIT : CG_NO_UNIT;

  string code = infoInvocation(left, right, result);
  if (!tr.isModifiable()) code += createCopy(result, tr);

  code += fmt::format("{}({}, {}, {}, {}, {}, 1.0, {}, {}, {}, {});\n",
                      CBLAS_DDITRSV, LAYOUT, side_diag, uplo_tr, diag_tr,
                      result.getRowName(), dataMatrix(diag), strideMatrix(diag),
                      dataMatrix(result), strideMatrix(result));

  if (diag.isModifiable()) code += freeMatrix(diag);

  return code;
}

std::string KernelDitrsv::generateCost(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  return fmt::format("{} += 0.5 * {} * {}; {}", COST_VAR, result.getRowName(),
                     result.getColName(), infoInvocation(left, right, result));
}

double KernelDitrsv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(result.getNcols()) * 0.5;
}

std::vector<Variant> KernelDitrsv::getCoveredVariants() const {
  RangeVariant range{{{Structure::Diagonal},
                      {Property::FullRank, Property::SPD},
                      {Trans::N},
                      {Inversion::Y}},
                     {triangular, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

std::string KernelDitrsv::infoInvocation(const Matrix& left,
                                         const Matrix& right,
                                         const Matrix& result) const {
  return Kernel::infoInvocation(CG_DITRSV, left, right, result);
}

void KernelDitrsv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _tr = _left.isDiagonal() ? _right : _left;
  CBLAS_SIDE Side = _left.isDiagonal() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploB = _tr.isLower() ? CblasLower : CblasUpper;
  CBLAS_DIAG DiagB = _tr.isUnit() ? CblasUnit : CblasNonUnit;

  dMatrix& diag = _left.isDiagonal() ? left : right;
  dMatrix& tr = _left.isDiagonal() ? right : left;

  int M = left.ROWS;

  cblas_dditrsv(CblasColMajor, Side, UploB, DiagB, M, 1.0, diag.DATA,
                diag.STRIDE, tr.DATA, tr.STRIDE);

  result = std::move(tr);
}

void KernelDitrsv::loadModel() { model.read(getPathModel()); }

double KernelDitrsv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  const Matrix& tr = (left.isDiagonal()) ? right : left;
  if (!left.isDiagonal()) mdl::setBitLR(key);
  if (tr.isLower()) mdl::setBitUploB(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
