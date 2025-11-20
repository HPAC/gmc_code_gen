#include "trsymm.hpp"

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

bool KernelTrsymm::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelTrsymm::needsNewMatrix() const {
  return {false, false};
}

void KernelTrsymm::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& symmetric = (left.isSymmetric()) ? left : right;
  result.setName((symmetric.isModifiable())
                     ? symmetric.getName()
                     : PREFIX_COPY + symmetric.getName());
}

string KernelTrsymm::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo_tr, uplo_sy, trans_tr, diag_tr;

  const Matrix& tr = (left.isTriangular()) ? left : right;
  const Matrix& symm = (left.isTriangular()) ? right : left;

  side = (left.isTriangular()) ? CG_LEFT : CG_RIGHT;
  uplo_tr = (tr.isLower()) ? CG_LOWER : CG_UPPER;
  trans_tr = (tr.isTransposed()) ? CG_TRANS : CG_NO_TRANS;
  diag_tr = (tr.isUnit()) ? CG_UNIT : CG_NO_UNIT;
  uplo_sy = (symm.isSymmetricLower()) ? CG_LOWER : CG_UPPER;

  string code = infoInvocation(left, right, result);
  if (!symm.isModifiable()) code += createCopy(result, symm);

  code +=
      fmt::format("{}({}, {}, {}, {}, {}, {}, {}, 1.0, {}, {}, {}, {});\n",
                  CBLAS_DTRSYMM, LAYOUT, side, uplo_tr, uplo_sy, trans_tr,
                  diag_tr, result.getRowName(), dataMatrix(tr),
                  strideMatrix(tr), dataMatrix(result), strideMatrix(result));
  if (tr.isModifiable()) code += freeMatrix(tr);

  return code;
}

string KernelTrsymm::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return fmt::format(
      "{} += {} * {} * {}; {}", COST_VAR, result.getRowName(),
      (!left.isTransposed()) ? left.getColName() : left.getRowName(),
      result.getColName(), infoInvocation(left, right, result));
}

double KernelTrsymm::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  double m = static_cast<double>(result.getNrows());
  // return m * m * m + 2.0 * m * m;
  return m * m * m;
}

std::vector<Variant> KernelTrsymm::getCoveredVariants() const {
  RangeVariant range{{triangular, all_properties, all_trans, {Inversion::N}},
                     {{Structure::Symmetric_L, Structure::Symmetric_U},
                      all_properties,
                      {Trans::N},
                      {Inversion::N}}};
  return range.generateVariants();
}

string KernelTrsymm::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_TRSYMM, left, right, result);
}

void KernelTrsymm::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _tr = _left.isTriangular() ? _left : _right;
  const Matrix& _symm = _left.isTriangular() ? _right : _left;

  CBLAS_SIDE Side = _left.isTriangular() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _tr.isLower() ? CblasLower : CblasUpper;
  CBLAS_UPLO UploB = _symm.isSymmetricLower() ? CblasLower : CblasUpper;
  CBLAS_TRANSPOSE TransA = _tr.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG Diag = _tr.isUnit() ? CblasUnit : CblasNonUnit;

  dMatrix& tr = _left.isTriangular() ? left : right;
  dMatrix& symm = _left.isTriangular() ? right : left;

  int M = left.ROWS;

  cblas_dtrsymm(CblasColMajor, Side, UploA, UploB, TransA, Diag, M, 1.0,
                tr.DATA, tr.STRIDE, symm.DATA, symm.STRIDE);

  result = std::move(symm);
}

void KernelTrsymm::loadModel() { model.read(getPathModel()); }

double KernelTrsymm::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = (0x00);

  const Matrix& tr = left.isTriangular() ? left : right;
  const Matrix& sym = left.isTriangular() ? right : left;

  if (right.isTriangular()) mdl::setBitLR(key);
  if (tr.isLower()) mdl::setBitUploA(key);
  if (sym.isSymmetricLower()) mdl::setBitUploB(key);
  if (tr.isTransposed()) mdl::setBitTransA(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
