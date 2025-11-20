#include "trtrmm.hpp"

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

bool KernelTrtrmm::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelTrtrmm::needsNewMatrix() const {
  return {true, false};
}

void KernelTrtrmm::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {}

string KernelTrtrmm::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string uplo_l, uplo_r, trans_l, trans_r, diag_l, diag_r;

  uplo_l = (left.isLower()) ? CG_LOWER : CG_UPPER;
  trans_l = (left.isTransposed()) ? CG_TRANS : CG_NO_TRANS;
  diag_l = (left.isUnit()) ? CG_UNIT : CG_NO_UNIT;

  uplo_r = (right.isLower()) ? CG_LOWER : CG_UPPER;
  trans_r = (right.isTransposed()) ? CG_TRANS : CG_NO_TRANS;
  diag_r = (right.isUnit()) ? CG_UNIT : CG_NO_UNIT;

  string code = infoInvocation(left, right, result);
  code += createMatrix(result);
  code += fmt::format(
      "{}({}, {}, {}, {}, {}, {}, {}, {}, 1.0, {}, {}, {}, {}, {}, {});\n",
      CBLAS_DTRTRMM, LAYOUT, uplo_l, uplo_r, trans_l, trans_r, diag_l, diag_r,
      result.getRowName(), dataMatrix(left), strideMatrix(left),
      dataMatrix(right), strideMatrix(right), dataMatrix(result),
      strideMatrix(result));

  if (left.isModifiable()) code += freeMatrix(left);
  if (right.isModifiable()) code += freeMatrix(right);

  return code;
}

string KernelTrtrmm::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  auto size = result.getRowName();

  if (result.isTriangular()) {
    return fmt::format(
        "{0} += {1} * {1} * {1} / 3.0 + 2.5 * {1} * {1} + 13.0 / 6.0 * {1}; "
        "{2}",
        COST_VAR, size, infoInvocation(left, right, result));
  } else {
    return fmt::format(
        "{0} += (2.0 / 3.0) * {1} * {1} * {1} + 4.0 * {1} * {1} + {1} / 3.0; "
        "{2}",
        COST_VAR, size, infoInvocation(left, right, result));
  }
}

double KernelTrtrmm::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  double m = static_cast<double>(result.getNrows());

  // can only be triangular if structure(op(A)) == structure(op(B))
  if (result.isTriangular()) {
    return m * m * m / 3.0 + 2.5 * m * m + 13.0 / 6.0 * m;
  } else {
    return 2.0 / 3.0 * m * m * m + 4.0 * m * m + m / 3.0;
  }
}

std::vector<Variant> KernelTrtrmm::getCoveredVariants() const {
  RangeVariant range{{triangular, all_properties, all_trans, {Inversion::N}},
                     {triangular, all_properties, all_trans, {Inversion::N}}};
  return range.generateVariants();
}

string KernelTrtrmm::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_TRTRMM, left, right, result);
}

void KernelTrtrmm::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  CBLAS_UPLO UploA = _left.isLower() ? CblasLower : CblasUpper;
  CBLAS_UPLO UploB = _right.isLower() ? CblasLower : CblasUpper;
  CBLAS_TRANSPOSE TransA = _left.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE TransB = _right.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG DiagA = _left.isUnit() ? CblasUnit : CblasNonUnit;
  CBLAS_DIAG DiagB = _right.isUnit() ? CblasUnit : CblasNonUnit;

  int M = left.ROWS;

  cblas_dtrtrmm(CblasColMajor, UploA, UploB, TransA, TransB, DiagA, DiagB, M,
                1.0, left.DATA, left.STRIDE, right.DATA, right.STRIDE,
                result.DATA, result.STRIDE);
}

void KernelTrtrmm::loadModel() { model.read(getPathModel()); }

double KernelTrtrmm::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = (0x00);

  if (left.isLower()) mdl::setBitUploA(key);
  if (right.isLower()) mdl::setBitUploB(key);
  if (left.isTransposed()) mdl::setBitTransA(key);
  if (right.isTransposed()) mdl::setBitTransB(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
