#include "trsm.hpp"

#include <fmt/core.h>
#include <openblas/cblas.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../kernel.hpp"
#include "../macros.hpp"
#include "../matrix.hpp"
#include "../models/model2d.hpp"
#include "../models/model_common.hpp"
#include "../variant.hpp"

using std::string;

namespace cg {

bool KernelTrsm::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& dense = (left.isDense()) ? left : right;
  return dense.isTransposed();
}

std::array<bool, 2U> KernelTrsm::needsNewMatrix() const {
  return {false, false};
}

void KernelTrsm::deduceName(const Matrix& left, const Matrix& right,
                            Matrix& result) const {
  const Matrix& dense = (left.isDense()) ? left : right;
  result.setName(dense.isModifiable() ? dense.getName()
                                      : PREFIX_COPY + dense.getName());
}

string KernelTrsm::generateCode(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  string side, uplo, trans, diag;
  string m, n;

  const Matrix& tr = (left.isTriangular()) ? left : right;
  const Matrix& dense = (left.isTriangular()) ? right : left;

  side = (left.isTriangular()) ? CG_LEFT : CG_RIGHT;
  uplo = (tr.isLower()) ? CG_LOWER : CG_UPPER;
  trans = (tr.isTransposed()) ? CG_TRANS : CG_NO_TRANS;
  diag = (tr.isUnit()) ? CG_UNIT : CG_NO_UNIT;

  string code = infoInvocation(left, right, result);
  if (!dense.isModifiable()) code += createCopy(result, dense);

  m = result.getRowName();
  n = result.getColName();

  code += fmt::format("{}({}, {}, {}, {}, {}, {}, {}, 1.0, {}, {}, {}, {});\n",
                      CBLAS_DTRSM, LAYOUT, side, uplo, trans, diag, m, n,
                      dataMatrix(tr), strideMatrix(tr), dataMatrix(result),
                      strideMatrix(result));

  if (tr.isModifiable()) code += freeMatrix(tr);

  return code;
}

std::string KernelTrsm::generateCost(const Matrix& left, const Matrix& right,
                                     const Matrix& result) const {
  return fmt::format(
      "{} += {} * {} * {}; {}", COST_VAR, result.getRowName(),
      (!left.isTransposed()) ? left.getColName() : left.getRowName(),
      result.getColName(), infoInvocation(left, right, result));
}

double KernelTrsm::computeFLOPs(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  unsigned shared_size =
      (!left.isTransposed()) ? left.getNcols() : left.getNrows();
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(shared_size) *
         static_cast<double>(result.getNcols());
}

std::vector<Variant> KernelTrsm::getCoveredVariants() const {
  RangeVariant range{
      {triangular, {Property::FullRank}, all_trans, {Inversion::Y}},
      {{Structure::Dense}, all_properties, all_trans, {Inversion::N}}};
  // A dense matrix cannot be SPD (not all combinations are legal)
  // but that combination will never be generated since it's not legal.
  return range.generateVariants();
}

std::string KernelTrsm::infoInvocation(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  return Kernel::infoInvocation(CG_TRSM, left, right, result);
}

void KernelTrsm::execute(const Matrix& _left, const Matrix& _right,
                         dMatrix& left, dMatrix& right, dMatrix& result) const {
  const Matrix& _tr = _left.isTriangular() ? _left : _right;

  CBLAS_SIDE Side = _left.isTriangular() ? CblasLeft : CblasRight;
  CBLAS_UPLO Uplo = _tr.isLower() ? CblasLower : CblasUpper;
  CBLAS_TRANSPOSE TransA = _tr.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG Diag = _tr.isUnit() ? CblasUnit : CblasNonUnit;

  dMatrix& tr = _left.isTriangular() ? left : right;
  dMatrix& dense = _left.isTriangular() ? right : left;

  int M = left.ROWS;
  int N = right.COLS;

  cblas_dtrsm(CblasColMajor, Side, Uplo, TransA, Diag, M, N, 1.0, tr.DATA,
              tr.STRIDE, dense.DATA, dense.STRIDE);

  result = std::move(dense);
}

void KernelTrsm::loadModel() { model.read(getPathModel()); }

double KernelTrsm::predictTime(const Matrix& left, const Matrix& right,
                               const Matrix& result) const {
  uint8_t key = (0x00);

  const Matrix& lhs = (left.isInverted()) ? left : right;
  if (right.isInverted()) mdl::setBitLR(key);
  if (lhs.isLower()) mdl::setBitUploA(key);
  if (lhs.isTransposed()) mdl::setBitTransA(key);

  unsigned m = result.getNrows();
  unsigned n = result.getNcols();
  return computeFLOPs(left, right, result) / model.predict(key, m, n);
}

}  // namespace cg
