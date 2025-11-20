#include "gemm.hpp"

#include <fmt/core.h>
#include <openblas/cblas.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../kernel.hpp"
#include "../macros.hpp"
#include "../matrix.hpp"
#include "../models/model3d.hpp"
#include "../models/model_common.hpp"
#include "../variant.hpp"

using std::string;

namespace cg {

bool KernelGemm::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelGemm::needsNewMatrix() const {
  return {true, false};
}

void KernelGemm::deduceName(const Matrix& left, const Matrix& right,
                            Matrix& result) const {}

string KernelGemm::generateCode(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  string transa, transb;
  string m, k, n;

  if (!left.isTransposed()) {
    transa = CG_NO_TRANS;
    m = left.getRowName();
    k = left.getColName();
  } else {
    transa = CG_TRANS;
    m = left.getColName();
    k = left.getRowName();
  }

  if (!right.isTransposed()) {
    transb = CG_NO_TRANS;
    n = right.getColName();
  } else {
    transb = CG_TRANS;
    n = right.getRowName();
  }

  string code = infoInvocation(left, right, result);
  code += createMatrix(result);

  code += fmt::format(
      "{}({}, {}, {}, {}, {}, {}, 1.0, {}, {}, {}, {}, 0.0, {}, {});\n",
      CBLAS_DGEMM, LAYOUT, transa, transb, m, n, k, dataMatrix(left),
      strideMatrix(left), dataMatrix(right), strideMatrix(right),
      dataMatrix(result), strideMatrix(result));

  if (left.isModifiable()) code += freeMatrix(left);
  if (right.isModifiable()) code += freeMatrix(right);

  return code;
}

string KernelGemm::generateCost(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  return fmt::format(
      "{} += 2.0 * {} * {} * {}; {}", COST_VAR, result.getRowName(),
      (!left.isTransposed()) ? left.getColName() : left.getRowName(),
      result.getColName(), infoInvocation(left, right, result));
}

double KernelGemm::computeFLOPs(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  unsigned shared_size =
      (!left.isTransposed()) ? left.getNcols() : left.getNrows();
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(shared_size) *
         static_cast<double>(result.getNcols()) * 2.0;
}

std::vector<Variant> KernelGemm::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Dense}, all_properties, all_trans, {Inversion::N}},
      {{Structure::Dense}, all_properties, all_trans, {Inversion::N}}};
  return range.generateVariants();
}

string KernelGemm::infoInvocation(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return Kernel::infoInvocation(CG_GEMM, left, right, result);
}

void KernelGemm::execute(const Matrix& _left, const Matrix& _right,
                         dMatrix& left, dMatrix& right, dMatrix& result) const {
  CBLAS_TRANSPOSE TransA = _left.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE TransB = _right.isTransposed() ? CblasTrans : CblasNoTrans;

  int M = _left.isTransposed() ? left.COLS : left.ROWS;
  int N = _right.isTransposed() ? right.ROWS : right.COLS;
  int K = _left.isTransposed() ? left.ROWS : left.COLS;

  cblas_dgemm(CblasColMajor, TransA, TransB, M, N, K, 1.0, left.DATA,
              left.STRIDE, right.DATA, right.STRIDE, 0.0, result.DATA,
              result.STRIDE);
}

void KernelGemm::loadModel() { model.read(getPathModel()); }

double KernelGemm::predictTime(const Matrix& left, const Matrix& right,
                               const Matrix& result) const {
  uint8_t key = {0x00};

  if (left.isTransposed()) mdl::setBitTransA(key);
  if (right.isTransposed()) mdl::setBitTransB(key);

  unsigned m = result.getNrows();
  unsigned k = left.isTransposed() ? left.getNrows() : left.getNcols();
  unsigned n = result.getNcols();
  return computeFLOPs(left, right, result) / model.predict(key, m, k, n);
}

}  // namespace cg
