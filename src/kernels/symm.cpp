#include "symm.hpp"

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

bool KernelSymm::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& dense = (left.isDense()) ? left : right;
  return dense.isTransposed();
}

std::array<bool, 2U> KernelSymm::needsNewMatrix() const {
  return {true, false};
}

void KernelSymm::deduceName(const Matrix& left, const Matrix& right,
                            Matrix& result) const {}

string KernelSymm::generateCode(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  string side, uplo;

  const Matrix& symm = (left.isSymmetric()) ? left : right;
  const Matrix& dense = (!left.isSymmetric()) ? left : right;
  side = (left.isSymmetric()) ? CG_LEFT : CG_RIGHT;

  uplo = (symm.isSymmetricLower()) ? CG_LOWER : CG_UPPER;

  string code = infoInvocation(left, right, result);
  code += createMatrix(result);

  code += fmt::format(
      "{}({}, {}, {}, {}, {}, 1.0, {}, {}, {}, {}, 0.0, {}, {}); \n",
      CBLAS_DSYMM, LAYOUT, side, uplo, result.getRowName(), result.getColName(),
      dataMatrix(symm), strideMatrix(symm), dataMatrix(dense),
      strideMatrix(dense), dataMatrix(result), strideMatrix(result));

  if (symm.isModifiable()) code += freeMatrix(symm);
  if (dense.isModifiable()) code += freeMatrix(dense);

  return code;
}

string KernelSymm::generateCost(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  return fmt::format(
      "{} += 2.0 * {} * {} * {}; {}", COST_VAR, result.getRowName(),
      (!left.isTransposed()) ? left.getColName() : left.getRowName(),
      result.getColName(), infoInvocation(left, right, result));
}

double KernelSymm::computeFLOPs(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  unsigned shared_size =
      (!left.isTransposed()) ? left.getNcols() : left.getNrows();
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(shared_size) *
         static_cast<double>(result.getNcols()) * 2.0;
}

std::vector<Variant> KernelSymm::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Symmetric_L, Structure::Symmetric_U},
       all_properties,
       {Trans::N},
       {Inversion::N}},
      {{Structure::Dense}, all_properties, all_trans, {Inversion::N}}};
  return range.generateVariants();
}

string KernelSymm::infoInvocation(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return Kernel::infoInvocation(CG_SYMM, left, right, result);
}

void KernelSymm::execute(const Matrix& _left, const Matrix& _right,
                         dMatrix& left, dMatrix& right, dMatrix& result) const {
  const Matrix& _symm = _left.isSymmetric() ? _left : _right;

  CBLAS_SIDE Side = _left.isSymmetric() ? CblasLeft : CblasRight;
  CBLAS_UPLO Uplo = _symm.isSymmetricLower() ? CblasLower : CblasUpper;

  dMatrix& symm = _left.isSymmetric() ? left : right;
  dMatrix& dense = _left.isSymmetric() ? right : left;

  int M = left.ROWS;
  int N = right.COLS;

  cblas_dsymm(CblasColMajor, Side, Uplo, M, N, 1.0, symm.DATA, symm.STRIDE,
              dense.DATA, dense.STRIDE, 0.0, result.DATA, result.STRIDE);
}

void KernelSymm::loadModel() { model.read(getPathModel()); }

double KernelSymm::predictTime(const Matrix& left, const Matrix& right,
                               const Matrix& result) const {
  uint8_t key = (0x00);

  const Matrix& sym = (left.isSymmetric()) ? left : right;
  if (right.isSymmetric()) mdl::setBitLR(key);
  if (sym.isSymmetricLower()) mdl::setBitUploA(key);

  unsigned m = result.getNrows();
  unsigned n = result.getNcols();
  return computeFLOPs(left, right, result) / model.predict(key, m, n);
}

}  // namespace cg
