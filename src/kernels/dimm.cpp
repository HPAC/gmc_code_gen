#include "dimm.hpp"

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

bool KernelDimm::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& dense = (left.isDense()) ? left : right;
  return dense.isTransposed();
}

std::array<bool, 2U> KernelDimm::needsNewMatrix() const {
  return {false, false};
}

void KernelDimm::deduceName(const Matrix& left, const Matrix& right,
                            Matrix& result) const {
  const Matrix& dense = (left.isDense()) ? left : right;
  result.name = (dense.isModifiable()) ? dense.name : PREFIX_COPY + dense.name;
}

string KernelDimm::generateCode(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  string side, m, n;

  side = (left.isDiagonal()) ? CG_LEFT : CG_RIGHT;
  const Matrix& diag = (left.isDiagonal()) ? left : right;
  const Matrix& dense = (left.isDiagonal()) ? right : left;

  m = result.getRowName();
  n = result.getColName();

  string code = infoInvocation(left, right, result);
  if (!dense.isModifiable()) code += createCopy(result, dense);

  code += fmt::format("{}({}, {}, {}, {}, 1.0, {}, {}, {}, {});\n", CBLAS_DDIMM,
                      LAYOUT, side, m, n, dataMatrix(diag), strideMatrix(diag),
                      dataMatrix(result), strideMatrix(result));
  if (diag.isModifiable()) code += freeMatrix(diag);

  return code;
}

string KernelDimm::generateCost(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  return fmt::format("{} += {} * {}; {}", COST_VAR, result.getRowName(),
                     result.getColName(), infoInvocation(left, right, result));
}

double KernelDimm::computeFLOPs(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(result.getNcols());
}

std::vector<Variant> KernelDimm::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Diagonal}, all_properties, {Trans::N}, {Inversion::N}},
      {{Structure::Dense}, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

string KernelDimm::infoInvocation(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return Kernel::infoInvocation(CG_DIMM, left, right, result);
}

void KernelDimm::execute(const Matrix& _left, const Matrix& _right,
                         dMatrix& left, dMatrix& right, dMatrix& result) const {
  dMatrix& diag = _left.isDiagonal() ? left : right;
  dMatrix& dense = _left.isDiagonal() ? right : left;
  CBLAS_SIDE Side = _left.isDiagonal() ? CblasLeft : CblasRight;

  int M = left.ROWS;
  int N = right.COLS;

  cblas_ddimm(CblasColMajor, Side, M, N, 1.0, diag.DATA, diag.STRIDE,
              dense.DATA, dense.STRIDE);

  result = std::move(dense);
}

void KernelDimm::loadModel() { model.read(getPathModel()); }

double KernelDimm::predictTime(const Matrix& left, const Matrix& right,
                               const Matrix& result) const {
  uint8_t key = {0x00};

  if (right.isDiagonal()) mdl::setBitLR(key);

  unsigned m = result.getNrows();
  unsigned n = result.getNcols();

  return computeFLOPs(left, right, result) / model.predict(key, m, n);
}

}  // namespace cg
