#include "disv.hpp"

#include <fmt/core.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../../cblas_extended/cblas_extended.hpp"
#include "../kernel.hpp"
#include "../macros.hpp"
#include "../matrix.hpp"
#include "../models/model2d.hpp"
#include "../models/model_common.hpp"
#include "../variant.hpp"

using std::string;

namespace cg {

bool KernelDisv::tweakTransposition(Matrix& left, Matrix& right) const {
  const Matrix& dense = (left.isDense()) ? left : right;
  return dense.isTransposed();
}

std::array<bool, 2U> KernelDisv::needsNewMatrix() const {
  return {false, false};
}

void KernelDisv::deduceName(const Matrix& left, const Matrix& right,
                            Matrix& result) const {
  const Matrix& dense = (left.isDense()) ? left : right;
  result.name = (dense.isModifiable()) ? dense.name : PREFIX_COPY + dense.name;
}

std::string KernelDisv::generateCode(const Matrix& left, const Matrix& right,
                                     const Matrix& result) const {
  string side, m, n;

  side = left.isDiagonal() ? CG_LEFT : CG_RIGHT;
  const Matrix& diag = left.isDiagonal() ? left : right;
  const Matrix& dense = left.isDiagonal() ? right : left;

  m = result.getRowName();
  n = result.getColName();

  string code = infoInvocation(left, right, result);
  if (!dense.isModifiable()) code += createCopy(result, dense);

  code += fmt::format("{}({}, {}, {}, {}, 1.0, {}, {}, {}, {});\n", CBLAS_DDISV,
                      LAYOUT, side, m, n, dataMatrix(diag), strideMatrix(diag),
                      dataMatrix(result), strideMatrix(result));

  if (diag.isModifiable()) code += freeMatrix(diag);

  return code;
}

std::string KernelDisv::generateCost(const Matrix& left, const Matrix& right,
                                     const Matrix& result) const {
  return fmt::format("{} += {} * {}; {}", COST_VAR, result.getRowName(),
                     result.getColName(), infoInvocation(left, right, result));
}

double KernelDisv::computeFLOPs(const Matrix& left, const Matrix& right,
                                const Matrix& result) const {
  return static_cast<double>(result.getNrows()) *
         static_cast<double>(result.getNcols());
}

std::vector<Variant> KernelDisv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Diagonal},
       {Property::FullRank, Property::SPD},
       {Trans::N},
       {Inversion::Y}},
      {{Structure::Dense}, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

std::string KernelDisv::infoInvocation(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  return Kernel::infoInvocation(CG_DISV, left, right, result);
}

void KernelDisv::execute(const Matrix& _left, const Matrix& _right,
                         dMatrix& left, dMatrix& right, dMatrix& result) const {
  dMatrix& lhs = _left.isDiagonal() ? left : right;
  dMatrix& rhs = _left.isDiagonal() ? right : left;
  CBLAS_SIDE Side = _left.isDiagonal() ? CblasLeft : CblasRight;

  int M = left.ROWS;
  int N = right.COLS;

  cblas_ddisv(CblasColMajor, Side, M, N, 1.0, lhs.DATA, lhs.STRIDE, rhs.DATA,
              rhs.STRIDE);

  result = std::move(rhs);
}

void KernelDisv::loadModel() { model.read(getPathModel()); }

double KernelDisv::predictTime(const Matrix& left, const Matrix& right,
                               const Matrix& result) const {
  uint8_t key = {0x00};

  if (!left.isDiagonal()) mdl::setBitLR(key);

  unsigned m = result.getNrows();
  unsigned n = result.getNcols();

  return computeFLOPs(left, right, result) / model.predict(key, m, n);
}

}  // namespace cg
