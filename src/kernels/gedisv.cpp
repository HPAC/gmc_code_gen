#include "gedisv.hpp"

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

bool KernelGedisv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelGedisv::needsNewMatrix() const {
  return {true, true};
}

void KernelGedisv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {}

string KernelGedisv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, trans_lhs, m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& diag_rhs = left.isInverted() ? right : left;

  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  trans_lhs = lhs.isTransposed() ? CG_TRANS : CG_NO_TRANS;
  m = result.getRowName();

  string code = infoInvocation(left, right, result);
  code += createMatrix(result);

  code += fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {}, {});\n",
                      CBLAS_DGEDISV, LAYOUT, side, trans_lhs, m,
                      dataMatrix(lhs), strideMatrix(lhs), dataMatrix(diag_rhs),
                      strideMatrix(diag_rhs), dataMatrix(result),
                      strideMatrix(result));

  if (lhs.isModifiable()) code += freeMatrix(lhs);
  if (diag_rhs.isModifiable()) code += freeMatrix(diag_rhs);

  return code;
}

string KernelGedisv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return fmt::format("{0} += 2.0 * {1} * {1} * {1};{2}", COST_VAR,
                     result.getRowName(), infoInvocation(left, right, result));
}

double KernelGedisv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  double m = static_cast<double>(result.getNrows());
  return 2.0 * m * m * m;
}

std::vector<Variant> KernelGedisv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Dense}, {Property::FullRank}, all_trans, {Inversion::Y}},
      {{Structure::Diagonal}, all_properties, all_trans, {Inversion::N}}};

  return range.generateVariants();
}

string KernelGedisv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_GEDISV, left, right, result);
}

void KernelGedisv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;
  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_TRANSPOSE TransA = _lhs.isTransposed() ? CblasTrans : CblasNoTrans;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& diag = _left.isInverted() ? right : left;
  int M = left.ROWS;

  cblas_dgedisv(CblasColMajor, Side, TransA, M, lhs.DATA, lhs.STRIDE, diag.DATA,
                diag.STRIDE, result.DATA, result.STRIDE);
}

void KernelGedisv::loadModel() { model.read(getPathModel()); }

double KernelGedisv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  const Matrix& lhs = (left.isInverted()) ? left : right;
  if (left.isInverted()) mdl::setBitLR(key);
  if (lhs.isTransposed()) mdl::setBitTransA(key);

  unsigned m = result.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
