#include "gesysv.hpp"

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

bool KernelGesysv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelGesysv::needsNewMatrix() const {
  return {false, false};
}

void KernelGesysv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  result.setName(rhs.isModifiable() ? rhs.getName()
                                    : PREFIX_COPY + rhs.getName());
}

string KernelGesysv::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string side, uplo_rhs, trans_lhs;
  string m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;
  side = left.isInverted() ? CG_LEFT : CG_RIGHT;
  uplo_rhs = rhs.isSymmetricLower() ? CG_LOWER : CG_UPPER;
  trans_lhs = lhs.isTransposed() ? CG_TRANS : CG_NO_TRANS;

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  m = result.getRowName();
  code +=
      fmt::format("{}({}, {}, {}, {}, {}, {}, {}, {}, {});\n", CBLAS_DGESYSV,
                  LAYOUT, side, uplo_rhs, trans_lhs, m, dataMatrix(lhs),
                  strideMatrix(lhs), dataMatrix(result), strideMatrix(result));
  if (lhs.isModifiable()) code += freeMatrix(lhs);

  return code;
}

string KernelGesysv::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string m = left.getRowName();
  return fmt::format("{0} += 8.0 / 3.0 * {1} * {1} * {1}; {2}", COST_VAR, m,
                     infoInvocation(left, right, result));
}

double KernelGesysv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const double m = left.getNrows();
  return 8.0 / 3.0 * m * m * m;
}

std::vector<Variant> KernelGesysv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Dense}, {Property::FullRank}, all_trans, {Inversion::Y}},
      {{Structure::Symmetric_L, Structure::Symmetric_U},
       all_properties,
       {Trans::N},
       {Inversion::N}}};

  return range.generateVariants();
}

string KernelGesysv::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_GESYSV, left, right, result);
}

void KernelGesysv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;
  const Matrix& _rhs = _left.isInverted() ? _right : _left;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploB = _rhs.isSymmetricLower() ? CblasLower : CblasUpper;
  CBLAS_TRANSPOSE TransA = _lhs.isTransposed() ? CblasTrans : CblasNoTrans;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;

  cblas_dgesysv(CblasColMajor, Side, UploB, TransA, M, lhs.DATA, lhs.STRIDE,
                rhs.DATA, rhs.STRIDE);

  result = std::move(rhs);
}

void KernelGesysv::loadModel() { model.read(getPathModel()); }

double KernelGesysv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};

  const Matrix& lhs = (left.isInverted()) ? left : right;
  const Matrix& rhs = (left.isInverted()) ? right : left;

  if (!left.isInverted()) mdl::setBitLR(key);
  if (rhs.isSymmetricLower()) mdl::setBitUploB(key);
  if (lhs.isTransposed()) mdl::setBitTransA(key);

  unsigned m = left.getNrows();
  return computeFLOPs(left, right, result) / model.predict(key, m);
}

}  // namespace cg
