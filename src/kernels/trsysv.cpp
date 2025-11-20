#include "trsysv.hpp"

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

bool KernelTrsysv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelTrsysv::needsNewMatrix() const {
  return {false, false};
}

void KernelTrsysv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;

  result.setName(rhs.isModifiable() ? rhs.getName()
                                    : PREFIX_COPY + rhs.getName());
}

std::string KernelTrsysv::generateCode(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  string side_lhs, uplo_lhs, uplo_rhs, trans_lhs, diag_lhs, m;

  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;
  side_lhs = left.isInverted() ? CG_LEFT : CG_RIGHT;

  uplo_lhs = lhs.isLower() ? CG_LOWER : CG_UPPER;
  uplo_rhs = rhs.isSymmetricLower() ? CG_LOWER : CG_UPPER;
  trans_lhs = lhs.isTransposed() ? CG_TRANS : CG_NO_TRANS;
  diag_lhs = lhs.isUnit() ? CG_UNIT : CG_NO_UNIT;
  m = lhs.getRowName();

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  code +=
      fmt::format("{}({}, {}, {}, {}, {}, {}, {}, 1.0, {}, {}, {}, {});\n",
                  CBLAS_DTRSYSV, LAYOUT, side_lhs, uplo_lhs, uplo_rhs,
                  trans_lhs, diag_lhs, m, dataMatrix(lhs), strideMatrix(lhs),
                  dataMatrix(result), strideMatrix(result));
  if (lhs.isModifiable()) code += freeMatrix(lhs);

  return code;
}

std::string KernelTrsysv::generateCost(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  return fmt::format("{0} += {1} * {1} * {1}; {2}", COST_VAR, left.getRowName(),
                     infoInvocation(left, right, result));
}

double KernelTrsysv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  const double m = static_cast<double>(left.getNrows());
  return m * m * m;
}

std::vector<Variant> KernelTrsysv::getCoveredVariants() const {
  RangeVariant range{
      {triangular, {Property::FullRank}, all_trans, {Inversion::Y}},
      {{Structure::Symmetric_L, Structure::Symmetric_U},
       all_properties,
       {Trans::N},
       {Inversion::N}}};

  return range.generateVariants();
}

std::string KernelTrsysv::infoInvocation(const Matrix& left,
                                         const Matrix& right,
                                         const Matrix& result) const {
  return Kernel::infoInvocation(CG_TRSYSV, left, right, result);
}

void KernelTrsysv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  const Matrix& _lhs = _left.isInverted() ? _left : _right;
  const Matrix& _rhs = _left.isInverted() ? _right : _left;

  CBLAS_SIDE Side = _left.isInverted() ? CblasLeft : CblasRight;
  CBLAS_UPLO UploA = _lhs.isLower() ? CblasLower : CblasUpper;
  CBLAS_UPLO UploB = _rhs.isSymmetricLower() ? CblasLower : CblasUpper;
  CBLAS_TRANSPOSE TransA = _lhs.isTransposed() ? CblasTrans : CblasNoTrans;
  CBLAS_DIAG DiagA = _lhs.isUnit() ? CblasUnit : CblasNonUnit;

  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;

  int M = left.ROWS;

  cblas_dtrsysv(CblasColMajor, Side, UploA, UploB, TransA, DiagA, M, 1.0,
                lhs.DATA, lhs.STRIDE, rhs.DATA, rhs.STRIDE);

  result = std::move(rhs);
}

void KernelTrsysv::loadModel() { model.read(getPathModel()); }

double KernelTrsysv::predictTime(const Matrix& left, const Matrix& right,
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
