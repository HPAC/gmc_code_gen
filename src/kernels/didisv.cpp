#include "didisv.hpp"

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

bool KernelDidisv::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelDidisv::needsNewMatrix() const {
  return {false, false};
}

void KernelDidisv::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  const Matrix& rhs = left.isInverted() ? right : left;
  result.setName((rhs.isModifiable()) ? rhs.getName()
                                      : PREFIX_COPY + rhs.getName());
}

std::string KernelDidisv::generateCode(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  const Matrix& lhs = left.isInverted() ? left : right;
  const Matrix& rhs = left.isInverted() ? right : left;

  string code = infoInvocation(left, right, result);
  if (!rhs.isModifiable()) code += createCopy(result, rhs);

  code +=
      fmt::format("{}({}, {}, 1.0, {}, {}, {}, {});\n", CBLAS_DDIDISV, LAYOUT,
                  result.getRowName(), dataMatrix(lhs), strideMatrix(lhs),
                  dataMatrix(result), strideMatrix(result));
  if (left.isModifiable()) code += freeMatrix(left);
  return code;
}

std::string KernelDidisv::generateCost(const Matrix& left, const Matrix& right,
                                       const Matrix& result) const {
  return fmt::format("{} += {}; {}", COST_VAR, result.getRowName(),
                     infoInvocation(left, right, result));
}

double KernelDidisv::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return static_cast<double>(result.getNrows());
}

std::vector<Variant> KernelDidisv::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Diagonal},
       {Property::FullRank, Property::SPD},
       {Trans::N},
       {Inversion::Y}},
      {{Structure::Diagonal}, all_properties, {Trans::N}, {Inversion::N}}};

  return range.generateVariants();
}

std::string KernelDidisv::infoInvocation(const Matrix& left,
                                         const Matrix& right,
                                         const Matrix& result) const {
  return Kernel::infoInvocation(CG_DIDISV, left, right, result);
}

void KernelDidisv::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  dMatrix& lhs = _left.isInverted() ? left : right;
  dMatrix& rhs = _left.isInverted() ? right : left;
  int M = left.ROWS;

  cblas_ddidisv(CblasColMajor, M, 1.0, lhs.DATA, lhs.STRIDE, rhs.DATA,
                rhs.STRIDE);

  result = std::move(rhs);
}

void KernelDidisv::loadModel() { model.read(getPathModel()); }

double KernelDidisv::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};
  return computeFLOPs(left, right, result) /
         model.predict(key, result.getNrows());
}

}  // namespace cg
