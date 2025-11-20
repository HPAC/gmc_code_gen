#include "didimm.hpp"

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

bool KernelDidimm::tweakTransposition(Matrix& left, Matrix& right) const {
  return false;
}

std::array<bool, 2U> KernelDidimm::needsNewMatrix() const {
  return {false, false};
}

void KernelDidimm::deduceName(const Matrix& left, const Matrix& right,
                              Matrix& result) const {
  result.setName((right.isModifiable()) ? right.getName()
                                        : PREFIX_COPY + right.getName());
}

string KernelDidimm::generateCode(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  string code = infoInvocation(left, right, result);
  if (!right.isModifiable()) code += createCopy(result, right);

  code +=
      fmt::format("{}({}, {}, 1.0, {}, {}, {}, {});\n", CBLAS_DDIDIMM, LAYOUT,
                  result.getRowName(), dataMatrix(left), strideMatrix(left),
                  dataMatrix(result), strideMatrix(result));
  if (left.isModifiable()) code += freeMatrix(left);
  return code;
}

string KernelDidimm::generateCost(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return fmt::format("{} += {}; {}", COST_VAR, result.getRowName(),
                     infoInvocation(left, right, result));
}

double KernelDidimm::computeFLOPs(const Matrix& left, const Matrix& right,
                                  const Matrix& result) const {
  return static_cast<double>(result.getNrows());
}

std::vector<Variant> KernelDidimm::getCoveredVariants() const {
  RangeVariant range{
      {{Structure::Diagonal}, all_properties, {Trans::N}, {Inversion::N}},
      {{Structure::Diagonal}, all_properties, {Trans::N}, {Inversion::N}}};

  return range.generateVariants();
}

string KernelDidimm::infoInvocation(const Matrix& left, const Matrix& right,
                                    const Matrix& result) const {
  return Kernel::infoInvocation(CG_DIDIMM, left, right, result);
}

void KernelDidimm::execute(const Matrix& _left, const Matrix& _right,
                           dMatrix& left, dMatrix& right,
                           dMatrix& result) const {
  int M = left.ROWS;

  cblas_ddidimm(CblasColMajor, M, 1.0, left.DATA, left.STRIDE, right.DATA,
                right.STRIDE);

  result = std::move(right);
}

void KernelDidimm::loadModel() { model.read(getPathModel()); }

double KernelDidimm::predictTime(const Matrix& left, const Matrix& right,
                                 const Matrix& result) const {
  uint8_t key = {0x00};
  return computeFLOPs(left, right, result) /
         model.predict(key, result.getNrows());
}

}  // namespace cg
