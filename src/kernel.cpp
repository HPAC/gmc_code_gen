#include "kernel.hpp"

#include <fmt/core.h>

#include <string>
#include <vector>

#include "macros.hpp"
#include "matrix.hpp"
#include "variant.hpp"

using std::string;

namespace cg {

string Kernel::generateTranspose(const Matrix& result) const {
  return fmt::format("{}.{}();\n", result.getName(), STR(TRANSPOSE));
}

string Kernel::createMatrix(const Matrix& mat) const {
  return fmt::format("{} {}({}, {});\n", STR(MATRIX), mat.getName(),
                     mat.getRowName(), mat.getColName());
}

string Kernel::createCopy(const Matrix& cpy, const Matrix& mat) const {
  return fmt::format("{} {}({});\n", STR(MATRIX), cpy.getName(), mat.getName());
}

string Kernel::freeMatrix(const Matrix& mat) const {
  return fmt::format("{}.{}();\n", mat.getName(), STR(DEALLOCATE));
}

string Kernel::infoInvocation(const string& kernel_name, const Matrix& left,
                              const Matrix& right, const Matrix& result) const {
  return fmt::format("// {} <-- {}({}, {})\n", simpleText(result), kernel_name,
                     simpleText(left), simpleText(right));
}

string Kernel::dataMatrix(const Matrix& mat) const {
  return fmt::format("{}.{}", mat.getName(), STR(DATA));
}

string Kernel::strideMatrix(const Matrix& mat) const {
  return fmt::format("{}.{}", mat.getName(), STR(STRIDE));
}

string Kernel::simpleText(const Matrix& mat) const {
  string text = mat.getName();
  if (mat.isTransposed() and mat.isInverted())
    text += "^-T";
  else if (mat.isTransposed())
    text += "^T";
  else if (mat.isInverted())
    text += "^-1";
  return text;
}

}  // namespace cg