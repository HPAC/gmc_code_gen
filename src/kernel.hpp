#ifndef KERNEL_H
#define KERNEL_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "matrix.hpp"
#include "models/model_common.hpp"
#include "utils/dMatrix.hpp"
#include "variant.hpp"

namespace cg {

class Kernel {
 private:
  uint8_t _args = {};
  std::string _path_model = {};

 public:
  Kernel(const uint8_t args) : _args{args} {}

  Kernel(const uint8_t args, const std::string& path_model)
      : _args{args}, _path_model{path_model} {}

  virtual ~Kernel() = default;

  inline uint8_t getArgs() const { return _args; }

  inline std::string getPathModel() const { return _path_model; }

  // these functions are specialized for each kernel.
  virtual bool tweakTransposition(Matrix& left, Matrix& right) const = 0;

  virtual std::array<bool, 2U> needsNewMatrix() const = 0;

  virtual void deduceName(const Matrix& left, const Matrix& right,
                          Matrix& result) const = 0;

  virtual std::string generateCode(const Matrix& left, const Matrix& right,
                                   const Matrix& result) const = 0;

  virtual std::string generateCost(const Matrix& left, const Matrix& right,
                                   const Matrix& result) const = 0;

  virtual double computeFLOPs(const Matrix& left, const Matrix& right,
                              const Matrix& result) const = 0;

  virtual std::vector<Variant> getCoveredVariants() const = 0;

  /**
   * @brief Executes the kernel invocation. The operands are left, right, and
   * result (if needed). _left and _right are symbolic representations of the
   * operands.
   *
   * @param _left   Symbolic representation of left  (Matrix)
   * @param _right  Symbolic representation of right (Matrix)
   * @param left    Operand on the left  (dMatrix)
   * @param right   Operand on the right (dMatrix)
   * @param result  Operand holding the result (dMatrix)
   */
  virtual void execute(const Matrix& _left, const Matrix& _right, dMatrix& left,
                       dMatrix& right, dMatrix& result) const = 0;

  virtual void loadModel() = 0;

  virtual double predictTime(const Matrix& left, const Matrix& right,
                             const Matrix& result) const = 0;

  std::string generateTranspose(const Matrix& result) const;

 protected:
  std::string createMatrix(const Matrix& mat) const;

  std::string createCopy(const Matrix& cpy, const Matrix& mat) const;

  std::string freeMatrix(const Matrix& mat) const;

  std::string infoInvocation(const std::string& kernel_name, const Matrix& left,
                             const Matrix& right, const Matrix& result) const;

  std::string dataMatrix(const Matrix& mat) const;

  std::string strideMatrix(const Matrix& mat) const;

  std::string simpleText(const Matrix& mat) const;
};

}  // namespace cg

#endif