#ifndef POGESV_H
#define POGESV_H

#include <cstdint>
#include <string>
#include <vector>

#include "../kernel.hpp"
#include "../matrix.hpp"
#include "../models/model2d.hpp"
#include "../models/settings_models.hpp"
#include "../variant.hpp"

namespace cg {

class KernelPogesv : public Kernel {
 private:
  mdl::Model2D model{};

 public:
  KernelPogesv(const uint8_t args) : Kernel(args, FNAME_POGESV) {}

  virtual bool tweakTransposition(Matrix& left, Matrix& right) const override;

  virtual std::array<bool, 2U> needsNewMatrix() const override;

  virtual void deduceName(const Matrix& left, const Matrix& right,
                          Matrix& result) const override;

  virtual std::string generateCode(const Matrix& left, const Matrix& right,
                                   const Matrix& result) const override;

  virtual std::string generateCost(const Matrix& left, const Matrix& right,
                                   const Matrix& result) const override;

  virtual double computeFLOPs(const Matrix& left, const Matrix& right,
                              const Matrix& result) const override;

  virtual std::vector<Variant> getCoveredVariants() const override;

  std::string infoInvocation(const Matrix& left, const Matrix& right,
                             const Matrix& result) const;

  virtual void execute(const Matrix& _left, const Matrix& _right, dMatrix& left,
                       dMatrix& right, dMatrix& result) const override;

  virtual void loadModel() override;

  virtual double predictTime(const Matrix& left, const Matrix& right,
                             const Matrix& result) const override;
};

}  // namespace cg

#endif