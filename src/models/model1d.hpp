#ifndef MODEL_KERNEL_1D_H
#define MODEL_KERNEL_1D_H

#include <cstdint>
#include <iostream>
#include <map>

#include "base1d.hpp"

namespace cg {

namespace mdl {

class Model1D {
 public:
  uint8_t _options{};
  std::map<uint8_t, Base1D> _models{};

 public:
  Model1D() = default;

  /**
   * @brief Construct a new Model1D.
   * Contains a different base model (Base1D) for each possible combination of
   * options.
   *
   * @param options uint8_t where the bits set to 1 indicate the option exists.
   */
  Model1D(const uint8_t options) : _options{options} {}

  /**
   * @brief Inserts a new Base1D object; will be identified by key.
   *
   * @param key  uint8_t containing the options of the call.
   * @param base Base copied/moved into the Model.
   */
  void insertBase(const uint8_t key, const Base1D& base);

  void emplaceBase(const uint8_t key, const std::vector<unsigned>& sizes_m,
                   const std::vector<double>& values);

  /**
   * @brief Predicts the performance of the kernel with options key and
   * sizes (m,n).
   *
   * @param key      uint8_t indicating the options of the call.
   * @param m        Number of rows and columns of the result.
   * @return double  Predicted performance.
   */
  double predict(const uint8_t key, const unsigned m) const;

  void read(std::istream& is);

  void read(const std::string& filename);

  void write(std::ostream& os) const;

  void write(const std::string& filename) const;
};

}  // namespace mdl

}  // namespace cg

#endif