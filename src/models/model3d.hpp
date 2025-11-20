#ifndef MODEL_KERNEL_3D_H
#define MODEL_KERNEL_3D_H

#include <cstdint>
#include <iostream>
#include <map>

#include "base3d.hpp"

namespace cg {

namespace mdl {

class Model3D {
 private:
  uint8_t _options{};
  std::map<uint8_t, Base3D> _models{};

 public:
  Model3D() = default;

  /**
   * @brief Construct a new Model3D.
   * Contains a different base model (Base3D) for each possible combination of
   * options.
   *
   * @param options uint8_t where the bits set to 1 indicate the option exists.
   */
  Model3D(const uint8_t options) : _options{options} {}

  /**
   * @brief Inserts a new Base3D object; will be identified by key.
   *
   * @param key  uint8_t containing the options for the call.
   * @param base Base copied/moved into the Model.
   */
  void insertBase(const uint8_t key, const Base3D& base);

  void emplaceBase(const uint8_t key, const std::vector<unsigned>& sizes_m,
                   const std::vector<unsigned>& sizes_k,
                   const std::vector<unsigned>& sizes_n,
                   const std::vector<double>& values);

  /**
   * @brief Predicts the performance of the kernel with options key and
   * sizes (m,n).
   *
   * @param key      uint8_t indicating the options of the call.
   * @param m        Number of rows of the result.
   * @param k        Shared dimension.
   * @param n        Number of columns of the result.
   * @return double  Predicted performance.
   */
  double predict(const uint8_t key, const unsigned m, const unsigned k,
                 const unsigned n) const;

  void read(std::istream& is);

  void read(const std::string& filename);

  void write(std::ostream& os) const;

  void write(const std::string& filename) const;
};

}  // namespace mdl

}  // namespace cg

#endif