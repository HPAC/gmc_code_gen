#ifndef BASE_3D_H
#define BASE_3D_H

#include <iostream>
#include <vector>

#include "interpolation.hpp"

namespace cg {

namespace mdl {

class Base3D {
 private:
  std::vector<unsigned> _sizes_m{};
  std::vector<unsigned> _sizes_k{};
  std::vector<unsigned> _sizes_n{};
  std::vector<double> _values{};

 public:
  Base3D() : _sizes_m(0U), _sizes_k(0U), _sizes_n(0U), _values(0U) {}

  Base3D(const unsigned points_m, const unsigned points_k,
         const unsigned points_n)
      : _sizes_m(points_m, 0U),
        _sizes_k(points_k, 0U),
        _sizes_n(points_n, 0U),
        _values(points_m * points_k * points_n, 0.0) {}

  Base3D(const std::vector<unsigned>& sizes_m,
         const std::vector<unsigned>& sizes_k,
         const std::vector<unsigned>& sizes_n,
         const std::vector<double>& values)
      : _sizes_m{sizes_m},
        _sizes_k{sizes_k},
        _sizes_n{sizes_n},
        _values{values} {}

  inline void setSizes(const std::vector<unsigned>& sizes_m,
                       const std::vector<unsigned>& sizes_k,
                       const std::vector<unsigned>& sizes_n) noexcept {
    _sizes_m = sizes_m;
    _sizes_k = sizes_k;
    _sizes_n = sizes_n;
  }

  inline void setValues(const std::vector<double>& values) noexcept {
    _values = values;  // check size = n_points_x * n_points_y?
  }

  inline void setValues(std::vector<double>&& values) noexcept {
    _values = std::move(values);
  }

  std::array<std::vector<unsigned>, 3U> getSizes() const noexcept {
    return {_sizes_m, _sizes_k, _sizes_n};
  }

  std::vector<double> getValues() const noexcept { return _values; }

  /**
   * @brief Predicts the performance for (m,k,n).
   *
   * @param m       first size parameter.
   * @param k       second size parameter.
   * @param n       third size parameter.
   * @return double -- the result of the interpolation.
   */
  double predict(const unsigned m, const unsigned k, const unsigned n) const;

  /**
   * @brief Read the model from a input stream.
   *
   * @param is input stream.
   */
  void read(std::istream& is);

  /**
   * @brief Write the contents of the model to an output stream.
   *
   * @param os output stream.
   */
  void write(std::ostream& os) const;
};

}  // namespace mdl

}  // namespace cg

#endif