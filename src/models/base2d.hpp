#ifndef BASE_2D_H
#define BASE_2D_H

#include <iostream>
#include <vector>

#include "interpolation.hpp"

namespace cg {

namespace mdl {

class Base2D {
 private:
  std::vector<unsigned> _sizes_m{};
  std::vector<unsigned> _sizes_n{};
  std::vector<double> _values{};

 public:
  Base2D() : _sizes_m(0U), _sizes_n(0U), _values(0U) {}

  Base2D(const unsigned points_m, const unsigned points_n)
      : _sizes_m(points_m, 0U),
        _sizes_n(points_n, 0U),
        _values(points_m * points_n, 0.0) {}

  Base2D(const std::vector<unsigned>& sizes_m,
         const std::vector<unsigned>& sizes_n,
         const std::vector<double>& values)
      : _sizes_m{sizes_m}, _sizes_n{sizes_n}, _values{values} {}

  inline void setSizes(const std::vector<unsigned>& sizes_m,
                       const std::vector<unsigned>& sizes_n) noexcept {
    _sizes_m = sizes_m;
    _sizes_n = sizes_n;
  }

  inline void setValues(const std::vector<double>& values) noexcept {
    _values = values;
  }

  inline void setValues(std::vector<double>&& values) noexcept {
    _values = std::move(values);
  }

  std::array<std::vector<unsigned>, 2U> getSizes() const noexcept {
    return {_sizes_m, _sizes_n};
  }

  std::vector<double> getValues() const noexcept { return _values; }

  /**
   * @brief Predicts the performance for sizes x and y.
   *
   * @param m   first size parameter.
   * @param n   second size parameter.
   * @return double -- the result of the interpolation.
   */
  double predict(const unsigned m, const unsigned n) const;

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