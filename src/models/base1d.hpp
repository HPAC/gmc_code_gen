#ifndef BASE_1D_H
#define BASE_1D_H

#include <iostream>
#include <vector>

#include "interpolation.hpp"

namespace cg {

namespace mdl {

class Base1D {
 private:
  std::vector<unsigned> _sizes_m{};
  std::vector<double> _values{};

 public:
  Base1D() : _sizes_m(0U), _values(0U) {}

  Base1D(const unsigned n_points)
      : _sizes_m(n_points, 0U), _values(n_points, 0.0) {}

  Base1D(const std::vector<unsigned>& sizes, const std::vector<double>& values)
      : _sizes_m{sizes}, _values{values} {}

  inline void setSizes(const std::vector<unsigned>& sizes) noexcept {
    _sizes_m = sizes;
  }

  inline void setValues(const std::vector<double>& values) noexcept {
    _values = values;
  }

  inline void setValues(std::vector<double>&& values) noexcept {
    _values = std::move(values);
  }

  inline std::vector<unsigned> getSizes() const noexcept { return _sizes_m; }

  inline std::vector<double> getValues() const noexcept { return _values; }

  double predict(const unsigned m) const;

  // Read from disk.
  void read(std::istream& is);

  // Write to disk.
  void write(std::ostream& os) const;
};

}  // namespace mdl

}  // namespace cg

#endif