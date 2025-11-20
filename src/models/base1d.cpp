#include "base1d.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include "../utils/common.hpp"
#include "interpolation.hpp"

namespace cg {

namespace mdl {

double Base1D::predict(const unsigned m) const {
  const unsigned M = static_cast<unsigned>(_sizes_m.size());

  if (m <= _sizes_m[0])
    return _values[0];
  else if (m >= _sizes_m[M - 1])
    return _values[M - 1];

  // finds pointer to first element no smaller than x.
  auto it = std::lower_bound(_sizes_m.begin(), _sizes_m.end(), m);
  unsigned idm = static_cast<unsigned>(it - _sizes_m.begin());
  if (m == *it)
    return _values[idm];
  else {
    Bound m_bound{_sizes_m[idm - 1], _sizes_m[idm]};
    Array2d values{_values[idm - 1], _values[idm]};
    return linearInterp(m, m_bound, values);
  }
}

void Base1D::read(std::istream& is) {
  unsigned M{};
  is >> M;

  _sizes_m.resize(M);
  for (unsigned i = 0; i < M; i++) is >> _sizes_m[i];

  _values.resize(M);
  for (unsigned i = 0; i < M; i++) is >> _values[i];
}

void Base1D::write(std::ostream& os) const {
  os << static_cast<unsigned>(_sizes_m.size()) << '\n';
  os << _sizes_m << '\n';
  os << _values << '\n';
}

}  // namespace mdl

}  // namespace cg
