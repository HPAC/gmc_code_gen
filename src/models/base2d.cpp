#include "base2d.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include "../utils/common.hpp"
#include "interpolation.hpp"

namespace cg {

namespace mdl {

double Base2D::predict(const unsigned m, const unsigned n) const {
  const unsigned N = static_cast<unsigned>(_sizes_n.size());

  unsigned m_tmp = saturate(m, _sizes_m);
  unsigned n_tmp = saturate(n, _sizes_n);

  auto it_m = std::lower_bound(_sizes_m.begin(), _sizes_m.end(), m_tmp);
  unsigned id_m_a = static_cast<unsigned>(it_m - _sizes_m.begin());
  unsigned id_m_b = (m_tmp == *it_m) ? id_m_a : id_m_a - 1;

  auto it_n = std::lower_bound(_sizes_n.begin(), _sizes_n.end(), n_tmp);
  unsigned id_n_a = static_cast<unsigned>(it_n - _sizes_n.begin());
  unsigned id_n_b = (n_tmp == *it_n) ? id_n_a : id_n_a - 1;

  Bound x_bound = {_sizes_m[id_m_b], _sizes_m[id_m_a]};
  Bound y_bound = {_sizes_n[id_n_b], _sizes_n[id_n_a]};
  Array4d values = {_values[id_m_b * N + id_n_b], _values[id_m_b * N + id_n_a],
                    _values[id_m_a * N + id_n_b], _values[id_m_a * N + id_n_a]};

  return bilinearInterp(m_tmp, n_tmp, x_bound, y_bound, values);
}

void Base2D::read(std::istream& is) {
  unsigned M{}, N{};
  is >> M;
  is >> N;

  _sizes_m.resize(M);
  for (unsigned i = 0; i < M; i++) is >> _sizes_m[i];

  _sizes_n.resize(N);
  for (unsigned i = 0; i < N; i++) is >> _sizes_n[i];

  _values.resize(M * N);
  for (unsigned i = 0; i < M * N; i++) is >> _values[i];
}

void Base2D::write(std::ostream& os) const {
  os << static_cast<unsigned>(_sizes_m.size()) << ' '
     << static_cast<unsigned>(_sizes_n.size()) << '\n';
  os << _sizes_m << '\n';
  os << _sizes_n << '\n';
  os << _values << '\n';
}

}  // namespace mdl

}  // namespace cg
