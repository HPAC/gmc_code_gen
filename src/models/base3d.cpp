#include "base3d.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

#include "../utils/common.hpp"
#include "interpolation.hpp"

namespace cg {

namespace mdl {

double Base3D::predict(const unsigned m, const unsigned k,
                       const unsigned n) const {
  const unsigned K = static_cast<unsigned>(_sizes_k.size());
  const unsigned N = static_cast<unsigned>(_sizes_n.size());

  unsigned m_tmp = saturate(m, _sizes_m);
  unsigned k_tmp = saturate(k, _sizes_k);
  unsigned n_tmp = saturate(n, _sizes_n);

  auto it_m = std::lower_bound(_sizes_m.begin(), _sizes_m.end(), m_tmp);
  unsigned id_m_a = static_cast<unsigned>(it_m - _sizes_m.begin());
  unsigned id_m_b = (m_tmp == *it_m) ? id_m_a : id_m_a - 1;

  auto it_k = std::lower_bound(_sizes_k.begin(), _sizes_k.end(), k_tmp);
  unsigned id_k_a = static_cast<unsigned>(it_k - _sizes_k.begin());
  unsigned id_k_b = (k_tmp == *it_k) ? id_k_a : id_k_a - 1;

  auto it_n = std::lower_bound(_sizes_n.begin(), _sizes_n.end(), n_tmp);
  unsigned id_n_a = static_cast<unsigned>(it_n - _sizes_n.begin());
  unsigned id_n_b = (n_tmp == *it_n) ? id_n_a : id_n_a - 1;

  Bound m_bound{_sizes_m[id_m_b], _sizes_m[id_m_a]};
  Bound k_bound{_sizes_k[id_k_b], _sizes_k[id_k_a]};
  Bound n_bound{_sizes_n[id_n_b], _sizes_n[id_n_a]};

  const unsigned BLOCK_KN = K * N;
  Array8d values = {_values[id_m_b * BLOCK_KN + id_k_b * N + id_n_b],
                    _values[id_m_b * BLOCK_KN + id_k_b * N + id_n_a],
                    _values[id_m_b * BLOCK_KN + id_k_a * N + id_n_b],
                    _values[id_m_b * BLOCK_KN + id_k_a * N + id_n_a],
                    _values[id_m_a * BLOCK_KN + id_k_b * N + id_n_b],
                    _values[id_m_a * BLOCK_KN + id_k_b * N + id_n_a],
                    _values[id_m_a * BLOCK_KN + id_k_a * N + id_n_b],
                    _values[id_m_a * BLOCK_KN + id_k_a * N + id_n_a]};

  return trilinearInterp(m_tmp, k_tmp, n_tmp, m_bound, k_bound, n_bound,
                         values);
}

void Base3D::read(std::istream& is) {
  unsigned M{}, K{}, N{};

  is >> M;
  is >> K;
  is >> N;

  _sizes_m.resize(M);
  for (unsigned i = 0; i < M; i++) is >> _sizes_m[i];

  _sizes_k.resize(K);
  for (unsigned i = 0; i < K; i++) is >> _sizes_k[i];

  _sizes_n.resize(N);
  for (unsigned i = 0; i < N; i++) is >> _sizes_n[i];

  _values.resize(M * K * N);
  for (unsigned i = 0; i < M * K * N; i++) is >> _values[i];
}

void Base3D::write(std::ostream& os) const {
  os << static_cast<unsigned>(_sizes_m.size()) << ' '
     << static_cast<unsigned>(_sizes_k.size()) << ' '
     << static_cast<unsigned>(_sizes_n.size()) << '\n';
  os << _sizes_m << '\n';
  os << _sizes_k << '\n';
  os << _sizes_n << '\n';
  os << _values << '\n';
}

}  // namespace mdl

}  // namespace cg