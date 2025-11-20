#include "model2d.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>

#include "model_common.hpp"

namespace cg {

namespace mdl {

void Model2D::insertBase(const uint8_t key, const Base2D& base) {
  _models[key] = base;
}

void Model2D::emplaceBase(const uint8_t key,
                          const std::vector<unsigned>& sizes_m,
                          const std::vector<unsigned>& sizes_n,
                          const std::vector<double>& values) {
  _models.emplace(std::piecewise_construct, std::forward_as_tuple(key),
                  std::forward_as_tuple(sizes_m, sizes_n, values));
}

double Model2D::predict(const uint8_t key, const unsigned m,
                        const unsigned n) const {
  return _models.at(key).predict(m, n);
}

void Model2D::read(std::istream& is) {
  int key_int;
  uint8_t key;
  Base2D base;

  is >> key_int;
  _options = static_cast<uint8_t>(key_int);
  unsigned n_base = static_cast<unsigned>(0b1 << countSetBits(_options));

  for (unsigned i = 0; i < n_base; i++) {
    is >> key_int;
    key = static_cast<uint8_t>(key_int);
    base.read(is);
    _models[key] = base;
  }
}

void Model2D::read(const std::string& filename) {
  std::ifstream ifile;
  ifile.open(filename);
  if (ifile.fail()) {
    std::cerr << "Error: cannot open " << filename << '\n';
    exit(-1);
  }
  read(ifile);
  ifile.close();
}

void Model2D::write(std::ostream& os) const {
  os << static_cast<int>(_options) << '\n';
  for (const auto& k : _models) {
    os << static_cast<int>(k.first) << '\n';
    k.second.write(os);
  }
}

void Model2D::write(const std::string& filename) const {
  std::ofstream ofile;
  ofile.open(filename);
  if (ofile.fail()) {
    std::cerr << "Error: cannot open " << filename << '\n';
    exit(-1);
  }
  write(ofile);
  ofile.close();
}

}  // namespace mdl

}  // namespace cg