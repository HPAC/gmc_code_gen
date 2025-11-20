#include "common.hpp"

#include <iostream>
#include <vector>

#include "../definitions.hpp"

namespace cg {

std::ostream& operator<<(std::ostream& os, const cg::Instance& instance) {
  for (unsigned i = 0U; i < instance.size(); i++) {
    os << instance[i];
    if (i < instance.size() - 1U) os << ' ';
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<cg::Instance>& S) {
  for (unsigned i = 0U; i < S.size(); i++) {
    os << S[i];
    if (i < S.size() - 1U) os << ' ';
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec) {
  for (unsigned i = 0U; i < vec.size(); i++) {
    os << vec[i];
    if (i < vec.size() - 1U) os << ' ';
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::set<unsigned>& Z) {
  os << "{";
  for (const auto& id : Z) {
    os << id << ' ';
  }
  os << "}";
  return os;
}

}  // namespace cg
