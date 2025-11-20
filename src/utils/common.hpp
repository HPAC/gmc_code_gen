#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <set>
#include <vector>

#include "../definitions.hpp"

namespace cg {

std::ostream& operator<<(std::ostream& os, const cg::Instance& instance);

std::ostream& operator<<(std::ostream& os, const std::vector<cg::Instance>& S);

std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec);

std::ostream& operator<<(std::ostream& os, const std::set<unsigned>& Z);

}  // namespace cg

#endif
