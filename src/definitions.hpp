#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <vector>

#include "matrix.hpp"

namespace cg {

using MatrixChain = std::vector<cg::Matrix>;
using Permutation = std::vector<unsigned>;
using Instance = std::vector<unsigned>;

}  // namespace cg

#endif