#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <array>
#include <utility>
#include <vector>

namespace cg {

namespace mdl {
using Bound = std::array<unsigned, 2>;
using Array2d = std::array<double, 2>;
using Array4d = std::array<double, 4>;
using Array8d = std::array<double, 8>;

unsigned saturate(const unsigned x, const std::vector<unsigned>& sizes);

double linearInterp(const unsigned x, const Bound& x_bound,
                    const Array2d& values);

double bilinearInterp(const unsigned x, const unsigned y, const Bound& x_bound,
                      const Bound& y_bound, const Array4d& values);

double trilinearInterp(const unsigned x, const unsigned y, const unsigned z,
                       const Bound& x_bound, const Bound& y_bound,
                       const Bound& z_bound, const Array8d& values);

}  // namespace mdl

}  // namespace cg

#endif