#include "interpolation.hpp"

namespace cg {

namespace mdl {

unsigned saturate(const unsigned x, const std::vector<unsigned>& sizes) {
  unsigned x_tmp = x;
  if (x_tmp <= sizes[0])
    x_tmp = sizes[0];
  else if (x_tmp >= sizes.back())
    x_tmp = sizes.back();

  return x_tmp;
}

double linearInterp(const unsigned x, const Bound& x_bound,
                    const Array2d& values) {
  if (x_bound[0] == x_bound[1])
    return values[0];
  else
    return values[0] + static_cast<double>(x - x_bound[0]) *
                           (values[1] - values[0]) /
                           static_cast<double>(x_bound[1] - x_bound[0]);
}

double bilinearInterp(const unsigned x, const unsigned y, const Bound& x_bound,
                      const Bound& y_bound, const Array4d& values) {
  double interp_x0 = linearInterp(y, y_bound, {values[0], values[1]});
  double interp_x1 = linearInterp(y, y_bound, {values[2], values[3]});
  return linearInterp(x, x_bound, {interp_x0, interp_x1});
}

double trilinearInterp(const unsigned x, const unsigned y, const unsigned z,
                       const Bound& x_bound, const Bound& y_bound,
                       const Bound& z_bound, const Array8d& values) {
  Array4d values_x0 = {values[0], values[1], values[2], values[3]};
  double interp_x0 = bilinearInterp(y, z, y_bound, z_bound, values_x0);

  Array4d values_x1 = {values[4], values[5], values[6], values[7]};
  double interp_x1 = bilinearInterp(y, z, y_bound, z_bound, values_x1);

  return linearInterp(x, x_bound, {interp_x0, interp_x1});
}

}  // namespace mdl

}  // namespace cg
