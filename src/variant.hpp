/**
 * @file variant.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Implements `Variant` and `RangeVariant`. These classes are used by
 * `Kernel` and `Matcher` to determine the mapping between pairs of `Features`
 * and `Kernels`.
 * @version 0.1
 *
 * @copyright Copyright (c) 2023
 */

#ifndef VARIANT_H
#define VARIANT_H

#include <iostream>

#include "features.hpp"
#include "matrix.hpp"

namespace cg {

/**
 * @brief A `Variant` is a pair of `Features` whose association is supported by
 * some `Kernel`.
 *
 */
class Variant {
 public:
  Features left;
  Features right;

  Variant() = default;

  Variant(const Features& left, const Features& right)
      : left{left}, right{right} {}

  Variant(const Variant& other) = default;

  Variant& operator=(const Variant& rhs) = default;

  bool operator==(const Variant& rhs) const;

  friend std::ostream& operator<<(std::ostream& os, const Variant& variant);

  friend std::string to_qualified_string(const Variant& variant);

  friend class std::hash<Variant>;
};

/**
 * @brief A `RangeVariant` is a compact representation of all the combinations
 * of features a `Kernel` support. The function `generateVariants()` produces
 * the Cartesian product of `A` and `B`.
 *
 */
struct RangeVariant {
  RangeMatrix A, B;

  /**
   * @brief Returns the Cartesian product of `A` and `B`.
   *
   * @return std::vector<Variant>
   */
  std::vector<Variant> generateVariants() const;
};

}  // namespace cg

template <>
struct std::hash<cg::Variant> {
  std::size_t operator()(const cg::Variant& variant) const noexcept {
    std::size_t seed = 0;
    hash_combine(seed, variant.left);
    hash_combine(seed, variant.right);

    return seed;
  }
};

#endif