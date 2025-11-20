#include "variant.hpp"

#include <algorithm>
#include <iostream>

#include "features.hpp"
#include "matrix.hpp"

namespace cg {

bool Variant::operator==(const Variant& rhs) const {
  return (left == rhs.left and right == rhs.right);
}

std::ostream& operator<<(std::ostream& os, const Variant& variant) {
  os << variant.left << " " << variant.right;
  return os;
}

std::string to_qualified_string(const Variant& variant) {
  return "{" + to_qualified_string(variant.left) + ", " +
         to_qualified_string(variant.right) + "}";
}

std::vector<Variant> RangeVariant::generateVariants() const {
  std::vector<Variant> variants;
  Variant variant, variant_swap;

  for (const auto& structure_L : A.structure) {
    variant.left.structure = structure_L;
    for (const auto& property_L : A.property) {
      variant.left.property = property_L;
      for (const auto& trans_L : A.trans) {
        variant.left.trans = trans_L;
        for (const auto& inv_L : A.inv) {
          variant.left.inversion = inv_L;
          variant.left.simplify();

          if (variant.left.isLegal() and !variant.left.isIdentity()) {
            for (const auto& structure_R : B.structure) {
              variant.right.structure = structure_R;
              for (const auto& property_R : B.property) {
                variant.right.property = property_R;
                for (const auto& trans_R : B.trans) {
                  variant.right.trans = trans_R;
                  for (const auto& inv_R : B.inv) {
                    variant.right.inversion = inv_R;
                    variant.right.simplify();

                    if (variant.right.isLegal() and
                        !variant.right.isIdentity()) {
                      auto it =
                          std::find(variants.begin(), variants.end(), variant);
                      if (it == variants.end()) {
                        variants.emplace_back(variant);
                        variant_swap = variant;
                        std::swap(variant_swap.left, variant_swap.right);
                        it = std::find(variants.begin(), variants.end(),
                                       variant_swap);
                        if (it == variants.end())
                          variants.emplace_back(variant_swap);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return variants;
}

}  // namespace cg
