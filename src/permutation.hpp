/**
 * @file permutation.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Convenience functions used to canonically represent orders of
 * execution for parenthesizations. This is only used in `generator.cpp`.
 * @version 0.1
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef PERMUTATION_H
#define PERMUTATION_H

#include <vector>

namespace cg {

/**
 * @brief Collection of functions that produce the order of execution of a
 * parenthesization in its canonical form. This class is not to be instantiated.
 *
 */
struct PermutationTransformer {
  // Private class. To be used only inside the functions in
  // `PermutationTransformer`.
  struct InfoEntry {
    int left{-1}, right{-1};
    bool computed{false};
  };

  static std::vector<InfoEntry> table_info;

  /**
   * @brief Returns a permutation in its canonical form.
   *
   * This function is the main point of interaction with the whole class.
   *
   * @param perm
   * @return std::vector<unsigned>
   */
  static std::vector<unsigned> canonicalize(const std::vector<unsigned>& perm);

 private:
  // clears `table_info` making it return to the default state.
  static void clearTable();

  static void buildRepresentation(const std::vector<unsigned>& perm);

  static void addDependencies(const unsigned& p);

  static std::vector<unsigned> buildPermutation(
      const std::vector<unsigned>& perm);

  static void buildRecursive(const unsigned p,
                             std::vector<unsigned>& canonical_perm);
};

}  // namespace cg

#endif