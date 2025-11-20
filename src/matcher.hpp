/**
 * @file matcher.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Implements a mapping from a `Variant` (pairs of `Feature`) to kernels.
 * @version 0.1
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef MATCHER_H
#define MATCHER_H

#include <unordered_map>
#include <vector>

#include "kernel.hpp"
#include "variant.hpp"

namespace cg {

class Matcher {
 public:
  std::unordered_map<Variant, Kernel*, std::hash<Variant>> map;

  // keeps track of the kernels in the Matcher
  std::vector<Kernel*> kernels;

 public:
  Matcher() = default;

  // Copy ctor
  Matcher(const Matcher& other) : map{other.map}, kernels{other.kernels} {}

  // User-defined ctor from a vector of `Kernel` pointers.
  Matcher(const std::vector<Kernel*>& input_kernels);

  /**
   * @brief Checks whether the `Matcher` is not initialized.
   *
   * @return true/false
   */
  bool isEmpty() const noexcept;

  /**
   * @brief Removes mappings in the `Matcher`.
   *
   */
  void clear() noexcept;

  /**
   * @brief Adds a number of `Kernel`s to the `Matcher`.
   *
   * @param input_kernels vector of `Kernel`.
   */
  void addSetKernels(const std::vector<Kernel*>& input_kernels);

  /**
   * @brief Overloaded operator to access kernels.
   *
   * @param variant   `Variant`: Pair of features in the association.
   * @return Kernel*  Pointer to the `Kernel` that computes the association.
   */
  Kernel* operator[](const Variant& variant) const;

  /**
   * @brief Adds a `Kernel` to the `Matcher`.
   *
   * @param kernel Pointer to `Kernel`.
   */
  void addKernel(Kernel* const kernel);

  /**
   * @brief Semantically better name for operator[].
   *
   * @param variant  `Variant`.
   * @return Kernel* Pointer to `Kernel` that computes the association.
   */
  Kernel* getKernel(const Variant& variant) const;

  /**
   * @brief Instructs all added kernels to load models.
   *
   * Each `Kernel` loads its corresponding model. Models should be in place for
   * this to work.
   *
   */
  void loadModels() const;
};

}  // namespace cg

#endif