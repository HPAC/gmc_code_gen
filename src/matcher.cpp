/**
 * @file matcher.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Implements a mapping from a `Variant` (pairs of `Feature`) to kernels.
 * @version 0.1
 *
 * `Kernel`s (found in /src/kernels/) contain the information of the `Variant`s
 * they support. This information is transferred to the `Matcher` when a
 * `Kernel` is added.
 * The current implementation can be improved in the following ways:
 *    - There is no check that an input `Kernel` is already in the `Matcher`.
 *    - If there is no mapping for a pair of `Feature`s, the execution halts. We
 * could resort to a fallback kernel depending on the association.
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "matcher.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>

#include "kernel.hpp"
#include "variant.hpp"

namespace cg {

Matcher::Matcher(const std::vector<Kernel*>& input_kernels) {
  for (const auto& k : input_kernels) {
    this->addKernel(k);
  }
}

bool Matcher::isEmpty() const noexcept { return kernels.empty(); }

void Matcher::clear() noexcept {
  map.clear();
  kernels.clear();
}

void Matcher::addSetKernels(const std::vector<Kernel*>& input_kernels) {
  for (const auto& k : input_kernels) this->addKernel(k);
}

Kernel* Matcher::operator[](const Variant& variant) const {
  try {
    return map.at(variant);
  } catch (const std::out_of_range& e) {
    std::cout << variant << "\n";
    exit(-1);
  }
}

void Matcher::addKernel(Kernel* const kernel) {
  // @warning: should check whether the kernel already exists in the vector.
  auto variants = kernel->getCoveredVariants();
  for (const auto variant : variants) {
    map[variant] = kernel;
  }
  this->kernels.emplace_back(kernel);
}

Kernel* Matcher::getKernel(const Variant& variant) const {
  return this->operator[](variant);
}

void Matcher::loadModels() const {
  for (auto& ker : kernels) ker->loadModel();
}

}  // namespace cg