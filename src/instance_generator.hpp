/**
 * @file instance_generator.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Implementation of `InstanceGenerator`. This class is responsible for
 * generating random instances for input shapes.
 * @version 0.1
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef INSTANCE_GENERATOR_H
#define INSTANCE_GENERATOR_H

#include <random>

#include "definitions.hpp"

namespace cg {

struct ConfigInstanceGenerator {
  // Default min/max sizes for operands: used when randomly generating instances
  unsigned min_size{2U};
  unsigned max_size{1'000U};

  ConfigInstanceGenerator() = default;

  ConfigInstanceGenerator(const unsigned min_size, const unsigned max_size)
      : min_size{min_size}, max_size{max_size} {}
};

class InstanceGenerator {
 public:
  ConfigInstanceGenerator settings{};
  std::uniform_int_distribution<unsigned> dist{};
  std::mt19937 random_generator{};

  InstanceGenerator();

  InstanceGenerator(const unsigned min_size, const unsigned max_size);

  /**
   * @brief Generates a random instance for the input chain.
   *
   * @param chain     MatrixChain
   * @return Instance
   */
  Instance rndInstance(const MatrixChain& chain);

  /**
   * @brief Generates N random instances for the passed chain.
   *
   * @param chain MatrixChain
   * @param N     unsigned - number of instances to generate.
   * @return std::vector<Instance>
   */
  std::vector<Instance> rndInstances(const MatrixChain& chain,
                                     const unsigned N);
};

}  // namespace cg

#endif