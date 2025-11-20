/**
 * @file instance_generator.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Implementation of `InstanceGenerator`. This class is responsible for
 * generating random instances for input shapes.
 * @version 0.1
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "instance_generator.hpp"

#include <random>
#include <vector>

using std::vector;

namespace cg {

static std::random_device rand_dev;
static std::mt19937 mt_random_engine(rand_dev());

InstanceGenerator::InstanceGenerator() {
  std::random_device rd;
  random_generator = std::mt19937(rd());
  dist = std::uniform_int_distribution<unsigned>(settings.min_size,
                                                 settings.max_size);
}

InstanceGenerator::InstanceGenerator(const unsigned min_size,
                                     const unsigned max_size)
    : settings{min_size, max_size} {
  std::random_device rd;
  random_generator = std::mt19937(rd());
  dist = std::uniform_int_distribution<unsigned>(min_size, max_size);
}

Instance InstanceGenerator::rndInstance(const MatrixChain& chain) {
  Instance instance(chain.size() + 1);
  instance[0] = dist(random_generator);

  for (unsigned i = 0; i < chain.size(); ++i) {
    // If the matrix is dense and not invertible, it is not square.
    instance[i + 1] = (chain[i].isDense() and !chain[i].isInvertible())
                          ? dist(random_generator)
                          : instance[i];
  }
  return instance;
}

vector<Instance> InstanceGenerator::rndInstances(const MatrixChain& chain,
                                                 const unsigned N) {
  vector<Instance> S;
  for (unsigned i = 0U; i < N; i++) {
    S.emplace_back(rndInstance(chain));
  }
  return S;
}

}  // namespace cg
